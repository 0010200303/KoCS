// Proof of concept, actual implementation will greatly reduce boilerplate

#include "../../include/kocs.hpp"

#include <vector>
#include <array>

enum class CellType {
  Serosa,
  Anchor,
  Pole,
  Embryo
};

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_PAIR_FINDER(pair_finders::BinnedGabriel)
  CONFIG_COM_FIXER(com_fixers::NoComFixer)
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const double dt = 0.001;
const int steps = 100;
const int steps_per_reduction = 500;
const Scalar r_max = 1.0f;

// spatial acceleration structure for surface lookup
struct SurfaceBin {
  int begin;
  int count;
};

KOKKOS_INLINE_FUNCTION
int flatten_bin_index(
  int ix, int iy, int iz,
  int nx, int ny, int nz
) {
  return ix + nx * (iy + ny * iz);
}

KOKKOS_INLINE_FUNCTION
int get_bin_index(
  const Vector& p,
  const Vector& min_bound,
  Scalar bin_size,
  int nx, int ny, int nz
) {
  int ix = static_cast<int>((p.x() - min_bound.x()) / bin_size);
  int iy = static_cast<int>((p.y() - min_bound.y()) / bin_size);
  int iz = static_cast<int>((p.z() - min_bound.z()) / bin_size);

  ix = Kokkos::max(0, Kokkos::min(nx - 1, ix));
  iy = Kokkos::max(0, Kokkos::min(ny - 1, iy));
  iz = Kokkos::max(0, Kokkos::min(nz - 1, iz));

  return flatten_bin_index(ix, iy, iz, nx, ny, nz);
}

int main() {
  /*
  * The cells need a suface acting as an outward force to pretend the shape to collapse
  * Therefore the surface and the normal vectors for the surface points is loaded
  * The surface has about 10 times the number of points then cells in the system
  */
  // read surface
  HighFive::File surface_file("./examples/tribolium/surface.h5", HighFive::File::AccessMode::ReadOnly);
  HighFive::DataSet surface_dataset = surface_file.getDataSet("POINTS");
  std::vector<std::array<Scalar, 3>> surface_read_data;
  surface_dataset.read(surface_read_data);

  // read main cells
  HighFive::File serosa_file("./examples/tribolium/serosa.h5", HighFive::File::AccessMode::ReadOnly);
  HighFive::DataSet serosa_points_dataset = serosa_file.getDataSet("POINTS");

  std::vector<std::array<Scalar, 3>> serosa_points;
  serosa_points_dataset.read(serosa_points);

  HighFive::DataSet serosa_types_dataset = serosa_file.getDataSet("cell_type");
  std::vector<int> serosa_types;
  serosa_types_dataset.read(serosa_types);

  // creating simulation
  Simulation<SimulationConfig> sim(serosa_points.size(), "./output/serosa", r_max);

  auto positions_view = sim.get_view<FIELD(Vector, positions)>();
  View<int> cell_types_view("cell_types", sim.get_agent_count());

  // surface points
  View<Vector> grid_view("grid_data", surface_read_data.size());
  auto grid_host_view = Kokkos::create_mirror_view(grid_view);
  for (int i = 0; i < surface_read_data.size(); ++i)
    grid_host_view(i) = 15 * Vector(surface_read_data[i]);
  Kokkos::deep_copy(grid_view, grid_host_view);

  // surface normals
  View<Vector> normals_view("normals_data", surface_read_data.size());
  auto normals_host_view = Kokkos::create_mirror_view(normals_view);
  for (int i = 0; i < surface_read_data.size(); ++i)
    normals_host_view(i) = Vector(surface_read_data[i]);
  Kokkos::deep_copy(normals_view, normals_host_view);

  // initial positions
  auto positions_host_view = Kokkos::create_mirror_view(positions_view);
  for (int i = 0; i < serosa_points.size(); ++i)
    positions_host_view(i) = Vector(serosa_points[i]);
  Kokkos::deep_copy(positions_view, positions_host_view);

  // poles and types
  std::vector<Vector> pole_positions;
  auto cell_types_host_view = Kokkos::create_mirror_view(cell_types_view);

  for (int i = 0; i < serosa_types.size(); ++i) {
    int type = serosa_types[i];
    cell_types_host_view(i) = type;

    if (type == static_cast<int>(CellType::Pole))
      pole_positions.push_back(Vector(serosa_points[i]));
  }

  Kokkos::deep_copy(cell_types_view, cell_types_host_view);

  // build spatial bins for surface lookup
  const Scalar bin_size = 1.0f;
  Vector min_bound = grid_host_view(0);
  Vector max_bound = grid_host_view(0);
  for (int i = 1; i < grid_view.size(); ++i) {
    Vector p = grid_host_view(i);

    min_bound.x() = std::min(min_bound.x(), p.x());
    min_bound.y() = std::min(min_bound.y(), p.y());
    min_bound.z() = std::min(min_bound.z(), p.z());

    max_bound.x() = std::max(max_bound.x(), p.x());
    max_bound.y() = std::max(max_bound.y(), p.y());
    max_bound.z() = std::max(max_bound.z(), p.z());
  }

  Vector extent = max_bound - min_bound;
  const int nx = static_cast<int>(extent.x() / bin_size) + 1;
  const int ny = static_cast<int>(extent.y() / bin_size) + 1;
  const int nz = static_cast<int>(extent.z() / bin_size) + 1;
  const int total_bins = nx * ny * nz;

  // sort points into bins
  std::vector<std::vector<int>> temp_bins(total_bins);
  for (int i = 0; i < grid_view.size(); ++i) {
    int bin_index = get_bin_index(grid_host_view(i), min_bound, bin_size, nx, ny, nz);
    temp_bins[bin_index].push_back(i);
  }

  // flatten bins into compact arrays
  View<int> sorted_indices("sorted_indices", grid_view.size());
  View<SurfaceBin> bins("surface_bins", total_bins);

  auto sorted_host = Kokkos::create_mirror_view(sorted_indices);
  auto bins_host = Kokkos::create_mirror_view(bins);

  int offset = 0;
  for (int b = 0; b < total_bins; ++b) {
    bins_host(b).begin = offset;
    bins_host(b).count = temp_bins[b].size();

    for (int idx : temp_bins[b])
      sorted_host(offset++) = idx;
  }

  Kokkos::deep_copy(sorted_indices, sorted_host);
  Kokkos::deep_copy(bins, bins_host);

  // pairwise cell forces
  auto force_between_cells = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position)) {
    const CellType this_type = static_cast<CellType>(cell_types_view(i));
    const CellType other_type = static_cast<CellType>(cell_types_view(j));

    if (this_type == CellType::Anchor || this_type == CellType::Pole) {
      // do nothing
    }
    else if (this_type == CellType::Serosa) {
      position.delta += (Kokkos::exp(-distance) - 0.5f) * displacement;
    }
    else {
      if (other_type == CellType::Pole)
        position.delta += (-Kokkos::exp(-distance)) * displacement;
      position.delta += (Kokkos::exp(-distance) - 0.4f) * displacement;
    }

    // drag
    if (this_type != CellType::Anchor && this_type != CellType::Pole)
      friction += 1.0;
  };

  // surface force using spatial bins
  auto system_forces = GENERIC_FORCE(GENERIC_REF(Vector, position)) {
    int center_bin = get_bin_index(position.self, min_bound, bin_size, nx, ny, nz);

    int rem = center_bin % (nx * ny);
    int cx = rem % nx;
    int cy = rem / nx;
    int cz = center_bin / (nx * ny);

    // search neighboring bins
    float min_distance = 1e9;
    int nearest_grid_point = -1;
    for (int dz = -1; dz <= 1; ++dz) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          int bx = cx + dx;
          int by = cy + dy;
          int bz = cz + dz;
          if (bx < 0 || bx >= nx || by < 0 || by >= ny || bz < 0 || bz >= nz)
            continue;

          int bin_index = flatten_bin_index(bx, by, bz, nx, ny, nz);
          SurfaceBin bin = bins(bin_index);

          for (int k = 0; k < bin.count; ++k) {
            int point = sorted_indices(bin.begin + k);
            float distance = position.self.distance_to_squared(grid_view(point));

            if (distance < min_distance) {
              min_distance = distance;
              nearest_grid_point = point;
            }
          }
        }
      }
    }

    if (nearest_grid_point >= 0) {
      Vector relative_position = position.self - grid_view(nearest_grid_point);
      Vector normal = normals_view(nearest_grid_point);

      Scalar F = -12.0f * relative_position.dot(normal);
      position.delta += F * normal;
    }
  };

  sim.write(cell_types_view);

  // simulation loop
  for (int i = 0; i < steps; ++i) {
    Kokkos::printf("%d step %d\n", sim.get_agent_count(), i);

    for (int j = 0; j < steps_per_reduction; ++j)
      sim.take_step(dt, force_between_cells, system_forces);

    // remove embryo cells near poles
    Kokkos::deep_copy(positions_host_view, positions_view);
    Kokkos::deep_copy(cell_types_host_view, cell_types_view);

    int new_cell_count = 0;
    for (int k = 0; k < sim.get_agent_count(); ++k) {
      Vector position_k = positions_host_view(k);
      int cell_type_k = cell_types_host_view(k);

      bool keep = true;
      if (cell_type_k == static_cast<int>(CellType::Embryo)) {
        for (const auto& pole_position : pole_positions) {
          if (position_k.distance_to_squared(pole_position) < 0.25f) {
            keep = false;
            break;
          }
        }
      }

      if (keep == true) {
        positions_host_view(new_cell_count) = position_k;
        cell_types_host_view(new_cell_count) = cell_type_k;
        ++new_cell_count;
      }
    }
    Kokkos::deep_copy(positions_view, positions_host_view);
    Kokkos::deep_copy(cell_types_view, cell_types_host_view);
    sim.set_agent_count(new_cell_count);

    sim.write(cell_types_view);
  }

  return 0;
}
