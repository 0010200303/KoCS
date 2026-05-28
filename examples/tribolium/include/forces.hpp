#include "../../../include/kocs.hpp"

using namespace kocs;

enum class CellType {
  Serosa,
  Anchor,
  Pole,
  Embryo
};

template<typename SimulationConfig>
struct SystemForces {
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

  // spatial acceleration structure for surface lookup
  struct SurfaceBin {
    int begin;
    int count;
  };

  SystemForces(
    View<Vector> grid_view_,
    View<Vector> normals_view_
  ) : grid_view(grid_view_)
    , normals_view(normals_view_)
  {
    auto grid_host_view = Kokkos::create_mirror_view(grid_view);
    Kokkos::deep_copy(grid_host_view, grid_view);

    min_bound = grid_host_view(0);
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
    nx = static_cast<int>(extent.x() / bin_size) + 1;
    ny = static_cast<int>(extent.y() / bin_size) + 1;
    nz = static_cast<int>(extent.z() / bin_size) + 1;
    const int total_bins = nx * ny * nz;

    // sort points into bins
    std::vector<std::vector<int>> temp_bins(total_bins);
    for (int i = 0; i < grid_view.size(); ++i) {
      int bin_index = get_bin_index(grid_host_view(i), min_bound, bin_size, nx, ny, nz);
      temp_bins[bin_index].push_back(i);
    }

    // flatten bins into compact arrays
    sorted_indices = View<int>("sorted_indices", grid_view.size());
    bins = View<SurfaceBin>("surface_bins", total_bins);

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
  }

  View<Vector> grid_view;
  View<Vector> normals_view;
  View<int> sorted_indices;
  View<SurfaceBin> bins;

  const Scalar bin_size = 1.0f;
  Vector min_bound;

  int nx;
  int ny;
  int nz;

  KOKKOS_INLINE_FUNCTION
  int flatten_bin_index(
    const int ix, const int iy, const int iz,
    const int nx, const int ny, const int nz
  ) const {
    return ix + nx * (iy + ny * iz);
  }

  KOKKOS_INLINE_FUNCTION
  int get_bin_index(
    const Vector& p,
    const Vector& min_bound,
    const Scalar bin_size,
    const int nx, const int ny, const int nz
  ) const {
    int ix = static_cast<int>((p.x() - min_bound.x()) / bin_size);
    int iy = static_cast<int>((p.y() - min_bound.y()) / bin_size);
    int iz = static_cast<int>((p.z() - min_bound.z()) / bin_size);

    ix = Kokkos::max(0, Kokkos::min(nx - 1, ix));
    iy = Kokkos::max(0, Kokkos::min(ny - 1, iy));
    iz = Kokkos::max(0, Kokkos::min(nz - 1, iz));

    return flatten_bin_index(ix, iy, iz, nx, ny, nz);
  }

  GENERIC_FORCE_OP(GENERIC_REF(Vector, position)) {
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
  }
};



template<typename SimulationConfig>
struct ForceBetweenCellsCustom {
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

  ForceBetweenCellsCustom(View<int> cell_types_view_) : cell_types_view(cell_types_view_) { }

  View<int> cell_types_view;

  PAIRWISE_FORCE_OP(PAIRWISE_REF(Vector, position)) {
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
  }
};

template<typename SimulationConfig>
struct ForceBetweenCellsMorse {
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

  ForceBetweenCellsMorse(View<int> cell_types_view_) : cell_types_view(cell_types_view_) { }

  View<int> cell_types_view;

  PAIRWISE_FORCE_OP(PAIRWISE_REF(Vector, position)) {
    const CellType this_type = static_cast<CellType>(cell_types_view(i));
    const CellType other_type = static_cast<CellType>(cell_types_view(j));

    if (this_type == CellType::Anchor || this_type == CellType::Pole) {
      // do nothing
    }
    else if (this_type == CellType::Serosa) {
      position.delta += forces::Morse(displacement, distance, 1.0f, 2.0f);
    }
    else {
      if (other_type == CellType::Pole)
        position.delta += position.delta += forces::Morse(displacement, distance, 1.0f, 2.0f);
      position.delta += position.delta += forces::Morse(displacement, distance, 1.0f, 2.0f);
    }

    // drag
    if (this_type != CellType::Anchor && this_type != CellType::Pole)
      friction += 1.0;
  }
};
