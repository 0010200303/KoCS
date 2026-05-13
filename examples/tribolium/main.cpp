// Proof of concept, actual implementation will greatly reduce boilerplate

#include "../../include/kocs.hpp"

enum class CellType {
  Serosa,
  Anchor,
  Pole,
  Embryo
};

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_PAIR_FINDER(pair_finders::NaiveGabriel)
  CONFIG_COM_FIXER(com_fixers::NoComFixer)
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const double dt = 0.001;
const int steps = 1;
const int steps_per_reduction = 500;
const Scalar r_max = 1.0f;
const Scalar r_min = 0.5f;

int main() {
  /*
  * The cells need a suface acting as an outward force to pretend the shape to collapse
  * Therefore the surface and the normal vectors for the surface points is loaded
  * The surface has about 10 times the number of points then cells in the system
  */
  // reading
  HighFive::File surface_file("./examples/tribolium/surface.h5", HighFive::File::AccessMode::ReadOnly);
  HighFive::DataSet surface_dataset = surface_file.getDataSet("POINTS");
  std::vector<std::array<Scalar, 3>> surface_read_data;
  surface_dataset.read(surface_read_data);

  HighFive::File serosa_file("./examples/tribolium/serosa.h5", HighFive::File::AccessMode::ReadOnly);

  HighFive::DataSet serosa_points_dataset = serosa_file.getDataSet("POINTS");
  std::vector<std::array<Scalar, 3>> serosa_points;
  serosa_points_dataset.read(serosa_points);

  HighFive::DataSet serosa_types_dataset = serosa_file.getDataSet("cell_type");
  std::vector<int> serosa_types;
  serosa_types_dataset.read(serosa_types);

  int n_cells = serosa_points.size();

  Simulation<SimulationConfig> sim(n_cells, "./output/serosa", r_max);
  auto positions_view = sim.get_view<FIELD(Vector, positions)>();
  View<int> cell_types_view("cell_types", n_cells);

  // assign grid
  View<Vector> grid_view("grid_data", surface_read_data.size());
  auto grid_host_view = Kokkos::create_mirror_view(grid_view);
  for (int i = 0; i < surface_read_data.size(); ++i)
    grid_host_view(i) = 15 * Vector(surface_read_data[i]);
  Kokkos::deep_copy(grid_view, grid_host_view);

  // assign normals
  View<Vector> normals_view("normals_data", surface_read_data.size());
  auto normals_host_view = Kokkos::create_mirror_view(normals_view);
  for (int i = 0; i < surface_read_data.size(); ++i)
    normals_host_view(i) = Vector(surface_read_data[i]);
  Kokkos::deep_copy(normals_view, normals_host_view);

  // assign positions
  auto positions_host_view = Kokkos::create_mirror_view(positions_view);
  for (int i = 0; i < serosa_points.size(); ++i)
    positions_host_view(i) = Vector(serosa_points[i]);
  Kokkos::deep_copy(positions_view, positions_host_view);

  // assign types
  std::vector<int> pole_indices;

  auto cell_types_host_view = Kokkos::create_mirror_view(cell_types_view);
  for (int i = 0; i < serosa_types.size(); ++i) {
    int type = serosa_types[i];
    cell_types_host_view(i) = type;

    if (type == static_cast<int>(CellType::Pole))
      pole_indices.push_back(i);
  }
  Kokkos::deep_copy(cell_types_view, cell_types_host_view);

  sim.write(cell_types_view);



  // define forces
  /*
  * A force is acting between to cells if they are neighbours.
  * Therefore two cells are difined as neighbours, if the distance between them is less then
  * a predifined maximum "r_max" and the two cells are the closest possible cells to their midpoint.
  * The force acting between the two cells is based on the distance between them.
  */
  auto force_between_cells = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position)) {
    const CellType this_type = static_cast<CellType>(cell_types_view(i));
    const CellType other_type = static_cast<CellType>(cell_types_view(j));

    if (this_type == CellType::Anchor || this_type == CellType::Pole) {
      // do nothing
    }
    else if (this_type == CellType::Serosa) {
      position.delta += (Kokkos::exp(-distance) - 0.5) * displacement;
    }
    else {
      if (other_type == CellType::Pole)
        position.delta += -Kokkos::exp(-distance) * displacement;
      position.delta += (Kokkos::exp(-distance) - 0.4) * displacement;
    }

    // drag
    if (this_type != CellType::Anchor && this_type != CellType::Pole)
      friction += 1.0;
  };

  auto system_forces = GENERIC_FORCE(GENERIC_REF(Vector, position)) {
    float distance_to_grid = position.self.distance_to_squared(grid_view(0));

    float min_distance = distance_to_grid;
    int nearest_grid_point = 0;
    for (int point = 0; point < grid_view.size(); ++point) {
      float temp_distance = position.self.distance_to_squared(grid_view(point));
      if (temp_distance < min_distance) {
        min_distance = temp_distance;
        nearest_grid_point = point;
      }
    }
    min_distance = Kokkos::sqrt(min_distance);

    Vector relative_position = position.self - grid_view(nearest_grid_point);
    Vector normal = normals_view(nearest_grid_point);
    float F = -12 * relative_position.dot(normal);

    position.delta += F * normal;
  };



  // simulation loop
  for (int i = 0; i < steps; ++i) {
    for (int j = 0; j < steps_per_reduction; ++j) {
      sim.take_step(dt, force_between_cells, system_forces);
      sim.write(cell_types_view);

      // TODO: remove as it should already be done by inside write
      Kokkos::deep_copy(positions_host_view, positions_view);
      Kokkos::deep_copy(cell_types_host_view, cell_types_view);

      std::vector<unsigned int> cells_near_pole;
      for (unsigned int k = 0; k < n_cells; ++k) {
        if (static_cast<CellType>(cell_types_host_view(k)) != CellType::Embryo)
          continue;

        for (const auto& pole_index : pole_indices) {
          if (positions_host_view(k).distance_to_squared(positions_host_view(pole_index)) < 0.25f) {
            cells_near_pole.push_back(k);
            break;
          }
        }
      }

      for (unsigned int k = 1; k <= cells_near_pole.size(); ++k) {
        const int dst = cells_near_pole[cells_near_pole.size() - k];
        const int src = n_cells - k;

        positions_host_view(dst) = positions_host_view(src);
        cell_types_host_view(dst) = cell_types_host_view(src);
      }
      Kokkos::deep_copy(positions_view, positions_host_view);
      Kokkos::deep_copy(cell_types_view, cell_types_host_view);

      n_cells -= cells_near_pole.size();
      sim.set_agent_count(n_cells);

      // TODO: think about resizing view occasionally
      // positions_host_view.resize(n_cells);
      // cell_types_host_view.resize(n_cells);
    }
  }

  return 0;
}
