// Proof of concept, actual implementation will greatly reduce boilerplate
#include <vector>
#include <array>

#include "../../include/kocs.hpp"
#include "include/forces.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_PAIR_FINDER(pair_finders::BinnedGabriel)
  CONFIG_COM_FIXER(com_fixers::NoComFixer)
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const double dt = 0.001;
const int steps = 100;
const int steps_per_reduction = 500;
Scalar r_max = 1.0f;
Scalar r_min = 0.0f;

int main(int argc, char* argv[]) {
  // argparse settings
  std::string function;
  std::string output_path;
  std::string input_path;
  std::string surface_path;
  bool ok = Arguments("tribolium")
    .add_argument("-rmax", "--rmax", r_max, 1.0f)
    .add_argument("-rmin", "--rmin", r_min, 0.0f)
    .add_argument("-f", "--function", function, "custom", "function for force evaluation",
      "custom", "morse")  // only allowed choices
    .add_argument("-o", "--output", output_path, "./output/serosa")
    .add_argument("-i", "--input", input_path, "./examples/tribolium/initial_data/serosa.h5")
    .add_argument("-s", "--surface", surface_path, "./examples/tribolium/initial_data/surface.h5")
    .parse(argc, argv);
  
  if (ok == false)
    return 1;
  std::cout << "==================== SIMULATION CONFIG ====================" << "\n";
  std::cout << "\tr_max: " << r_max << "\n";
  std::cout << "\tr_min: " << r_min << "\n";
  std::cout << "\toutput path: " << output_path << "\n";
  std::cout << "\tinput path: " << input_path << "\n";
  std::cout << "\tsurface path: " << surface_path << "\n";
  std::cout << "\tfunction used: " << function << "\n";
  std::cout << "===========================================================" << std::endl;

  /*
  * The cells need a suface acting as an outward force to pretend the shape to collapse
  * Therefore the surface and the normal vectors for the surface points is loaded
  * The surface has about 10 times the number of points then cells in the system
  */
  // read surface
  HighFive::File surface_file(surface_path, HighFive::File::AccessMode::ReadOnly);
  HighFive::DataSet surface_dataset = surface_file.getDataSet("POINTS");
  std::vector<std::array<Scalar, 3>> surface_read_data;
  surface_dataset.read(surface_read_data);

  // read main cells
  HighFive::File serosa_file(input_path, HighFive::File::AccessMode::ReadOnly);
  HighFive::DataSet serosa_points_dataset = serosa_file.getDataSet("POINTS");

  std::vector<std::array<Scalar, 3>> serosa_points;
  serosa_points_dataset.read(serosa_points);

  HighFive::DataSet serosa_types_dataset = serosa_file.getDataSet("cell_type");
  std::vector<int> serosa_types;
  serosa_types_dataset.read(serosa_types);

  // creating simulation
  Simulation<SimulationConfig> sim(serosa_points.size(), output_path, r_max);

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



  sim.write(cell_types_view);

  // simulation loop
  auto system_forces = SystemForces<SimulationConfig>(grid_view, normals_view);
  for (int i = 0; i < steps; ++i) {
    Kokkos::printf("%d step %d\n", sim.get_agent_count(), i);

    for (int j = 0; j < steps_per_reduction; ++j) {
      // choose force
      if (function == "custom")
        sim.take_step(dt, ForceBetweenCellsCustom<SimulationConfig>(cell_types_view), system_forces);
      else if (function == "morse")
        sim.take_step(dt, ForceBetweenCellsMorse<SimulationConfig>(cell_types_view), system_forces);
    }

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
