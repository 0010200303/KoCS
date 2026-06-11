#include "../../include/kocs.hpp"

enum CellType {
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
EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

const double dt = 0.001;
const int steps = 100;
const int steps_per_reduction = 500;
Scalar r_max = 1.0f;
Scalar r_min = 0.0f;

int main(int argc, char* argv[]) {
  // argparse settings
  std::string output_path;
  std::string input_path;
  std::string surface_path;
  bool ok = Arguments("tribolium")
    .add_argument("-rmax", "--rmax", r_max, 1.0f)
    .add_argument("-rmin", "--rmin", r_min, 0.0f)
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
  std::cout << "===========================================================" << std::endl;

  /*
  * The cells need a suface acting as an outward force to pretend the shape to collapse
  * Therefore the surface and the normal vectors for the surface points is loaded
  * The surface has about 10 times the number of points then cells in the system
  */

  // prepare readers
  io::HDF5_Reader serosa_reader(input_path);
  io::HDF5_Reader surface_reader(surface_path);

  // create simulation
  PairFinder::Settings pair_finder_settings;
  pair_finder_settings.rebuild_every_n = 20;
  Simulation<SimulationConfig> sim(
    serosa_reader.get_dataset_dimensions("POINTS")[0],
    output_path,
    r_max,
    pair_finder_settings
  );

  // read initial positions
  auto positions_view = sim.get_view<FIELD(Vector, position)>();
  auto positions_host = Kokkos::create_mirror_view(positions_view);
  serosa_reader.read_dataset("POINTS", positions_view, positions_host);

  // read cell types
  Kokkos::View<int*> cell_types_view("cell_types", sim.get_agent_count());
  auto cell_types_host = Kokkos::create_mirror_view(cell_types_view);
  serosa_reader.read_dataset("cell_type", cell_types_view, cell_types_host);

  // read normals
  Kokkos::View<Vector*> normals_view("normals_data", surface_reader.get_dataset_dimensions("POINTS")[0]);
  auto normals_host = Kokkos::create_mirror_view(normals_view);
  surface_reader.read_dataset("POINTS", normals_view, normals_host);

  // surface points
  Kokkos::View<Vector*> grid_view("grid_data", normals_view.extent(0));
  auto grid_host = Kokkos::create_mirror_view(grid_view);
  for (int i = 0; i < normals_view.extent(0); ++i)
    grid_host(i) = 15 * normals_host(i);
  Kokkos::deep_copy(grid_view, grid_host);



  sim.write(cell_types_host);

  // simulation loop
  acceleration::Grid<Vector> grid(grid_view);
  grid.rebuild();
  auto system_forces = MAKE_GENERIC_FORCE_NAMED({
    int nearest_grid_point_idx = grid.get_nearest_point_index(f.position.self);
    if (nearest_grid_point_idx >= 0) {
      Vector relative_position = f.position.self - grid_view(nearest_grid_point_idx);
      Vector normal = normals_view(nearest_grid_point_idx);

      Scalar F = -12.0f * relative_position.dot(normal);
      f.position.delta += F * normal;
    }
  });

  auto force_between_cells = MAKE_PAIRWISE_FORCE_NAMED({
    const int this_type = cell_types_view(i);
    const Scalar exp_neg_d = Kokkos::exp(-distance);

    if (this_type == CellType::Anchor || this_type == CellType::Pole) {
      // do nothing
    }
    else if (this_type == CellType::Serosa) {
      f.position.delta += (exp_neg_d - 0.5f) * displacement;
    }
    else {
      if (cell_types_view(j) == CellType::Pole)
        f.position.delta += (-exp_neg_d) * displacement;
      f.position.delta += (exp_neg_d - 0.4f) * displacement;
    }

    // drag
    if (this_type != CellType::Anchor && this_type != CellType::Pole)
      friction += 1.0;
  });

  // pre-allocate vectors for host-side cell removal
  std::vector<Vector> pole_positions;
  pole_positions.reserve(sim.get_agent_count() / 10);

  for (int i = 0; i < steps; ++i) {
    std::cout << sim.get_agent_count() << " step " << i << "\n";

    for (int j = 0; j < steps_per_reduction; ++j)
      sim.take_step(dt, force_between_cells(), system_forces());

    // remove embryo cells near poles
    Kokkos::deep_copy(positions_host, positions_view);
    Kokkos::deep_copy(cell_types_host, cell_types_view);

    pole_positions.clear();
    for (int k = 0; k < sim.get_agent_count(); ++k) {
      if (cell_types_host(k) == CellType::Pole)
        pole_positions.push_back(positions_host(k));
    }

    int new_cell_count = 0;
    for (int k = 0; k < sim.get_agent_count(); ++k) {
      Vector position_k = positions_host(k);
      int cell_type_k = cell_types_host(k);

      bool keep = true;
      if (cell_type_k == CellType::Embryo) {
        for (const auto& pole_position : pole_positions) {
          if (position_k.distance_to_squared(pole_position) < 0.25f) {
            keep = false;
            break;
          }
        }
      }

      if (keep == true) {
        positions_host(new_cell_count) = position_k;
        cell_types_host(new_cell_count) = cell_type_k;
        ++new_cell_count;
      }
    }
    Kokkos::deep_copy(positions_view, positions_host);
    Kokkos::deep_copy(cell_types_view, cell_types_host);
    sim.set_agent_count(new_cell_count);

    sim.write(cell_types_host);
  }

  return 0;
}
