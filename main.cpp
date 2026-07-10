#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_PAIR_FINDER(pair_finders::NaiveDelaunay)
  CONFIG_INTEGRATOR(integrators::Euler)
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const unsigned int n_cells = 8;
const double dt = 1.0;
const Scalar r_max = 1.0f;

int main() {
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/main");
  settings.cutoff_distance = r_max;
  settings.link_capacity = n_cells * n_cells * n_cells * n_cells;

  Simulation<SimulationConfig> sim(settings);
  auto& links = sim.get_links();
  sim.init_random_hollow_sphere(1.0);

  DeviceVar<int> link_counter("links", links.get_active_count());
  auto generate_links = PAIRWISE_FORCE(
    int link_n = Kokkos::atomic_add_fetch(link_counter.data(), 1) - 1;
    links(link_n) = Link(i, j);
  );

  sim.take_step(0.0, generate_links);
  links.set_active_count(link_counter);
  sim.write(0.0);

  return 0;
}
