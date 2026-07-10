#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_PAIR_FINDER(pair_finders::BinnedAllPairs)
  CONFIG_INTEGRATOR(integrators::Euler)
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const unsigned int n_cells = 16384;
const double dt = 1.0;
const Scalar r_max = 1.0f;

int main() {
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/main");
  settings.cutoff_distance = r_max;

  Simulation<SimulationConfig> sim(settings);
  sim.init_random_filled_sphere(16.0);

  auto pairwise_force = PAIRWISE_FORCE(
    ctx.position.delta += forces::Spring(displacement, distance, 0.5f);
  );

  for (int i = 0; i < 100; ++i) {
    sim.take_step(0.0, pairwise_force);
    sim.write(0.0);
  }

  return 0;
}
