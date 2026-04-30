#include <Kokkos_Core.hpp>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, positions)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  Simulation<SimulationConfig> sim(3, "./output/tust");
  sim.init_random_filled_sphere(2.0);
  sim.write();

  auto generic_force_pos = GENERIC_FORCE(GENERIC_REF(Vector, position)) {

  };

  auto friction = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position)) {

  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step(0.001, generic_force_pos, friction);
    sim.write();
  }

  return 0;
}
