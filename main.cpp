#include <Kokkos_Core.hpp>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, positions),
    FIELD(Vector, velocities),
    FIELD(Polarity, polarities)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  Simulation<SimulationConfig> sim(3, "./output/tust");
  sim.init_random_filled_sphere(2.0);
  sim.write();

  auto interactions_mech = PAIRWISE_FORCE(
    PAIRWISE_REF(Vector, position),
    PAIRWISE_REF(Vector, velocitiy),
    PAIRWISE_REF(Polarity, polarity)
  ) {

  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step(0.001, interactions_mech);
    sim.write();
  }

  return 0;
}
