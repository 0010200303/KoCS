#include <Kokkos_Core.hpp>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, "positions")
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  Simulation<SimulationConfig> sim(2, "./output/tust");
  sim.init_random_hollow_sphere(2.0);
  sim.write();

  auto generic_force_pos = GENERIC_FORCE(
    unsigned int i,
    Random& rng,
    Vector& force
  ) {
    if (i == 0)
      return;

    force += Vector(0.0, 100.0, 0.0);
  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step(0.001, generic_force_pos);
    sim.write();
  }

  return 0;
}
