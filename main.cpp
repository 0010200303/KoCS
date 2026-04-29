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
  Simulation<SimulationConfig> sim(3, "./output/tust");
  sim.init_random_hollow_sphere(2.0);
  sim.write();

  auto generic_force_pos = GENERIC_FORCE(
    unsigned int i,
    Random& rng,
    Vector& force
  ) {
    if (i == 0)
      return;

    if (i == 1)
      force += Vector(0.0, 100.0, 0.0);
    else
      force += Vector(0.0, 000.0, 100.0);
  };

  auto friction = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Random& rng,
    Scalar& friction,
    Vector& force
  ) {
    if (i != 0)
      return;

    friction += 1.0;
    // friction += 0.0000001;
  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step(0.001, generic_force_pos, friction);
    sim.write();
  }

  return 0;
}
