#include <Kokkos_Core.hpp>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, "positions"),
    FIELD(float, "masses")
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
    detail::GenericFieldRef<Vector> position,
    detail::GenericFieldRef<float> mass
  ) {
    if (i == 0)
      return;

    if (i == 1)
      position.delta += Vector(0.0, 100.0, 0.0);
    else
      position.delta += Vector(0.0, -100.0, 0.0);
  };

  auto friction = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Random& rng,
    Scalar& friction,
    auto position,
    detail::PairwiseFieldRef<float> mass
  ) {
    friction += 1.0;
  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step(0.001, generic_force_pos, friction);
    sim.write();
  }

  return 0;
}
