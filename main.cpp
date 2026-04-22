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
  Simulation<SimulationConfig> sim(16);
  auto& positions = sim.get_view<FIELD(Vector, "positions")>();
  auto& masses = sim.get_view<FIELD(float, "masses")>();

  initializer::Line<SimulationConfig> init(positions);
  sim.init(init);

  Writer<SimulationConfig> writer("./output/tust");
  writer.write(0, sim);

  auto generic_force = GENERIC_FORCE(unsigned int i, Random& rng, Vector& force, float& mass) {
    mass += rng.drand(-10.0, 10.0);
  };

  auto pairwise_force = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Random& rng,
    Vector& force,
    float& mass
  ) {
    force += Vector(rng.rand(-100.0, 100.0));
  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step_rng(0.01, generic_force);
    sim.take_step_rng(0.01, pairwise_force);
    writer.write(i, sim);
  }
  
  return 0;
}
