#include <iostream>

#include <Kokkos_Core.hpp>
#include <cmath>
#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  using Fields = FieldList<
    Field<Vector*, "positions">,
    Field<float*, "masses">
  >;
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  Simulation<SimulationConfig> sim(16);
  auto& positions = sim.get_view<Field<Vector*, "positions">>();
  auto& masses = sim.get_view<Field<float*, "masses">>();

  initializer::Line<SimulationConfig> init(positions);
  sim.init(init);

  Writer<SimulationConfig> writer("./output/tust");
  writer.write(0, sim);

  auto generic_force = GENERIC_FORCE(unsigned int i, Vector& force, float& mass) {
    const float stiffness = 1.0f;
    force += -stiffness * positions(i);
    mass += 1.0;
  };

  auto pairwise_force = PAIRWISE_FORCE(unsigned int i, unsigned int j, Vector& force, float& mass) {
    const float stiffness = 0.5f;

    Vector r = positions(i) - positions(j);
    float dist = r.length();

    force += r * (stiffness - dist) / dist;
  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step(0.01, generic_force);
    sim.take_step(0.01, pairwise_force);
    writer.write(i, sim);
  }
  
  return 0;
}
