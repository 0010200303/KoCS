#include <iostream>

#include <Kokkos_Core.hpp>

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

  auto tust = GENERIC_FORCE(unsigned int i, Vector& force, float& mass) {
    const float stiffness = 0.25f;
    const Vector& pos = positions(i);

    force += -stiffness * positions(i);
    mass += 1.0;
  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step(1.0, tust);
    writer.write(i, sim);
  }
  
  return 0;
}
