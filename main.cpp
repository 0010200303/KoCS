#include <iostream>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  // using Fields = std::tuple<
  //   Field<VectorView, "positions">,
  //   Field<VectorView, "velocities">
  // >;
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  Simulation<SimulationConfig> sim(128);
  auto& positions = sim.get_view<"positions">();

  initializer::RandomHollowSphere<SimulationConfig> init(2.0f, positions);
  sim.init(init);

  Writer<SimulationConfig> writer("./output/tust");
  writer.write(0, sim);

  auto force = KOKKOS_LAMBDA(
    const int i,
    const int j,
    // auto& rng,
    Vector& position
  ) {
    position += Vector(28.0f, 0.0f, 7.0f);
    // position.x() = rng.drand(0.0f, 28.0f);
  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step(force, 0.00001f);
    writer.write(i, sim);
  }
  
  return 0;
}
