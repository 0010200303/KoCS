// example translated from https://github.com/germannp/yalla/blob/main/examples/springs.cu

#include "../include/kocs.hpp"

using namespace kocs;
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(DefaultSimulationConfig)

const int n_bodies = 1024;
const int steps = 100;
const float dt = 0.0001;
const float L_0 = 0.5f;

int main() {
  auto pairwise_force = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force
  ) {
    force += displacement * (L_0 - distance) / distance;
  };

  Simulation<DefaultSimulationConfig> sim(n_bodies);
  auto& positions = sim.get_view<FIELD(Vector, "positions")>();

  initializers::RandomHollowSphere<DefaultSimulationConfig> init(2.0, positions);
  sim.init(init);

  // writes to output/springs.h5 & output/springs.xmf
  Writer<DefaultSimulationConfig> writer("./output/springs");
  writer.write(0, sim); // write full simulation data

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, pairwise_force);
    writer.write(i, sim);
  }

  return 0;
}
