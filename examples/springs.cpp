// example translated from https://github.com/germannp/yalla/blob/main/examples/springs.cu

#include "../include/kocs.hpp"

using namespace kocs;
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(DefaultSimulationConfig)

const int n_bodies = 1024;
const int steps = 100;
const float dt = 0.001;
const float L_0 = 0.5f;

int main() {
  auto pairwise_force = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Random& rng,
    Scalar& friction,
    Vector& force
  ) {
    force += displacement * (L_0 - distance) / distance;
  };

  Simulation<DefaultSimulationConfig> sim(n_bodies, "./output/springs");
  sim.init_random_filled_sphere(3.0);
  sim.write();

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, pairwise_force);
    sim.write();
  }

  return 0;
}
