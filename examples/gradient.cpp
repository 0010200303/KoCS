// example translated from https://github.com/germannp/yalla/blob/main/examples/gradient.cu

#include "../include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, positions),
    FIELD(Scalar, gradients)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 61;
const int steps = 200;
const double dt = 0.005;
const Scalar r_max = 1.0;
const Scalar D = 10;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/gradient");
  sim.init_regular_hexagon(0.75);

  auto gradient_view = sim.get_view<FIELD(Scalar, gradients)>();
  auto gradient_init = GENERIC_FORCE() {
    if (i == 11)
      gradient_view(i) = Scalar(1);
  };
  sim.init(gradient_init);
  sim.write();

  auto diffusion = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position), PAIRWISE_REF(Scalar, gradient)) {
    if (distance > r_max || i == 11)
      return;
    gradient.delta += -(gradient.i - gradient.j) * D;
  };

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, diffusion);
    sim.write();
  }

  return 0;
}
