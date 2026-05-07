// Simulate cell sorting by forces strength

#include "../include/kocs.hpp"

using namespace kocs;
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(DefaultSimulationConfig)

const int n_cells = 100;
const int steps = 300;
const double dt = 0.05;
const Scalar r_max = 1.0;

int main() {
  Simulation<DefaultSimulationConfig> sim(n_cells, "./output/sorting", r_max);
  View<bool> types("types", n_cells);
  auto types_init = INIT_FUNC() {
    types(i) = (i < n_cells / 2) ? 0 : 1;
  };
  sim.init_random_filled_sphere(1.0, types_init);
  sim.write(types);

  auto differential_adhesion = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position)) {
    // mixed: tau_{1,1} = tau_{2,2} = tau_{1,2}
    // Scalar tau = Scalar(2.0);

    // separated: tau_{1,1} = tau_{2,2} > tau_{1,2}
    Scalar tau = (types(i) == types(j)) ? Scalar(2.0) : Scalar(1.0);

    // engulfed: tau_{1,1} > tau_{2,2} = tau_{1,2}
    // Scalar tau = (types(i) == true || types(j) == true) ? Scalar(2.0) : Scalar(1.0);
    position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 0.8f, 1.0f, tau);
  };

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, differential_adhesion);
    sim.write(types);
  }

  return 0;
}
