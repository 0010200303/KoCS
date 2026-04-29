// example translated from https://github.com/germannp/yalla/blob/main/examples/sorting.cu

#include "../include/kocs.hpp"

using namespace kocs;
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(DefaultSimulationConfig)

const int n_cells = 100;
const int steps = 300;
const double dt = 0.05;
const Scalar r_min = 0.5;
const Scalar r_max = 1.0;

int main() {
  Simulation<DefaultSimulationConfig> sim(n_cells, "./output/sorting");
  sim.init_random_filled_sphere(1.5);

  Kokkos::View<int*> types("types", n_cells);
  auto types_init = GENERIC_FORCE(unsigned int i) {
    types(i) = (i < n_cells / 2) ? 0 : 1;
  };
  Kokkos::parallel_for("init_types", n_cells, types_init);

  sim.write(types);

  auto differential_adhesion = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Random& rng,
    Scalar& friction,
    Vector& force
  ) {
    if (distance > r_max)
      return;
    
    Scalar strength = (1 + 2 * (j < n_cells / 2)) * (1 + 2 * (i < n_cells / 2));
    Scalar F = 2 * (r_min - distance) * (r_max - distance) + Kokkos::pow(r_max - distance, 2);
    force += strength * displacement * F / distance;
  };

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, differential_adhesion);
    sim.write(types);
  }

  return 0;
}
