// example translated from https://github.com/germannp/yalla/blob/main/examples/polarization.cu
// Simulate randomly migrating cell

#include <numbers>
#include "../include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, positions),
    FIELD(Polarity, polarities)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 200;
const int steps = 300;
const double dt = 0.025;
const Scalar r_max = 1.0;
const Scalar r_min = 0.6;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/polarization", r_max);
  auto& polarities_view = sim.get_view<FIELD(Polarity, polarities)>();
  auto initial_conditions = INIT_FUNC() {
    polarities_view(i) = Polarity{
      Kokkos::acos(2.0 * rng.drand(0.0, 1.0) - 1.0),
      2.0 * Scalar(std::numbers::pi) * rng.drand(0.0, 1.0)
    };
  };

  sim.init_random_filled_sphere(2.0, initial_conditions);
  sim.write();

  auto relu_w_migration = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position), PAIRWISE_REF(Polarity, polarity)) {
    Scalar F = 2.0 * (r_min - distance) * (r_max - distance) + Kokkos::pow(r_max - distance, 2);
    position.delta += displacement * F / distance;
    
    // U_Pol = -(Σ(n_i . n_j)^2) / 2
    polarity.delta += polarity.self.bidirectional_polarization_force(polarity.other);
  };

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, relu_w_migration);
    sim.write();
  }

  return 0;
}
