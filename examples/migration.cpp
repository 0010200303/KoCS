// example translated from https://github.com/germannp/yalla/blob/main/examples/migration.cu
// Simulate mono-polar migration

#include "../include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, positions),
    FIELD(Polarity, polarities)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 262;
const int steps = 100;
const double dt = 0.05;
const Scalar r_max = 1.0;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/migration", r_max);
  auto& positions_view = sim.get_view<FIELD(Vector, positions)>();
  auto& polarities_view = sim.get_view<FIELD(Polarity, polarities)>();
  auto initial_conditions = INIT_FUNC() {
    if (i == 0) {
      positions_view(i) = Vector{0.0};
      polarities_view(i) = Polarity{0.0, 0.01};
    }
  };

  sim.init_relaxed_cuboid(Vector{-1.5, -1.5, 0.0}, Vector{1.5, 1.5, 10.0}, 2000, initial_conditions);
  sim.write();

  auto relu_w_migration = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position), PAIRWISE_REF(Polarity, polarity)) {
    Scalar F = Kokkos::fmax(0.7 - distance, 0) * 2 - Kokkos::fmax(distance - 0.8, 0);
    position.delta += displacement * F / distance + polarity.self.migration_force(displacement, polarity.other, distance);
  };

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, relu_w_migration);
    sim.write();
  }

  return 0;
}
