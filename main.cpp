#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, positions),
    FIELD(Polarity, polarities)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  const int n_cells = 100;
  const double dt = 0.1;
  const float r_max = 1.2f;
  const int steps = 100;

  Simulation<SimulationConfig> sim(n_cells, "./output/rotation", r_max);
  auto& positions_view = sim.get_view<FIELD(Vector, positions)>();
  auto& polarities_view = sim.get_view<FIELD(Polarity, polarities)>();
  View<int> types("types", n_cells);

  auto initial_conditions = GENERIC_FORCE() {
    polarities_view(i) = Polarity(positions_view(i));
    types(i) = i < n_cells / 2;
  };
  sim.init_random_filled_sphere(2.0, initial_conditions);

  auto generic_force = GENERIC_FORCE(
    GENERIC_REF(Vector, position),
    GENERIC_REF(Polarity, polarity)
  ) {

  };

  auto pairwise_force = PAIRWISE_FORCE(
    PAIRWISE_REF(Vector, position),
    PAIRWISE_REF(Polarity, polarity)
  ) {

  };

  for (int i = 0; i <= steps; ++i) {
    sim.take_step(dt, generic_force, pairwise_force);
    sim.write(types);
  }

  return 0;
}
