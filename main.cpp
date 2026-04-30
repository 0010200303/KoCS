#include <Kokkos_Core.hpp>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_PAIR_FINDER(pair_finders::NaiveGabriel)
  CONFIG_FIELDS(
    FIELD(Vector, positions),
    FIELD(Vector, velocities),
    FIELD(Polarity, polarities)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  const int n_cells = 100;
  const double dt = 0.1;
  const float r_max = 1.2f;

  // settings
  const float polarity_strength = 0.3f;
  const float walker_probability = 0.5f;

  Simulation<SimulationConfig> sim(n_cells, "./output/rotation");
  auto& positions_view = sim.get_view<FIELD(Vector, positions)>();
  auto& velocities_view = sim.get_view<FIELD(Vector, velocities)>();
  auto& polarities_view = sim.get_view<FIELD(Polarity, polarities)>();
  Kokkos::View<int*> types("types", n_cells);

  auto initial_conditions = GENERIC_FORCE() {
    polarities_view(i) = Polarity(positions_view(i));
    velocities_view(i) = Vector(rng.frand(0.1), rng.frand(0.1), rng.frand(0.1));
    types(i) = rng.frand(1.0) < walker_probability;
  };
  sim.init_random_filled_sphere(2.0);
  sim.init(initial_conditions);

  auto mechanical_interactions = PAIRWISE_FORCE(
    PAIRWISE_REF(Vector, position),
    PAIRWISE_REF(Vector, velocity),
    PAIRWISE_REF(Polarity, polarity)
  ) {
    // TODO: move r_max into pair finder for performance
    if (distance > r_max)
      return;

    // mechanical interactions
    Scalar F = fmax(0.7 - distance, 0) * 2 - fmax(distance - 0.8, 0);
    position.delta += displacement * F / distance;

    // bending force
    auto bending_force = polarity.i.bending_force(displacement, polarity.j, distance);
    position.delta += bending_force.vector * polarity_strength;
    polarity.delta += bending_force.polarity * polarity_strength;
  };

  sim.write(types);
  const int relaxation_steps = 2000;
  for (int i = 0; i <= relaxation_steps; ++i) {
    sim.take_step(dt, mechanical_interactions);
    sim.write(types);
  }
  sim.write(types);
  


  // for (int i = 1; i <= 10; ++i) {
  //   sim.take_step(0.001, mechanical_interactions);
  //   sim.write(types);
  // }

  return 0;
}
