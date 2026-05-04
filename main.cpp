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
  const int t_max = 337;
  const int steps_per_30_min = 120;

  // settings
  const float const_1 = 0.5f;
  const float const_2 = 0.1f;
  const float polarity_strength = 0.3f;
  const float D = 0.25f;
  const float walker_probability = 0.2f;

  const float sqrt_stochastic = Kokkos::sqrt(D) * Kokkos::sqrt(dt);

  Simulation<SimulationConfig> sim(n_cells, "./output/rotation");
  auto& positions_view = sim.get_view<FIELD(Vector, positions)>();
  auto& velocities_view = sim.get_view<FIELD(Vector, velocities)>();
  auto& polarities_view = sim.get_view<FIELD(Polarity, polarities)>();
  Kokkos::View<int*> types("types", n_cells);

  auto initial_conditions = GENERIC_FORCE() {
    polarities_view(i) = Polarity(positions_view(i));
    velocities_view(i) = Vector(rng.normal(0, 0.1), rng.normal(0, 0.1), rng.normal(0, 0.1));
    types(i) = rng.frand(1.0) < walker_probability;
  };
  sim.init_random_filled_sphere(2.0);
  sim.init(initial_conditions);

  auto pairwise_mechanical_interactions = PAIRWISE_FORCE(
    PAIRWISE_REF(Vector, position),
    PAIRWISE_REF(Vector, velocity),
    PAIRWISE_REF(Polarity, polarity)
  ) {
    // TODO: move r_max into pair finder for performance
    if (distance > r_max)
      return;

    // mechanical interactions
    Scalar F = Kokkos::fmax(0.7 - distance, 0) * 2 - Kokkos::fmax(distance - 0.8, 0);
    position.delta += displacement * F / distance;

    // bending force
    auto bending_force = polarity.i.bending_force(displacement, polarity.j, distance);
    position.delta += bending_force.vector * polarity_strength;
    polarity.delta += bending_force.polarity * polarity_strength;
  };

  auto self_interactions = GENERIC_FORCE(
    GENERIC_REF(Vector, position),
    GENERIC_REF(Vector, velocity),
    GENERIC_REF(Polarity, polarity)
  ) {
    // orthogonalize velocity to polarity
    Vector polarity_vector = polarity.i.to_vector3();
    Scalar dot = velocity.i.dot(polarity_vector);
    velocity.delta -= 0.5 * dot * polarity_vector;

    // stochastic force on position
    position.delta += Vector(rng.normal(0, sqrt_stochastic), rng.normal(0, sqrt_stochastic), rng.normal(0, sqrt_stochastic));

    // calculate force contributions on velocity
    Scalar magnitude = velocity.i.length();
    velocity.delta += (const_2 - magnitude) * (velocity.i / magnitude) - velocity.i;

    // direction dependent drift
    position.delta += velocity.i * types(i);
  };

  auto pairwise_interactions = PAIRWISE_FORCE(
    PAIRWISE_REF(Vector, position),
    PAIRWISE_REF(Vector, velocity),
    PAIRWISE_REF(Polarity, polarity)
  ) {
    if (distance > r_max)
      return;

    // mechanical interactions
    Scalar F = Kokkos::fmax(0.7 - distance, 0) * 2 - Kokkos::fmax(distance - 0.8, 0);
    position.delta += displacement * F / distance;

    // bending force
    auto bending_force = polarity.i.bending_force(displacement, polarity.j, distance);
    position.delta += bending_force.vector * polarity_strength;
    polarity.delta += bending_force.polarity * polarity_strength;

    // "mechanotransduction" impact of mechanical forces on velocity
    velocity.delta += const_1 * F * displacement / distance;
  };

  const int relaxation_steps = 2000;
  for (int i = 0; i <= relaxation_steps; ++i)
    sim.take_step(dt, pairwise_mechanical_interactions);
  sim.write(types);

  for (int i = 0; i <= t_max; ++i) {
    for (int j = 0; j < steps_per_30_min; ++j) {
      sim.take_step(dt, self_interactions, pairwise_interactions);
    }
    sim.write(types);
  }

  return 0;
}
