// simulate organoid rotation

#include "../include/kocs.hpp"

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
  const float D = 0.05f;
  const float walker_probability = 0.5f;

  const float sqrt_stochastic = Kokkos::sqrt(D) * Kokkos::sqrt(dt);

  Simulation<SimulationConfig> sim(n_cells, "./output/rotation", r_max);
  auto& positions_view = sim.get_view<FIELD(Vector, positions)>();
  auto& velocities_view = sim.get_view<FIELD(Vector, velocities)>();
  auto& polarities_view = sim.get_view<FIELD(Polarity, polarities)>();
  View<bool> types("types", n_cells);

  auto initial_conditions = INIT_FUNC() {
    polarities_view(i) = Polarity(positions_view(i));
    velocities_view(i) = Vector(rng.normal(0, 0.1), rng.normal(0, 0.1), rng.normal(0, 0.1));
    types(i) = rng.frand(1.0) < walker_probability;
  };
  sim.init_random_filled_sphere(2.0, initial_conditions);

  auto pairwise_mechanical_interactions = PAIRWISE_FORCE(
    PAIRWISE_REF(Vector, position),
    PAIRWISE_REF(Vector, velocity),
    PAIRWISE_REF(Polarity, polarity)
  ) {
    // mechanical interactions
    position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 0.8f);

    // bending force
    auto bending_force = polarity.self.bending_force(displacement, polarity.other, distance);
    position.delta += bending_force.vector * polarity_strength;
    polarity.delta += bending_force.polarity * polarity_strength;
  };

  auto self_interactions = GENERIC_FORCE(
    GENERIC_REF(Vector, position),
    GENERIC_REF(Vector, velocity),
    GENERIC_REF(Polarity, polarity)
  ) {
    // orthogonalize velocity to polarity
    Vector polarity_vector = polarity.self.to_vector3();
    Scalar dot = velocity.self.dot(polarity_vector);
    velocity.delta -= 0.5 * dot * polarity_vector;

    // stochastic force on position
    position.delta += Vector(rng.normal(0, sqrt_stochastic), rng.normal(0, sqrt_stochastic), rng.normal(0, sqrt_stochastic));

    // calculate force contributions on velocity
    Scalar magnitude = velocity.self.length();
    velocity.delta += (const_2 - magnitude) * (velocity.self / magnitude) - velocity.self;

    // direction dependent drift
    position.delta += velocity.self * types(i);
  };

  auto pairwise_interactions = PAIRWISE_FORCE(
    PAIRWISE_REF(Vector, position),
    PAIRWISE_REF(Vector, velocity),
    PAIRWISE_REF(Polarity, polarity)
  ) {
    // mechanical interactions
    Vector F = forces::PiecewiseLinear(displacement, distance, 0.7f, 0.8f);
    position.delta += F;

    // bending force
    auto bending_force = polarity.self.bending_force(displacement, polarity.other, distance);
    position.delta += bending_force.vector * polarity_strength;
    polarity.delta += bending_force.polarity * polarity_strength;

    // "mechanotransduction" impact of mechanical forces on velocity
    velocity.delta += const_1 * F;
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
