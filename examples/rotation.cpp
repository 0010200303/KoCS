// simulate organoid rotation

#include "../include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_PAIR_FINDER(pair_finders::TustGabriel)
  CONFIG_FIELDS(
    FIELD(Vector, positions),
    FIELD(Vector, velocities),
    FIELD(Polarity, polarities)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main(int argc, char* argv[]) {
  const double dt = 0.1;
  const float r_max = 1.2f;
  const int t_max = 337;
  const int steps_per_30_min = 120;

  // argparse settings
  int n_cells;
  float const_1;
  float const_2;
  float polarity_strength;
  float D;
  float walker_probability;
  uint64_t seed;
  std::string output_path;

  bool ok = Arguments("organoid rotation")
    .add_argument("-n", "--n_cells", n_cells, 100, "number of cells")
    .add_argument("-c1", "--const_1", const_1, 0.5f, "determines how strongly a cell adjusts it's \
active migration in response to mechanical force F (interaction with neighbouring cells)")
    .add_argument("-c2", "--const_2", const_2, 0.1f, "defines the target speed of active cell migration")
    .add_argument("-p", "--polarity_strength", polarity_strength, 0.3f, "strength of cell polarity")
    .add_argument("-D", "--stochasticity", D, 0.05f, "stochasticity")
    .add_argument("-P", "--walker_probability", walker_probability, 0.5f, "probability of a cell to be a walker")
    .add_argument("-s", "--seed", seed, 2807, "seed of the whole simulation")
    .add_argument("-o", "--output", output_path)
    .parse(argc, argv);
  
  if (ok == false)
    return 1;
  
  if (output_path.empty() == true)
    output_path = "./output/out_" + std::to_string(n_cells) + "_cells_c1_" + std::to_string(const_1) +
      "_c2_" + std::to_string(const_2) + "_D_" + std::to_string(D) + "_walkers_" + std::to_string(walker_probability);

  const float sqrt_stochastic = Kokkos::sqrt(D) * Kokkos::sqrt(dt);

  Simulation<SimulationConfig> sim(n_cells, output_path, r_max, seed);
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
