// example translated from https://github.com/germannp/yalla/blob/main/examples/passive_growth.cu
// Simulate growing mesenchyme enveloped by epithelium
// mesenchyme proliferation only starts at step 100

#include "../include/kocs.hpp"

enum CellType {
  Mesenchyme,
  Epithelium
};

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_COM_FIXER(com_fixers::NoComFixer)
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 200;
const int steps = 500;
const double dt = 0.2;
const Scalar r_max = 1.0;
const Scalar mean_distance = 0.75;
const Scalar proliferation_rate = 0.006;

int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/passive_growth", r_max);
  auto& positions = sim.get_view<FIELD(Vector, position)>();
  auto& polarities = sim.get_view<FIELD(Polarity, polarity)>();
  View<int> types("cell_types", n_cells);
  View<int> mesenchyme_neighbours("mesenchyme_neighbours", n_cells);
  View<int> epithelium_neighbours("epithelium_neighbours", n_cells);

  sim.set_agent_count(n_cells);
  sim.init_relaxed_sphere(1.0);

  // forces
  auto relu_w_epithelium = PAIRWISE_FORCE(
    if (types(j) == CellType::Mesenchyme)
      Kokkos::atomic_add(&mesenchyme_neighbours(i), 1);
    else
      Kokkos::atomic_add(&epithelium_neighbours(i), 1);

    if (types(i) == types(j))
      ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 0.8f, 2.0f, 1.0f);
    else
      ctx.position.delta += forces::PiecewiseLinear(displacement, distance, 0.8f, 0.9f, 2.0f, 1.0f);
    
    if (types(i) == CellType::Epithelium && types(j) == CellType::Epithelium) {
      auto result = ctx.polarity.self.bending_force(displacement, ctx.polarity.other, distance);
      ctx.position.delta += result.vector * 0.15;
      ctx.polarity.delta += result.polarity * 0.15;
    }
  );

  Kokkos::View<int> counter("counter");
  Kokkos::View<Scalar> rate("proliferation_rate");
  auto proliferate = INIT_FUNC(
    if (types(i) == CellType::Mesenchyme && rng.drand(0.0, 1.0) > rate())
      return;
    else if (types(i) == CellType::Epithelium && epithelium_neighbours(i) > mesenchyme_neighbours(i))
      return;

    int n = Kokkos::atomic_fetch_add(&counter(), 1);
    
    Polarity temp_polarity = Polarity(
      Kokkos::acos(2.0 * rng.drand(0.0, 1.0) - 1),
      rng.drand(0.0, 1.0) * 2.0 * Kokkos::numbers::pi_v<Scalar>
    );
    auto pos_i = Vector(positions(i));
positions(n) = pos_i + mean_distance / 4 * temp_polarity.to_vector3();
    polarities(n) = polarities(i);
    types(n) = types(i);
  );

  // find epithelium
  sim.take_step(dt, relu_w_epithelium);
  sim.init(INIT_FUNC(
    if (mesenchyme_neighbours(i) < 12 * 2) {  // *2 for 2nd order solver
      types(i) = CellType::Epithelium;
      polarities(i) = Polarity(positions(i));
    }
  ));

  sim.write(types, mesenchyme_neighbours, epithelium_neighbours);
  for (int i = 0; i < steps; ++i) {
    mesenchyme_neighbours.deep_copy(0);
    epithelium_neighbours.deep_copy(0);
    sim.take_step(dt, relu_w_epithelium());



    // ensure capacity is high enough to store all possible cells
    if (sim.get_agent_count() * 2 > sim.get_capacity())
      sim.set_capacity(sim.get_agent_count() * 4, types, mesenchyme_neighbours, epithelium_neighbours);

    Kokkos::deep_copy(counter, sim.get_agent_count());
    Kokkos::deep_copy(rate, proliferation_rate * (i > 100));
    sim.init(proliferate());

    auto counter_host = Kokkos::create_mirror_view(counter);
    Kokkos::deep_copy(counter_host, counter);
    sim.set_agent_count(counter_host());

    

    sim.write(types, mesenchyme_neighbours, epithelium_neighbours);
  }
}
