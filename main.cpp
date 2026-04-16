#include <iostream>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  // using Fields = std::tuple<
  //   Field<VectorView, "positions">,
  //   Field<VectorView, "velocities">
  // >;
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  Simulation<SimulationConfig> sim(3);
  auto& positions = sim.get_view<"positions">();

  initializer::Line<SimulationConfig> init(positions);
  sim.init(init);

  auto host_view = Kokkos::create_mirror_view(sim.get_view<"positions">());
  Kokkos::deep_copy(host_view, sim.get_view<"positions">());
  std::cout << host_view(3).x() << std::endl;

  //Writer<SimulationConfig> writer("./output/tust");
  //writer.write(0, sim);

  auto force = KOKKOS_LAMBDA(
    const int i,
    const int j,
    // auto& rng,
    Vector& position
  ) {
    position += Vector(28.0f, 0.0f, 7.0f);
    // position.x() = rng.drand(0.0f, 28.0f);
  };

  // integrators::Euler integrator(sim);
  // integrator(force, 0.0001);

  // for (int i = 1; i <= 10; ++i) {
  //   sim.take_step(force, 0.00001);
  //   // integrator(force, 0.0001);
  //   writer.write(i, sim);
  // }
  
  return 0;
}



// struct Storage;
// struct LocalValues;
// struct PairFinder;
// struct Heun;

// Storage state;
// PairFinder pair_finder;
// Heun heun;

// template<typename ForceFn>
// void take_step(ForceFn force, const double dt = 1.0) {
//   Storage local_state{};

//   heun_(force, dt, state, local_state);
// }

// void heun_() {
//   pair_finder.for_each(state, local_state, force);

//   euler(state, local_state, dt);

//   pair_finder.for_each(state, local_state, force);

//   heun(state, local_state, dt);
// }
