#include <iostream>

#include <Kokkos_Core.hpp>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  using Fields = FieldList<
    Field<Vector*, "positions">,
    Field<float*, "masses">
  >;
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  Simulation<SimulationConfig> sim(16);
  // auto& positions = get<Field<Vector*, "positions">>(sim.storage);
  // auto& masses = get<Field<float*, "masses">>(sim.storage);
  auto& positions = sim.get_view<Field<Vector*, "positions">>();
  auto& masses = sim.get_view<Field<float*, "masses">>();

  auto tust = KOKKOS_LAMBDA(unsigned int i, Vector& pos) {
    pos = Vector(28.0f);
    pos[1] = 0.0f;
    pos.z() = 7.0f;
  };

  Kokkos::parallel_for("tust", positions.extent(0) - 1, KOKKOS_LAMBDA(unsigned int i) {
    tust(i, positions(i));
  });



  auto host = Kokkos::create_mirror_view(positions);
  Kokkos::deep_copy(host, positions);

  for (int i = 0; i < positions.extent(0); ++i) {
    std::cout << host(i).data[0] << " " << host(i).data[1] << " " << host(i).data[2] << std::endl;
  }







  // initializer::Line<SimulationConfig> init(positions);
  // sim.init(init);

  // Writer<SimulationConfig> writer("./output/tust");
  // writer.write(0, sim);

  // auto force = KOKKOS_LAMBDA(
  //   const int i,
  //   const int j,
  //   // auto& rng,
  //   Vector& position,
  //   Vector& velocity
  // ) {
  //   position += Vector(28.0f, 0.0f, 7.0f);
  //   // position.x() = rng.drand(0.0f, 28.0f);
  // };

  // // integrators::Euler integrator(sim);
  // // integrator(force, 0.0001);

  // for (int i = 1; i <= 10; ++i) {
  //   sim.take_step(force, 0.001);
  // //   // integrator(force, 0.0001);
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
