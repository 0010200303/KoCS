#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    (Vector, position),
    (Polarity, polarity)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const unsigned int n_cells = 4;
const double dt = 1.0;
const Scalar r_max = 1.0f;




int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/main", r_max);
  View<int> view("view", sim.get_capacity());
  sim.init_line();

  auto generic = GENERIC_FORCE(
    view(i) = i;
    ctx.position.delta += Vector(0.0, 1.0, 0.0);
    ctx.polarity.delta += Polarity(1.0);
  );

  sim.write();
  for (int i = 0; i < 4; ++i) {
    sim.take_step(dt, generic());
    sim.write();
  }
  view.sync_device_to_host();

  Kokkos::parallel_for(sim.get_agent_count(), KOKKOS_LAMBDA(const unsigned int i) {
    Kokkos::printf("%d\n", view(i));
  });
  Kokkos::fence();

  Kokkos::printf("\n");

  for (int i = 0; i < sim.get_agent_count(); ++i)
    Kokkos::printf("%d\n", view(i));

  Kokkos::printf("=================================================\n");
  unsigned int prev_size = sim.get_capacity();
  Kokkos::printf("%d\n", sim.get_capacity());
  sim.set_capacity(8);
  sim.set_agent_count(8);
  view.resize(8);
  Kokkos::printf("%d\n", sim.get_capacity());
  Kokkos::printf("=================================================\n");

  for (int i = prev_size; i < sim.get_agent_count(); ++i)
    view(i) = 10 + i;
  view.sync_host_to_device();
  Kokkos::fence();

  Kokkos::parallel_for(sim.get_agent_count(), KOKKOS_LAMBDA(const unsigned int i) {
    Kokkos::printf("%d\n", view(i));
  });
  Kokkos::fence();

  Kokkos::printf("\n");

  for (int i = 0; i < sim.get_agent_count(); ++i)
    Kokkos::printf("%d\n", view(i));

  return 0;
}
