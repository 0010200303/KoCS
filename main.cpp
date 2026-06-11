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

  auto generic = MAKE_GENERIC_FORCE_NAMED({
    view(i) = i;
    f.position.delta += Vector(0.0, 1.0, 0.0);
    f.polarity.delta += Polarity(1.0);
  });

  sim.write();
  for (int i = 0; i < 4; ++i) {
    sim.take_step(dt, generic());
    sim.write();
  }

  view.sync_device_to_host();

  Kokkos::parallel_for(view.get_size(), KOKKOS_LAMBDA(const unsigned int i) {
    Kokkos::printf("%d\n", view(i));
  });
  Kokkos::fence();

  Kokkos::printf("\n");

  for (int i = 0; i < view.get_size(); ++i)
    Kokkos::printf("%d\n", view(i));

  Kokkos::printf("=================================================\n");
  unsigned int prev_size = view.get_size();
  Kokkos::printf("%d\n", view.get_size());
  view.resize(8);
  Kokkos::printf("%d\n", view.get_size());
  Kokkos::printf("=================================================\n");

  for (int i = prev_size; i < view.get_size(); ++i)
    view(i) = 10 + i;
  view.sync_host_to_device();
  Kokkos::fence();

  Kokkos::parallel_for(view.get_size(), KOKKOS_LAMBDA(const unsigned int i) {
    Kokkos::printf("%d\n", view(i));
  });
  Kokkos::fence();

  Kokkos::printf("\n");

  for (int i = 0; i < view.get_size(); ++i)
    Kokkos::printf("%d\n", view(i));

  return 0;
}
