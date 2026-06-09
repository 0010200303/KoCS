#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  const int n_cells = 10;
  const float r_max = 1.0f;

  Simulation<SimulationConfig> sim(n_cells, "./output/rotation", r_max);
  auto& _positions = sim.get_view<FIELD(Vector, positions)>();
  // auto positions_view = track_view(_positions);

  sim.init_random_filled_sphere(2.0);

  // sim.init(INIT_FUNC() {
  //   Kokkos::printf("%f %f %f\n", positions_view(i).x(), positions_view(i).y(), positions_view(i).z());
  // });
  // Kokkos::fence();

  // Kokkos::printf("=================================================\n");

  // for (int i = 0; i < _positions.extent(0); ++i)
  //   Kokkos::printf("%f %f %f\n", positions_view(i).x(), positions_view(i).y(), positions_view(i).z());



  View<int> view("view", 4);
  Kokkos::parallel_for(view.get_size(), KOKKOS_LAMBDA(const unsigned int i) {
    view(i) = i;
  });
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
