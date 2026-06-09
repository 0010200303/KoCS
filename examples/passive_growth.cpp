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
    FIELD(Vector, positions),
    FIELD(Polarity, polarities)
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const int n_cells = 200;
const int steps = 500;
const double dt = 0.2;
const Scalar r_max = 1.0;
const Scalar mean_distance = 0.75;
const Scalar proliferation_rate = 0.006;



template<typename T>
struct TrackedView {
    Kokkos::View<uintptr_t> data_ptr;   // 0D view, stores device address

    TrackedView() : data_ptr("tracked_view") {}

    KOKKOS_INLINE_FUNCTION
    T& operator()(int i) const {
        return reinterpret_cast<T*>(data_ptr())[i];
    }

    KOKKOS_INLINE_FUNCTION
    T* data() const {
        return reinterpret_cast<T*>(data_ptr());
    }

    // subscript for convenience
    KOKKOS_INLINE_FUNCTION
    T& operator[](int i) const { return (*this)(i); }
};

// Factory — creates a TrackedView from a Kokkos View
template<typename T>
TrackedView<T> track_view(Kokkos::View<T*>& view) {
    TrackedView<T> tv;
    auto host = Kokkos::create_mirror_view(tv.data_ptr);
    host() = reinterpret_cast<uintptr_t>(view.data());
    Kokkos::deep_copy(tv.data_ptr, host);
    return tv;
}

// Update — refresh the stored pointer after resize
template<typename T>
void update_tracked(TrackedView<T>& tv, Kokkos::View<T*>& view) {
    auto host = Kokkos::create_mirror_view(tv.data_ptr);
    Kokkos::deep_copy(host, tv.data_ptr);
    host() = reinterpret_cast<uintptr_t>(view.data());
    Kokkos::deep_copy(tv.data_ptr, host);
}



int main() {
  Simulation<SimulationConfig> sim(n_cells, "./output/passive_growth", r_max);
  auto& _positions = sim.get_view<FIELD(Vector, positions)>();
  auto& _polarities = sim.get_view<FIELD(Polarity, polarities)>();
  View<int> _types("cell_types", n_cells);
  View<int> _mesenchyme_neighbours("mesenchyme_neighbours", n_cells);
  View<int> _epithelium_neighbours("epithelium_neighbours", n_cells);

  auto positions = track_view(_positions);
  auto polarities = track_view(_polarities);
  auto types = track_view(_types);
  auto mesenchyme_neighbours = track_view(_mesenchyme_neighbours);
  auto epithelium_neighbours = track_view(_epithelium_neighbours);

  sim.set_agent_count(n_cells);
  sim.init_relaxed_sphere(1.0);

  // forces
  auto relu_w_epithelium = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position), PAIRWISE_REF(Polarity, polarity)) {
    if (types(j) == CellType::Mesenchyme)
      Kokkos::atomic_add(&mesenchyme_neighbours(i), 1);
    else
      Kokkos::atomic_add(&epithelium_neighbours(i), 1);

    if (types(i) == types(j))
      position.delta += forces::PiecewiseLinear(displacement, distance, 0.7f, 0.8f, 2.0f, 1.0f);
    else
      position.delta += forces::PiecewiseLinear(displacement, distance, 0.8f, 0.9f, 2.0f, 1.0f);
    
    if (types(i) == CellType::Epithelium && types(j) == CellType::Epithelium) {
      auto result = polarity.self.bending_force(displacement, polarity.other, distance);
      position.delta += result.vector * 0.15;
      polarity.delta += result.polarity * 0.15;
    }
  };

  Kokkos::View<int> counter("counter");
  Kokkos::View<Scalar> rate("proliferation_rate");
  auto proliferate = INIT_FUNC() {
    if (types(i) == CellType::Mesenchyme && rng.drand(0.0, 1.0) > rate())
      return;
    else if (types(i) == CellType::Epithelium && epithelium_neighbours(i) > mesenchyme_neighbours(i))
      return;

    int n = Kokkos::atomic_fetch_add(&counter(), 1);
    
    Polarity temp_polarity = Polarity(
      Kokkos::acos(2.0 * rng.drand(0.0, 1.0) - 1),
      rng.drand(0.0, 1.0) * 2.0 * Kokkos::numbers::pi_v<Scalar>
    );
    positions(n) = positions(i) + mean_distance / 4 * temp_polarity.to_vector3();
    polarities(n) = polarities(i);
    types(n) = types(i);
  };

  // find epithelium
  sim.take_step(dt, relu_w_epithelium);
  sim.init(INIT_FUNC() {
    if (mesenchyme_neighbours(i) < 12 * 2) {  // *2 for 2nd order solver
      types(i) = CellType::Epithelium;
      polarities(i) = Polarity(positions(i));
    }
  });

  sim.write(_types, _mesenchyme_neighbours, _epithelium_neighbours);
  for (int i = 0; i < steps; ++i) {
    Kokkos::deep_copy(_mesenchyme_neighbours, 0);
    Kokkos::deep_copy(_epithelium_neighbours, 0);
    sim.take_step(dt, relu_w_epithelium);



    // ensure capacity is high enough to store all possible cells
    if (sim.get_agent_count() * 2 > sim.get_capacity()) {
      sim.set_capacity(sim.get_agent_count() * 4, _types, _mesenchyme_neighbours, _epithelium_neighbours);
      update_tracked(positions, _positions);
      update_tracked(polarities, _polarities);
      update_tracked(types, _types);
      update_tracked(mesenchyme_neighbours, _mesenchyme_neighbours);
      update_tracked(epithelium_neighbours, _epithelium_neighbours);
    }

    Kokkos::deep_copy(counter, sim.get_agent_count());
    Kokkos::deep_copy(rate, proliferation_rate * (i > 100));
    sim.init(proliferate);

    auto counter_host = Kokkos::create_mirror_view(counter);
    Kokkos::deep_copy(counter_host, counter);
    sim.set_agent_count(counter_host());

    

    sim.write(_types, _mesenchyme_neighbours, _epithelium_neighbours);
  }
}
