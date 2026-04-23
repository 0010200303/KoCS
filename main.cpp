#include <Kokkos_Core.hpp>
#include <tuple>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, "positions"),
    FIELD(float, "masses")
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

template <std::size_t I = 0, typename... Forces>
KOKKOS_INLINE_FUNCTION void invoke_merged_force(
  const std::tuple<Forces...>& forces,
  unsigned int i,
  unsigned int j,
  const Vector& displacement,
  const Scalar& distance,
  Vector& force,
  float& mass
) {
  if constexpr (I < sizeof...(Forces)) {
    std::get<I>(forces)(i, j, displacement, distance, force, mass);
    invoke_merged_force<I + 1>(forces, i, j, displacement, distance, force, mass);
  }
}

template<typename... Forces>
struct MergedForce {
  std::tuple<Forces...> forces;

  KOKKOS_INLINE_FUNCTION void operator()(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force,
    float& mass
  ) const {
    invoke_merged_force(forces, i, j, displacement, distance, force, mass);
  }
};

template<typename... Forces>
auto merge(Forces... forces) {
  return MergedForce<Forces...>{std::make_tuple(forces...)} | detail::pairwise_force;
}



int main() {
  Simulation<SimulationConfig> sim(128);
  auto& positions = sim.get_view<FIELD(Vector, "positions")>();
  auto& masses = sim.get_view<FIELD(float, "masses")>();

  initializers::RandomHollowSphere<SimulationConfig> init(2.0, positions);
  sim.init(init);

  Writer<SimulationConfig> writer("./output/tust");
  writer.write(0, sim);

  auto generic_force = GENERIC_FORCE(unsigned int i, Random& rng, Vector& force, float& mass) {
    mass += rng.drand(-100.0, 100.0);
  };

  const float stiffness = 0.1f;
  auto pairwise_force_x = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force,
    float& mass
  ) {
    force.x() += displacement.x() * (stiffness - distance) / distance;
  };

  auto pairwise_force_y = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force,
    float& mass
  ) {
    force.y() += displacement.y() * (stiffness - distance) / distance;
  };

  auto pairwise_force_z = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force,
    float& mass
  ) {
    force.z() += displacement.z() * (stiffness - distance) / distance;
  };

  auto pairwise_force_merged = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force,
    float& mass
  ) {
    pairwise_force_x(i, j, displacement, distance, force, mass);
    pairwise_force_y(i, j, displacement, distance, force, mass);
    pairwise_force_z(i, j, displacement, distance, force, mass);
  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step_rng(0.001, generic_force);
    // sim.take_step(0.001, pairwise_force_x);
    // sim.take_step(0.001, pairwise_force_y);
    // sim.take_step(0.001, pairwise_force_z);

    // sim.take_step(0.001, pairwise_force_x, pairwise_force_y, pairwise_force_z);

    sim.take_step(0.001, merge(pairwise_force_x, pairwise_force_y, pairwise_force_z));

    writer.write(i, sim);
  }
  
  return 0;
}
