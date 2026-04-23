#include <Kokkos_Core.hpp>

#include <tuple>
#include <type_traits>
#include <utility>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, "positions"),
    FIELD(float, "masses")
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

template<typename Tag, typename... Forces>
struct Merger;

template<typename Tag>
struct Merger<Tag> {
  Merger() = default;

  using tag = Tag;

  template<typename... Args>
  KOKKOS_INLINE_FUNCTION
  void operator()(Args&&...) const { }
};

template<typename Tag, typename FirstForce, typename... RestForces>
struct Merger<Tag, FirstForce, RestForces...> : Merger<Tag, RestForces...> {
  using base_type = Merger<Tag, RestForces...>;

  FirstForce force;

  KOKKOS_INLINE_FUNCTION
  Merger() = default;

  KOKKOS_INLINE_FUNCTION
  Merger(FirstForce first, RestForces... rest)
    : base_type(rest...)
    , force(first) { }

  template<typename... Args>
  KOKKOS_INLINE_FUNCTION
  void operator()(Args&&... args) const {
    force(static_cast<Args&&>(args)...);
    static_cast<const base_type&>(*this)(static_cast<Args&&>(args)...);
  }
};

template<typename Tag, typename... Forces>
Merger(Tag, Forces...) -> Merger<Tag, Forces...>;

template<typename Tag, typename Force>
auto collect_tagged_force(Force&& force) {
  using decayed = std::decay_t<Force>;

  if constexpr (std::is_same_v<typename decayed::tag, Tag>) {
    return std::tuple<decayed>(std::forward<Force>(force));
  } else {
    return std::tuple<>{};
  }
}

template<typename Tag, typename... Forces>
auto merge_for_tag(Forces&&... forces) {
  auto selected = std::tuple_cat(
    collect_tagged_force<Tag>(std::forward<Forces>(forces))...
  );

  return std::apply([](auto&&... tagged_forces) {
    return Merger<Tag, std::decay_t<decltype(tagged_forces)>...>{
      std::forward<decltype(tagged_forces)>(tagged_forces)...
    };
  }, selected);
}

template<typename... Forces>
auto merge(Forces&&... forces) {
  return std::make_tuple(
    merge_for_tag<detail::GenericForceTag>(std::forward<Forces>(forces)...),
    merge_for_tag<detail::PairwiseForceTag>(std::forward<Forces>(forces)...)
  );
}

int main() {
  Simulation<SimulationConfig> sim(128);
  auto& positions = sim.get_view<FIELD(Vector, "positions")>();
  auto& masses = sim.get_view<FIELD(float, "masses")>();

  initializers::RandomHollowSphere<SimulationConfig> init(2.0, positions);
  sim.init(init);

  Writer<SimulationConfig> writer("./output/tust");
  writer.write(0, sim);

  auto generic_force_mass = GENERIC_FORCE(unsigned int i, Vector& force, float& mass) {
    mass += 1.0;
  };

  auto generic_force_pos = GENERIC_FORCE(unsigned int i, Vector& force, float& mass) {
    force += Vector(100.0, 0.0, 0.0);
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

  auto merged_forces = merge(
    pairwise_force_x,
    pairwise_force_y,
    pairwise_force_z,
    generic_force_pos,
    generic_force_mass
  );

  for (int i = 1; i <= 10; ++i) {
    std::apply([&](auto&&... args) {
        sim.take_step(0.001, args...);
      }, merged_forces
    );

    writer.write(i, sim);
  }
  
  return 0;
}
