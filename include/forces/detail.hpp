#ifndef KOCS_FORCES_DETAIL_HPP
#define KOCS_FORCES_DETAIL_HPP

#include <type_traits>
#include <utility>

namespace kocs::detail {
  struct GenericForceTag { };
  struct PairwiseForceTag { };

  template<typename Tag, typename Force>
  struct TaggedForce {
    Force force;

    KOKKOS_INLINE_FUNCTION
    TaggedForce(Force force_) : force(force_) { }

    using tag = Tag;

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(Args&&... args) const {
      force(static_cast<Args&&>(args)...);
    }
  };

  template<typename Tag>
  struct ForceTagger {
    template<typename Force>
    friend auto operator|(ForceTagger, Force force) {
      return TaggedForce<Tag, Force>(force);
    }

    template<typename Force>
    friend auto operator|(Force force, ForceTagger) {
      return TaggedForce<Tag, Force>(force);
    }
  };

  constexpr ForceTagger<GenericForceTag> generic_force{};
  constexpr ForceTagger<PairwiseForceTag> pairwise_force{};

  template<typename T, typename = void>
  struct HasMemberForce : std::false_type { };

  template<typename T>
  struct HasMemberForce<T, std::void_t<decltype(std::declval<T&>().force)>> : std::true_type { };

  template<typename T>
  inline constexpr bool has_member_force_v = HasMemberForce<T>::value;

  template<typename T, bool = has_member_force_v<std::remove_reference_t<T>>>
  struct ForceCallable;

  template<typename T>
  struct ForceCallable<T, true> {
    using type = std::remove_reference_t<decltype(std::declval<std::remove_reference_t<T>&>().force)>;
  };

  template<typename T>
  struct ForceCallable<T, false> {
    using type = std::remove_reference_t<T>;
  };

  template<typename T>
  using force_callable_t = typename ForceCallable<T>::type;

  template<typename RandomPool>
  using random_state_t = std::remove_reference_t<decltype(std::declval<RandomPool&>().get_state())>;

  template<typename Force, typename RandomPool, typename... Args>
  constexpr bool force_takes_rng_v = std::is_invocable_v<
    std::add_lvalue_reference_t<const force_callable_t<Force>>,
    unsigned int,
    random_state_t<RandomPool>&,
    Args...
  >;

  template<typename Force>
  KOKKOS_INLINE_FUNCTION
  decltype(auto) force_target(Force&& force) {
    if constexpr (has_member_force_v<std::remove_reference_t<Force>>) {
      return (force.force);
    } else {
      return (force);
    }
  }

  template<typename Force, typename RandomPool, typename... Args>
  KOKKOS_INLINE_FUNCTION
  void invoke_force_with_optional_rng(Force&& force, RandomPool& random_pool, unsigned int i, Args&&... args) {
    auto&& target = force_target(force);

    if constexpr (force_takes_rng_v<Force, RandomPool, Args...>) {
      auto generator = random_pool.get_state();
      target(i, generator, static_cast<Args&&>(args)...);
      random_pool.free_state(generator);
    } else {
      target(i, static_cast<Args&&>(args)...);
    }
  }

  template<typename T>
  struct AccumulatorSlot {
    T value{};

    KOKKOS_INLINE_FUNCTION
    AccumulatorSlot() = default;

    KOKKOS_INLINE_FUNCTION
    AccumulatorSlot(const T& v) : value(v) { }

    KOKKOS_INLINE_FUNCTION
    T& get() {
      return value;
    }

    KOKKOS_INLINE_FUNCTION
    const T& get() const {
      return value;
    }

    KOKKOS_INLINE_FUNCTION
    AccumulatorSlot& operator+=(const AccumulatorSlot& other) {
      value += other.value;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    AccumulatorSlot& operator+=(const T& other) {
      value += other;
      return *this;
    }
  };

  // TODO: remove default constructor?
  template<typename... Slots>
  struct AccumulatorPack : Slots... {
    KOKKOS_INLINE_FUNCTION
    AccumulatorPack() = default;

    KOKKOS_INLINE_FUNCTION
    AccumulatorPack(Slots... types) : Slots(types)... { }

    KOKKOS_INLINE_FUNCTION
    AccumulatorPack& operator+=(const AccumulatorPack& other) {
      ((static_cast<Slots&>(*this) += static_cast<const Slots&>(other)), ...);
      return *this;
    }

    template<typename Func>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(Func&& func) {
      return func(static_cast<Slots&>(*this).get()...);
    }

    template<typename Func>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(Func&& func) const {
      return func(static_cast<const Slots&>(*this).get()...);
    }
  };

  template<typename... Views>
  KOKKOS_INLINE_FUNCTION
  static auto make_accumulator_pack(const ViewPack<Views...>& pack) {
    return AccumulatorPack<AccumulatorSlot<typename Views::value_type>...>(
      AccumulatorSlot<typename Views::value_type>{}...
    );
  }
} // namespace kocs::detail

#endif // KOCS_FORCES_DETAIL_HPP
