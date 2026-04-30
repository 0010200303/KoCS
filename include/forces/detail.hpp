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



  template<std::size_t Index, typename T>
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

  template<typename... Views, std::size_t... Is>
  KOKKOS_INLINE_FUNCTION
  static auto make_accumulator_pack_impl(std::index_sequence<Is...>) {
    return AccumulatorPack<AccumulatorSlot<Is, typename Views::value_type>...>(
      AccumulatorSlot<Is, typename Views::value_type>{}...
    );
  }

  template<typename... Views>
  KOKKOS_INLINE_FUNCTION
  static auto make_accumulator_pack(const ViewPack<Views...>& pack) {
    return make_accumulator_pack_impl<Views...>(std::index_sequence_for<Views...>{});
  }



  template<typename T>
  struct GenericFieldRef {
    const T& i;
    T& delta;
  };

  template<typename T>
  struct PairwiseFieldRef {
    const T& i;
    const T& j;
    T& delta;
  };
} // namespace kocs::detail

namespace Kokkos {
  template<typename T>
  struct reduction_identity<kocs::detail::AccumulatorSlot<0, T>> {
    KOKKOS_INLINE_FUNCTION
    static constexpr kocs::detail::AccumulatorSlot<0, T> sum() {
      return kocs::detail::AccumulatorSlot<0, T>{};
    }
  };

  template<typename... Slots>
  struct reduction_identity<kocs::detail::AccumulatorPack<Slots...>> {
    KOKKOS_INLINE_FUNCTION
    static constexpr kocs::detail::AccumulatorPack<Slots...> sum() {
      return kocs::detail::AccumulatorPack<Slots...>{};
    }
  };
} // namespace Kokkos

#endif // KOCS_FORCES_DETAIL_HPP
