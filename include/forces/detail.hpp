#ifndef KOCS_FORCES_DETAIL_HPP
#define KOCS_FORCES_DETAIL_HPP

namespace kocs::detail {
  struct GenericForceTag { };
  struct PairwiseForceTag { };

  template<typename Tag, typename Force>
  struct TaggedForce {
    Force force;

    TaggedForce(Force force_) : force(force_) { }

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

#define GENERIC_FORCE kocs::detail::generic_force | KOKKOS_LAMBDA
#define PAIRWISE_FORCE kocs::detail::pairwise_force | KOKKOS_LAMBDA
} // namespace kocs::detail

#endif // KOCS_FORCES_DETAIL_HPP
