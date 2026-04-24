#ifndef KOCS_FORCES_KERNEL_FUSER_HPP
#define KOCS_FORCES_KERNEL_FUSER_HPP

#include <tuple>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "detail.hpp"

namespace kocs::detail {
  template<typename Tag, typename... Forces>
  struct KernelFuser;

  template<typename Tag>
  struct KernelFuser<Tag> {
    // KernelFuser() = default;

    using tag = Tag;

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(Args&&...) const { }
  };

  template<typename Tag, typename FirstForce, typename... RestForces>
  struct KernelFuser<Tag, FirstForce, RestForces...> : KernelFuser<Tag, RestForces...> {
    using base_type = KernelFuser<Tag, RestForces...>;

    FirstForce force;

    KOKKOS_INLINE_FUNCTION
    KernelFuser() = default;

    KOKKOS_INLINE_FUNCTION
    KernelFuser(FirstForce first, RestForces... rest)
      : base_type(std::move(rest)...), force(std::move(first)) { }

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(Args&&... args) const {
      force(static_cast<Args&&>(args)...);
      static_cast<const base_type&>(*this)(static_cast<Args&&>(args)...);
    }
  };

  template<typename Tag, typename... Forces>
  KernelFuser(Tag, Forces...) -> KernelFuser<Tag, Forces...>;

  template<typename Tag1, typename Tag2, typename Force>
  auto collect_tagged_for_two_tags(Force&& force) {
    using force_t = std::decay_t<Force>;

    if constexpr (std::is_same_v<typename force_t::tag, Tag1>) {
      using pure_force_t = std::decay_t<decltype(std::forward<Force>(force).force)>;
      return std::pair{
        std::tuple<pure_force_t>(std::forward<Force>(force).force),
        std::tuple<>{}
      };
    } else if constexpr (std::is_same_v<typename force_t::tag, Tag2>) {
      using pure_force_t = std::decay_t<decltype(std::forward<Force>(force).force)>;
      return std::pair{
        std::tuple<>{},
        std::tuple<pure_force_t>(std::forward<Force>(force).force)
      };
    } else {
      return std::pair{std::tuple<>{}, std::tuple<>{}};
    }
  }

  template<typename Tag1, typename Tag2>
  auto partition_forces() {
    return std::pair{std::tuple<>{}, std::tuple<>{}};
  }

  template<typename Tag1, typename Tag2, typename Force, typename... Rest>
  auto partition_forces(Force&& force, Rest&&... rest) {
    auto tail = partition_forces<Tag1, Tag2>(std::forward<Rest>(rest)...);
    auto head = collect_tagged_for_two_tags<Tag1, Tag2>(std::forward<Force>(force));

    return std::pair{
      std::tuple_cat(std::move(head.first), std::move(tail.first)),
      std::tuple_cat(std::move(head.second), std::move(tail.second))
    };
  }

  // TODO: add more tags
  template<typename... Forces>
  auto fuse_forces(Forces&&... forces) {
    auto grouped = partition_forces<detail::GenericForceTag, detail::PairwiseForceTag>(
      std::forward<Forces>(forces)...
    );

    return std::make_tuple(
      std::apply([](auto&&... kernels) {
        return KernelFuser<detail::GenericForceTag, std::decay_t<decltype(kernels)>...>{
          std::forward<decltype(kernels)>(kernels)...
        };
      }, std::move(grouped.first)),
      std::apply([](auto&&... kernels) {
        return KernelFuser<detail::PairwiseForceTag, std::decay_t<decltype(kernels)>...>{
          std::forward<decltype(kernels)>(kernels)...
        };
      }, std::move(grouped.second))
    );
  }
} // namespace kocs::detail

#endif // KOCS_FORCES_KERNEL_FUSER_HPP
