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
    KernelFuser(FirstForce first, RestForces... rest) : base_type(rest...), force(first.force) { }

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(Args&&... args) const {
      force(static_cast<Args&&>(args)...);
      static_cast<const base_type&>(*this)(static_cast<Args&&>(args)...);
    }
  };

  template<typename Tag, typename... Forces>
  KernelFuser(Tag, Forces...) -> KernelFuser<Tag, Forces...>;

  template<typename Tag, typename Force>
  auto collect_tagged_force(Force&& force) {
    if constexpr (std::is_same_v<typename std::decay_t<Force>::tag, Tag>)
      return std::tuple<std::decay_t<Force>>(std::forward<Force>(force));
    else
      return std::tuple<>{};
  }

  template<typename Tag, typename... Forces>
  auto fuse_forces_for_tag(Forces&&... forces) {
    auto selected = std::tuple_cat(collect_tagged_force<Tag>(std::forward<Forces>(forces))...);

    return std::apply([](auto&&... tagged_kernels) {
      return KernelFuser<Tag, std::decay_t<decltype(tagged_kernels)>...> {
        std::forward<decltype(tagged_kernels)>(tagged_kernels)...
      };
    }, selected);
  }

  // TODO: add more tags
  template<typename... Forces>
  auto fuse_forces(Forces&&... forces) {
    return std::make_tuple(
      fuse_forces_for_tag<detail::GenericForceTag>(std::forward<Forces>(forces)...),
      fuse_forces_for_tag<detail::PairwiseForceTag>(std::forward<Forces>(forces)...)
    );
  }
} // namespace kocs::detail

#endif // KOCS_FORCES_KERNEL_FUSER_HPP
