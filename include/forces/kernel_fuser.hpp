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
    using tag = Tag;

    template<typename... Args>
    KOKKOS_INLINE_FUNCTION
    void operator()(Args&&...) const { }
  };

  template<typename Tag, typename FirstForce, typename... RestForces>
  struct KernelFuser<Tag, FirstForce, RestForces...> : KernelFuser<Tag, RestForces...> {
    using base_type = KernelFuser<Tag, RestForces...>;

    FirstForce force;

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

  template<typename Tag, typename Force>
  auto collect_tagged_force(Force&& force) {
    if constexpr (std::is_same_v<typename std::decay_t<Force>::tag, Tag>) {
      using pure_force_t = std::decay_t<decltype(std::forward<Force>(force).force)>;
      return std::tuple<pure_force_t>(std::forward<Force>(force).force);
    } else {
      return std::tuple<>{};
    }
  }

  template<typename Tag, typename... Forces>
  auto fuse_forces_for_tag(Forces&&... forces) {
    return std::apply([](auto&&... kernels) {
      return KernelFuser<Tag, std::decay_t<decltype(kernels)>...> {
        std::forward<decltype(kernels)>(kernels)...
      };
    }, std::tuple_cat(collect_tagged_force<Tag>(std::forward<Forces>(forces))...));
  }

  template<typename T>
  auto as_tuple_if_not_empty(T&& value) {
    if constexpr (std::is_empty_v<std::decay_t<T>>)
      return std::tuple<>{};
    else
      return std::tuple<std::decay_t<T>>(std::forward<T>(value));
  }

  template<typename... Forces>
  auto fuse_forces(Forces&&... forces) {
    return std::tuple_cat(
      as_tuple_if_not_empty(fuse_forces_for_tag<detail::GenericForceTag>(std::forward<Forces>(forces)...)),
      as_tuple_if_not_empty(fuse_forces_for_tag<detail::PairwiseForceTag>(std::forward<Forces>(forces)...))
    );
  }
} // namespace kocs::detail

#endif // KOCS_FORCES_KERNEL_FUSER_HPP
