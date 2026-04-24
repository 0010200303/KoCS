#ifndef KOCS_INTEGRATORS_BASE_HPP
#define KOCS_INTEGRATORS_BASE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <type_traits>
#include <utility>

#include "detail.hpp"
#include "../forces/detail.hpp"

#include "../pair_finders/all_pairs.hpp"

namespace kocs::integrators {
  template<typename PairFinder, const unsigned int N, typename... Views>
  struct Base {
    Base(unsigned int agent_count_, Views... views)
      : agent_count(agent_count_), stage_pack(detail::ViewPack<Views...>(views...)) { }

    unsigned int agent_count;
    detail::StagePack<N, Views...> stage_pack;

    template<typename Force>
    void evaluate_force_impl(Force force, detail::GenericForceTag, detail::ViewPack<Views...>& view_pack) {
      Kokkos::parallel_for(
        "apply_generic_force",
        agent_count,
        KOKKOS_LAMBDA(const unsigned int i) {
          view_pack.apply([&](auto&... views) {
            force(i, views(i)...);
          });
        }
      );
    }

    template<typename Force>
    void evaluate_force_impl(Force force, detail::PairwiseForceTag, detail::ViewPack<Views...>& view_pack) {
      auto pair_finders = pair_finders::NaiveAllPairs(
        agent_count,
        10000.0f,
        detail::first(this->stage_pack[0]),
        view_pack
      );
      pair_finders.evaluate_force(force);
    }

    template<typename Force>
    void evaluate_force_one(Force force, detail::ViewPack<Views...>& view_pack) {
      evaluate_force_impl(force, typename Force::tag{}, view_pack);
    }

    template<typename... Forces>
    void evaluate_force(detail::ViewPack<Views...>& view_pack, Forces... forces) {
      (evaluate_force_one(forces, view_pack), ...);
    }

    template<typename Force>
    void evaluate_force_impl_single(
      Force force,
      detail::GenericForceTag,
      detail::ViewPack<Views...>& view_pack
    ) {
      Kokkos::parallel_for(
        "apply_generic_force_single",
        agent_count,
        KOKKOS_LAMBDA(const unsigned int i) {
          view_pack.apply([&](auto&... views) {
            force(i, views(i)...);
          });
        }
      );
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_impl_single(
      RandomPool& random_pool,
      Force force,
      detail::PairwiseForceTag,
      detail::ViewPack<Views...>& view_pack
    ) {
      auto pair_finders = pair_finders::NaiveAllPairs(
        agent_count,
        10000.0f,
        detail::first(this->stage_pack[0]),
        view_pack
      );
      pair_finders.evaluate_force_single(force);
    }

    template<typename Force>
    void evaluate_force_single(Force force, detail::ViewPack<Views...>& view_pack) {
      evaluate_force_impl_single(force, typename Force::tag{}, view_pack);
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_impl_rng(
      RandomPool& random_pool,
      Force force,
      detail::GenericForceTag,
      detail::ViewPack<Views...>& view_pack
    ) {
      // if constexpr (force_takes_rng)
      //   Kokkos::printf("RNG");
      // else
      //   Kokkos::printf("No RNG");

      Kokkos::parallel_for(
        "apply_generic_force_rng",
        agent_count,
        KOKKOS_LAMBDA(const unsigned int i) {
          view_pack.apply([&](auto&... views) {
            detail::invoke_force_with_optional_rng(force, random_pool, i, views(i)...);
          });
        }
      );
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_impl_rng(
      RandomPool& random_pool,
      Force force,
      detail::PairwiseForceTag,
      detail::ViewPack<Views...>& view_pack
    ) {
      auto pair_finders = pair_finders::NaiveAllPairs(
        agent_count,
        10000.0f,
        detail::first(this->stage_pack[0]),
        view_pack
      );
      pair_finders.evaluate_force_rng(random_pool, force);
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_rng(RandomPool& random_pool, Force force, detail::ViewPack<Views...>& view_pack) {
      evaluate_force_impl_rng(random_pool, force, typename Force::tag{}, view_pack);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_BASE_HPP
