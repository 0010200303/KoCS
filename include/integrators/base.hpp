#ifndef KOCS_INTEGRATORS_BASE_HPP
#define KOCS_INTEGRATORS_BASE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "detail.hpp"
#include "../forces/detail.hpp"
#include "../pair_finders/all_pairs.hpp"

namespace kocs::integrators {
  template<typename PairFinder, const unsigned int N, typename... Views>
  struct Base {
    Base(
      unsigned int agent_count_,
      PairFinder& pair_finder_,
      Views... views)
      : agent_count(agent_count_)
      , pair_finder(pair_finder_)
      , stage_pack(detail::ViewPack<Views...>(views...))
      , old_velocities("integrator_base_old_velocities", agent_count_) { }
    
    using PositionsView = typename PairFinder::positions_view_type;

    PairFinder pair_finder;

    unsigned int agent_count;
    detail::StagePack<N, Views...> stage_pack;
    PositionsView old_velocities;

    template<typename RandomPool, typename Force>
    void evaluate_force_impl(
      RandomPool& random_pool,
      Force force,
      detail::GenericForceTag,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      Kokkos::parallel_for(
        "apply_generic_force",
        agent_count,
        KOKKOS_LAMBDA(const unsigned int i) {
          auto generator = random_pool.get_state();
          in_view_pack.apply([&](auto&... in_views) {
            out_view_pack.apply([&](auto&... out_views) {
              force(i, generator, detail::GenericFieldRef{in_views(i), out_views(i)}...);
            });
          });
          random_pool.free_state(generator);
        }
      );
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_impl(
      RandomPool& random_pool,
      Force force,
      detail::PairwiseForceTag,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      pair_finder.evaluate_force(in_view_pack, out_view_pack, old_velocities, random_pool, force);
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_one(
      RandomPool& random_pool,
      Force force,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      evaluate_force_impl(random_pool, force, typename Force::tag{}, in_view_pack, out_view_pack);
    }

    template<typename RandomPool, typename... Forces>
    void evaluate_forces(
      RandomPool& random_pool,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      Forces... forces
    ) {      
      (evaluate_force_one(random_pool, forces, in_view_pack, out_view_pack), ...);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_BASE_HPP
