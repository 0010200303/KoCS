#ifndef KOCS_INTEGRATORS_BASE_HPP
#define KOCS_INTEGRATORS_BASE_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "detail.hpp"
#include "../forces/detail.hpp"
#include "../pair_finders/all_pairs.hpp"

namespace kocs::integrators {
  template<
    typename PairFinder,
    typename ComFixer,
    const unsigned int N,
    typename GenericForceFields,
    typename PairwiseForceFields,
    typename LinkForceFields,
    typename... Views
  >
  struct Base {
    Base(
      unsigned int agent_count_,
      PairFinder& pair_finder_,
      ComFixer& com_fixer_,
      Kokkos::View<Link*> links_,
      Views... views
    ) : agent_count(agent_count_)
      , pair_finder(pair_finder_)
      , com_fixer(com_fixer_)
      , stage_pack(detail::ViewPack<Views...>(views...))
      , old_velocities("integrator_base_old_velocities", agent_count_)
      , links(links_) { }
    
    using PositionsView = typename PairFinder::positions_view_type;

    PairFinder& pair_finder;
    ComFixer& com_fixer;

    unsigned int agent_count;
    detail::StagePack<N, Views...> stage_pack;
    PositionsView old_velocities;

    Kokkos::View<Link*> links;

    template<typename... NewViews>
    void reattach_stage_0(const NewViews&... new_views) {
      stage_pack[0] = detail::ViewPack<Views...>(new_views...);
    }

    inline void set_capacity(const unsigned int value) {
      Kokkos::resize(old_velocities, value);

      // resize all views from auxiliary stages
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((stage_pack[Is + 1].apply_host([&](auto&... views) {
          ((Kokkos::realloc(views, value)), ...);
        })), ...);
      }(std::make_index_sequence<N - 1>{});
    }

    inline void set_agent_count(const unsigned int value) {
      agent_count = value;
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_impl(
      bool is_full_step,
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
              force(i, generator, GenericForceFields{detail::GenericFieldRef{in_views(i), out_views(i)}...});
            });
          });
          random_pool.free_state(generator);
        }
      );
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_impl(
      bool is_full_step,
      RandomPool& random_pool,
      Force force,
      detail::PairwiseForceTag,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      pair_finder.template evaluate_force<RandomPool, Force, PairwiseForceFields>(
        in_view_pack, out_view_pack, old_velocities, random_pool, force, is_full_step
      );
    }

    template<typename RandomPool, typename Func>
    void evaluate_force_impl(
      bool is_full_step,
      RandomPool& random_pool,
      Func function,
      detail::UpdateFuncTag,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      Kokkos::parallel_for(
        "apply_update_func",
        agent_count,
        KOKKOS_LAMBDA(const unsigned int i) {
          auto generator = random_pool.get_state();
          function(i, generator);
          random_pool.free_state(generator);
        }
      );
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_impl(
      bool is_full_step,
      RandomPool& random_pool,
      Force force,
      detail::LinkForceTag,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      Kokkos::parallel_for(
        "apply_link_force",
        links.extent(0),
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          Link link = links(i);
          if (link.a == link.b)
            return;

          // TODO: fix data race (accumulate locally (maybe reducer) then add back to original out_views)
          auto generator = random_pool.get_state();
          in_view_pack.apply([&](auto&... in_views) {
            out_view_pack.apply([&](auto&... out_views) {
              force(link, generator, LinkForceFields{detail::LinkFieldRef{
                in_views(link.a),
                in_views(link.b),
                out_views(link.a),
                out_views(link.b)
              }...});
            });
          });
          random_pool.free_state(generator);
        }
      );
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_one(
      bool is_full_step,
      RandomPool& random_pool,
      Force force,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack
    ) {
      evaluate_force_impl(is_full_step, random_pool, force, typename Force::tag{}, in_view_pack, out_view_pack);
    }

    template<typename RandomPool, typename... Forces>
    void evaluate_forces(
      bool is_full_step,
      RandomPool& random_pool,
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      Forces... forces
    ) {
      (evaluate_force_one(is_full_step, random_pool, forces, in_view_pack, out_view_pack), ...);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_BASE_HPP
