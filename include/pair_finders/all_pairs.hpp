#ifndef KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
#define KOCS_PAIR_FINDERS_ALL_PAIRS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

namespace kocs::pair_finders {
  template<typename PositionsView, typename... Views>
  struct NaiveAllPairs {
    NaiveAllPairs(
      unsigned int agent_count_,
      float cutoff_distance,
      PositionsView& positions_,
      detail::ViewPack<Views...>& view_pack_)
      : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance)
      , positions(positions_)
      , view_pack(view_pack_) { }
    
    unsigned int agent_count;
    float cutoff_distance_squared;
    PositionsView positions;
    detail::ViewPack<Views...> view_pack;

    template<typename Force>
    void evaluate_force_one(Force force) {
      Kokkos::parallel_for(
        "naive_all_pairs_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          auto& position_i = positions(i); 

          auto total = detail::make_accumulator_pack(view_pack);

          // TODO: maybe you can actually have the total be references into the current view???
          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, agent_count),
            [&](const int j, auto& local) {
              if (i == j)
                return;

              const auto displacement = position_i - positions(j);
              const auto distance_squared = displacement.length_squared();

              if (distance_squared >= cutoff_distance_squared)
                return;

              // TODO: check this
              local.apply([&](auto&... values) {
                force(i, j, displacement, Kokkos::sqrt(distance_squared), values...);
              });
            },
            total
          );

          Kokkos::single(
            Kokkos::PerTeam(team_member),
            [&]() {
              // TODO: create better syntax
              view_pack.apply([&](auto&... views) {
                total.apply([&](auto&... values) {
                  ((views(i) += values), ...);
                });
              });
            }
          );
        }
      );
    }

    // TODO: fix this
    template<typename... Forces>
    void evaluate_force(Forces... forces) {
      (evaluate_force_one(forces), ...);
    }

    template<typename RandomPool, typename Force>
    void evaluate_force_rng(RandomPool& random_pool, Force force) {
      Kokkos::parallel_for(
        "naive_all_pairs_apply_force_rng",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          auto& position_i = positions(i); 

          auto total = detail::make_accumulator_pack(view_pack);

          // TODO: maybe you can actually have the total be references into the current view???
          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, agent_count),
            [&](const int j, auto& local) {
              if (i == j)
                return;

              const auto displacement = position_i - positions(j);
              const auto distance_squared = displacement.length_squared();

              if (distance_squared >= cutoff_distance_squared)
                return;

              auto generator = random_pool.get_state();

              local.apply([&](auto&... values) {
                force(i, j, displacement, Kokkos::sqrt(distance_squared), generator, values...);
              });

              random_pool.free_state(generator);
            },
            total
          );

          Kokkos::single(
            Kokkos::PerTeam(team_member),
            [&]() {
              // TODO: create better syntax
              view_pack.apply([&](auto&... views) {
                total.apply([&](auto&... values) {
                  ((views(i) += values), ...);
                });
              });
            }
          );
        }
      );
    }
  };
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
