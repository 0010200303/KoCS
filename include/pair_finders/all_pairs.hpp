#ifndef KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
#define KOCS_PAIR_FINDERS_ALL_PAIRS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

namespace kocs::pair_finders {
  template<typename PositionsView>
  struct NaiveAllPairs {
    using positions_view_type = PositionsView;

    NaiveAllPairs(
      unsigned int agent_count_,
      float cutoff_distance,
      PositionsView& positions_)
      : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance)
      , positions(positions_) { }
    
    unsigned int agent_count;
    float cutoff_distance_squared;

    PositionsView positions;

    template<typename RandomPool, typename Force, typename... Views>
    void evaluate_force(
      detail::ViewPack<Views...> view_pack,
      PositionsView& old_velocities,
      RandomPool& random_pool,
      Force force
    ) {
      Kokkos::parallel_for(
        "naive_all_pairs_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          auto& position_i = positions(i);

          auto total_delta_i = detail::make_accumulator_pack(view_pack);
          // TODO: float should be Scalar
          float total_friction_i = 0.0;
          typename PositionsView::value_type total_velocity_i{0.0};

          // TODO: maybe you can actually have the total be references into the current view???
          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, agent_count),
            [&](const int j, auto& local_delta, auto& local_friction, auto& local_velocity) {
              if (i == j)
                return;

              const auto displacement = position_i - positions(j);
              const auto distance_squared = displacement.length_squared();

              if (distance_squared >= cutoff_distance_squared)
                return;
              const auto distance = Kokkos::sqrt(distance_squared);

              float pair_friction = 0.0;

              auto generator = random_pool.get_state();
              local_delta.apply([&](auto&... values) {
                force(i, j, displacement, distance, generator, pair_friction, values...);
              });
              random_pool.free_state(generator);

              local_friction += pair_friction;
              local_velocity += pair_friction * old_velocities(j);
            },
            Kokkos::Sum<decltype(total_delta_i)>(total_delta_i),
            Kokkos::Sum<float>(total_friction_i),
            Kokkos::Sum<typename PositionsView::value_type>(total_velocity_i)
          );

          Kokkos::single(
            Kokkos::PerTeam(team_member),
            [&]() {
              view_pack.apply([&](auto&... views) {
                total_delta_i.apply([&](auto&... values) {
                  ((views(i) += values), ...);
                });
              });

              // TODO: don't branch here, instead:
              // view_pack.first()(i) += total_velocity / max(total_friction, epsilon);
              if (total_friction_i != 0.0)
                view_pack.first()(i) += total_velocity_i / total_friction_i;
            }
          );
        }
      );
    }
  };
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
