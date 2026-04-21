#ifndef KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
#define KOCS_PAIR_FINDERS_ALL_PAIRS_HPP

#include <Kokkos_Core.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

namespace kocs::pair_finders {
  template<typename Force, typename... Views>
  static void NaiveAllPairs(unsigned int agent_count, Force force, detail::ViewPack<Views...>& view_pack) {
    Kokkos::parallel_for(
      "naive_all_pairs_apply_force",
      Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        const int i = team_member.league_rank();

        auto total = detail::make_accumulator_pack(view_pack);

        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, agent_count),
          [&](const int j, auto& local) {
            local.apply([&](auto&... values) {
              force(i, j, values...);
            });
          },
          total
        );



        // Kokkos::parallel_for(
        //   Kokkos::TeamThreadRange(team, agent_count),
        //   [&](const int j) {
        //     if (i == j)
        //       return;

        //     force(i, j, static_cast<const Views&>(view_pack)(i)...);
        //   }
        // );

        // for (int j = 0; j < static_cast<int>(agent_count); ++j) {
        //   if (i == j)
        //     continue;

        //   force(i, j, static_cast<const Views&>(view_pack)(i)...);
        // }
      }
    );
  }
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
