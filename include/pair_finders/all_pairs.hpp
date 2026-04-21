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

        // TODO: maybe you can actually have the total be references into the current view???
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, agent_count),
          [&](const int j, auto& local) {
            if (i == j)
              return;

            local.apply([&](auto&... values) {
              force(i, j, values...);
            });
          },
          total
        );

        Kokkos::single(
          Kokkos::PerTeam(team_member),
          [&]() {
            total.apply([&](auto&... values) {
              ((static_cast<const Views&>(view_pack)(i) += values), ...);
            });
          }
        );
      }
    );
  }
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
