#ifndef KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
#define KOCS_PAIR_FINDERS_ALL_PAIRS_HPP

#include <Kokkos_Core.hpp>

#include "../integrators/detail.hpp"

namespace kocs::pair_finders {
  template<typename Force, typename... Views>
  static void NaiveAllPairs(unsigned int agent_count, Force force, detail::ViewPack<Views...>& view_pack) {
    Kokkos::parallel_for(
      "naive_all_pairs_apply_force",
      Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();

        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, agent_count),
          KOKKOS_LAMBDA(const int j) {
            if (i == j)
              return;

            force(i, j, static_cast<const Views&>(view_pack)(i)...);
          });
      });
  }
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
