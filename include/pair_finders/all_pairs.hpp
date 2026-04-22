#ifndef KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
#define KOCS_PAIR_FINDERS_ALL_PAIRS_HPP

#include <Kokkos_Core.hpp>

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
    void evaluate_force(Force force) {
      // Kokkos::printf("%f\n", kek(1).x());

      Kokkos::parallel_for(
        "apply_euler",
        agent_count,
        KOKKOS_LAMBDA(const unsigned int i) {
          Kokkos::printf("%f\n", positions(i).x());
      });
    }
  };
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
