#ifndef KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
#define KOCS_PAIR_FINDERS_ALL_PAIRS_HPP

#include <Kokkos_Core.hpp>

namespace kocs::pair_finders {
  template<typename Force, typename... Views>
  static void NaiveAllPairs(unsigned int agent_count, Force force, detail::ViewPack<Views...> view_pack) {
    Kokkos::parallel_for(
      "apply_force",
      agent_count,
      KOKKOS_LAMBDA(const unsigned int i) {
        force(i, static_cast<const Views&>(view_pack)(i)...);
      }
    );
  }
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
