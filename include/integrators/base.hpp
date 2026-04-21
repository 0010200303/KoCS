#ifndef KOCS_INTEGRATORS_BASE_HPP
#define KOCS_INTEGRATORS_BASE_HPP

#include <Kokkos_Core.hpp>

#include "detail.hpp"
#include "../pair_finders/all_pairs.hpp"

namespace kocs::integrators {
  template<const unsigned int N, typename... Views>
  struct Base {
    Base(unsigned int agent_count, Views... views)
      : agent_count(agent_count), stage_pack(detail::ViewPack<Views...>(views...)) { }

    unsigned int agent_count;
    mutable detail::StagePack<N, Views...> stage_pack;

    template<typename Force>
    void evaluate_force(Force force, detail::ViewPack<Views...>& view_pack) {
      pair_finders::NaiveAllPairs(agent_count, force, view_pack);
    }
  };
} // namespace kocs::integrator

#endif // KOCS_INTEGRATORS_BASE_HPP
