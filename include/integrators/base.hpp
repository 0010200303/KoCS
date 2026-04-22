#ifndef KOCS_INTEGRATORS_BASE_HPP
#define KOCS_INTEGRATORS_BASE_HPP

#include <Kokkos_Core.hpp>

#include "detail.hpp"
#include "../forces/detail.hpp"

#include "../pair_finders/all_pairs.hpp"

namespace kocs::integrators {
  template<typename PairFinder, const unsigned int N, typename... Views>
  struct Base {
    Base(unsigned int agent_count_, Views... views)
      : agent_count(agent_count_), stage_pack(detail::ViewPack<Views...>(views...)) { }

    unsigned int agent_count;
    mutable detail::StagePack<N, Views...> stage_pack;
    // TODO: remove mutable?

    template<typename Force>
    void evaluate_force_impl(Force force, detail::GenericForceTag, detail::ViewPack<Views...>& view_pack) {
      Kokkos::parallel_for(
        "apply_generic_force",
        agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          force(i, static_cast<const Views&>(view_pack)(i)...);

          Kokkos::printf("%d\n", detail::first(this->stage_pack[0]).extent(0));
        }
      );
    }

    template<typename Force>
    void evaluate_force_impl(Force force, detail::PairwiseForceTag, detail::ViewPack<Views...>& view_pack) {
      auto pair_finders = pair_finders::NaiveAllPairs(
        agent_count,
        10000.0f,
        detail::first(this->stage_pack[0]),
        view_pack
      );
      pair_finders.evaluate_force(force);
    }

    template<typename Force>
    void evaluate_force(Force force, detail::ViewPack<Views...>& view_pack) {
      evaluate_force_impl(force, typename Force::tag{}, view_pack);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_BASE_HPP
