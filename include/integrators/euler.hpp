#ifndef KOCS_INTEGRATORS_EULER_HPP
#define KOCS_INTEGRATORS_EULER_HPP

#include <Kokkos_Core.hpp>

#include "base.hpp"

namespace kocs::integrators {
  template<typename PairFinder, typename ComFixer, typename... Views>
  struct Euler : public Base<PairFinder, ComFixer, 2, Views...> {
    using Base<PairFinder, ComFixer, 2, Views...>::Base;

    void apply_euler(double dt) {
      const auto com_fix_delta = this->com_fixer.fix(this->stage_pack[0].first());

      Kokkos::parallel_for(
        "apply_euler",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          // write old velocities
          this->old_velocities(i) = this->stage_pack[1].first()(i);

          this->stage_pack[1].first()(i) -= com_fix_delta;

          this->stage_pack[0].zip_apply([&](auto& current_views, const auto& delta_views) {
            current_views(i) += delta_views(i) * dt;

            // clear views (faster than new deep_copy call)
            delta_views(i) = std::remove_cv_t<std::remove_reference_t<decltype(delta_views(i))>>{};
          }, this->stage_pack[1]);
        }
      );
    }

    template<typename RandomPool, typename... Forces>
    void integrate(double dt, RandomPool& random_pool, Forces... forces) {
      this->evaluate_forces(random_pool, this->stage_pack[0], this->stage_pack[1], forces...);
      apply_euler(dt);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_EULER_HPP
