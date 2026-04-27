#ifndef KOCS_INTEGRATORS_EULER_HPP
#define KOCS_INTEGRATORS_EULER_HPP

#include <Kokkos_Core.hpp>

#include "base.hpp"

namespace kocs::integrators {
  template<typename PairFinder, typename... Views>
  struct Euler : public Base<PairFinder, 2, Views...> {
    using Base<PairFinder, 2, Views...>::Base;

    void apply_euler(double dt) {
      Kokkos::parallel_for(
        "apply_euler",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          this->stage_pack[0].apply([&](auto&... current_views) {
            this->stage_pack[1].apply([&](auto&... delta_views) {
              ((current_views(i) += delta_views(i) * dt), ...);

              // clear views (faster than new deep_copy)
              ((delta_views(i) = 0), ...);

              // heun for later
              // ((delta_views_1(i) = (delta_views_1(i) + delta_views_2(i)) * 0.5), ...);
            });
          });
        }
      );
    }

    template<typename RandomPool, typename... Forces>
    void integrate(double dt, RandomPool& random_pool, Forces... forces) {
      this->evaluate_forces(random_pool, this->stage_pack[1], forces...);
      apply_euler(dt);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_EULER_HPP
