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
            });
          });
        }
      );
    }

    template<typename Force>
    void integrate(double dt, Force force) {
      this->evaluate_force(force, this->stage_pack[1]);
      apply_euler(dt);
    }

    template<typename RandomPool, typename Force>
    void integrate_rng(double dt, RandomPool& random_pool, Force force) {
      this->evaluate_force_rng(random_pool, force, this->stage_pack[1]);
      apply_euler(dt);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_EULER_HPP
