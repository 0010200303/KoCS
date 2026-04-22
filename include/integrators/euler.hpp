#ifndef KOCS_INTEGRATORS_EULER_HPP
#define KOCS_INTEGRATORS_EULER_HPP

#include <Kokkos_Core.hpp>

#include "base.hpp"

namespace kocs::integrators {
  template<typename PairFinder, typename... Views>
  struct Euler : public Base<PairFinder, 2, Views...> {
    using Base<PairFinder, 2, Views...>::Base;

    template<typename Force>
    void integrate(double dt, Force force) {
      this->evaluate_force(force, this->stage_pack[1]);

      Kokkos::parallel_for(
        "apply_euler",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          this->stage_pack[0].apply([&](auto&... current_views) {
            this->stage_pack[1].apply([&](auto&... delta_views) {
              ((current_views(i) += delta_views(i) * dt), ...);
            });
          });
        });
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_EULER_HPP
