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
          ( (static_cast<Views&>(this->stage_pack[0])(i) += static_cast<Views&>(this->stage_pack[1])(i) * dt), ... );
      });
    }
  };
} // namespace kocs::integrator

#endif // KOCS_INTEGRATORS_EULER_HPP
