#ifndef KOCS_INTEGRATORS_EULER_HPP
#define KOCS_INTEGRATORS_EULER_HPP

#include <Kokkos_Core.hpp>

#include "base.hpp"

namespace kocs::integrator {
  template<typename... Views>
  struct Euler : public Base<2, Views...> {
    using Base<2, Views...>::Base;

    template<typename Force>
    void integrate(Force force) {
      Kokkos::parallel_for("integrate_euler", this->agent_count, KOKKOS_CLASS_LAMBDA(const unsigned int i) {
        force(i, static_cast<Views&>(this->stage_pack[1])(i)...);
      });

      Kokkos::parallel_for("apply_euler", this->agent_count, KOKKOS_CLASS_LAMBDA(const unsigned int i) {
        ( (static_cast<Views&>(this->stage_pack[0])(i) += static_cast<Views&>(this->stage_pack[1])(i)), ... );
      });
    }
  };
} // namespace kocs::integrator

#endif // KOCS_INTEGRATORS_EULER_HPP
