#ifndef KOCS_SIMULATION_HPP
#define KOCS_SIMULATION_HPP

#include <iostream>

#include <Kokkos_Core.hpp>

#include "runtime_guard.hpp"

namespace kocs {
  template <typename Scalar, unsigned int dimensions>
  class Simulation {
    public:
      Simulation() { get_runtime_guard(); }

    private:
      // runtime guard (kokkos initialize & finalize)
      static RuntimeGuard& get_runtime_guard() {
        static RuntimeGuard guard;
        return guard;
      }
  };
} // namespace kocs

#endif // KOCS_SIMULATION_HPP
