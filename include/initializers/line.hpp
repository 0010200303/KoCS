#ifndef KOCS_INITIALIZERS_LINE_INIT_HPP
#define KOCS_INITIALIZERS_LINE_INIT_HPP

#include <Kokkos_Core.hpp>

#include "../utils.hpp"

namespace kocs::initializer {
  template<typename SimulationConfig>
  struct Line {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    VectorView positions_view;

    Line(VectorView positions) : positions_view(positions) { }

    KOKKOS_INLINE_FUNCTION
    void operator() (const unsigned int i) const {
      positions_view(i) = Vector(float(i));

      Kokkos::printf("%d %f\n", i, float(i));
    }
  };
} // namespace kocs

#endif // KOCS_INITIALIZERS_LINE_INIT_HPP
