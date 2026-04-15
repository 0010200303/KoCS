#ifndef KOCS_LINE_INIT_HPP
#define KOCS_LINE_INIT_HPP

#include <Kokkos_Core.hpp>

#include "../utils.hpp"

namespace kocs::initializer {
  template <typename SimulationConfig>
  struct Line {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    VectorView positions_view;

    Line(VectorView positions) : positions_view(positions) { }

    KOKKOS_INLINE_FUNCTION
    void operator() (const unsigned int i, auto& generator) const {
      positions_view(i) = Vector(Scalar(i));
    }
  };
} // namespace kocs

#endif // KOCS_LINE_INIT_HPP
