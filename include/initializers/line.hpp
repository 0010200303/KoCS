#ifndef KOCS_INITIALIZERS_LINE_INIT_HPP
#define KOCS_INITIALIZERS_LINE_INIT_HPP

#include "../utils/utils.hpp"

namespace kocs::initializers {
  template<typename SimulationConfig>
  struct Line {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    VectorView positions_view;
    const Scalar distance;

    template<typename ViewType>
    Line(ViewType positions, const Scalar distance_) 
      : positions_view(positions)
      , distance(distance_) { }

    KOKKOS_INLINE_FUNCTION
    void operator() (const unsigned int i, Random& generator) const {
      positions_view(i) = Vector(Scalar(i) * distance);
    }
  };
} // namespace kocs::initalizers

#endif // KOCS_INITIALIZERS_LINE_INIT_HPP
