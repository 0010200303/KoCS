#ifndef KOCS_INITIALIZERS_RECTANGLE_INIT_HPP
#define KOCS_INITIALIZERS_RECTANGLE_INIT_HPP

#include "../utils/utils.hpp"

namespace kocs::initializers {
  template<typename SimulationConfig>
  struct RegularRectangle {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    VectorView positions_view;
    Scalar distance_nb;
    unsigned int nx;

    template<typename ViewType>
    RegularRectangle(ViewType positions, Scalar distance_to_neighbour, unsigned int nx_)
      : positions_view(positions)
      , distance_nb(distance_to_neighbour)
      , nx(nx_) { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Random& generator) const {
      const unsigned int row = i / nx;
      const unsigned int col = i % nx;

      // row spacing = sqrt(d_nb^2 - (d_nb/2)^2) = d_nb * sqrt(3)/2
      const Scalar row_height = distance_nb * Scalar(0.5) * Kokkos::sqrt(Scalar(3.0));
      const Scalar row_offset = (row % 2 != 0) ? (distance_nb / Scalar(2.0)) : Scalar(0.0);

      positions_view(i) = Vector(
        row_offset + Scalar(col) * distance_nb,
        Scalar(row) * row_height,
        Scalar(0.0)
      );
    }
  };
} // namespace kocs::initializers

#endif // KOCS_INITIALIZERS_RECTANGLE_INIT_HPP
