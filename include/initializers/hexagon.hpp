#ifndef KOCS_INITIALIZERS_HEXAGON_INIT_HPP
#define KOCS_INITIALIZERS_HEXAGON_INIT_HPP

#include <numbers>

#include <Kokkos_Core.hpp>

#include "../utils.hpp"

namespace kocs::initializers {
  template<typename SimulationConfig>
  struct RegularHexagon {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    VectorView positions_view;
    Scalar distance_nb;

    template<typename ViewType>
    RegularHexagon(ViewType positions, Scalar distance_to_neighbour)
      : positions_view(positions)
      , distance_nb(distance_to_neighbour) { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Random& generator) const {
      if (i == 0)
        return;

      auto& position = positions_view(i);

      unsigned int remaining = i - 1;
      unsigned int ring = 1;
      while (remaining >= 6 * ring) {
        remaining -= 6 * ring;
        ++ring;
      }

      const Scalar beta = Scalar(std::numbers::pi) / Scalar(3);
      const unsigned int side = remaining / ring;
      const unsigned int step = remaining % ring;

      const Scalar radius = distance_nb * Scalar(ring);
      const Scalar t = Scalar(step) / Scalar(ring);

      Vector start{};
      start[0] = -radius * Kokkos::sin(beta * Scalar(side));
      start[1] =  radius * Kokkos::cos(beta * Scalar(side));
      start[2] = Scalar(0);

      Vector end{};
      end[0] = -radius * Kokkos::sin(beta * Scalar(side + 1));
      end[1] =  radius * Kokkos::cos(beta * Scalar(side + 1));
      end[2] = Scalar(0);

      position[0] = start[0] + (end[0] - start[0]) * t;
      position[1] = start[1] + (end[1] - start[1]) * t;
      position[2] = Scalar(0);
    }
  };
} // namespace kocs::initializers

#endif // KOCS_INITIALIZERS_HEXAGON_INIT_HPP
