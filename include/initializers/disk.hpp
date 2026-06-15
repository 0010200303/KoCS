#ifndef KOCS_INITIALIZERS_DISK_INIT_HPP
#define KOCS_INITIALIZERS_DISK_INIT_HPP

#include "../utils/utils.hpp"

namespace kocs::initializers {
  template<typename SimulationConfig>
  struct RandomDisk {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    VectorView positions_view;
    Scalar distance_nb;

    template<typename ViewType>
    RandomDisk(ViewType positions, Scalar distance_to_neighbour)
      : positions_view(positions)
      , distance_nb(distance_to_neighbour) { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Random& generator) const {
      const Scalar r_max = Kokkos::sqrt(Scalar(positions_view.extent(0)) / Scalar(0.9069)) * distance_nb / Scalar(2.0);
      const Scalar r  = r_max * Kokkos::sqrt(static_cast<Scalar>(generator.drand()));
      const Scalar phi = static_cast<Scalar>(generator.drand()) * Scalar(2.0) * Kokkos::numbers::pi_v<Scalar>;

      positions_view(i)[0] = Scalar(0.0);
      positions_view(i)[1] = r * Kokkos::sin(phi);
      positions_view(i)[2] = r * Kokkos::cos(phi);
    }
  };
} // namespace kocs::initializers

#endif // KOCS_INITIALIZERS_DISK_INIT_HPP
