#ifndef KOCS_FORCES_PREDEFINED_HPP
#define KOCS_FORCES_PREDEFINED_HPP

#include <Kokkos_Core.hpp>

#include "detail.hpp"

namespace kocs::forces {
  template<typename Scalar, typename Vector>
  static Vector PiecewiseLinearSpring(
    const Vector& displacement,
    const Scalar distance,
    const Scalar min,
    const Scalar max,
    const Scalar min_scale = Scalar(2.0),
    const Scalar max_scale = Scalar(1.0)
  ) {
    Scalar F = Kokkos::fmax(min - distance, 0) * min_scale -
      Kokkos::fmax(distance - max, 0) * max_scale;
    return displacement * F / distance;
  }
} // namespace kocs::forces

#endif // KOCS_FORCES_PREDEFINED_HPP
