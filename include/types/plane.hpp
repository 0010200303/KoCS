#ifndef KOCS_TYPES_PLANE_HPP
#define KOCS_TYPES_PLANE_HPP

#include <Kokkos_Core.hpp>

#include "vector.hpp"

namespace kocs {
  template<typename Scalar, unsigned int dimensions, unsigned int Align = alignof(Scalar)>
  struct alignas(Align) PlaneN {
    using Vector = VectorN<Scalar, dimensions>;

    VectorN<Scalar, dimensions + 1> coefficients;

    KOKKOS_INLINE_FUNCTION
    PlaneN() {
      coefficients[dimensions - 1] = Scalar(1);
    }

    template<typename... Scalars>
    KOKKOS_INLINE_FUNCTION
    PlaneN(const Scalars... scalars) : coefficients(static_cast<Scalar>(scalars)...) {
      Scalar squared_sum = Scalar(0);
      for (unsigned int i = 0; i < dimensions; ++i)
        squared_sum += coefficients[i] * coefficients[i];

      Scalar inv_norm = Scalar(1) / Kokkos::sqrt(squared_sum);
      for (unsigned int i = 0; i < dimensions + 1; ++i)
        coefficients[i] *= inv_norm;
    }

    KOKKOS_INLINE_FUNCTION
    Scalar signed_distance(const Vector& point) const {
      Scalar result = coefficients[dimensions];
      for (unsigned int i = 0; i < dimensions; ++i)
        result += coefficients[i] * point[i];
      return result;
    }

    KOKKOS_INLINE_FUNCTION
    Vector normal() const {
      Vector normal;
      for (unsigned int i = 0; i < dimensions; ++i)
        normal[i] = coefficients[i];
      return normal;
    }
  };

  template<typename Scalar>
  using Plane1 = PlaneN<Scalar, 1>;

  template<typename Scalar>
  using Plane2 = PlaneN<Scalar, 2>;

  template<typename Scalar>
  using Plane3 = PlaneN<Scalar, 3>;

  template<typename Scalar>
  using Plane4 = PlaneN<Scalar, 4>;
} // namespace kocs

#endif // KOCS_TYPES_PLANE_HPP
