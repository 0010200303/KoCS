#ifndef KOCS_POLARITY_HPP
#define KOCS_POLARITY_HPP

#include <vector.hpp>

namespace kocs {
  template<typename Scalar, unsigned int Align = alignof(Scalar)>
  struct alignas(Align) Polarity_ : public VectorN<Scalar, 2, Align> {
    using Base = VectorN<Scalar, 2, Align>;
    using Base::Base;

    static const constexpr Scalar epsilon = Scalar(1e-6);

    struct BendingForceResult {
      Vector3<Scalar> vector;
      Polarity_ polarity;
    };

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_() = default;

    KOKKOS_INLINE_FUNCTION
    constexpr explicit Polarity_(Scalar value) : Base(value) { }

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_(Scalar theta, Scalar phi) : Base(theta, phi) { }

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_(const Base& value) : Base(value) { }

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_(const Vector3<Scalar>& vector, const Scalar distance) : Base(
      Kokkos::acos(vector[2] / distance),
      Kokkos::atan2(vector[1], vector[0])
    ) { }

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_(const Vector3<Scalar>& vector) : Base(
      Kokkos::acos(vector[2] / vector.length()),
      Kokkos::atan2(vector[1], vector[0])
    ) { }

    KOKKOS_INLINE_FUNCTION constexpr Scalar& theta() { return this->data[0]; }
    KOKKOS_INLINE_FUNCTION constexpr Scalar& phi() { return this->data[1]; }

    KOKKOS_INLINE_FUNCTION constexpr const Scalar& theta() const { return this->data[0]; }
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& phi() const { return this->data[1]; }

    KOKKOS_INLINE_FUNCTION
    static constexpr Polarity_ from_vector3(const Vector3<Scalar>& vector, const Scalar distance) {
      return Polarity_(vector, distance);
    }

    KOKKOS_INLINE_FUNCTION
    static constexpr Polarity_ from_vector3(const Vector3<Scalar>& vector) {
      return Polarity_(vector);
    }

    KOKKOS_INLINE_FUNCTION
    static constexpr Vector3<Scalar> to_vector3(const Polarity_& polarity) {
      return Vector3<Scalar>{
        Kokkos::sin(polarity[0]) * Kokkos::cos(polarity[1]),
        Kokkos::sin(polarity[0]) * Kokkos::sin(polarity[1]),
        Kokkos::cos(polarity[0])
      };
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Vector3<Scalar> to_vector3() const {
      return to_vector3(*this);
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Scalar dot(const Polarity_& rhs) const {
      return Kokkos::sin(this->data[0]) * Kokkos::sin(rhs[0]) *
        Kokkos::cos(this->data[1] - rhs[1]) +
        Kokkos::cos(this->data[0]) * Kokkos::cos(rhs[0]);
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_ unidirectional_polarization_force(const Polarity_& other) const {
      Polarity_ result{
        Kokkos::cos(this->data[0]) * Kokkos::sin(other[0]) * Kokkos::cos(this->data[1] - other[1]) - 
        Kokkos::sin(this->data[0]) * Kokkos::cos(other[0]),
        0
      };

      Scalar sin_theta = Kokkos::sin(this->data[0]);
      if (Kokkos::abs(sin_theta) > epsilon)
        result[1] = -Kokkos::sin(other[0]) * Kokkos::sin(this->data[1] - other[1]) / sin_theta;
      return result;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_ bidirectional_polarization_force(const Polarity_& other) const {
      return dot(other) * unidirectional_polarization_force(other);
    }

    KOKKOS_INLINE_FUNCTION
    constexpr BendingForceResult bending_force(
      const Vector3<Scalar>& displacement,
      const Polarity_& other_polarity,
      const Scalar distance
    ) const {
      Vector3<Scalar> this_vector = to_vector3();
      Scalar prod_i = this_vector.dot(displacement) / distance;

      Vector3<Scalar> other_vector = to_vector3(other_polarity);
      Scalar prod_j = other_vector.dot(displacement) / distance;

      return BendingForceResult{
        -prod_i / distance * this_vector + Kokkos::pow(prod_i, 2) / Kokkos::pow(distance, 2) * displacement +
        -prod_j / distance * other_vector + Kokkos::pow(prod_j, 2) / Kokkos::pow(distance, 2) * displacement,
        -prod_i * unidirectional_polarization_force(from_vector3(displacement, distance))
      };
    }
  };
}

#endif // KOCS_POLARITY_HPP
