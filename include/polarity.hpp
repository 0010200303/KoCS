#ifndef KOCS_POLARITY_HPP
#define KOCS_POLARITY_HPP

#include <vector.hpp>

namespace kocs {
  template<typename Scalar, unsigned int Align = alignof(Scalar)>
  struct alignas(Align) Polarity_ : public VectorN<Scalar, 2, Align> {
    using Base = VectorN<Scalar, 2, Align>;
    using Base::Base;

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_() = default;

    KOKKOS_INLINE_FUNCTION
    constexpr explicit Polarity_(Scalar value) : Base(value) { }

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_(Scalar theta, Scalar phi) : Base(theta, phi) { }

    KOKKOS_INLINE_FUNCTION constexpr Scalar& theta() { return this->data[0]; }
    KOKKOS_INLINE_FUNCTION constexpr Scalar& phi() { return this->data[1]; }

    KOKKOS_INLINE_FUNCTION constexpr const Scalar& theta() const { return this->data[0]; }
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& phi() const { return this->data[1]; }

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_& operator=(const Base& rhs) {
      this->data[0] = rhs.data[0];
      this->data[1] = rhs.data[1];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Polarity_& operator=(Scalar value) {
      this->data[0] = value;
      this->data[1] = value;
      return *this;
    }
  };
}

#endif // KOCS_POLARITY_HPP
