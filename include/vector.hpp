#ifndef KOCS_VECTOR_HPP
#define KOCS_VECTOR_HPP

#include <type_traits>
#include <array>

#include <Kokkos_Core.hpp>
 
namespace kocs {
  template<typename Scalar, unsigned int dimensions, unsigned int Align = alignof(Scalar)>
  struct alignas(Align) VectorN {
    static_assert(dimensions > 0, "Vector dimensions must be greater than 0");
    static_assert(std::is_trivially_copyable<Scalar>::value, "Scalar must be trivially copyable");

    Scalar data[dimensions];

    // constructors
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN() : data{} { }

    KOKKOS_INLINE_FUNCTION
    constexpr explicit VectorN(Scalar value) {
      for (int i = 0; i < dimensions; ++i)
        data[i] = value;
    }

    template<typename... Args, typename = std::enable_if<sizeof...(Args) == dimensions>>
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN(Args... args) : data{static_cast<Scalar>(args)...} { }

    // indexing
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar& operator[](int i) { return data[i]; }

    KOKKOS_INLINE_FUNCTION
    constexpr const Scalar& operator[](int i) const { return data[i]; }

    template<int I>
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar& get() {
      static_assert(I < dimensions, "Index out of bounds");
      return data[I];
    }

    template<int I>
    KOKKOS_INLINE_FUNCTION
    constexpr const Scalar& get() const {
      static_assert(I < dimensions, "Index out of bounds");
      return data[I];
    }

    // named accessors
    KOKKOS_INLINE_FUNCTION constexpr Scalar& x() { static_assert(dimensions > 0); return data[0]; }
    KOKKOS_INLINE_FUNCTION constexpr Scalar& y() { static_assert(dimensions > 1); return data[1]; }
    KOKKOS_INLINE_FUNCTION constexpr Scalar& z() { static_assert(dimensions > 2); return data[2]; }
    KOKKOS_INLINE_FUNCTION constexpr Scalar& w() { static_assert(dimensions > 3); return data[3]; }

    KOKKOS_INLINE_FUNCTION constexpr const Scalar& x() const { static_assert(dimensions > 0); return data[0]; }
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& y() const { static_assert(dimensions > 1); return data[1]; }
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& z() const { static_assert(dimensions > 2); return data[2]; }
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& w() const { static_assert(dimensions > 3); return data[3]; }
  
    // arithmetic operators
    KOKKOS_INLINE_FUNCTION
    VectorN& operator+=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data[i] += rhs.data[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator-=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data[i] -= rhs.data[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator*=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data[i] *= rhs.data[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator/=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data[i] /= rhs.data[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator+(VectorN lhs, const VectorN& rhs) {
      lhs += rhs;
      return lhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator-(VectorN lhs, const VectorN& rhs) {
      lhs -= rhs;
      return lhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator*(VectorN lhs, const VectorN& rhs) {
      lhs *= rhs;
      return lhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator/(VectorN lhs, const VectorN& rhs) {
      lhs /= rhs;
      return lhs;
    }

    // scalar arithmetic operations
    KOKKOS_INLINE_FUNCTION
    VectorN& operator+=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data[i] += rhs;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator-=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data[i] -= rhs;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator*=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data[i] *= rhs;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator/=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data[i] /= rhs;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator+(VectorN lhs, Scalar rhs) {
      lhs += rhs;
      return lhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator-(VectorN lhs, Scalar rhs) {
      lhs -= rhs;
      return lhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator*(VectorN lhs, Scalar rhs) {
      lhs *= rhs;
      return lhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator/(VectorN lhs, Scalar rhs) {
      lhs /= rhs;
      return lhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator+(Scalar lhs, VectorN rhs) {
      rhs += lhs;
      return rhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator-(Scalar lhs, VectorN rhs) {
      for (int i = 0; i < dimensions; ++i)
        rhs.data[i] = lhs - rhs.data[i];
      return rhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator*(Scalar lhs, VectorN rhs) {
      rhs *= lhs;
      return rhs;
    }

    KOKKOS_INLINE_FUNCTION
    friend VectorN operator/(Scalar lhs, VectorN rhs) {
      for (int i = 0; i < dimensions; ++i)
        rhs.data[i] = lhs / rhs.data[i];
      return rhs;
    }

    // vector math
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar dot(const VectorN& rhs) const {
      Scalar result{};
      for (int i = 0; i < dimensions; ++i)
        result += data[i] * rhs.data[i];
      return result;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Scalar length_squared() const {
      return dot(*this);
    }

    KOKKOS_INLINE_FUNCTION
    Scalar length() const {
      return Kokkos::sqrt(length_squared());
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Scalar distance_to_squared(const VectorN& rhs) const {
      Scalar result{};
      for (int i = 0; i < dimensions; ++i) {
        const Scalar d = data[i] - rhs.data[i];
        result += d * d;
      }
      return result;
    }

    KOKKOS_INLINE_FUNCTION
    Scalar distance_to(const VectorN& rhs) const {
      return Kokkos::sqrt(distance_to_squared(rhs));
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& normalize() {
      const Scalar len = length();
      for (int i = 0; i < dimensions; ++i)
        data[i] /= len;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN normalized() const {
      VectorN result = *this;
      result.normalize();
      return result;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr VectorN cross(const VectorN& rhs) const {
      static_assert(dimensions == 3, "Cross product is only defined for 3D vectors");
      return VectorN{
        data[1] * rhs.data[2] - data[2] * rhs.data[1],
        data[2] * rhs.data[0] - data[0] * rhs.data[2],
        data[0] * rhs.data[1] - data[1] * rhs.data[0]
      };
    }

    // enable easy HighFive writing
    // TODO: optimize
    KOKKOS_INLINE_FUNCTION
    std::array<Scalar, dimensions> to_array() const {
      std::array<Scalar, dimensions> out{};
      for (int i = 0; i < dimensions; ++i)
        out[i] = data[i];
      return out;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr const unsigned int get_dimensions() const {
      return dimensions;
    }

    // unary operations
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN operator+() const {
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr VectorN operator-() const {
      VectorN result;
      for (int i = 0; i < dimensions; ++i)
        result.data[i] = -data[i];
      return result;
    }
  };

  template<typename Scalar>
  using Vector1 = VectorN<Scalar, 1>;

  template<typename Scalar>
  using Vector2 = VectorN<Scalar, 2>;

  template<typename Scalar>
  using Vector3 = VectorN<Scalar, 3>;

  template<typename Scalar>
  using Vector4 = VectorN<Scalar, 4>;
} // namespace kocs

#endif // KOCS_VECTOR_HPP
