#ifndef KOCS_TYPES_VECTOR_HPP
#define KOCS_TYPES_VECTOR_HPP

#include <type_traits>
#include <array>

#include <Kokkos_Core.hpp>
#include <highfive/H5File.hpp>
 
namespace kocs {
  template<typename Scalar_, unsigned int dimensions_, unsigned int Align = alignof(Scalar_)>
  struct alignas(Align) VectorN {
    using Scalar = Scalar_;
    static constexpr unsigned int dimensions = dimensions_;

    static_assert(dimensions > 0, "Vector dimensions must be greater than 0");
    static_assert(std::is_trivially_copyable<Scalar>::value, "Scalar must be trivially copyable");

    using value_type = Scalar;
    using pointer = Scalar*;
    using const_pointer = const Scalar*;
    using reference = Scalar&;
    using const_reference = const Scalar&;
    using iterator = Scalar*;
    using const_iterator = const Scalar*;

    Scalar data_[dimensions];

    // constructors
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN() : data_{} { }

    KOKKOS_INLINE_FUNCTION
    constexpr explicit VectorN(Scalar value) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] = value;
    }

    template<typename... Args, std::enable_if_t<sizeof...(Args) == dimensions, int> = 0>
    KOKKOS_INLINE_FUNCTION
    constexpr VectorN(Args... args) : data_{static_cast<Scalar>(args)...} { }

    constexpr explicit VectorN(const std::array<Scalar, dimensions>& array) {
      for (unsigned int i = 0; i < dimensions; ++i)
        data_[i] = array[i];
    }

    // std algorithms / container compatibility
    KOKKOS_INLINE_FUNCTION constexpr std::size_t size() const { return dimensions; }
    KOKKOS_INLINE_FUNCTION constexpr Scalar* data() { return data_; }
    KOKKOS_INLINE_FUNCTION constexpr const Scalar* data() const { return data_; }
    KOKKOS_INLINE_FUNCTION constexpr iterator begin() { return data_; }
    KOKKOS_INLINE_FUNCTION constexpr iterator end() { return data_ + dimensions; }
    KOKKOS_INLINE_FUNCTION constexpr const_iterator begin() const { return data_; }
    KOKKOS_INLINE_FUNCTION constexpr const_iterator end() const { return data_ + dimensions; }
    KOKKOS_INLINE_FUNCTION constexpr const_iterator cbegin() const { return data_; }
    KOKKOS_INLINE_FUNCTION constexpr const_iterator cend() const { return data_ + dimensions; }

    // indexing
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar& operator[](int i) { return data_[i]; }

    KOKKOS_INLINE_FUNCTION
    constexpr const Scalar& operator[](int i) const { return data_[i]; }

    template<int I>
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar& get() {
      static_assert(I < dimensions, "Index out of bounds");
      return data_[I];
    }

    template<int I>
    KOKKOS_INLINE_FUNCTION
    constexpr const Scalar& get() const {
      static_assert(I < dimensions, "Index out of bounds");
      return data_[I];
    }

    // named accessors
    KOKKOS_INLINE_FUNCTION constexpr Scalar& x() { static_assert(dimensions > 0); return data_[0]; }
    KOKKOS_INLINE_FUNCTION constexpr Scalar& y() { static_assert(dimensions > 1); return data_[1]; }
    KOKKOS_INLINE_FUNCTION constexpr Scalar& z() { static_assert(dimensions > 2); return data_[2]; }
    KOKKOS_INLINE_FUNCTION constexpr Scalar& w() { static_assert(dimensions > 3); return data_[3]; }

    KOKKOS_INLINE_FUNCTION constexpr const Scalar& x() const { static_assert(dimensions > 0); return data_[0]; }
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& y() const { static_assert(dimensions > 1); return data_[1]; }
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& z() const { static_assert(dimensions > 2); return data_[2]; }
    KOKKOS_INLINE_FUNCTION constexpr const Scalar& w() const { static_assert(dimensions > 3); return data_[3]; }
  
    // arithmetic operators
    KOKKOS_INLINE_FUNCTION
    VectorN& operator+=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] += rhs.data_[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator-=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] -= rhs.data_[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator*=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] *= rhs.data_[i];
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator/=(const VectorN& rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] /= rhs.data_[i];
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
        data_[i] += rhs;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator-=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] -= rhs;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator*=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] *= rhs;
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    VectorN& operator/=(Scalar rhs) {
      for (int i = 0; i < dimensions; ++i)
        data_[i] /= rhs;
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
        rhs.data_[i] = lhs - rhs.data_[i];
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
        rhs.data_[i] = lhs / rhs.data_[i];
      return rhs;
    }

    // vector math
    KOKKOS_INLINE_FUNCTION
    constexpr Scalar dot(const VectorN& rhs) const {
      Scalar result{};
      for (int i = 0; i < dimensions; ++i)
        result += data_[i] * rhs.data_[i];
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
        const Scalar d = data_[i] - rhs.data_[i];
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
        data_[i] /= len;
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
        data_[1] * rhs.data_[2] - data_[2] * rhs.data_[1],
        data_[2] * rhs.data_[0] - data_[0] * rhs.data_[2],
        data_[0] * rhs.data_[1] - data_[1] * rhs.data_[0]
      };
    }

    KOKKOS_INLINE_FUNCTION
    constexpr VectorN orthonormal(const VectorN& rhs) const {
      VectorN normal = *this - dot(rhs) * rhs;
      return normal / Kokkos::sqrt(normal.dot(normal));
    }

    // TODO: remove, use dimensions member variable instead
    constexpr unsigned int get_dimensions() const {
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
        result.data_[i] = -data_[i];
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



namespace Kokkos {
  template<typename Scalar, unsigned int dimensions, unsigned int Align>
  struct reduction_identity<kocs::VectorN<Scalar, dimensions, Align>> {
    using value_type = kocs::VectorN<Scalar, dimensions, Align>;

    KOKKOS_FORCEINLINE_FUNCTION
    static constexpr value_type sum() {
      return value_type{};
    }
  };
} // namespace Kokkos



namespace HighFive::details {
  template <typename Scalar, unsigned int dimensions, unsigned int Align>
  struct inspector<kocs::VectorN<Scalar, dimensions, Align>> {
    using type = kocs::VectorN<Scalar, dimensions, Align>;
    using base_type = typename inspector<Scalar>::base_type;
    using hdf5_type = typename inspector<Scalar>::hdf5_type;

    static constexpr size_t ndim = inspector<Scalar>::ndim + 1;
    static constexpr size_t min_ndim = inspector<Scalar>::min_ndim + 1;
    static constexpr size_t max_ndim = inspector<Scalar>::max_ndim + 1;
    static constexpr bool is_trivially_nestable = inspector<Scalar>::is_trivially_nestable;
    static constexpr bool is_trivially_copyable =
      std::is_trivially_copyable<type>::value && inspector<Scalar>::is_trivially_copyable;

    static size_t getRank(const type& val) {
      return ndim + inspector<Scalar>::getRank(val[0]);
    }

    static std::vector<size_t> getDimensions(const type& val) {
      std::vector<size_t> result = inspector<Scalar>::getDimensions(val[0]);
      result.insert(result.begin(), dimensions);
      return result;
    }

    static void prepare(type& value, const std::vector<size_t>& next_dims) {
      for (unsigned int i = 0; i < dimensions; ++i)
        inspector<Scalar>::prepare(value[i], next_dims);
    }

    static hdf5_type* data(type& value) {
      return inspector<Scalar>::data(value[0]);
    }

    static const hdf5_type* data(const type& value) {
      return inspector<Scalar>::data(value[0]);
    }
  };
} // namespace HighFive::details

#endif // KOCS_TYPES_VECTOR_HPP
