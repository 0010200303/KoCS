#ifndef KOCS_UTILS_FUNCTION_HPP
#define KOCS_UTILS_FUNCTION_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "utils.hpp"
#include "../types/vector.hpp"
#include "../types/view.hpp"

namespace kocs::utils {
  template<typename Vector>
  struct Bounds {
    Vector min;
    Vector max;
  };

  template<typename Vector>
  struct MinPerDimension {
    EXTRACT_VECTOR(Vector)

    View<Vector> points_view;

    MinPerDimension(const View<Vector> points_view_) : points_view(points_view_) { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Vector& local) const {
      Vector point = points_view(i);
      for (unsigned int d = 0; d < dimensions; ++d) {
        const Scalar value = point[d];
        if (value < local[d])
          local[d] = value;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void init(Vector& value) const {
      value = Vector(Kokkos::Experimental::finite_max_v<Scalar>);
    }

    KOKKOS_INLINE_FUNCTION
    void join(Vector& dst, const Vector& src) const {
      for (unsigned int d = 0; d < dimensions; ++d) {
        if (src[d] < dst[d])
          dst[d] = src[d];
      }
    }
  };

  template<typename Vector>
  struct MaxPerDimension {
    EXTRACT_VECTOR(Vector)

    View<Vector> points_view;

    MaxPerDimension(const View<Vector> points_view_) : points_view(points_view_) { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Vector& local) const {
      Vector point = points_view(i);
      for (unsigned int d = 0; d < dimensions; ++d) {
        const Scalar value = point[d];
        if (value > local[d])
          local[d] = value;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void init(Vector& value) const {
      value = Vector(Kokkos::Experimental::finite_min_v<Scalar>);
    }

    KOKKOS_INLINE_FUNCTION
    void join(Vector& dst, const Vector& src) const {
      for (unsigned int d = 0; d < dimensions; ++d) {
        if (src[d] > dst[d])
          dst[d] = src[d];
      }
    }
  };

  template<typename Vector>
  struct BoundsPerDimension {
    EXTRACT_VECTOR(Vector)

    View<Vector> points_view;

    BoundsPerDimension(const View<Vector> points_view_) : points_view(points_view_) { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Bounds<Vector>& local) const {
      Vector point = points_view(i);
      for (unsigned int d = 0; d < dimensions; ++d) {
        const Scalar value = point[d];
        if (value < local.min[d])
          local.min[d] = value;
        if (value > local.max[d])
          local.max[d] = value;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void init(Bounds<Vector>& value) const {
      value = Bounds<Vector>(
        Vector(Kokkos::Experimental::finite_max_v<Scalar>),
        Vector(Kokkos::Experimental::finite_min_v<Scalar>)
      );
    }

    KOKKOS_INLINE_FUNCTION
    void join(Bounds<Vector>& dst, const Bounds<Vector>& src) const {
      for (unsigned int d = 0; d < dimensions; ++d) {
        if (src.min[d] < dst.min[d])
          dst.min[d] = src.min[d];
        if (src.max[d] > dst.max[d])
          dst.max[d] = src.max[d];
      }
    }
  };

  template<typename Vector>
  Vector get_min_bounds(const View<Vector> points_view) {
    Vector result;
    Kokkos::parallel_reduce(
      "utils_get_min_bounds",
      points_view.extent(0),
      MinPerDimension(points_view),
      result
    );
    return result;
  }

  template<typename Vector>
  Vector get_max_bounds(const View<Vector> points_view) {
    Vector result;
    Kokkos::parallel_reduce(
      "utils_get_max_bounds",
      points_view.extent(0),
      MaxPerDimension(points_view),
      result
    );
    return result;
  }

  template<typename Vector>
  Bounds<Vector> get_bounds(const View<Vector> points_view) {
    Bounds<Vector> result;
    Kokkos::parallel_reduce(
      "utils_get_bounds",
      points_view.extent(0),
      BoundsPerDimension(points_view),
      result
    );
    return result;
  }
} // namespace kocs::utils

#endif // KOCS_UTILS_FUNCTION_HPP
