#ifndef KOCS_UTILS_HPP
#define KOCS_UTILS_HPP

#define EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Scalar = typename __SIMULATION_CONFIG__::Scalar; \
  static constexpr unsigned int dimensions = __SIMULATION_CONFIG__::dimensions; \
  using Vector = kocs::VectorN<Scalar, dimensions>; \
  using VectorView = Kokkos::View<Vector*>;

#endif // KOCS_UTILS_HPP
