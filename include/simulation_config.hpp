#ifndef KOCS_SIMULATION_CONFIG
#define KOCS_SIMULATION_CONFIG

#include <string_view>

#include "integrators/detail.hpp"
#include "integrators/euler.hpp"
#include "pair_finders/all_pairs.hpp"

namespace kocs {
  namespace detail {
    template<template<typename...> typename IntegratorT, typename... Views>
    using integrator_t = IntegratorT<Views...>;

    template<template<typename...> typename PairFinderT, typename... Views>
    using pair_finder_t = PairFinderT<first_type_t<Views...>, Views...>;
  } // namespace detail

  template <std::size_t N>
  struct fixed_string {
    char data[N];

    consteval fixed_string(const char (&str)[N]) {
      for (std::size_t i = 0; i < N; ++i)
        data[i] = str[i];
    }

    constexpr operator std::string_view() const {
      return {data, N - 1};
    }
  };

  template<typename T, fixed_string Name>
  struct Field {
    using type = T;
    static constexpr auto name = Name;
  };

  template <typename... Fields>
  struct FieldList {};



#define CONFIG_SCALAR(__SCALAR__) \
  using Scalar = __SCALAR__;

#define CONFIG_DIMENSIONS(__DIMENSIONS__) \
  static constexpr int dimensions = __DIMENSIONS__;

#define CONFIG_RANDOM_POOL(__RANDOM_POOL__) \
  using RandomPoolT = __RANDOM_POOL__<>;

#define CONFIG_INTEGRATOR(__INTEGRATOR__) \
  template<typename... Views> \
  using IntegratorT = kocs::detail::integrator_t<__INTEGRATOR__, Views...>;

#define CONFIG_PAIR_FINDER(__PAIR_FINDER_TYPE__) \
  template<typename PositionsView, typename... Views> \
  using PairFinderT = kocs::detail::pair_finder_t<__PAIR_FINDER_TYPE__, Views...>;

#define FIELD(__SCALAR_TYPE__, __FIELD_NAME__) \
  kocs::Field<__SCALAR_TYPE__, __FIELD_NAME__>

#define CONFIG_FIELDS(...) \
  using Fields = kocs::FieldList<__VA_ARGS__>;

  // default simulation configs
  struct DefaultSimulationConfig {
    CONFIG_SCALAR(float)
    CONFIG_DIMENSIONS(3)

    using Vector = VectorN<Scalar, dimensions>;
    using VectorView = Kokkos::View<Vector*>;

    CONFIG_RANDOM_POOL(Kokkos::Random_XorShift64_Pool)
    CONFIG_INTEGRATOR(integrators::Euler)
    CONFIG_PAIR_FINDER(pair_finders::NaiveAllPairs)

    CONFIG_FIELDS(
      FIELD(Vector, "positions")
    )
  };
} // namespace kocs

#endif // KOCS_SIMULATION_CONFIG
