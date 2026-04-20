#ifndef KOCS_SIMULATION_CONFIG
#define KOCS_SIMULATION_CONFIG

#include <string_view>

#include "integrators/euler.hpp"

namespace kocs {
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

    // friend consteval bool operator==(fixed_string const& a, fixed_string const& b) {
    //   for (std::size_t i = 0; i < N; ++i) {
    //     if (a.data[i] != b.data[i])
    //       return false;
    //   }
    //   return true;
    // }
  };

  template<typename T, fixed_string Name>
  struct Field {
    using type = T;
    static constexpr auto name = Name;

    // static type make(unsigned int n) {
    //   return type(Name.data, n);
    // }
  };

  template <typename... Fields>
  struct FieldList {};

  

  // default simulation configs
  struct DefaultSimulationConfig {
    using Scalar = float;
    static constexpr int dimensions = 3;

    using Vector = VectorN<Scalar, dimensions>;
    using VectorView = Kokkos::View<Vector*>;

    using RandomPoolT = Kokkos::Random_XorShift64_Pool<>;

    template<typename... Views>
    using Integrator = integrator::Euler<Views...>;

    using Fields = FieldList<
      Field<Vector*, "positions">
    >;
  };
} // namespace kocs

#endif // KOCS_SIMULATION_CONFIG
