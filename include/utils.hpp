#ifndef KOCS_UTILS_HPP
#define KOCS_UTILS_HPP

#include <tuple>

#include <Kokkos_Core.hpp>

#include "vector.hpp"

namespace kocs {
  // check if every Field is a Kokkos::View
  template<typename T>
  struct is_kokkos_view : std::false_type { };

  template<typename DataType, typename... Args>
  struct is_kokkos_view<Kokkos::View<DataType, Args...>> : std::true_type { };

  template<typename Tuple, std::size_t... I>
  constexpr bool tuple_all_kokkos_views_impl(std::index_sequence<I...>) {
    return (is_kokkos_view<std::tuple_element_t<I, Tuple>>::value && ...);
  }

  template<typename Tuple>
  constexpr bool all_kokkos_views_v =
    tuple_all_kokkos_views_impl<Tuple>(std::make_index_sequence<std::tuple_size<Tuple>::value>{});



  // helper functions for IntegrationFields from simulation config
  template<typename Fields>
  struct ViewsFromFields;

  template<typename... Views>
  struct ViewsFromFields<std::tuple<Views...>> {
    using type = std::tuple<Views...>;
  };
    
  template<typename Fields>
  struct ValuesFromFields;

  template<typename... Specs>
  struct ValuesFromFields<std::tuple<Specs...>> {
    struct type {
      public:
        std::tuple<typename Specs::value_type...> data;

        KOKKOS_INLINE_FUNCTION type() : data{} { }

        template<typename... Args,
          typename = std::enable_if_t<sizeof...(Args) == sizeof...(Specs)>>
        KOKKOS_INLINE_FUNCTION explicit type(Args... args) : data(args...) { }

        KOKKOS_INLINE_FUNCTION type& operator+=(const type& rhs) {
          add_impl(rhs, std::make_index_sequence<sizeof...(Specs)>{});
          return *this;
        }

      private:
        template<std::size_t... I>
        KOKKOS_INLINE_FUNCTION void add_impl(const type& rhs, std::index_sequence<I...>) {
          ((std::get<I>(data) = std::get<I>(data) + std::get<I>(rhs.data)), ...);
        }
    };
  };
} // namespace kocs

#define EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Scalar = typename __SIMULATION_CONFIG__::Scalar; \
  static constexpr unsigned int dimensions = __SIMULATION_CONFIG__::dimensions; \
  using Vector = kocs::VectorN<Scalar, dimensions>; \
  using VectorView = Kokkos::View<Vector*>;

#define EXTRACT_ALL_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using FieldSpecs = typename SimulationConfig::IntegrationFields; \
  using Fields = typename ViewsFromFields<FieldSpecs>::type; \
  using LocalValues = typename ValuesFromFields<FieldSpecs>::type; \
  static constexpr const char* const* FieldNames = SimulationConfig::IntegrationFieldNames; \
  static constexpr std::size_t FieldNamesCount = \
    std::extent<decltype(SimulationConfig::IntegrationFieldNames)>::value;

// default simulation configs
struct DefaultSimulationConfig {
  using Scalar = float;
  static constexpr int dimensions = 3;

  using Vector = kocs::VectorN<Scalar, dimensions>;
  using VectorView = Kokkos::View<Vector*>;

  using IntegrationFields = std::tuple<
    VectorView // positions
  >;

  static constexpr const char* IntegrationFieldNames[] = {
    "positions"
  };
};

#define MAKE_DEFAULT_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  struct __SIMULATION_CONFIG__ : public DefaultSimulationConfig { }; \
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

#endif // KOCS_UTILS_HPP
