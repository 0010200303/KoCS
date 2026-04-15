#ifndef KOCS_UTILS_HPP
#define KOCS_UTILS_HPP

#include <tuple>
#include <string_view>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "vector.hpp"
#include "simulation_config.hpp"

namespace kocs {
  // extract types from Fields
  template<typename Tuple>
  struct extract_types;

  template<typename... Fs>
  struct extract_types<std::tuple<Fs...>> {
    using type = std::tuple<typename Fs::type...>;
  };

  template<typename Tuple>
  using storage_t = typename extract_types<Tuple>::type;



template<typename Fields>
struct ValuesFromFields;

template<typename... Specs>
struct ValuesFromFields<std::tuple<Specs...>> {
    struct type {
        using tuple_type = std::tuple<typename Specs::type::value_type...>;

        tuple_type data;

        KOKKOS_INLINE_FUNCTION
        type() = default;

        KOKKOS_INLINE_FUNCTION
        explicit type(const typename Specs::type&... args)
            : data(args...) {}

        KOKKOS_INLINE_FUNCTION
        type& operator+=(const type& rhs) {
            add_impl(rhs, std::make_index_sequence<sizeof...(Specs)>{});
            return *this;
        }

    private:
        template<std::size_t... I>
        KOKKOS_INLINE_FUNCTION
        void add_impl(const type& rhs, std::index_sequence<I...>) {
            ((std::get<I>(data) += std::get<I>(rhs.data)), ...);
        }
    };
};
} // namespace kocs

#define EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Scalar = typename __SIMULATION_CONFIG__::Scalar; \
  static constexpr unsigned int dimensions = __SIMULATION_CONFIG__::dimensions; \
  using Vector = kocs::VectorN<Scalar, dimensions>; \
  using VectorView = Kokkos::View<Vector*>; \
  using RandomPool = typename __SIMULATION_CONFIG__::RandomPoolT;

#define EXTRACT_ALL_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Fields = typename __SIMULATION_CONFIG__::Fields; \
  using Storage = kocs::storage_t<Fields>; \
  using LocalValues = typename ValuesFromFields<Fields>::type;

#define MAKE_DEFAULT_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  struct __SIMULATION_CONFIG__ : public DefaultSimulationConfig { }; \
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__)

#endif // KOCS_UTILS_HPP
