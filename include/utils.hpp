#ifndef KOCS_UTILS_HPP
#define KOCS_UTILS_HPP

#include <array>
#include <tuple>
#include <string_view>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "vector.hpp"

namespace kocs {
  // extract types from Fields
  template<typename Tuple>
  struct extract_types;

  template<typename... Fs>
  struct extract_types<std::tuple<Fs...>> {
    static_assert(sizeof...(Fs) > 0, "Storage requires at least one field");
    using first_type = typename std::tuple_element_t<0, std::tuple<Fs...>>::type;
    static_assert((std::is_same_v<first_type, typename Fs::type> && ...),
      "All Storage field types must match");

    using type = std::array<first_type, sizeof...(Fs)>;
  };

  template<typename Tuple>
  using storage_t = typename extract_types<Tuple>::type;

  

  template<typename Fields>
  struct ValuesFromFields;

  template<typename... Specs>
  struct ValuesFromFields<std::tuple<Specs...>> {
    struct type {
      public:
        static_assert(sizeof...(Specs) > 0, "ValuesFromFields requires at least one field");

        using value_type = typename std::tuple_element_t<0, std::tuple<Specs...>>::type::value_type;
        static_assert((std::is_same_v<value_type, typename Specs::type::value_type> && ...),
          "All field value types must match");

        using tuple_type = value_type[sizeof...(Specs)];

        tuple_type data{};

        KOKKOS_INLINE_FUNCTION
        type() = default;

        template<typename... Args, std::enable_if_t<sizeof...(Args) == sizeof...(Specs), int> = 0>
        KOKKOS_INLINE_FUNCTION
        explicit type(Args&&... args) : data{static_cast<value_type>(std::forward<Args>(args))...} { }

        KOKKOS_INLINE_FUNCTION
        type& operator+=(const type& rhs) {
          add_impl(rhs, std::make_index_sequence<sizeof...(Specs)>{});
          return *this;
        }

    private:
      template<std::size_t... I>
      KOKKOS_INLINE_FUNCTION
      void add_impl(const type& rhs, std::index_sequence<I...>) {
        ((data[I] += rhs.data[I]), ...);
      }
    };
  };
} // namespace kocs

// force macros
#define GENERIC_FORCE_IMPL kocs::detail::generic_force | KOKKOS_LAMBDA
#define PAIRWISE_FORCE_IMPL kocs::detail::pairwise_force | KOKKOS_LAMBDA

#define GENERIC_REF_IMPL(__TYPE__) detail::GenericFieldRef<__TYPE__>
#define PAIRWISE_REF_IMPL(__TYPE__) detail::PairwiseFieldRef<__TYPE__>

#define GENERIC_REF(__TYPE__, __NAME__) GENERIC_REF_IMPL(__TYPE__) __NAME__
#define PAIRWISE_REF(__TYPE__, __NAME__) PAIRWISE_REF_IMPL(__TYPE__) __NAME__

#define GENERIC_FORCE(...) GENERIC_FORCE_IMPL( \
  const unsigned int i, \
  Random& rng __VA_OPT__(,) __VA_ARGS__)
#define PAIRWISE_FORCE(...) PAIRWISE_FORCE_IMPL(__VA_ARGS__)


#define EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Scalar = typename __SIMULATION_CONFIG__::Scalar; \
  static constexpr unsigned int dimensions = __SIMULATION_CONFIG__::dimensions; \
  using Vector = kocs::VectorN<Scalar, dimensions>; \
  using VectorView = Kokkos::View<Vector*>; \
  using Polarity = kocs::Polarity_<Scalar>; \
  using RandomPool = typename __SIMULATION_CONFIG__::RandomPoolT; \
  using Random = typename RandomPool::generator_type;

#define EXTRACT_ALL_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Fields = typename __SIMULATION_CONFIG__::Fields; \
  using Integrator = typename __SIMULATION_CONFIG__::template IntegratorT<__SIMULATION_CONFIG__>; \
  using PairFinder = typename __SIMULATION_CONFIG__::template PairFinderT<__SIMULATION_CONFIG__>; \
  using Writer = typename __SIMULATION_CONFIG__::template WriterT<__SIMULATION_CONFIG__>;

// #define MAKE_DEFAULT_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
//   struct __SIMULATION_CONFIG__ : public DefaultSimulationConfig { }; \
//   EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__)

#endif // KOCS_UTILS_HPP
