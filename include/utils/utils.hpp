#ifndef KOCS_UTILS_UTILS_HPP
#define KOCS_UTILS_UTILS_HPP

#include <array>
#include <tuple>
#include <string_view>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "types/vector.hpp"
#include "types/polarity.hpp"
#include "types/link.hpp"

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



  // free function to support read, write and access function for Kokkos::View in internal use
  template<typename T, typename = void>
  struct has_read : std::false_type {};

  template<typename T>
  struct has_read<T, std::void_t<decltype(std::declval<const T&>().read(0))>> : std::true_type {};

  template<typename T, typename = void>
  struct has_write : std::false_type {};

  template<typename T>
  struct has_write<T, std::void_t<decltype(std::declval<T&>().write(0))>> : std::true_type {};

  template<typename T, typename = void>
  struct has_access : std::false_type {};

  template<typename T>
  struct has_access<T, std::void_t<decltype(std::declval<T&>().access(0))>> : std::true_type {};

  template<typename ViewType>
  KOKKOS_INLINE_FUNCTION
  auto read_element(const ViewType& v, const int i) -> decltype(v(i)) {
    if constexpr (has_read<ViewType>::value)
      return v.read(i);
    else
      return v(i);
  }

  template<typename ViewType>
  KOKKOS_INLINE_FUNCTION
  auto write_element(ViewType& v, const int i) -> decltype(v(i)) {
    if constexpr (has_write<ViewType>::value)
      return v.write(i);
    else
      return v(i);
  }

  template<typename ViewType>
  KOKKOS_INLINE_FUNCTION
  auto access_element(ViewType& v, const int i) -> decltype(v(i)) {
    if constexpr (has_access<ViewType>::value)
      return v.access(i);
    else
      return v(i);
  }
} // namespace kocs

// force macros
#define GENERIC_REF(__TYPE__, __NAME__) detail::GenericFieldRef<__TYPE__> __NAME__
#define PAIRWISE_REF(__TYPE__, __NAME__) detail::PairwiseFieldRef<__TYPE__> __NAME__

#define GENERIC_FORCE_IMPL kocs::detail::generic_force | KOKKOS_LAMBDA
#define PAIRWISE_FORCE_IMPL kocs::detail::pairwise_force | KOKKOS_LAMBDA
#define UPDATE_FUNC_IMPL kocs::detail::update_func | KOKKOS_LAMBDA
#define LINK_FORCE_IMPL kocs::detail::link_force | KOKKOS_LAMBDA

#define GENERIC_FORCE_PARAMETERS const bool is_full_step, const unsigned int i, Random& rng, const GenericForceFields& ctx
#define PAIRWISE_FORCE_PARAMETERS const bool is_full_step, const unsigned int i, const unsigned int j, \
  const Vector& displacement, const Scalar distance, Random& rng, Scalar& drag, const PairwiseForceFields& ctx
#define UPDATE_FUNC_PARAMETERS const unsigned int i, Random& rng
#define LINK_FORCE_PARAMETERS const bool is_full_step, const Link& link, Random& rng, const LinkForceFields& ctx

#define GENERIC_FORCE(...) [&]() { return GENERIC_FORCE_IMPL(GENERIC_FORCE_PARAMETERS) { __VA_ARGS__ }; }
#define PAIRWISE_FORCE(...) [&]() { return PAIRWISE_FORCE_IMPL(PAIRWISE_FORCE_PARAMETERS) { __VA_ARGS__ }; }
#define UPDATE_FUNC(...) [&]() { return UPDATE_FUNC_IMPL(UPDATE_FUNC_PARAMETERS) { __VA_ARGS__ }; }
#define LINK_FORCE(...) [&]() { return LINK_FORCE_IMPL(LINK_FORCE_PARAMETERS) { __VA_ARGS__ }; }

#define GENERIC_FORCE_OP() \
  using tag = kocs::detail::GenericForceTag; \
  KOKKOS_INLINE_FUNCTION void operator()(GENERIC_FORCE_PARAMETERS) const

#define PAIRWISE_FORCE_OP() \
  using tag = kocs::detail::PairwiseForceTag; \
  KOKKOS_INLINE_FUNCTION void operator()(PAIRWISE_FORCE_PARAMETERS) const

#define UPDATE_FUNC_OP() \
  using tag = kocs::detail::UpdateFuncTag; \
  KOKKOS_INLINE_FUNCTION void operator()(UPDATE_FUNC_PARAMETERS) const

#define LINK_FORCE_OP() \
  using tag = kocs::detail::LinkForceTag; \
  KOKKOS_INLINE_FUNCTION void operator()(LINK_FORCE_PARAMETERS) const

// special version for inits
#define INIT_FUNC(...) [&]() { return KOKKOS_LAMBDA(const unsigned int i, Random& rng) { __VA_ARGS__ }; }



#define EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Scalar = typename __SIMULATION_CONFIG__::Scalar; \
  static constexpr unsigned int dimensions = __SIMULATION_CONFIG__::dimensions; \
  using Vector = kocs::VectorN<Scalar, dimensions>; \
  using VectorI = kocs::VectorN<int, dimensions>; \
  using VectorView = kocs::View<Vector>; \
  using Polarity = kocs::Polarity_<Scalar>; \
  using Plane = kocs::PlaneN<Scalar, dimensions>; \
  using RandomPool = typename __SIMULATION_CONFIG__::RandomPoolT; \
  using Random = typename RandomPool::generator_type; \
  using GenericForceFields = typename __SIMULATION_CONFIG__::template ForceFields<kocs::detail::GenericFieldRef>; \
  using PairwiseForceFields = typename __SIMULATION_CONFIG__::template ForceFields<kocs::detail::PairwiseFieldRef>; \
  using LinkForceFields = typename __SIMULATION_CONFIG__::template ForceFields<kocs::detail::LinkFieldRef>;

#define EXTRACT_ALL_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(__SIMULATION_CONFIG__) \
  using Fields = typename __SIMULATION_CONFIG__::Fields; \
  using Integrator = typename __SIMULATION_CONFIG__::template IntegratorT<__SIMULATION_CONFIG__>; \
  using PairFinder = typename __SIMULATION_CONFIG__::template PairFinderT<__SIMULATION_CONFIG__>; \
  using ComFixer = typename __SIMULATION_CONFIG__::template ComFixerT<__SIMULATION_CONFIG__>; \
  using Writer = typename __SIMULATION_CONFIG__::template WriterT<__SIMULATION_CONFIG__>;

#define EXTRACT_VECTOR(__VECTOR__) \
  using Scalar = typename __VECTOR__::Scalar; \
  static constexpr unsigned int dimensions = __VECTOR__::dimensions; \

#endif // KOCS_UTILS_UTILS_HPP
