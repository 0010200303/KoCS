#ifndef KOCS_SIMULATION_CONFIG
#define KOCS_SIMULATION_CONFIG

#include <string_view>

#include <Kokkos_Random.hpp>

#include "types/view.hpp"
#include "integrators/detail.hpp"
#include "integrators/heun.hpp"
#include "pair_finders/naive_all_pairs.hpp"
#include "com_fixers/com_fixers.hpp"
#include "io/hdf5_writer.hpp"

namespace kocs {
  namespace detail {
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



    template<typename U>
    struct pointer_depth { static constexpr std::size_t value = 0; };
    template<typename U>
    struct pointer_depth<U*> { static constexpr std::size_t value = 1 + pointer_depth<U>::value; };

    template <typename Field>
    struct ViewFromField {
      using field_type = std::remove_cv_t<typename Field::type>;
      static_assert(pointer_depth<field_type>::value <= 1,
                    "Field element types must not be pointer-to-pointer or higher (depth >= 2)");
      using type = View<std::remove_pointer_t<field_type>>;
    };

    // Device-only view used for internal integrator/pair_finder buffers that never need host sync
    template <typename Field>
    struct DeviceViewFromField {
      using field_type = std::remove_cv_t<typename Field::type>;
      using type = Kokkos::View<std::remove_pointer_t<field_type>*>;
    };

    template <typename Field>
    auto make_view(std::size_t n) {
      using view_type = typename ViewFromField<Field>::type;
      return view_type(std::string(Field::name), n);
    }

    template <typename FieldList>
    struct FirstFieldFromList;

    template <typename Field, typename... Rest>
    struct FirstFieldFromList<FieldList<Field, Rest...>> {
      using type = Field;
    };

    template <template<typename, typename, int> typename PairFinderT, typename SimulationConfig, typename FieldList>
    struct PairFinderFromFields;

    template <template<typename, typename, int> typename PairFinderT, typename SimulationConfig, typename... Fields>
    struct PairFinderFromFields<PairFinderT, SimulationConfig, FieldList<Fields...>> {
      using FirstField = typename FirstFieldFromList<FieldList<Fields...>>::type;
      using type = PairFinderT<
        typename DeviceViewFromField<FirstField>::type,
        typename SimulationConfig::Scalar, SimulationConfig::dimensions
      >;
    };

    template <template<typename, typename...> typename IntegratorT, typename SimulationConfig, typename FieldList>
    struct IntegratorFromFields;

    template <template<typename, typename...> typename IntegratorT, typename SimulationConfig, typename... Fields>
    struct IntegratorFromFields<IntegratorT, SimulationConfig, FieldList<Fields...>> {
      // Integrators receive device only Kokkos::Views for internal storage
      // (The simulation's storage retains kocs::View for host sync management)
      using type = IntegratorT<
        typename SimulationConfig::template PairFinderT<SimulationConfig>,
        typename SimulationConfig::template ComFixerT<SimulationConfig>,
        typename SimulationConfig::template ForceFields<kocs::detail::GenericFieldRef>,
        typename SimulationConfig::template ForceFields<kocs::detail::PairwiseFieldRef>,
        typename SimulationConfig::template ForceFields<kocs::detail::LinkFieldRef>,
        typename DeviceViewFromField<Fields>::type...
      >;
    };

    template<template<typename, typename...> typename IntegratorT, typename SimulationConfig>
    using integrator_t = typename detail::IntegratorFromFields<IntegratorT, SimulationConfig, typename SimulationConfig::Fields>::type;

    template<template<typename, typename, int> typename PairFinderT, typename SimulationConfig>
    using pair_finder_t = typename detail::PairFinderFromFields<PairFinderT, SimulationConfig, typename SimulationConfig::Fields>::type;

    template<template<typename> typename WriterT, typename SimulationConfig>
    using writer_t = WriterT<SimulationConfig>;

    

    template <typename Field>
    struct FieldHolder {
      using view_type = typename ViewFromField<Field>::type;
      view_type view;

      FieldHolder(view_type v) : view(v) { }
    };

    template <typename... Fields>
    struct FieldStorage : FieldHolder<Fields>... {
      FieldStorage(std::size_t n) : FieldHolder<Fields>{make_view<Fields>(n)}... { }
    };

    template <typename FieldList>
    struct FieldStorageFromList;

    template <typename... Fields>
    struct FieldStorageFromList<detail::FieldList<Fields...>> {
      using type = FieldStorage<Fields...>;
    };

    template <typename Field, typename Storage>
    inline auto& get(Storage& s) {
      using holder = FieldHolder<Field>;
      return static_cast<holder&>(s).view;
    }

    template <typename Field, typename Storage>
    inline const auto& get(const Storage& s) {
      using holder = FieldHolder<Field>;
      return static_cast<const holder&>(s).view;
    }

    template <typename FieldList, typename Storage>
    struct ViewsFromStorage;

    template <typename... Fields, typename Storage>
    struct ViewsFromStorage<FieldList<Fields...>, Storage> {
      static auto get(Storage& storage) {
        return std::forward_as_tuple(detail::get<Fields>(storage)...);
      }

      static auto get(const Storage& storage) {
        return std::forward_as_tuple(detail::get<Fields>(storage)...);
      }
    };
  } // namespace detail

#define CONFIG_SCALAR(__SCALAR__) \
  using Scalar = __SCALAR__;

#define CONFIG_DIMENSIONS(__DIMENSIONS__) \
  static constexpr int dimensions = __DIMENSIONS__; \
  using Vector = kocs::VectorN<Scalar, dimensions>;

#define CONFIG_RANDOM_POOL(__RANDOM_POOL__) \
  using RandomPoolT = __RANDOM_POOL__<>;

#define CONFIG_INTEGRATOR(__INTEGRATOR__) \
  template<typename SimulationConfig> \
  using IntegratorT = kocs::detail::integrator_t<__INTEGRATOR__, SimulationConfig>;

#define CONFIG_PAIR_FINDER(__PAIR_FINDER__) \
  template<typename SimulationConfig> \
  using PairFinderT = kocs::detail::pair_finder_t<__PAIR_FINDER__, SimulationConfig>;

#define CONFIG_COM_FIXER(__COM_FIXER__) \
  template<typename SimulationConfig> \
  using ComFixerT = __COM_FIXER__<SimulationConfig>;

#define CONFIG_WRITER(__WRITER__) \
  template<typename SimulationConfig> \
  using WriterT = kocs::detail::writer_t<__WRITER__, SimulationConfig>;

#define FIELD(__SCALAR_TYPE__, __FIELD_NAME__) \
  kocs::detail::Field<__SCALAR_TYPE__, #__FIELD_NAME__>

// #define CONFIG_FIELDS(...) \
//   using Fields = kocs::detail::FieldList<__VA_ARGS__>;



#define APPLY_PAIR(M, PAIR) M PAIR

// PP_NARG: count the number of arguments
#define PP_NARG(...) PP_NARG_IMPL(__VA_ARGS__,8,7,6,5,4,3,2,1,0)
#define PP_NARG_IMPL(x8,x7,x6,x5,x4,x3,x2,x1,n,...) n

// CAT with argument expansion (avoids ## not expanding args)
#define CAT(a, b) CAT_IMPL(a, b)
#define CAT_IMPL(a, b) a ## b

// FOR_EACH_PAIR: iterate over (type, name) pairs (comma-separated for FieldList)
#define FOR_EACH_PAIR_1(M, A) APPLY_PAIR(M, A)
#define FOR_EACH_PAIR_2(M, A, B) APPLY_PAIR(M, A), APPLY_PAIR(M, B)
#define FOR_EACH_PAIR_3(M, A, B, C) APPLY_PAIR(M, A), APPLY_PAIR(M, B), APPLY_PAIR(M, C)
#define FOR_EACH_PAIR_4(M, A, B, C, D) APPLY_PAIR(M, A), APPLY_PAIR(M, B), APPLY_PAIR(M, C), APPLY_PAIR(M, D)

#define FOR_EACH_PAIR(M, ...) CAT(FOR_EACH_PAIR_, PP_NARG(__VA_ARGS__))(M, __VA_ARGS__)

// helpers for generating ForceFields struct members (semicolon-separated)
#define FIELDS_ITERATE_1(M, A) APPLY_PAIR(M, A);
#define FIELDS_ITERATE_2(M, A, B) APPLY_PAIR(M, A); APPLY_PAIR(M, B);
#define FIELDS_ITERATE_3(M, A, B, C) APPLY_PAIR(M, A); APPLY_PAIR(M, B); APPLY_PAIR(M, C);
#define FIELDS_ITERATE_4(M, A, B, C, D) APPLY_PAIR(M, A); APPLY_PAIR(M, B); APPLY_PAIR(M, C); APPLY_PAIR(M, D);

#define FIELDS_ITERATE(M, ...) CAT(FIELDS_ITERATE_, PP_NARG(__VA_ARGS__))(M, __VA_ARGS__)

#define FIELD_FORCE_MEMBER(TYPE, NAME) FieldRefT<TYPE> NAME

#define CONFIG_FIELDS(...) \
  using Fields = kocs::detail::FieldList<FOR_EACH_PAIR(FIELD, __VA_ARGS__)>; \
  template<template<typename> typename FieldRefT = kocs::detail::GenericFieldRef> \
  struct ForceFields { \
    FIELDS_ITERATE(FIELD_FORCE_MEMBER, __VA_ARGS__) \
  }; /* ForceFields */



  // default simulation configs
  struct DefaultSimulationConfig {
    CONFIG_SCALAR(float)
    CONFIG_DIMENSIONS(3)

    using VectorView = kocs::View<Vector>;
    using Polarity = kocs::Polarity_<Scalar>;

    CONFIG_FIELDS(
      (Vector, position)
    )

    CONFIG_RANDOM_POOL(Kokkos::Random_XorShift64_Pool)
    CONFIG_PAIR_FINDER(kocs::pair_finders::NaiveAllPairs)
    CONFIG_COM_FIXER(kocs::com_fixers::NoComFixer)
    CONFIG_INTEGRATOR(kocs::integrators::Heun)

    CONFIG_WRITER(kocs::io::HDF5_Writer)
  };
} // namespace kocs

#endif // KOCS_SIMULATION_CONFIG
