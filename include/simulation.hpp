#ifndef KOCS_SIMULATION_HPP
#define KOCS_SIMULATION_HPP

#include <tuple>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "runtime_guard.hpp"
#include "vector.hpp"

namespace kocs {
  template<typename...>
  struct dependent_false : std::false_type { };

  template<typename Name, typename View>
  struct NamedField {
    using name = Name;
    using view_type = View;
    using value_type = typename View::value_type;
  };

  template<typename T>
  struct is_named_field : std::false_type { };

  template<typename Name, typename View>
  struct is_named_field<NamedField<Name, View>> : std::true_type { };

  template<typename Tuple, std::size_t... I>
  constexpr bool tuple_all_named_fields_impl(std::index_sequence<I...>) {
    return (is_named_field<std::tuple_element_t<I, Tuple>>::value && ...);
  }

  template<typename Tuple>
  constexpr bool all_named_fields_v =
    tuple_all_named_fields_impl<Tuple>(std::make_index_sequence<std::tuple_size<Tuple>::value>{});



  template<typename Fields>
  struct ViewsFromFields;

  template<typename... Specs>
  struct ViewsFromFields<std::tuple<Specs...>> {
    using type = std::tuple<typename Specs::view_type...>;
  };

  template<typename Name, typename Tuple>
  struct field_index_by_name;

  template<typename Name>
  struct field_index_by_name<Name, std::tuple<>> {
    static_assert(dependent_false<Name>::value, "Unknown field name in DeltaState");
  };

  template<typename Name, typename View, typename... Rest>
  struct field_index_by_name<Name, std::tuple<NamedField<Name, View>, Rest...>> {
    static constexpr std::size_t value = 0;
  };

  template<typename Name, typename First, typename... Rest>
  struct field_index_by_name<Name, std::tuple<First, Rest...>> {
    static constexpr std::size_t value =
      1 + field_index_by_name<Name, std::tuple<Rest...>>::value;
  };

  template<typename Fields>
  struct DeltasFromFields;



  template<typename... Specs>
  struct DeltasFromFields<std::tuple<Specs...>> {
    struct type {
      std::tuple<typename Specs::value_type...> data;

      KOKKOS_INLINE_FUNCTION type() : data{} {}

      template<typename... Args,
        typename = std::enable_if_t<sizeof...(Args) == sizeof...(Specs)>>
      KOKKOS_INLINE_FUNCTION explicit type(Args... args) : data(args...) {}

      template<std::size_t I>
      KOKKOS_INLINE_FUNCTION auto& get() {
        return std::get<I>(data);
      }

      template<std::size_t I>
      KOKKOS_INLINE_FUNCTION const auto& get() const {
        return std::get<I>(data);
      }

      template<typename Name>
      KOKKOS_INLINE_FUNCTION auto& get() {
        return std::get<field_index_by_name<Name, std::tuple<Specs...>>::value>(data);
      }

      template<typename Name>
      KOKKOS_INLINE_FUNCTION const auto& get() const {
        return std::get<field_index_by_name<Name, std::tuple<Specs...>>::value>(data);
      }

      template<typename Name>
      KOKKOS_INLINE_FUNCTION auto& operator[](Name) {
        return get<Name>();
      }

      template<typename Name>
      KOKKOS_INLINE_FUNCTION const auto& operator[](Name) const {
        return get<Name>();
      }

      KOKKOS_INLINE_FUNCTION type& operator+=(const type& rhs) {
        add_impl(rhs, std::make_index_sequence<sizeof...(Specs)>{});
        return *this;
      }

    private:
      template<std::size_t... I>
      KOKKOS_INLINE_FUNCTION void add_impl(const type& rhs, std::index_sequence<I...>) {
        ((std::get<I>(data) += std::get<I>(rhs.data)), ...);
      }
    };
  };

  template <typename Config>
  class Simulation {
    public:
      using Scalar = typename Config::Scalar;
      static constexpr unsigned int dimensions = Config::dimensions;
      using FieldSpecs = typename Config::IntegrationFields;
      using Fields = typename ViewsFromFields<FieldSpecs>::type;
      using Vector = VectorN<Scalar, dimensions>;
      using DeltaState = typename DeltasFromFields<FieldSpecs>::type;

      static_assert(all_named_fields_v<FieldSpecs>,
        "Config::IntegrationFields must be a std::tuple of kocs::NamedField types"
      );

      Simulation(unsigned int _agent_count) : agent_count(_agent_count) {
        get_runtime_guard();

        fields = make_fields<Fields>(agent_count);
      }

    private:
      const unsigned int agent_count;

      Fields fields;

      static RuntimeGuard& get_runtime_guard() {
        static RuntimeGuard guard;
        return guard;
      }

      template<typename Tuple, std::size_t... I>
      static Tuple make_fields_impl(unsigned int n, std::index_sequence<I...>) {
        return Tuple{
          std::tuple_element_t<I, Tuple>(std::string("field") + std::to_string(I), n)...
        };
      }

      template<typename Tuple>
      static Tuple make_fields(unsigned int n) {
        return make_fields_impl<Tuple>(n, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
      }

    public:
      template<typename ForceFn>
      void take_step(ForceFn force) {
        Kokkos::parallel_for(
          "take_step",
          Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO),
          KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const int i = team.league_rank();

            DeltaState ds{};

            Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, agent_count),
              [&](const int j, DeltaState& local) {
                if (i == j)
                  return;

                force(i, j, local);
              },
              ds
            );

            Kokkos::printf("%f\n", ds.template get<0>().x());

            Kokkos::single(Kokkos::PerTeam(team), [&]() {

            });
          }
        );
      }
  };
} // namespace kocs

#endif // KOCS_SIMULATION_HPP
