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

  template <typename Config>
  class Simulation {
    public:
      using Scalar = typename Config::Scalar;
      static constexpr unsigned int dimensions = Config::dimensions;
      using FieldSpecs = typename Config::IntegrationFields;
      using Fields = typename ViewsFromFields<FieldSpecs>::type;
      using Vector = VectorN<Scalar, dimensions>;
      using LocalValues = typename ValuesFromFields<FieldSpecs>::type;

      static_assert(all_kokkos_views_v<FieldSpecs>,
        "Config::IntegrationFields must be a std::tuple of Kokkos::View types"
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

      template<typename ForceFn, typename... Values, std::size_t... I>
      KOKKOS_INLINE_FUNCTION static void invoke_force_impl(
        const ForceFn& force,
        const int i,
        const int j,
        std::tuple<Values...>& values,
        std::index_sequence<I...>
      ) {
        force(i, j, std::get<I>(values)...);
      }

      template<typename ForceFn, typename... Values>
      KOKKOS_INLINE_FUNCTION static void invoke_force(
        const ForceFn& force,
        const int i,
        const int j,
        std::tuple<Values...>& values
      ) {
        invoke_force_impl(
          force,
          i,
          j,
          values,
          std::make_index_sequence<sizeof...(Values)>{}
        );
      }

    public:
      template<typename ForceFn>
      void take_step(ForceFn force) {
        Kokkos::parallel_for(
          "take_step",
          Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO),
          KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const int i = team.league_rank();

            LocalValues local_values{};

            Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(team, agent_count),
              [&](const int j, LocalValues& local) {
                if (i == j) {
                  return;
                }

                invoke_force(force, i, j, local.data);
              },
              local_values
            );

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              Kokkos::printf("%f\n", std::get<0>(local_values.data).x());
            });
          }
        );
      }
  };
} // namespace kocs

#endif // KOCS_SIMULATION_HPP
