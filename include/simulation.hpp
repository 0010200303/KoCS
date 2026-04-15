#ifndef KOCS_SIMULATION_HPP
#define KOCS_SIMULATION_HPP

#include <tuple>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>

#include "runtime_guard.hpp"
#include "utils.hpp"
#include "vector.hpp"

namespace kocs {
  template <typename SimulationConfig>
  class Simulation {
    public:
      EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

      static_assert(all_kokkos_views_v<FieldSpecs>,
        "SimulationConfig::IntegrationFields must be a std::tuple of Kokkos::View types"
      );

      Simulation(unsigned int _agent_count) : agent_count(_agent_count) {
        get_runtime_guard();
        fields = make_fields<Fields>(agent_count);
      }

      // field accessors
      KOKKOS_INLINE_FUNCTION
      Fields& get_fields() noexcept { return fields; }

      KOKKOS_INLINE_FUNCTION
      const Fields& get_fields() const noexcept { return fields; }

      template <std::size_t I>
      KOKKOS_INLINE_FUNCTION
      auto& get_field() noexcept {
        return std::get<I>(fields);
      }

      template <std::size_t I>
      KOKKOS_INLINE_FUNCTION
      const auto& get_field() const noexcept {
        return std::get<I>(fields);
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
          std::tuple_element_t<I, Tuple>(
            (I < FieldNamesCount) ? FieldNames[I] : std::string("field") + std::to_string(I), n)...
        };
      }

      template<typename Tuple>
      static Tuple make_fields(unsigned int n) {
        return make_fields_impl<Tuple>(n, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
      }

      template<typename ForceFn, typename... Values, std::size_t... I>
      KOKKOS_INLINE_FUNCTION static void invoke_force_impl(
        ForceFn force,
        const int i,
        const int j,
        std::tuple<Values...>& values,
        std::index_sequence<I...>
      ) {
        force(i, j, std::get<I>(values)...);
      }

      template<typename ForceFn, typename... Values>
      KOKKOS_INLINE_FUNCTION static void invoke_force(
        ForceFn force,
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
      template<typename InitFn>
      void init(InitFn init) {
        Kokkos::parallel_for(
          "init",
          agent_count,
          KOKKOS_CLASS_LAMBDA (const unsigned int i) {
            init(i);
          }
        );
      }

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
