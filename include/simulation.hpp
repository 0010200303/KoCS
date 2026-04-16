#ifndef KOCS_SIMULATION_HPP
#define KOCS_SIMULATION_HPP

#include <tuple>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "runtime_guard.hpp"
#include "utils.hpp"
#include "vector.hpp"

namespace kocs {
  template<typename SimulationConfig>
  class Simulation {
    public:
      EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

      // TODO: reimplement
      // static_assert(all_kokkos_views_v<FieldSpecs>,
      //   "SimulationConfig::IntegrationFields must be a std::tuple of Kokkos::View types"
      // );

      Simulation(const unsigned int agent_count_, const uint64_t seed = 2807) 
        : agent_count(agent_count_) {
        get_runtime_guard();
        state = make_fields<Storage>(agent_count);

        random_pool = RandomPool(seed);
      }

      // getters
      constexpr Storage& get_views() noexcept { return state; }

      constexpr const Storage& get_views() const noexcept { return state; }

      template<std::size_t I>
      auto& get_view() noexcept {
        return std::get<I>(state);
      }

      template<std::size_t I>
      const auto& get_view() const noexcept {
        return std::get<I>(state);
      }

      template<fixed_string Name>
      static consteval std::size_t index_of_view() noexcept {
        constexpr std::size_t index = index_of_view_impl<Name>();
        static_assert(index != std::size_t(-1), "Field not found");
        return index;
      }

      template <fixed_string Name>
      auto& get_view() {
        return get_view<index_of_view<Name>()>();
      }

    public:
      const unsigned int agent_count;
      Storage state;

      RandomPool random_pool;

      static RuntimeGuard& get_runtime_guard() {
        static RuntimeGuard guard;
        return guard;
      }

      template<typename Tuple, std::size_t... I>
      static Tuple make_fields_impl(unsigned int n, std::index_sequence<I...>) {
        return Tuple{ std::tuple_element_t<I, Fields>::make(n)... };
      }

      template<typename Tuple>
      static Tuple make_fields(unsigned int n) {
        return make_fields_impl<Tuple>(n, std::make_index_sequence<std::tuple_size<Tuple>::value>{});
      }

      template<fixed_string Name, std::size_t I = 0>
      static consteval std::size_t index_of_view_impl() noexcept {
        if constexpr (I >= std::tuple_size_v<Fields>) {
          return std::size_t(-1);
        } else {
          using FieldT = std::tuple_element_t<I, Fields>;
          if constexpr (FieldT::name == Name) {
            return I;
          } else {
            return index_of_view_impl<Name, I + 1>();
          }
        }
      }

      // forces
      template<typename ForceFn, typename ValuesContainer, std::size_t... I>
      KOKKOS_INLINE_FUNCTION static void invoke_force_impl(
        ForceFn force,
        const int i,
        const int j,
        ValuesContainer& values,
        std::index_sequence<I...>
      ) {
        force(i, j, std::get<I>(values)...);
      }

      template<typename ForceFn, typename ValuesContainer>
      KOKKOS_INLINE_FUNCTION static void invoke_force(
        ForceFn force,
        const int i,
        const int j,
        ValuesContainer& values
      ) {
        invoke_force_impl(
          force,
          i,
          j,
          values,
          std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<ValuesContainer>>>{}
        );
      }

      template<typename ForceFn, typename ValuesContainer, std::size_t... I>
      KOKKOS_INLINE_FUNCTION static void invoke_force_impl_rng(
        ForceFn force,
        const int i,
        const int j,
        auto& generator,
        ValuesContainer& values,
        std::index_sequence<I...>
      ) {
        force(i, j, generator, std::get<I>(values)...);
      }

      template<typename ForceFn, typename ValuesContainer>
      KOKKOS_INLINE_FUNCTION static void invoke_force_rng(
        ForceFn force,
        const int i,
        const int j,
        auto& generator,
        ValuesContainer& values
      ) {
        invoke_force_impl_rng(
          force,
          i,
          j,
          generator,
          values,
          std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<ValuesContainer>>>{}
        );
      }

      // Euler integration helpers: update each Field's view at index i using the
      // corresponding accumulated value in LocalValues::data multiplied by dt.
      template<std::size_t... I>
      KOKKOS_INLINE_FUNCTION
      static void euler_update_impl(
        const Storage& state_ref,
        LocalValues& local_values,
        const int i,
        const double dt,
        std::index_sequence<I...>
      ) {
        // Expand updates for each tuple element; use comma-expression inside
        // initializer_list to force evaluation in device code.
        (void)std::initializer_list<int>{
          (
            ( std::get<I>(state_ref)(i) = std::get<I>(state_ref)(i) + (std::get<I>(local_values.data) * dt) ), 0
          )...
        };
      }

      KOKKOS_INLINE_FUNCTION
      static void euler_update(
        const Storage& state_ref,
        LocalValues& local_values,
        const int i,
        const double dt
      ) {
        euler_update_impl(
          state_ref,
          local_values,
          i,
          dt,
          std::make_index_sequence<std::tuple_size_v<Fields>>{}
        );
      }

    public:
      template<typename InitFn>
      void init(InitFn init) {
        Kokkos::parallel_for(
          "init",
          agent_count,
          KOKKOS_CLASS_LAMBDA (const unsigned int i) {
            auto generator = random_pool.get_state();
            init(i, generator);
            random_pool.free_state(generator);
          }
        );
      }

      template<typename ForceFn>
      void take_step(ForceFn force, const double dt = 1.0) {
        Kokkos::parallel_for(
          "take_step",
          Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO),
          KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const int i = team.league_rank();

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              LocalValues local_values{};

              for (int j = 0; j < static_cast<int>(agent_count); ++j) {
                if (i == j) continue;
                invoke_force(force, i, j, local_values.data);
              }

              euler_update(state, local_values, i, dt);
            });
          }
        );
      }

      template<typename ForceFn>
      void take_step_rng(ForceFn force) {
        Kokkos::parallel_for(
          "take_step",
          Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO),
          KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
            const int i = team.league_rank();

            Kokkos::single(Kokkos::PerTeam(team), [&]() {
              LocalValues local_values{};

              for (int j = 0; j < static_cast<int>(agent_count); ++j) {
                if (i == j) continue;

                auto generator = random_pool.get_state();
                invoke_force_rng(force, i, j, generator, local_values.data);
                random_pool.free_state(generator);
              }

              Kokkos::printf("%f\n", std::get<0>(local_values.data).x());
            });
          }
        );
      }
  };
} // namespace kocs

#endif // KOCS_SIMULATION_HPP
