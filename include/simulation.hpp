#ifndef KOCS_SIMULATION_HPP
#define KOCS_SIMULATION_HPP

#include <tuple>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "simulation_config.hpp"
#include "runtime_guard.hpp"
#include "utils.hpp"
#include "vector.hpp"
#include "forces/kernel_fuser.hpp"

#include "initializers/line.hpp"
#include "initializers/spheres.hpp"
#include "initializers/hexagon.hpp"
#include "initializers/cuboid.hpp"

namespace kocs {
  template<typename SimulationConfig>
  class Simulation {
    EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)
    using Storage = typename detail::FieldStorageFromList<Fields>::type;

    public:
      Simulation(
        const unsigned int agent_count_,
        const std::string& output_path,
        const Scalar cutoff_distance = Scalar(1'000'000),
        const uint64_t seed = 2807
      )
        : agent_count(agent_count_)
        , storage((get_runtime_guard(), Storage(agent_count_)))
        , random_pool(seed)
        , pair_finder(make_pair_finder(agent_count_, cutoff_distance))
        , com_fixer()
        , integrator(make_integrator(agent_count_, pair_finder, com_fixer, storage))
        , writer(output_path)
        , current_step(0) { }

    private:
      const unsigned int agent_count;
      Storage storage;

      RandomPool random_pool;

      // TODO: maybe add some specific options to construct these
      PairFinder pair_finder;
      ComFixer com_fixer;
      Integrator integrator;

      Writer writer;
      unsigned int current_step;

      static RuntimeGuard& get_runtime_guard() {
        static RuntimeGuard guard;
        return guard;
      }

      static PairFinder make_pair_finder(
        unsigned int agent_count_,
        Scalar cutoff_distance
      ) {
        return PairFinder(
          agent_count_,
          cutoff_distance
        );
      }

      static Integrator make_integrator(
        unsigned int agent_count_,
        PairFinder pair_finder_,
        ComFixer com_fixer_,
        Storage& storage_
      ) {
        return std::apply(
          [&](auto&... views) {
            return Integrator(agent_count_, pair_finder_, com_fixer_, views...);
          },
          detail::ViewsFromStorage<Fields, Storage>::get(storage_)
        );
      }
    
    public:
      template <typename Field>
      inline auto& get_view() {
        return detail::get<Field>(storage);
      }

      template <typename Field>
      inline const auto& get_view() const {
        return detail::get<Field>(storage);
      }

      inline auto get_views() {
        return detail::ViewsFromStorage<Fields, Storage>::get(storage);
      }

      inline auto get_views() const {
        return detail::ViewsFromStorage<Fields, Storage>::get(storage);
      }

      inline auto get_positions_view() const {
        return std::get<0>(get_views());
      }

    private:
      template<typename... Forces>
      void take_step_impl(double dt, Forces&&... forces) {
        integrator.integrate(dt, random_pool, static_cast<Forces&&>(forces)...);
      }

    public:
      template<typename... InitFuncs>
      inline void init_line(InitFuncs&&... init_functions) {
        initializers::Line<SimulationConfig> initializer(get_positions_view());
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_random_hollow_sphere(Scalar radius, InitFuncs&&... init_functions) {
        initializers::RandomHollowSphere<SimulationConfig> initializer(get_positions_view(), radius);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_random_filled_sphere(Scalar radius, InitFuncs&&... init_functions) {
        initializers::RandomFilledSphere<SimulationConfig> initializer(get_positions_view(), radius);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_regular_hexagon(Scalar distance_to_neighbour, InitFuncs&&... init_functions) {
        initializers::RegularHexagon<SimulationConfig> initializer(get_positions_view(), distance_to_neighbour);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_random_cuboid(const Vector& min, const Vector& max, InitFuncs&&... init_functions) {
        initializers::RandomCuboid<SimulationConfig> initializer(get_positions_view(), min, max);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        const unsigned int relaxation_steps = 2000,
        InitFuncs&&... init_functions
      ) {
        initializers::RelaxedCuboid<SimulationConfig> initializer(get_positions_view(), min, max, relaxation_steps);
        init(initializer);
        initializer.relax(*this);

        init(init_functions...);
      }

      template<typename Initializer>
      inline void init(Initializer initializer) {
        auto& random_pool_ = random_pool;
        Kokkos::parallel_for("init", agent_count, KOKKOS_LAMBDA(const unsigned int i) {
          Random generator = random_pool_.get_state();

          initializer(i, generator);

          random_pool_.free_state(generator);
        });
      }

      template<typename Initializer, typename... InitFuncs>
      inline void init(Initializer initializer, InitFuncs&&... init_functions) {
        init(initializer);
        if constexpr (sizeof...(init_functions) > 0)
          init(static_cast<InitFuncs&&>(init_functions)...);
      }

      template<typename... Forces>
      inline void take_step(double dt, Forces&&... forces) {
        auto fused_forces = detail::fuse_forces(static_cast<Forces&&>(forces)...);

        std::apply([&](auto&&... args) {
          take_step_impl(dt, static_cast<decltype(args)&&>(args)...);
        }, fused_forces);
      }

      inline void write() {
        std::apply([&](auto&&... args) {
          writer.write(current_step++, static_cast<decltype(args)&&>(args)...);
        }, get_views());
      }

      template<typename... Views>
      inline void write(Views&&... additional_views) {
        std::apply([&](auto&&... args) {
          writer.write(
            current_step++,
            static_cast<decltype(args)&&>(args)...,
            std::forward<Views>(additional_views)...
          );
        }, get_views());
      }
  };
} // namespace kocs

#endif // KOCS_SIMULATION_HPP
