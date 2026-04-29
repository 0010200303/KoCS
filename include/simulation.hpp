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

namespace kocs {
  template<typename SimulationConfig>
  class Simulation {
    EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)
    using Storage = typename detail::FieldStorageFromList<Fields>::type;

    public:
      Simulation(
        const unsigned int agent_count_,
        const std::string& output_path,
        const uint64_t seed = 2807
      )
        : agent_count(agent_count_)
        , storage((get_runtime_guard(), Storage(agent_count_)))
        , random_pool(seed)
        , pair_finder(make_pair_finder(agent_count_))
        , integrator(make_integrator(agent_count_, pair_finder, storage))
        , writer(output_path)
        , current_step(0) { }

    private:
      const unsigned int agent_count;
      Storage storage;

      RandomPool random_pool;

      // TODO: maybe add some specific options to construct these
      PairFinder pair_finder;
      Integrator integrator;

      Writer writer;
      unsigned int current_step;

      static RuntimeGuard& get_runtime_guard() {
        static RuntimeGuard guard;
        return guard;
      }

      static PairFinder make_pair_finder(
        unsigned int agent_count_,
        float cutoff_distance = 10'000.0f
      ) {
        return PairFinder(
          agent_count_,
          cutoff_distance
        );
      }

      static Integrator make_integrator(
        unsigned int agent_count_,
        PairFinder pair_finder_,
        Storage& storage_
      ) {
        return std::apply(
          [&](auto&... views) {
            return Integrator(agent_count_, pair_finder_, views...);
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
      inline void init_line() {
        initializers::Line<SimulationConfig> initializer(get_positions_view());
        init(initializer);
      }

      inline void init_random_hollow_sphere(Scalar radius) {
        initializers::RandomHollowSphere<SimulationConfig> initializer(get_positions_view(), radius);
        init(initializer);
      }

      inline void init_random_filled_sphere(Scalar radius) {
        initializers::RandomFilledSphere<SimulationConfig> initializer(get_positions_view(), radius);
        init(initializer);
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
