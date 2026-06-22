#ifndef KOCS_SIMULATION_HPP
#define KOCS_SIMULATION_HPP

#include <tuple>
#include <string>
#include <type_traits>
#include <utility>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "simulation_config.hpp"
#include "utils/runtime_guard.hpp"
#include "utils/utils.hpp"
#include "forces/kernel_fuser.hpp"

#include "types/vector.hpp"
#include "types/link.hpp"

#include "initializers/line.hpp"
#include "initializers/spheres.hpp"
#include "initializers/hexagon.hpp"
#include "initializers/cuboid.hpp"
#include "initializers/rectangle.hpp"
#include "initializers/disk.hpp"

namespace kocs {
  template<typename SimulationConfig>
  class Simulation {
    EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)
    using Storage = typename detail::FieldStorageFromList<Fields>::type;

    public:
      // legacy constructor
      // TODO: remove later
      struct Settings {
        Settings(const unsigned int agent_count_, const std::string& output_path_)
          : agent_count(agent_count_)
          , output_path(output_path_)
          , capacity(agent_count_) { }

        unsigned int agent_count;
        std::string output_path;

        unsigned int capacity;

        Scalar cutoff_distance = Scalar(1'000'000);
        uint64_t seed = 2807;

        PairFinder::Settings pair_finder_settings = {};
        Writer::Settings writer_settings = {};

        // settings for additional features
        unsigned int link_capacity = 0;
        unsigned int link_active_count = 0;
      };

      Simulation(
        const unsigned int agent_count_,
        const std::string& output_path,
        const Scalar cutoff_distance = Scalar(1'000'000),
        const PairFinder::Settings& pair_finder_settings = {},
        const Writer::Settings& writer_settings = {},
        const uint64_t seed = 2807
      )
        : agent_count(agent_count_)
        , capacity(agent_count_)
        , storage((get_runtime_guard(), Storage(agent_count_)))
        , random_pool(seed)
        , pair_finder(agent_count_, cutoff_distance, pair_finder_settings)
        , com_fixer()
        , integrator(make_integrator(agent_count_, agent_count_, pair_finder, com_fixer, storage, links))
        , writer(output_path, writer_settings)
        , current_step(0) { }

      Simulation(const Settings settings)
        : agent_count(settings.agent_count)
        , capacity(settings.capacity)
        , storage((get_runtime_guard(), Storage(settings.capacity)))
        , random_pool(settings.seed)
        , pair_finder(settings.agent_count, settings.cutoff_distance, settings.pair_finder_settings)
        , com_fixer()
        , links(settings.link_capacity != 0 ? View<Link>("links", settings.link_capacity) : View<Link>())
        , integrator(make_integrator(
            settings.agent_count,
            settings.capacity,
            pair_finder,
            com_fixer,
            storage,
            links
          ))
        , writer(settings.output_path, settings.writer_settings)
        , current_step(0) 
        {
          links.set_active_count(settings.link_active_count);
        }

    public:
      unsigned int capacity;
      unsigned int agent_count;
      Storage storage;

      View<Link> links;

      RandomPool random_pool;

      PairFinder pair_finder;
      ComFixer com_fixer;
      Integrator integrator;

      Writer writer;
      unsigned int current_step;

      static detail::RuntimeGuard& get_runtime_guard() {
        static detail::RuntimeGuard guard;
        return guard;
      }

      static Integrator make_integrator(
        unsigned int agent_count_,
        unsigned int capacity_,
        PairFinder& pair_finder_,
        ComFixer& com_fixer_,
        Storage& storage_,
        View<Link>& links_
      ) {
        return std::apply(
          [&](auto&... views) {
            // Pass device-only views (Kokkos::View) to the integrator,
            // since its internal buffers never need host-device sync.
            return Integrator(
              agent_count_,
              capacity_,
              pair_finder_,
              com_fixer_,
              links_.view_device(),
              views.view_device()...
            );
          },
          detail::ViewsFromStorage<Fields, Storage>::get(storage_)
        );
      }

      Integrator& get_integrator() const {
        return integrator;
      }

      PairFinder& get_pair_finder() const {
        return pair_finder;
      }
    
    public:
      // TODO: add static_assert for better error message
      template <typename Field>
      inline auto& get_view() {
        return detail::get<Field>(storage);
      }
      
      // TODO: add static_assert for better error message
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

      inline unsigned int get_capacity() const {
        return capacity;
      }

      inline void set_capacity(const unsigned int value) {
        capacity = value;
        integrator.set_capacity(value);

        std::apply([&](auto&... views) {
          ((views.resize(value)), ...);
          integrator.reattach_stage_0(views.view_device()...);
        }, get_views());
      }

      template<typename... AdditionalViews>
      inline void set_capacity(const unsigned int value, AdditionalViews&... additional_views) {
        set_capacity(value);
        ((additional_views.resize(value)), ...);
      }

      inline unsigned int get_agent_count() const {
        return agent_count;
      }

      // TODO: auto resize (shrink_to_fit & grow (vector like: double capacity))
      inline void set_agent_count(const unsigned int value) {
        agent_count = value;
        integrator.set_agent_count(value);
        pair_finder.set_agent_count(value);
        std::apply([&](auto&... views) {
          ((views.set_active_count(value)), ...);
        }, get_views());
      }

      template<typename... AdditionalViews>
      inline void set_agent_count(const unsigned int value, AdditionalViews&... additional_views) {
        set_agent_count(value);
        ((additional_views.set_active_count(value)), ...);
      }

      inline auto get_old_velocities_view_from_integrator() const {
        return integrator.old_velocities;
      }

      inline View<Link>& get_links() {
        return links;
      }

      inline const View<Link>& get_links() const {
        return links;
      }

      inline unsigned int get_link_count() const {
        return links.extent(0);
      }

      inline void resize_links(const unsigned int value) {
        links.resize(value);
      }

    private:
      template<typename... Forces>
      void take_step_impl(double dt, Forces&&... forces) {
        integrator.integrate(dt, random_pool, static_cast<Forces&&>(forces)...);

        std::apply([&](auto&... views) {
          ((views.modify_device()), ...);
        }, get_views());
      }

    public:
      template<typename... InitFuncs>
      inline void init_line(const Scalar distance, InitFuncs&&... init_functions) {
        initializers::Line<SimulationConfig> initializer(get_positions_view(), distance);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_random_hollow_sphere(const Scalar radius, InitFuncs&&... init_functions) {
        initializers::RandomHollowSphere<SimulationConfig> initializer(get_positions_view(), radius);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_random_filled_sphere(const Scalar radius, InitFuncs&&... init_functions) {
        initializers::RandomFilledSphere<SimulationConfig> initializer(get_positions_view(), radius);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_regular_hexagon(const Scalar distance_to_neighbour, InitFuncs&&... init_functions) {
        initializers::RegularHexagon<SimulationConfig> initializer(get_positions_view(), distance_to_neighbour);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_random_cuboid(const Vector& min, const Vector& max, InitFuncs&&... init_functions) {
        initializers::RandomCuboid<SimulationConfig> initializer(get_positions_view(), min, max);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_regular_rectangle(
        const Scalar distance_to_neighbour,
        const unsigned int nx,
        InitFuncs&&... init_functions
      ) {
        initializers::RegularRectangle<SimulationConfig> initializer(get_positions_view(), distance_to_neighbour, nx);
        init(initializer, init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_random_disk(const Scalar distance_to_neighbour, InitFuncs&&... init_functions) {
        initializers::RandomDisk<SimulationConfig> initializer(get_positions_view(), distance_to_neighbour);
        init(initializer, init_functions...);
      }

      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        const unsigned int relaxation_steps = 2000
      ) {
        initializers::RelaxedSphere<SimulationConfig> initializer(get_positions_view(), initial_radius, relaxation_steps);
        init(initializer);
        initializer.relax(*this);
      }

      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        const int relaxation_steps
      ) {
        initializers::RelaxedSphere<SimulationConfig> initializer(get_positions_view(), initial_radius, static_cast<unsigned int>(relaxation_steps));
        init(initializer);
        initializer.relax(*this);
      }

      template<typename... InitFuncs>
      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        const unsigned int relaxation_steps = 2000,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_sphere(initial_radius, relaxation_steps);
        init(init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        const int relaxation_steps,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_sphere(initial_radius, static_cast<unsigned int>(relaxation_steps));
        init(init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_relaxed_sphere(
        const Scalar initial_radius,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_sphere(initial_radius, 2000u);
        init(init_functions...);
      }

      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        const unsigned int relaxation_steps = 2000
      ) {
        initializers::RelaxedCuboid<SimulationConfig> initializer(get_positions_view(), min, max, relaxation_steps);
        init(initializer);
        initializer.relax(*this);
      }

      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        const int relaxation_steps
      ) {
        initializers::RelaxedCuboid<SimulationConfig> initializer(get_positions_view(), min, max, static_cast<unsigned int>(relaxation_steps));
        init(initializer);
        initializer.relax(*this);
      }

      template<typename... InitFuncs>
      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        const unsigned int relaxation_steps = 2000,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_cuboid(min, max, relaxation_steps);
        init(init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        const int relaxation_steps,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_cuboid(min, max, static_cast<unsigned int>(relaxation_steps));
        init(init_functions...);
      }

      template<typename... InitFuncs>
      inline void init_relaxed_cuboid(
        const Vector& min,
        const Vector& max,
        InitFuncs&&... init_functions
      ) {
        init_relaxed_cuboid(min, max, 2000u);
        init(init_functions...);
      }

      template<typename Initializer>
      inline void init(Initializer initializer) {
        auto& random_pool_ = random_pool;
        Kokkos::parallel_for("Simulation::init", agent_count, KOKKOS_LAMBDA(const unsigned int i) {
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

      template<typename Func>
      inline void run_custom(unsigned int count, Func function) {
        auto& random_pool_ = random_pool;
        Kokkos::parallel_for("Simulation::run", count, KOKKOS_LAMBDA(const unsigned int i) {
          Random generator = random_pool_.get_state();

          function(i, generator);

          random_pool_.free_state(generator);
        });
      }

      template<typename... Funcs>
      inline void run_custom(unsigned int count, Funcs&&... functions) {
        auto fused_funcs = detail::fuse_forces(static_cast<Funcs&&>(functions)...);

        std::apply([&](auto&&... args) {
          run_custom(count, static_cast<decltype(args)&&>(args)...);
        }, fused_funcs);
      }

      template<typename... Funcs>
      inline void run(Funcs&&... functions) {
        run_custom(agent_count, std::forward<Funcs>(functions)...);
      }

      template<typename... Funcs>
      inline void run_links(Funcs&&... functions) {
        run_custom(links.get_active_count(), std::forward<Funcs>(functions)...);
      }

      inline void write(const double time) {
        std::apply([&](auto&&... args) {
          writer.write(time, current_step++, static_cast<decltype(args)&&>(args)..., links);
        }, get_views());
      }

      template<typename... Views>
      inline void write(const double time, Views&&... additional_views) {
        std::apply([&](auto&&... args) {
          writer.write(
            time,
            current_step++,
            static_cast<decltype(args)&&>(args)...,
            std::forward<Views>(additional_views)...,
            links
          );
        }, get_views());
      }

      template<typename... Views>
      inline void write_static(Views&&... static_views) {
        writer.write_static(static_views...);
      } 
  };
} // namespace kocs

#endif // KOCS_SIMULATION_HPP
