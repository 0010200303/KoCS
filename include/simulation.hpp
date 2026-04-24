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
  namespace detail {
    template<typename U>
    struct pointer_depth { static constexpr std::size_t value = 0; };
    template<typename U>
    struct pointer_depth<U*> { static constexpr std::size_t value = 1 + pointer_depth<U>::value; };

    template <typename Field>
    struct ViewFromField {
      using field_type = std::remove_cv_t<typename Field::type>;
      static_assert(pointer_depth<field_type>::value <= 1,
                    "Field element types must not be pointer-to-pointer or higher (depth >= 2)");
      using type = Kokkos::View<std::remove_pointer_t<field_type>*>;
    };

    template <typename Field>
    auto make_view(std::size_t n) {
      using view_type = typename ViewFromField<Field>::type;
      // TODO: maybe use string_view instead
      return view_type(std::string(Field::name), n);
    }

    

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
    struct FieldStorageFromList<FieldList<Fields...>> {
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
        , writer(output_path)
        , current_step(0) { }

    private:
      const unsigned int agent_count;
      Storage storage;

      RandomPool random_pool;

      Writer writer;
      unsigned int current_step;

      static RuntimeGuard& get_runtime_guard() {
        static RuntimeGuard guard;
        return guard;
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

    private:
      template<typename... Forces>
      void take_step_impl(double dt, Forces&&... forces) {
        std::apply(
          [&](auto&&... views) {
            auto integrator = Integrator<
              PairFinder<Kokkos::View<Vector*>, std::decay_t<decltype(views)>...>,
              std::decay_t<decltype(views)>...
            >{ agent_count, views... };

            integrator.integrate(dt, static_cast<Forces&&>(forces)...);
          },
          get_views()
        );
      }
    
    public:
      inline void init_line() {
        initializers::Line<SimulationConfig> initializer(std::get<0>(get_views()));
        init(initializer);
      }

      inline void init_random_hollow_sphere(Scalar radius) {
        initializers::RandomHollowSphere<SimulationConfig> initializer(std::get<0>(get_views()), radius);
        init(initializer);
      }

      inline void init_random_filled_sphere(Scalar radius) {
        initializers::RandomFilledSphere<SimulationConfig> initializer(std::get<0>(get_views()), radius);
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

      // template<typename Force, typename... Views>
      // void take_step(double dt, Force force, Views... views) {
        // Integrator<PairFinder<Force, Views...>, Views...>{ agent_count, views... }.integrate(dt, force);
      // }

      template<typename... Forces>
      inline void take_step(double dt, Forces&&... forces) {
        auto fused_forces = detail::fuse_forces(static_cast<Forces&&>(forces)...);

        std::apply([&](auto&&... args) {
          take_step_impl(dt, static_cast<decltype(args)&&>(args)...);
        }, fused_forces);
      }



      template<typename Force, typename... Views>
      void take_step_single(double dt, Force force, Views... views) {
        auto fused_force = detail::fuse_forces(static_cast<Force&&>(force));

        auto integrator = Integrator<PairFinder<Force, Views...>, Views...>{ agent_count, views... };
        integrator.integrate_single(dt, fused_force);
      }

      template<typename Force>
      void take_step_single(double dt, Force force) {
        std::apply([this, dt, force](auto&&... args) { take_step_single(dt, force, args...); }, get_views());
      }



      template<typename Force, typename... Views>
      void take_step_rng(double dt, Force force, Views... views) {
        auto integrator = Integrator<PairFinder<Force, Views...>, Views...>{ agent_count, views... };
        integrator.integrate_rng(dt, random_pool, force);
      }

      template<typename Force>
      void take_step_rng(double dt, Force force) {
        std::apply([this, dt, force](auto&&... args) { take_step_rng(dt, force, args...); }, get_views());
      }

      inline void write() {
        std::apply([&](auto&&... args) {
          writer.write(current_step++, static_cast<decltype(args)&&>(args)...);
        }, get_views());
      }
  };
} // namespace kocs

#endif // KOCS_SIMULATION_HPP
