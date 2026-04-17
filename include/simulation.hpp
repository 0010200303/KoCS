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

namespace kocs {
  namespace detail {
    template<typename U>
    struct pointer_depth { static constexpr std::size_t value = 0; };
    template<typename U>
    struct pointer_depth<U*> { static constexpr std::size_t value = 1 + pointer_depth<U>::value; };

    template <typename Field>
    struct ViewFromField {
      using type = Kokkos::View<typename Field::type>;
    };

    template <typename T, fixed_string Name>
    struct ViewFromField<Field<T, Name>> {
      static_assert(pointer_depth<T>::value <= 1,
                    "Field element types must not be pointer-to-pointer or higher (depth >= 2)");
      using type = Kokkos::View<T>;
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
      Simulation(const unsigned int agent_count_, const uint64_t seed = 2807)
        : agent_count(agent_count_)
        , storage((get_runtime_guard(), Storage(agent_count_)))
        , random_pool(seed) { }

    private:
      const unsigned int agent_count;
      Storage storage;

      RandomPool random_pool;

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
    
    public:
      template<typename Initializer>
      inline void init(Initializer initializer) {
        Kokkos::parallel_for("init", agent_count, initializer);
      }

      struct Tust {
        Tust(Kokkos::View<Vector*> _pos, Kokkos::View<float*> _mass)
          : pos(_pos), mass(_mass) { }

        Kokkos::View<Vector*> pos;
        Kokkos::View<float*> mass;

        template<typename Force>
        KOKKOS_INLINE_FUNCTION
        void operator() (const unsigned int i, Force force) const {
          force(i, pos, mass);
        }
      };

      #define TUST(...) \
      struct Tust { \
      };

      template<typename Force>
      inline void take_step(Force force) {
        auto views = get_views();

        std::apply(
          [&](auto&&... expanded_views) {
            Tust tust(expanded_views...);
            Kokkos::parallel_for(agent_count, KOKKOS_LAMBDA(unsigned int i) {
              // tust(i, force);
            });
          },
          views
        );
      }
  };
} // namespace kocs

#endif // KOCS_SIMULATION_HPP
