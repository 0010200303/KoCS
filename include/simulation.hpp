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

    // create a fresh "delta" view for accumulating changes (same element type)
    template <typename Field>
    auto make_delta_view(std::size_t n) {
      using view_type = typename ViewFromField<Field>::type;
      return view_type(std::string(Field::name) + "_delta", n);
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

      template<typename... Views>
      struct Tust : Views... {
        KOKKOS_INLINE_FUNCTION
        Tust(Views... v) : Views(v)... {}

        template<typename Force>
        KOKKOS_INLINE_FUNCTION
        void operator()(unsigned int i, Force force) const {
          force(i, static_cast<const Views&>(*this)(i)...);
        }
      };

      template<typename Force, typename... Views>
      void take_step(Force force, Views... views) {
        Tust tust(views...);

        Kokkos::parallel_for("step", agent_count, KOKKOS_LAMBDA(const unsigned int i) {
          tust(i, force);
        });
      }

      private:
        // helper: expand (orig..., delta...) for calling force inside device code
        template<typename Force, typename OrigTuple, typename DeltaTuple, std::size_t... I, std::size_t... J>
        static KOKKOS_INLINE_FUNCTION
        void call_force_impl(unsigned int i, Force force, const OrigTuple& origs, DeltaTuple& deltas,
                                  std::index_sequence<I...>, std::index_sequence<J...>) {
          force(i, std::get<I>(origs)(i)..., std::get<J>(deltas)(i)...);
        }

        // helper: add deltas back into originals element-wise
        template<typename OrigTuple, typename DeltaTuple, std::size_t... I>
        static KOKKOS_INLINE_FUNCTION
        void apply_deltas_impl(unsigned int i, OrigTuple& origs, DeltaTuple& deltas, std::index_sequence<I...>) {
          (void)std::initializer_list<int>{((std::get<I>(origs)(i) += std::get<I>(deltas)(i)), 0)...};
        }
    
    public:
      template<typename Force>
      void take_step(Force force) {
        // get tuple of original views
        auto origs = get_views();

        // create tuple of empty delta views (one per Field)
        auto deltas = std::make_tuple(detail::make_delta_view<Fields>(agent_count)...);

        using orig_tuple_t = decltype(origs);
        using delta_tuple_t = decltype(deltas);

        // run step kernel: force(i, orig_0(i), ..., delta_0(i), ...)
        Kokkos::parallel_for("step", agent_count, KOKKOS_LAMBDA(const unsigned int i) {
          call_force_impl<Force, orig_tuple_t, delta_tuple_t>(
            i, force, origs, deltas,
            std::make_index_sequence<std::tuple_size_v<orig_tuple_t>>{},
            std::make_index_sequence<std::tuple_size_v<delta_tuple_t>>{});
        });

        // apply accumulated deltas back into original views
        Kokkos::parallel_for("apply_deltas", agent_count, KOKKOS_LAMBDA(const unsigned int i) {
          apply_deltas_impl(orig_tuple_t(), delta_tuple_t()); // avoid unused warning in device builds
          apply_deltas_impl(i, origs, deltas, std::make_index_sequence<std::tuple_size_v<orig_tuple_t>>{});
        });
      }
  };
} // namespace kocs

#endif // KOCS_SIMULATION_HPP
