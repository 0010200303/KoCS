#ifndef KOCS_INTEGRATORS_BASE_HPP
#define KOCS_INTEGRATORS_BASE_HPP

#include <Kokkos_Core.hpp>

namespace kocs {
  namespace detail {
    template<typename... Views>
    struct ViewPack : Views... {
      ViewPack() = default;
      ViewPack(Views... views) : Views(views)... { }
    };

    template<int N, typename... Views>
    struct StagePack {
      static_assert(N > 0, "StagePack must contain at least one stage");

      ViewPack<Views...> stages[N];

      static auto make_mirror_view_pack(const ViewPack<Views...>& pack) {
        return ViewPack<Views...>(
          Kokkos::create_mirror(Kokkos::DefaultExecutionSpace(), static_cast<const Views&>(pack))...
        );
      }

      StagePack(const ViewPack<Views...>& stage) {
        stages[0] = stage;
        for (int i = 1; i < N; ++i) {
          stages[i] = make_mirror_view_pack(stage);
        }
      }

      KOKKOS_INLINE_FUNCTION
      ViewPack<Views...>& operator[](int i) {
        return stages[i];
      }

      KOKKOS_INLINE_FUNCTION
      const ViewPack<Views...>& operator[](int i) const {
        return stages[i];
      }
    };
  } // namespace detail

  namespace integrator {
    template<const unsigned int N, typename... Views>
    struct Base {
      public:
        unsigned int agent_count;
        mutable detail::StagePack<N, Views...> stage_pack;

      public:
        Base(unsigned int agent_count, Views... views)
          : agent_count(agent_count), stage_pack(detail::ViewPack<Views...>(views...)) { }
    };
  } // namespace integrator
} // namespace kocs

#endif // KOCS_INTEGRATORS_BASE_HPP
