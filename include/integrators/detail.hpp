#ifndef KOCS_INTEGRATORS_DETAIL_HPP
#define KOCS_INTEGRATORS_DETAIL_HPP

namespace kocs::detail {
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
} // namespace kocs::detail

#endif // KOCS_INTEGRATORS_DETAIL_HPP
