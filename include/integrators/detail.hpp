#ifndef KOCS_INTEGRATORS_DETAIL_HPP
#define KOCS_INTEGRATORS_DETAIL_HPP

namespace kocs::detail {
  template<typename... Views>
  struct ViewPack;

  template<>
  struct ViewPack<> {
    ViewPack() = default;

    template<typename F>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(F&& f) const {
      return static_cast<F&&>(f)();
    }

    template<typename F>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(F&& f) {
      return static_cast<F&&>(f)();
    }

    template<typename F, typename... OtherPacks>
    KOKKOS_INLINE_FUNCTION
    void zip_apply(F&&, OtherPacks&&...) { }

    template<typename F, typename... OtherPacks>
    KOKKOS_INLINE_FUNCTION
    void zip_apply(F&&, OtherPacks&&...) const { }
  };

  template<typename FirstView, typename... RestViews>
  struct ViewPack<FirstView, RestViews...> : ViewPack<RestViews...> {
    using base_type = ViewPack<RestViews...>;

    FirstView first_value;

    ViewPack() = default;

    KOKKOS_INLINE_FUNCTION
    ViewPack(const FirstView& first, const RestViews&... rest)
      : base_type(rest...)
      , first_value(first) { }

    KOKKOS_INLINE_FUNCTION
    ViewPack(const FirstView& first, const base_type& rest)
      : base_type(rest)
      , first_value(first) { }

    KOKKOS_INLINE_FUNCTION
    FirstView& first() {
      return first_value;
    }

    KOKKOS_INLINE_FUNCTION
    const FirstView& first() const {
      return first_value;
    }

    template<typename F>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(F&& f) {
      return static_cast<base_type&>(*this).apply(
        [&](auto&... rest_values) -> decltype(auto) {
          return static_cast<F&&>(f)(first_value, rest_values...);
        }
      );
    }

    template<typename F>
    KOKKOS_INLINE_FUNCTION
    decltype(auto) apply(F&& f) const {
      return static_cast<const base_type&>(*this).apply(
        [&](const auto&... rest_values) -> decltype(auto) {
          return static_cast<F&&>(f)(first_value, rest_values...);
        }
      );
    }

    template<typename F, typename... OtherPacks>
    KOKKOS_INLINE_FUNCTION
    void zip_apply(F&& f, OtherPacks&&... others) {
      f(first_value, others.first()...);
      base_type::zip_apply(
        std::forward<F>(f),
        static_cast<base_type&>(others)...
      );
    }

    template<typename F, typename... OtherPacks>
    KOKKOS_INLINE_FUNCTION
    void zip_apply(F&& f, OtherPacks&&... others) const {
      f(first_value, others.first()...);
      static_cast<const base_type&>(*this).zip_apply(
        std::forward<F>(f),
        static_cast<const base_type&>(others)...
      );
    }
  };

  template<typename FirstView, typename... RestViews>
  KOKKOS_INLINE_FUNCTION
  FirstView& first(ViewPack<FirstView, RestViews...>& pack) {
    return pack.first();
  }

  template<typename FirstView, typename... RestViews>
  KOKKOS_INLINE_FUNCTION
  const FirstView& first(const ViewPack<FirstView, RestViews...>& pack) {
    return pack.first();
  }

  template<typename First, typename...>
  struct first_type {
    using type = First;
  };

  template<typename... Views>
  using first_type_t = typename first_type<Views...>::type;

  template<int N, typename... Views>
  struct StagePack {
    static_assert(N > 0, "StagePack must contain at least one stage");

    ViewPack<Views...> stages[N];

    template<typename FirstView, typename... RestViews>
    static auto make_mirror_view_pack_impl(const ViewPack<FirstView, RestViews...>& pack) {
      using mirror_first_type = FirstView;

      if constexpr (sizeof...(RestViews) == 0) {
        return ViewPack<mirror_first_type>(
          Kokkos::create_mirror(Kokkos::DefaultExecutionSpace(), pack.first())
        );
      } else {
        return ViewPack<mirror_first_type, RestViews...>(
          Kokkos::create_mirror(Kokkos::DefaultExecutionSpace(), pack.first()),
          make_mirror_view_pack_impl(static_cast<const ViewPack<RestViews...>&>(pack))
        );
      }
    }

    static auto make_mirror_view_pack(const ViewPack<Views...>& pack) {
      return make_mirror_view_pack_impl(pack);
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
