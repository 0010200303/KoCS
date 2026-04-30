#ifndef KOCS_INTEGRATORS_HEUN_HPP
#define KOCS_INTEGRATORS_HEUN_HPP

#include <type_traits>
#include <Kokkos_Core.hpp>

#include "base.hpp"

namespace kocs::integrators {
  template<typename PairFinder, typename... Views>
  struct Heun : public Base<PairFinder, 4, Views...> {
    using Base<PairFinder, 4, Views...>::Base;

  private:
    template<typename FirstView, typename... RestViews>
    KOKKOS_INLINE_FUNCTION
    static void reset_pack_at(detail::ViewPack<FirstView, RestViews...>& pack, const unsigned int i) {
      pack.zip_apply([&](auto& view) {
        using ValueT = std::remove_cv_t<std::remove_reference_t<decltype(view(i))>>;
        view(i) = ValueT{};
      });
    }

    template<typename FirstView, typename... RestViews>
    KOKKOS_INLINE_FUNCTION
    static void predict_pack_at(
      detail::ViewPack<FirstView, RestViews...>& dst,
      const detail::ViewPack<FirstView, RestViews...>& current,
      const detail::ViewPack<FirstView, RestViews...>& delta,
      const unsigned int i,
      const double dt
    ) {
      dst.zip_apply([&](auto& d, const auto& c, const auto& del) {
        d(i) = c(i) + del(i) * dt;
      }, current, delta);
    }

    template<typename FirstView, typename... RestViews>
    KOKKOS_INLINE_FUNCTION
    static void correct_pack_at(
      detail::ViewPack<FirstView, RestViews...>& current,
      const detail::ViewPack<FirstView, RestViews...>& delta0,
      const detail::ViewPack<FirstView, RestViews...>& delta1,
      const unsigned int i,
      const double dt
    ) {
      current.zip_apply([&](auto& c, const auto& d0, const auto& d1) {
        c(i) += (d0(i) + d1(i)) * 0.5 * dt;
      }, delta0, delta1);
    }

  public:
    void apply_euler_predictor(float dt) {
      auto& stage_pack = this->stage_pack;

      Kokkos::parallel_for(
        "apply_euler_predictor",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          stage_pack[2].zip_apply([&](auto& predicted, const auto& current, const auto& delta) {
            predicted(i) = current(i) + delta(i) * dt;
          }, stage_pack[0], stage_pack[1]);
        }
      );
    }

    void apply_heun_corrector(double dt) {
      auto& stage_pack = this->stage_pack;
      auto& old_velocities = this->old_velocities;

      Kokkos::parallel_for(
        "apply_heun_corrector",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          old_velocities(i) =
            (stage_pack[1].first()(i) + stage_pack[3].first()(i)) * 0.5;

          correct_pack_at(stage_pack[0], stage_pack[1], stage_pack[3], i, dt);

          reset_pack_at(stage_pack[1], i);
          reset_pack_at(stage_pack[2], i);
          reset_pack_at(stage_pack[3], i);
        }
      );
    }

    template<typename RandomPool, typename... Forces>
    void integrate(double dt, RandomPool& random_pool, Forces... forces) {
      this->evaluate_forces(random_pool, this->stage_pack[0], this->stage_pack[1], forces...);
      apply_euler_predictor(dt);

      this->evaluate_forces(random_pool, this->stage_pack[2], this->stage_pack[3], forces...);
      // apply_heun_corrector(dt);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_HEUN_HPP