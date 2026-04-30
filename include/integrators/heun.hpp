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
    static void reset_pack_at(const detail::ViewPack<FirstView, RestViews...>& pack, const unsigned int i) {
      pack.first()(i) = std::remove_cv_t<std::remove_reference_t<decltype(pack.first()(i))>>{};
      reset_pack_at(static_cast<const typename detail::ViewPack<FirstView, RestViews...>::base_type&>(pack), i);
    }

    KOKKOS_INLINE_FUNCTION
    static void reset_pack_at(const detail::ViewPack<>&, const unsigned int) { }

    template<typename FirstView, typename... RestViews>
    KOKKOS_INLINE_FUNCTION
    static void predict_pack_at(
      const detail::ViewPack<FirstView, RestViews...>& dst,
      const detail::ViewPack<FirstView, RestViews...>& current,
      const detail::ViewPack<FirstView, RestViews...>& delta,
      const unsigned int i,
      const double dt
    ) {
      dst.first()(i) = current.first()(i) + delta.first()(i) * dt;
      predict_pack_at(
        static_cast<const typename detail::ViewPack<FirstView, RestViews...>::base_type&>(dst),
        static_cast<const typename detail::ViewPack<FirstView, RestViews...>::base_type&>(current),
        static_cast<const typename detail::ViewPack<FirstView, RestViews...>::base_type&>(delta),
        i,
        dt
      );
    }

    KOKKOS_INLINE_FUNCTION
    static void predict_pack_at(
      const detail::ViewPack<>&,
      const detail::ViewPack<>&,
      const detail::ViewPack<>&,
      const unsigned int,
      const double
    ) { }

    template<typename FirstView, typename... RestViews>
    KOKKOS_INLINE_FUNCTION
    static void correct_pack_at(
      const detail::ViewPack<FirstView, RestViews...>& current,
      const detail::ViewPack<FirstView, RestViews...>& delta0,
      const detail::ViewPack<FirstView, RestViews...>& delta1,
      const unsigned int i,
      const double dt
    ) {
      current.first()(i) += (delta0.first()(i) + delta1.first()(i)) * 0.5 * dt;
      correct_pack_at(
        static_cast<const typename detail::ViewPack<FirstView, RestViews...>::base_type&>(current),
        static_cast<const typename detail::ViewPack<FirstView, RestViews...>::base_type&>(delta0),
        static_cast<const typename detail::ViewPack<FirstView, RestViews...>::base_type&>(delta1),
        i,
        dt
      );
    }

    KOKKOS_INLINE_FUNCTION
    static void correct_pack_at(
      const detail::ViewPack<>&,
      const detail::ViewPack<>&,
      const detail::ViewPack<>&,
      const unsigned int,
      const double
    ) { }

  public:
    void apply_euler_predictor(float dt) {
      Kokkos::parallel_for(
        "apply_euler_predictor",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          predict_pack_at(this->stage_pack[2], this->stage_pack[0], this->stage_pack[1], i, dt);
        }
      );
    }

    void apply_heun_corrector(double dt) {
      Kokkos::parallel_for(
        "apply_heun_corrector",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          this->old_velocities(i) =
            (this->stage_pack[1].first()(i) + this->stage_pack[3].first()(i)) * 0.5;

          correct_pack_at(this->stage_pack[0], this->stage_pack[1], this->stage_pack[3], i, dt);

          reset_pack_at(this->stage_pack[1], i);
          reset_pack_at(this->stage_pack[2], i);
          reset_pack_at(this->stage_pack[3], i);
        }
      );
    }

    template<typename RandomPool, typename... Forces>
    void integrate(double dt, RandomPool& random_pool, Forces... forces) {
      this->evaluate_forces(random_pool, this->stage_pack[0], this->stage_pack[1], forces...);
      apply_euler_predictor(dt);

      this->evaluate_forces(random_pool, this->stage_pack[2], this->stage_pack[3], forces...);
      apply_heun_corrector(dt);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_HEUN_HPP
