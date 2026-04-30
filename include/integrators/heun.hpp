#ifndef KOCS_INTEGRATORS_HEUN_HPP
#define KOCS_INTEGRATORS_HEUN_HPP

#include <type_traits>
#include <Kokkos_Core.hpp>

#include "base.hpp"

namespace kocs::integrators {
  template<typename PairFinder, typename... Views>
  struct Heun : public Base<PairFinder, 4, Views...> {
    using Base<PairFinder, 4, Views...>::Base;

  public:
    void apply_euler_predictor(double dt) {
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
          old_velocities(i) = (stage_pack[1].first()(i) + stage_pack[3].first()(i)) * 0.5;

          stage_pack[0].zip_apply([&](auto& current, const auto& delta_0, const auto& delta_1) {
            current(i) += (delta_0(i) + delta_1(i)) * 0.5 * dt;

            // clear deltas
            delta_0(i) = std::remove_cv_t<std::remove_reference_t<decltype(delta_0(i))>>{};
            delta_1(i) = std::remove_cv_t<std::remove_reference_t<decltype(delta_1(i))>>{};
          }, stage_pack[1], stage_pack[3]);
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