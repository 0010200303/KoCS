#ifndef KOCS_INTEGRATORS_HEUN_HPP
#define KOCS_INTEGRATORS_HEUN_HPP

#include <Kokkos_Core.hpp>

#include "base.hpp"

namespace kocs::integrators {
  template<typename PairFinder, typename... Views>
  struct Heun : public Base<PairFinder, 4, Views...> {
    using Base<PairFinder, 4, Views...>::Base;

    void apply_euler_predictor(double dt) {
      Kokkos::parallel_for(
        "apply_euler_predictor",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          this->stage_pack[2].apply([&](auto&... predicted_views) {
            this->stage_pack[1].apply([&](auto&... delta_views) {
              this->stage_pack[0].apply([&](auto&... current_views) {
                ((predicted_views(i) = current_views(i) + delta_views(i) * dt), ...);
              });
            });
          });
        }
      );
    }

    void apply_heun_corrector(double dt) {
      Kokkos::parallel_for(
        "apply_heun_corrector",
        this->agent_count,
        KOKKOS_CLASS_LAMBDA(const unsigned int i) {
          // write old velocities
          this->old_velocities(i) = (this->stage_pack[1].first()(i) + this->stage_pack[3].first()(i)) * 0.5;

          this->stage_pack[0].apply([&](auto&... current_views) {
            this->stage_pack[1].apply([&](auto&... delta_views_0) {
              this->stage_pack[2].apply([&](auto&... predicted_views) {
                this->stage_pack[3].apply([&](auto&... delta_views_1) {
                  ((current_views(i) += (delta_views_0(i) + delta_views_1(i)) * 0.5 * dt), ...);

                  // clear views
                  ((delta_views_0(i) = 0), ...);
                  ((delta_views_1(i) = 0), ...);
                });
              });
            });
          });
        }
      );
    }

    template<typename RandomPool, typename... Forces>
    void integrate(double dt, RandomPool& random_pool, Forces... forces) {
      this->evaluate_forces(random_pool, this->stage_pack[0].first(), this->stage_pack[1], forces...);
      apply_euler_predictor(dt);

      this->evaluate_forces(random_pool, this->stage_pack[2].first(), this->stage_pack[3], forces...);
      apply_heun_corrector(dt);
    }
  };
} // namespace kocs::integrators

#endif // KOCS_INTEGRATORS_HEUN_HPP
