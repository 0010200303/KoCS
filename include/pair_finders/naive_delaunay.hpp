#ifndef KOCS_PAIR_FINDERS_NAIVE_DELAUNEY_HPP
#define KOCS_PAIR_FINDERS_NAIVE_DELAUNEY_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

namespace kocs::pair_finders {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct NaiveDelaunay {
    using positions_view_type = PositionsView;

    struct Settings { };

    NaiveDelaunay(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      const Settings& settings
    ) : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance) { }

    unsigned int agent_count;
    Scalar cutoff_distance_squared;

    template<typename RandomPool, typename Force, typename ForceFields, typename... Views>
    void evaluate_force(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities,
      RandomPool& random_pool,
      Force force,
      bool is_full_step
    ) {
      using Vector = VectorN<Scalar, dimensions>;

      const auto& input_positions = in_view_pack.first();

      if constexpr (dimensions == 2) {
        const int block = (agent_count - 1) * (agent_count - 2);
        Kokkos::parallel_for(
          "naive_delauney_apply_force",
          agent_count * (agent_count - 1) * (agent_count - 2),
          KOKKOS_CLASS_LAMBDA(const int idx) {
            const int rem = idx % block;
            const int i = idx / block;
            int j = rem / (agent_count - 2);
            int k = rem % (agent_count - 2);

            j += j >= i;
            k += k >= i;
            k += k >= j;

            const auto position_i = input_positions(i);
            const auto position_j = input_positions(j);

            const auto displacement = position_i - position_j;
            const auto distance_squared = displacement.length_squared();

            if (distance_squared >= cutoff_distance_squared)
              return;

            const auto position_k = input_positions(k);

            const Scalar d = Scalar(2) * (
              position_i[0] * (position_j[1] - position_k[1]) +
              position_j[0] * (position_k[1] - position_i[1]) +
              position_k[0] * (position_i[1] - position_j[1])
            );
            if (d == Scalar(0))
              return;

            Vector circumcenter = Vector(
              (
                position_i.length_squared() * (position_j[1] - position_k[1]) +
                position_j.length_squared() * (position_k[1] - position_i[1]) +
                position_k.length_squared() * (position_i[1] - position_j[1])
              ) / d,
              (
                position_i.length_squared() * (position_k[0] - position_j[0]) +
                position_j.length_squared() * (position_i[0] - position_k[0]) +
                position_k.length_squared() * (position_j[0] - position_i[0])
              ) / d
            );
            const Scalar radius_squared = position_i.distance_to_squared(circumcenter);

            // check every other cell
            for (int l = 0; l < agent_count; ++l) {
              if (l == i || l == j || l == k)
                continue;

              if (input_positions(l).distance_to_squared(circumcenter) <= radius_squared)
                return;
            }

            const auto distance = Kokkos::sqrt(distance_squared);

            // drag

            auto generator = random_pool.get_state();

            auto _delta = detail::make_accumulator_pack(out_view_pack);
            Scalar kek = Scalar(0);
            in_view_pack.apply([&](auto&... views) {
              _delta.apply([&](auto&... deltas) {
                force(
                  is_full_step, i, j, displacement, distance, generator, kek,
                  ForceFields{detail::PairwiseFieldRef{views(i), views(j), deltas}...}
                );
              });
            });
            random_pool.free_state(generator);
            
            // drag
            // velocity
          }
        );
      }
    }
  };
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_NAIVE_DELAUNEY_HPP
