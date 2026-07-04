#ifndef KOCS_PAIR_FINDERS_NAIVE_DELAUNAY_HPP
#define KOCS_PAIR_FINDERS_NAIVE_DELAUNAY_HPP

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

    static_assert(dimensions >= 2, "NaiveDelaunay requires at least 2 dimensions");

    NaiveDelaunay(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      const Settings& settings
    ) : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance) { }

    unsigned int agent_count;
    Scalar cutoff_distance_squared;

    KOKKOS_INLINE_FUNCTION
    int comb3(const int a) const {
      return (a * (a - 1) * (a - 2)) / 6;
    }

    // benchmark different loops
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
        // using delta_type = decltype(detail::make_accumulator_pack(out_view_pack));
        // Kokkos::View<delta_type*> total_delta_view("naive_delaunay_apply_force_total_delta", agent_count);
        Kokkos::View<Scalar*> total_drag_view("naive_delaunay_apply_force_total_drag", agent_count);
        Kokkos::View<Vector*> total_velocity_view("naive_delaunay_apply_force_total_velocity", agent_count);

        const unsigned long long block = (agent_count - 1) * (agent_count - 2);
        Kokkos::parallel_for(
          "naive_delaunay_apply_force",
          Kokkos::RangePolicy<Kokkos::IndexType<unsigned long long>>(0, static_cast<unsigned long long>(agent_count) * block),
          KOKKOS_CLASS_LAMBDA(const unsigned long long idx) {
            // flat loop
            const int rem = idx % block;
            const int i = idx / block;
            int j = rem / (agent_count - 2);
            int k = rem % (agent_count - 2);

            j += j >= i;
            k += k >= (i < j ? i : j);
            k += k >= (i < j ? j : i);

            const auto position_i = input_positions(i);
            const auto position_j = input_positions(j);

            const auto displacement = position_i - position_j;
            const auto distance_squared = displacement.length_squared();
            if (distance_squared >= cutoff_distance_squared)
              return;

            const auto position_k = input_positions(k);

            const Scalar denominator = Scalar(2) * (
              position_i[0] * (position_j[1] - position_k[1]) +
              position_j[0] * (position_k[1] - position_i[1]) +
              position_k[0] * (position_i[1] - position_j[1])
            );

            // collinearity check
            if (Kokkos::abs(denominator) <= Kokkos::Experimental::epsilon_v<Scalar>)
              return;

            Vector circumcenter = Vector(
              (
                position_i.length_squared() * (position_j[1] - position_k[1]) +
                position_j.length_squared() * (position_k[1] - position_i[1]) +
                position_k.length_squared() * (position_i[1] - position_j[1])
              ) / denominator,
              (
                position_i.length_squared() * (position_k[0] - position_j[0]) +
                position_j.length_squared() * (position_i[0] - position_k[0]) +
                position_k.length_squared() * (position_j[0] - position_i[0])
              ) / denominator
            );
            const Scalar radius_squared = position_i.distance_to_squared(circumcenter);

            // check every other cell
            for (int m = 0; m < agent_count; ++m) {
              if (m == i || m == j || m == k)
                continue;

              if (input_positions(m).distance_to_squared(circumcenter) <= radius_squared)
                return;
            }

            const auto distance = Kokkos::sqrt(distance_squared);

            auto local_delta = detail::make_accumulator_pack(out_view_pack);
            Scalar pairwise_drag = 0.0;

            auto generator = random_pool.get_state();
            in_view_pack.apply([&](auto&... views) {
              local_delta.apply([&](auto&... deltas) {
                force(
                  is_full_step, i, j, displacement, distance, generator, pairwise_drag,
                  ForceFields{detail::PairwiseFieldRef{views(i), views(j), deltas}...}
                );
              });
            });
            random_pool.free_state(generator);

            // total_delta_view(i).apply([&](auto&... total_deltas) {
              // local_delta.apply([&](auto&... local_deltas) {
                // ((Kokkos::atomic_add(&total_deltas, local_deltas)), ...);
              // });
            // });

            out_view_pack.apply([&](auto&... views) {
              local_delta.apply([&](auto&... local_deltas) {
                ((Kokkos::atomic_add(&views(i), local_deltas)), ...);
              });
            });

            Kokkos::atomic_add(&total_drag_view(i), pairwise_drag);
            Kokkos::atomic_add(&total_velocity_view(i), pairwise_drag * old_velocities(j));

            return;
          }
        );

        Kokkos::parallel_for(
          "naive_delaunay_apply_changes",
          agent_count,
          KOKKOS_CLASS_LAMBDA(const int i) {
            // out_view_pack.apply([&](auto&... views) {
              // total_delta_view(i).apply([&](auto&... values) {
                // ((views(i) += values), ...);
              // });
            // });

            out_view_pack.first()(i) += total_velocity_view(i) /
              Kokkos::fmax(total_drag_view(i), Kokkos::Experimental::epsilon_v<Scalar>);
          }
        );



        // Kokkos::parallel_for(
        //   "naive_delaunay_apply_force",
        //   agent_count,
        //   KOKKOS_CLASS_LAMBDA(const int idx) {
        //     const int i = idx;
        //     const auto position_i = input_positions(i);

        //     auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
        //     Scalar total_drag_i = 0.0;
        //     typename PositionsView::value_type total_velocity_i{0.0};

        //     for (int j = 0; j < agent_count; ++j) {
        //       if (j == i)
        //         continue;
        //       const auto position_j = input_positions(j);
        //       const auto displacement = position_i - position_j;
        //       const auto distance_squared = displacement.length_squared();
        //       if (distance_squared >= cutoff_distance_squared)
        //         continue;
              
        //       for (int k = 0; k < agent_count; ++k) {
        //         if (k == i || k == j)
        //           continue;
        //         const auto position_k = input_positions(k);
                
        //         const Scalar d = Scalar(2) * (
        //           position_i[0] * (position_j[1] - position_k[1]) +
        //           position_j[0] * (position_k[1] - position_i[1]) +
        //           position_k[0] * (position_i[1] - position_j[1])
        //         );
        //         if (Kokkos::abs(d) <= Kokkos::Experimental::epsilon_v<Scalar>)
        //           continue;

        //         Vector circumcenter = Vector(
        //           (
        //             position_i.length_squared() * (position_j[1] - position_k[1]) +
        //             position_j.length_squared() * (position_k[1] - position_i[1]) +
        //             position_k.length_squared() * (position_i[1] - position_j[1])
        //           ) / d,
        //           (
        //             position_i.length_squared() * (position_k[0] - position_j[0]) +
        //             position_j.length_squared() * (position_i[0] - position_k[0]) +
        //             position_k.length_squared() * (position_j[0] - position_i[0])
        //           ) / d
        //         );
        //         const Scalar radius_squared = position_i.distance_to_squared(circumcenter);

        //         // check every other cell
        //         bool valid = true;
        //         for (int l = 0; l < agent_count; ++l) {
        //           if (l == i || l == j || l == k)
        //             continue;
                  
        //           if (input_positions(l).distance_to_squared(circumcenter) <= radius_squared) {
        //             valid = false;
        //             break;
        //           }
        //         }
        //         if (valid == false)
        //           continue;

        //         auto local_delta = detail::make_accumulator_pack(out_view_pack);
        //         Scalar pairwise_drag = 0.0;

        //         auto generator = random_pool.get_state();
        //         in_view_pack.apply([&](auto&... views) {
        //           total_delta_i.apply([&](auto&... deltas) {
        //             force(
        //               is_full_step, i, j, displacement, Kokkos::sqrt(distance_squared), generator, pairwise_drag,
        //               ForceFields{detail::PairwiseFieldRef{views(i), views(j), deltas}...}
        //             );
        //           });
        //         });
        //         random_pool.free_state(generator);

        //         total_drag_i += pairwise_drag;
        //         total_velocity_i += pairwise_drag * old_velocities(j);

        //         break;
        //       }
        //     }

        //     out_view_pack.apply([&](auto&... views) {
        //       total_delta_i.apply([&](auto&... values) {
        //         ((views(i) += values), ...);
        //       });
        //     });

        //     out_view_pack.first()(i) += total_velocity_i /
        //       Kokkos::fmax(total_drag_i, Kokkos::Experimental::epsilon_v<Scalar>);
        //   }
        // );
      }
    }
  };
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_NAIVE_DELAUNAY_HPP
