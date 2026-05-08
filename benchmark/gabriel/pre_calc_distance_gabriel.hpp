#ifndef BENCHMARK_PRE_CALC_GABRIEL_HPP
#define BENCHMARK_PRE_CALC_GABRIEL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "../../include/integrators/detail.hpp"
#include "../../include/forces/detail.hpp"

using namespace kocs;

template<typename PositionsView, typename Scalar>
struct BenchmarkPreCalcGabriel {
  using positions_view_type = PositionsView;

  KOKKOS_INLINE_FUNCTION
  static constexpr unsigned int n_to_array_size(const unsigned int N) {
    return (N * (N - 1)) / 2;
  }

  static constexpr unsigned int pair_to_index(unsigned int N, unsigned int i, unsigned int j) {
    if (i > j)
      Kokkos::kokkos_swap(i, j);
    return i * (2 * N - i - 1) / 2 + (j - i - 1);
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr unsigned int index_to_i(const unsigned int N, const unsigned int index) {
    return static_cast<unsigned int>(
      Kokkos::floor(((2 * N - 1) - Kokkos::sqrt(Kokkos::pow(2 * N - 1, 2) - 8 * index)) / 2)
    );
  }

  KOKKOS_INLINE_FUNCTION
  static constexpr unsigned int index_to_j(
    const unsigned int N,
    const unsigned int index,
    const unsigned int i
  ) {
    return i + 1 + (index - (i * (2 * N - i - 1)) / 2);
  }

  BenchmarkPreCalcGabriel(
    unsigned int agent_count_,
    Scalar cutoff_distance,
    Scalar gabriel_coefficient_ = 0.8f)
    : agent_count(agent_count_)
    , cutoff_distance_squared(cutoff_distance * cutoff_distance)
    , gabriel_coefficient(gabriel_coefficient_)
    , distances_view("distances", n_to_array_size(agent_count_)) { }

  static const constexpr Scalar epsilon = Scalar(1e-6);

  unsigned int agent_count;
  Scalar cutoff_distance_squared;
  Scalar gabriel_coefficient;

  Kokkos::View<Scalar*> distances_view;

  template<typename RandomPool, typename Force, typename... Views>
  void evaluate_force(
    detail::ViewPack<Views...>& in_view_pack,
    detail::ViewPack<Views...>& out_view_pack,
    PositionsView& old_velocities,
    RandomPool& random_pool,
    Force force
  ) {
    const auto& input_positions = in_view_pack.first();

    // calculate distances
    Kokkos::parallel_for(
      "pre_calc_dist_gabriel_compute_distances",
      n_to_array_size(agent_count),
      KOKKOS_CLASS_LAMBDA(const unsigned int i) {
        const unsigned int index_i = index_to_i(agent_count, i);
        const unsigned int index_j = index_to_j(agent_count, i, index_i);

        distances_view(i) = input_positions(index_i).distance_to_squared(input_positions(index_j));
      }
    );

    Kokkos::parallel_for(
      "pre_calc_dist_gabriel_apply_force",
      Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
      KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
        const int i = team_member.league_rank();
        const auto& position_i = input_positions(i);

        auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
        Scalar total_friction_i = 0.0;
        typename PositionsView::value_type total_velocity_i{0.0};

        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, agent_count),
          [&](const int j, auto& local_delta, auto& local_friction, auto& local_velocity) {
            if (i == j)
              return;

            // const auto& position_j = input_positions(j);
            // const auto displacement = position_i - position_j;
            // const auto distance_squared = displacement.length_squared();
            const Scalar distance_squared = distances_view(pair_to_index(agent_count, i, j));

            if (distance_squared >= cutoff_distance_squared)
              return;
            
            const auto displacement = position_i - input_positions(j);

            const auto midpoint = position_i - displacement * 0.5f;
            const auto radius_squared = distance_squared * 0.25f * gabriel_coefficient;
            for (int k = 0; k < agent_count; ++k) {
              if (k == i || k == j)
                continue;

              const auto offset = input_positions(k) - midpoint;
              if (offset.length_squared() < radius_squared)
                return;
            }

            const auto distance = Kokkos::sqrt(distance_squared);

            Scalar pair_friction = 0.0;

            auto generator = random_pool.get_state();
            in_view_pack.apply([&](auto&... views) {
              local_delta.apply([&](auto&... deltas) {
                force(
                  i, j, displacement, distance, generator, pair_friction,
                  detail::PairwiseFieldRef{views(i), views(j), deltas}...
                );
              });
            });
            random_pool.free_state(generator);

            local_friction += pair_friction;
            local_velocity += pair_friction * old_velocities(j);
          },
          Kokkos::Sum<decltype(total_delta_i)>(total_delta_i),
          Kokkos::Sum<Scalar>(total_friction_i),
          Kokkos::Sum<typename PositionsView::value_type>(total_velocity_i)
        );

        Kokkos::single(
          Kokkos::PerTeam(team_member),
          [&]() {
            out_view_pack.apply([&](auto&... views) {
              total_delta_i.apply([&](auto&... values) {
                ((views(i) += values), ...);
              });
            });

            out_view_pack.first()(i) += total_velocity_i / Kokkos::fmax(total_friction_i, epsilon);
          }
        );
      }
    );
  }
};

#endif // BENCHMARK_PRE_CALC_GABRIEL_HPP
