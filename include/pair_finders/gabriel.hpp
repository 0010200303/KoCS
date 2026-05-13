#ifndef KOCS_PAIR_FINDERS_GABRIEL_HPP
#define KOCS_PAIR_FINDERS_GABRIEL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

// TODO: optimization to test:
// compute all distances then construct gabriel graph using the precalculated distances

namespace kocs::pair_finders {
  template<typename PositionsView, typename Scalar>
  struct NaiveGabriel {
    using positions_view_type = PositionsView;

    NaiveGabriel(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      Scalar gabriel_coefficient_ = 0.8f)
      : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance)
      , gabriel_coefficient(gabriel_coefficient_) { }
    
    static const constexpr Scalar epsilon = Scalar(1e-6);

    unsigned int agent_count;
    Scalar cutoff_distance_squared;
    Scalar gabriel_coefficient;

    template<typename RandomPool, typename Force, typename... Views>
    void evaluate_force(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities,
      RandomPool& random_pool,
      Force force
    ) {
      Kokkos::parallel_for(
        "naive_gabriel_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          const auto& input_positions = in_view_pack.first();
          const auto& position_i = input_positions(i);

          auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar total_friction_i = 0.0;
          typename PositionsView::value_type total_velocity_i{0.0};

          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, agent_count),
            [&](const int j, auto& local_delta, auto& local_friction, auto& local_velocity) {
              if (i == j)
                return;

              const auto& position_j = input_positions(j);
              const auto displacement = position_i - position_j;
              const auto distance_squared = displacement.length_squared();

              if (distance_squared >= cutoff_distance_squared)
                return;

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

  template<typename PositionsView, typename Scalar>
  struct TustGabriel {
    using positions_view_type = PositionsView;

    TustGabriel(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      Scalar gabriel_coefficient_ = 0.8f)
      : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance)
      , gabriel_coefficient(gabriel_coefficient_)
      , particle_bin("gabriel_particle_bin", agent_count_)
      , permutation("gabriel_permutation", agent_count_) { }
    
    static const constexpr Scalar epsilon = Scalar(1e-6);

    unsigned int agent_count;
    Scalar cutoff_distance_squared;
    Scalar gabriel_coefficient;



    const Vector3<Scalar> _min = Vector3<Scalar>(-3.0f);
    const Vector3<Scalar> _max = Vector3<Scalar>( 3.0f);
    const Scalar bin_size = 0.5f;

    View<unsigned int> particle_bin;
    View<unsigned int> permutation;

    template<typename... Views>
    void build(detail::ViewPack<Views...>& in_view_pack) {
      const unsigned int nx = static_cast<unsigned int>(Kokkos::ceil((_max[0] - _min[0]) / bin_size));
      const unsigned int ny = static_cast<unsigned int>(Kokkos::ceil((_max[1] - _min[1]) / bin_size));
      const unsigned int nz = static_cast<unsigned int>(Kokkos::ceil((_max[2] - _min[2]) / bin_size));
      const unsigned int n_bins = nx * ny * nz;

      Kokkos::parallel_for("gabriel_build", agent_count, KOKKOS_CLASS_LAMBDA(const unsigned int i) {
        const auto& input_positions = in_view_pack.first();
        const auto& position_i = input_positions(i);

        int ix = static_cast<unsigned int>(Kokkos::floor((position_i[0] - _min[0]) / bin_size));
        int iy = static_cast<unsigned int>(Kokkos::floor((position_i[1] - _min[1]) / bin_size));
        int iz = static_cast<unsigned int>(Kokkos::floor((position_i[2] - _min[2]) / bin_size));
        const int bin = ix + nx * (iy + ny * iz);

        if (ix < 0) ix = 0;
        if (iy < 0) iy = 0;
        if (iz < 0) iz = 0;
        if (ix >= static_cast<int>(nx)) ix = static_cast<int>(nx) - 1;
        if (iy >= static_cast<int>(ny)) iy = static_cast<int>(ny) - 1;
        if (iz >= static_cast<int>(nz)) iz = static_cast<int>(nz) - 1;


        particle_bin(i) = bin;
        // permutation(i) = i;
      });
      // Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), particle_bin, permutation);

      Kokkos::BinOp1D<View<unsigned int>> bin_op(n_bins, 0, n_bins - 1);
      Kokkos::BinSort<View<unsigned int>, Kokkos::BinOp1D<View<unsigned int>>> sorter(particle_bin, 0, n_bins - 1, bin_op);

      sorter.create_permute_vector();
    }

    template<typename RandomPool, typename Force, typename... Views>
    void evaluate_force(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities,
      RandomPool& random_pool,
      Force force
    ) {
      build(in_view_pack);
    }
  };
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_GABRIEL_HPP
