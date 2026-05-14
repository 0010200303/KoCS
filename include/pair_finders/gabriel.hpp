#ifndef KOCS_PAIR_FINDERS_GABRIEL_HPP
#define KOCS_PAIR_FINDERS_GABRIEL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

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
      const auto& input_positions = in_view_pack.first();

      Kokkos::parallel_for(
        "naive_gabriel_apply_force",
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

      , nx(static_cast<unsigned int>(Kokkos::ceil((_max[0] - _min[0]) / bin_size)))
      , ny(static_cast<unsigned int>(Kokkos::ceil((_max[1] - _min[1]) / bin_size)))
      , nz(static_cast<unsigned int>(Kokkos::ceil((_max[2] - _min[2]) / bin_size)))
      , n_bins(nx * ny * nz)
      , particle_bin("gabriel_particle_bin", agent_count_)
      , permutation("gabriel_permutation", agent_count_)
      , bin_offset("gabriel_bin_offset", n_bins + 1) { }
    
    static const constexpr Scalar epsilon = Scalar(1e-6);

    unsigned int agent_count;
    Scalar cutoff_distance_squared;
    Scalar gabriel_coefficient;



    const Vector3<Scalar> _min = Vector3<Scalar>(-5.0f);
    const Vector3<Scalar> _max = Vector3<Scalar>( 5.0f);
    const Scalar bin_size = 1.0f * Kokkos::sqrt(cutoff_distance_squared);
    // TODO: bin_size = scale * cutoff_distance;

    unsigned int nx;
    unsigned int ny;
    unsigned int nz;
    unsigned int n_bins;

    View<unsigned int> particle_bin;
    View<unsigned int> permutation;
    View<unsigned int> bin_offset;

    int step_count = 0;
    int build_every_n = 20;

    template<typename... Views>
    void build(detail::ViewPack<Views...>& in_view_pack) {
      const auto& input_positions = in_view_pack.first();

      Kokkos::parallel_for("gabriel_build", agent_count, KOKKOS_CLASS_LAMBDA(const unsigned int i) {
        const auto& position_i = input_positions(i);

        int ix = static_cast<int>(Kokkos::floor((position_i[0] - _min[0]) / bin_size));
        int iy = static_cast<int>(Kokkos::floor((position_i[1] - _min[1]) / bin_size));
        int iz = static_cast<int>(Kokkos::floor((position_i[2] - _min[2]) / bin_size));

        ix = max(0, min(ix, nx - 1));
        iy = max(0, min(iy, ny - 1));
        iz = max(0, min(iz, nz - 1));
        const unsigned int bin = ix + nx * (iy + ny * iz);

        particle_bin(i) = bin;
        permutation(i) = i;
      });
      // Kokkos::BinOp1D<View<unsigned int>> bin_op(n_bins, 0, n_bins - 1);
      // Kokkos::BinSort<View<unsigned int>, Kokkos::BinOp1D<View<unsigned int>>> sorter(particle_bin, 0, n_bins - 1, bin_op);
      // sorter.create_permute_vector();

      Kokkos::Experimental::sort_by_key(Kokkos::DefaultExecutionSpace(), particle_bin, permutation);

      // build offsets
      Kokkos::deep_copy(bin_offset, -1);
      Kokkos::parallel_for("gabriel_build_offsets", agent_count, KOKKOS_CLASS_LAMBDA(const unsigned int i) {
        unsigned int b = particle_bin(i);

        if (i == 0 || particle_bin(i - 1) != b)
          bin_offset(b) = i;
      });

      // forward sweep
      auto bin_offset_host = Kokkos::create_mirror_view(bin_offset);
      Kokkos::deep_copy(bin_offset_host, bin_offset);

      int last = agent_count;
      for (int b = n_bins; b >= 0; --b) {
        if (bin_offset_host(b) == -1)
          bin_offset_host(b) = last;
        else
          last = bin_offset_host(b);
      }
      Kokkos::deep_copy(bin_offset, bin_offset_host);
    }

    template<typename RandomPool, typename Force, typename... Views>
    void evaluate_force(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities,
      RandomPool& random_pool,
      Force force
    ) {
      const auto& input_positions = in_view_pack.first();

      if (step_count == build_every_n) {
        build(in_view_pack);
        step_count = 0;
      }
      step_count++;

      Kokkos::parallel_for(
        "naive_gabriel_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          const auto& position_i = input_positions(i);

          auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar total_friction_i = 0.0;
          typename PositionsView::value_type total_velocity_i{0.0};

          // compute bin coords for particle i
          int ix = static_cast<int>(Kokkos::floor((position_i[0] - _min[0]) / bin_size));
          int iy = static_cast<int>(Kokkos::floor((position_i[1] - _min[1]) / bin_size));
          int iz = static_cast<int>(Kokkos::floor((position_i[2] - _min[2]) / bin_size));

          ix = max(0, min(ix, static_cast<int>(nx - 1)));
          iy = max(0, min(iy, static_cast<int>(ny - 1)));
          iz = max(0, min(iz, static_cast<int>(nz - 1)));

          // number of bins to search in each direction based on cutoff
          const Scalar cutoff = Kokkos::sqrt(cutoff_distance_squared);
          const int rb = static_cast<int>(Kokkos::ceil(cutoff / bin_size));
          const int side = 2 * rb + 1;
          const int n_bin_tasks = side * side * side;

          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, n_bin_tasks),
            [&](const int task_idx, auto& local_delta, auto& local_friction, auto& local_velocity) {
              // map task_idx -> dx ,dy, dz in [-rb, rb]
              int t = task_idx;
              const int dz = (t / (side * side)) - rb;
              t %= (side * side);
              const int dy = (t / side) - rb;
              const int dx = (t % side) - rb;

              const int nix = ix + dx;
              const int niy = iy + dy;
              const int niz = iz + dz;

              if (nix < 0 || nix >= nx || niy < 0 || niy >= ny || niz < 0 || niz >= nz)
                return;

              const unsigned int b = nix + nx * (niy + ny * niz);
              const unsigned int start = bin_offset(b);
              const unsigned int end = bin_offset(b + 1);



              for (unsigned int idx = start; idx < end; ++idx) {
                const int j = static_cast<int>(permutation(idx));
                if (j == i) continue;

                const auto& position_j = input_positions(j);
                const auto displacement = position_i - position_j;
                const auto distance_squared = displacement.length_squared();

                if (distance_squared >= cutoff_distance_squared) continue;

                const auto midpoint = position_i - displacement * Scalar(0.5);
                const auto radius_squared = distance_squared * Scalar(0.25) * gabriel_coefficient;

                // determine overlapping bins for the midpoint-sphere
                const Scalar radius = Kokkos::sqrt(radius_squared);
                int min_bx = static_cast<int>(Kokkos::floor((midpoint[0] - radius - _min[0]) / bin_size));
                int min_by = static_cast<int>(Kokkos::floor((midpoint[1] - radius - _min[1]) / bin_size));
                int min_bz = static_cast<int>(Kokkos::floor((midpoint[2] - radius - _min[2]) / bin_size));

                int max_bx = static_cast<int>(Kokkos::floor((midpoint[0] + radius - _min[0]) / bin_size));
                int max_by = static_cast<int>(Kokkos::floor((midpoint[1] + radius - _min[1]) / bin_size));
                int max_bz = static_cast<int>(Kokkos::floor((midpoint[2] + radius - _min[2]) / bin_size));

                min_bx = max(0, min(min_bx, static_cast<int>(nx - 1)));
                min_by = max(0, min(min_by, static_cast<int>(ny - 1)));
                min_bz = max(0, min(min_bz, static_cast<int>(nz - 1)));
                max_bx = max(0, min(max_bx, static_cast<int>(nx - 1)));
                max_by = max(0, min(max_by, static_cast<int>(ny - 1)));
                max_bz = max(0, min(max_bz, static_cast<int>(nz - 1)));

                bool blocked = false;
                for (int bx = min_bx; bx <= max_bx && !blocked; ++bx) {
                  for (int by = min_by; by <= max_by && !blocked; ++by) {
                    for (int bz = min_bz; bz <= max_bz && !blocked; ++bz) {

                      const unsigned int bb = static_cast<unsigned int>(bx + nx * (by + ny * bz));
                      const unsigned int s2 = bin_offset(bb);
                      const unsigned int e2 = bin_offset(bb + 1);

                      for (unsigned int idx2 = s2; idx2 < e2; ++idx2) {
                        const int k = static_cast<int>(permutation(idx2));
                        if (k == i || k == j)
                          continue;

                        const auto offset = input_positions(k) - midpoint;
                        if (offset.length_squared() < radius_squared) {
                          blocked = true;
                          break;
                        }
                      }
                    }
                  }
                }
                if (blocked)
                  continue;

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
              }



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
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_GABRIEL_HPP
