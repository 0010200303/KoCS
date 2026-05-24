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
      Scalar gabriel_coefficient = 0.8f)
      : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance)
      , gabriel_coefficient_squared(gabriel_coefficient * gabriel_coefficient) { }
    
    static const constexpr Scalar epsilon = Scalar(1e-6);

    unsigned int agent_count;
    Scalar cutoff_distance_squared;
    Scalar gabriel_coefficient_squared;

    template<typename RandomPool, typename Force, typename... Views>
    void evaluate_force(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities,
      RandomPool& random_pool,
      Force force,
      bool is_full_step
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

              const auto midpoint = position_i - displacement * Scalar(0.5);
              const auto radius_squared = distance_squared * Scalar(0.25) * gabriel_coefficient_squared;
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
    using BinOp = Kokkos::BinOp1D<View<int>>;

    TustGabriel(
      unsigned int agent_count_,
      Scalar cutoff_distance,
      Scalar gabriel_coefficient = 0.8f)
      : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance)
      , gabriel_coefficient_squared(gabriel_coefficient * gabriel_coefficient)

      , nx(static_cast<int>(Kokkos::ceil((_max[0] - _min[0]) / bin_size)))
      , ny(static_cast<int>(Kokkos::ceil((_max[1] - _min[1]) / bin_size)))
      , nz(static_cast<int>(Kokkos::ceil((_max[2] - _min[2]) / bin_size)))
      , n_bins(nx * ny * nz)
      , particle_bins("gabriel_particle_bins", agent_count_)
      , sorter(particle_bins, 0, agent_count_, BinOp{n_bins, 0, n_bins}, true) { }
    
    static const constexpr Scalar epsilon = Scalar(1e-6);

    unsigned int agent_count;
    Scalar cutoff_distance_squared;
    Scalar gabriel_coefficient_squared;



    const Vector3<Scalar> _min = Vector3<Scalar>(-20.0f);
    const Vector3<Scalar> _max = Vector3<Scalar>( 20.0f);
    const Scalar bin_size = 1.0f * Kokkos::sqrt(cutoff_distance_squared);
    // TODO: bin_size = scale * cutoff_distance;

    int nx;
    int ny;
    int nz;
    int n_bins;

    View<int> particle_bins;
    Kokkos::BinSort<View<int>, BinOp> sorter;

    int step_count = 0;
    int rebuild_every_n = 0;

    View<int> cell_types_view;

    template<typename... Views>
    void build(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities
    ) {
      const auto& input_positions = in_view_pack.first();

      Kokkos::parallel_for("gabriel_build", agent_count, KOKKOS_CLASS_LAMBDA(const unsigned int i) {
        const auto& position_i = input_positions(i);

        int ix = Kokkos::max(0, Kokkos::min(static_cast<int>(Kokkos::floor((position_i[0] - _min[0]) / bin_size)), nx - 1));
        int iy = Kokkos::max(0, Kokkos::min(static_cast<int>(Kokkos::floor((position_i[1] - _min[1]) / bin_size)), ny - 1));
        int iz = Kokkos::max(0, Kokkos::min(static_cast<int>(Kokkos::floor((position_i[2] - _min[2]) / bin_size)), nz - 1));

        int bin = ix + nx * (iy + ny * iz);

        particle_bins(i) = bin;
      });

      sorter = Kokkos::BinSort<View<int>, BinOp>(particle_bins, 0, agent_count, BinOp{n_bins, 0, n_bins});
      sorter.create_permute_vector();

      in_view_pack.apply([&](auto&... views) { (sorter.sort(views), ...); });
      out_view_pack.apply([&](auto&... views) { (sorter.sort(views), ...); });
      sorter.sort(old_velocities);
      sorter.sort(cell_types_view);
    }

    template<typename RandomPool, typename Force, typename... Views>
    void evaluate_force(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities,
      RandomPool& random_pool,
      Force force,
      bool is_full_step
    ) {
      const auto& input_positions = in_view_pack.first();
      // const auto& permutation = sorter.get_permute_vector();
      const auto& bin_offsets = sorter.get_bin_offsets();

      if (is_full_step == true) {
        if (step_count == 0 || step_count >= rebuild_every_n) {
          build(in_view_pack, out_view_pack, old_velocities);
          step_count = 0;
        }
        step_count++;
      }

      float cutoff_distance = Kokkos::sqrt(cutoff_distance_squared);

      Kokkos::parallel_for(
        "naive_gabriel_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
auto generator = random_pool.get_state();

          const int i = team_member.league_rank();
          const auto& position_i = input_positions(i);

          auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar total_friction_i = 0.0;
          typename PositionsView::value_type total_velocity_i{0.0};

          // compute bin coords for particle i
          int ix = Kokkos::max(0, Kokkos::min(static_cast<int>(Kokkos::floor((position_i[0] - _min[0]) / bin_size)), nx - 1));
          int iy = Kokkos::max(0, Kokkos::min(static_cast<int>(Kokkos::floor((position_i[1] - _min[1]) / bin_size)), ny - 1));
          int iz = Kokkos::max(0, Kokkos::min(static_cast<int>(Kokkos::floor((position_i[2] - _min[2]) / bin_size)), nz - 1));

          // number of bins to search in each direction based on cutoff
          const Scalar cutoff = cutoff_distance;
          const int rb = static_cast<int>(Kokkos::ceil(cutoff / bin_size));
          const int side = 2 * rb + 1;
          const int n_bin_tasks = side * side * side;

          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, n_bin_tasks),
            [&](const int task_idx, auto& local_delta, auto& local_friction, auto& local_velocity) {
              // map task_idx -> dx ,dy, dz in [-rb, rb]
              const int dz = (task_idx / (side * side)) - rb;
              const int t = task_idx % (side * side);
              const int dy = (t / side) - rb;
              const int dx = (t % side) - rb;

              const int nix = ix + dx;
              const int niy = iy + dy;
              const int niz = iz + dz;

              if (nix < 0 || nix >= nx || niy < 0 || niy >= ny || niz < 0 || niz >= nz)
                return;

              const unsigned int b = nix + nx * (niy + ny * niz);
              const unsigned int start = bin_offsets(b);
              const unsigned int end = bin_offsets(b + 1);



              for (unsigned int idx = start; idx < end; ++idx) {
                const int j = static_cast<int>(idx);
                if (j == i)
                  continue;

                const auto& position_j = input_positions(j);
                const auto displacement = position_i - position_j;
                const auto distance_squared = displacement.length_squared();

                if (distance_squared >= cutoff_distance_squared)
                  continue;

                const auto midpoint = position_i - displacement * Scalar(0.5);
                const auto radius_squared = distance_squared * Scalar(0.25) * gabriel_coefficient_squared;

                // determine overlapping bins for the midpoint-sphere
                const Scalar radius = Kokkos::sqrt(radius_squared);
                int min_bx = static_cast<int>(Kokkos::floor((midpoint[0] - radius - _min[0]) / bin_size));
                int min_by = static_cast<int>(Kokkos::floor((midpoint[1] - radius - _min[1]) / bin_size));
                int min_bz = static_cast<int>(Kokkos::floor((midpoint[2] - radius - _min[2]) / bin_size));

                int max_bx = static_cast<int>(Kokkos::floor((midpoint[0] + radius - _min[0]) / bin_size));
                int max_by = static_cast<int>(Kokkos::floor((midpoint[1] + radius - _min[1]) / bin_size));
                int max_bz = static_cast<int>(Kokkos::floor((midpoint[2] + radius - _min[2]) / bin_size));

                min_bx = Kokkos::max(0, Kokkos::min(min_bx, static_cast<int>(nx - 1)));
                min_by = Kokkos::max(0, Kokkos::min(min_by, static_cast<int>(ny - 1)));
                min_bz = Kokkos::max(0, Kokkos::min(min_bz, static_cast<int>(nz - 1)));
                max_bx = Kokkos::max(0, Kokkos::min(max_bx, static_cast<int>(nx - 1)));
                max_by = Kokkos::max(0, Kokkos::min(max_by, static_cast<int>(ny - 1)));
                max_bz = Kokkos::max(0, Kokkos::min(max_bz, static_cast<int>(nz - 1)));

                bool blocked = false;
                for (int bx = min_bx; bx <= max_bx && !blocked; ++bx) {
                  for (int by = min_by; by <= max_by && !blocked; ++by) {
                    for (int bz = min_bz; bz <= max_bz && !blocked; ++bz) {

                      const unsigned int bb = static_cast<unsigned int>(bx + nx * (by + ny * bz));
                      const unsigned int s2 = bin_offsets(bb);
                      const unsigned int e2 = bin_offsets(bb + 1);

                      for (unsigned int idx2 = s2; idx2 < e2; ++idx2) {
                        const int k = static_cast<int>(idx2);
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

                // auto generator = random_pool.get_state();
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

random_pool.free_state(generator);

          Kokkos::single(
            Kokkos::PerTeam(team_member),
            [&]() {
              out_view_pack.apply([&](auto&... views) {
                total_delta_i.apply([&](auto&... values) {
                  ((views(i) += values), ...);
                });
              });

              out_view_pack.first()(i) += total_velocity_i / Kokkos::fmax(total_friction_i, epsilon);
              // if (total_friction_i > 0)
                // out_view_pack.first()(i) += total_velocity_i / total_friction_i;
            }
          );
        }
      );
    }
  };
} // namespace kocs::pair_finders

#endif // KOCS_PAIR_FINDERS_GABRIEL_HPP
