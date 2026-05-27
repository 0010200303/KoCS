#ifndef KOCS_PAIR_FINDERS_BINNED_GABRIEL_HPP
#define KOCS_PAIR_FINDERS_BINNED_GABRIEL_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Sort.hpp>

#include "../integrators/detail.hpp"
#include "../forces/detail.hpp"

namespace kocs::pair_finders {
  template<typename PositionsView, typename Scalar, int dimensions>
  struct BinnedGabriel {
    using positions_view_type = PositionsView;
    using Vector = VectorN<Scalar, dimensions>;
    using VectorI = VectorN<int, dimensions>;
    using BinOp = Kokkos::BinOp1D<View<int>>;

    static inline constexpr VectorI get_bin_extents(const Vector& min_, const Vector& max_, const Scalar bin_size) {
      VectorI result;
      for (int d = 0; d < dimensions; ++d)
        result[d] = static_cast<int>(Kokkos::ceil((max_[d] - min_[d]) / bin_size));
      return result;
    }

    static inline constexpr int get_bin_count(const VectorI& bin_extents_) {
      int result = 1;
      for (int d = 0; d < dimensions; ++d)
        result *= bin_extents_[d];
      return result;
    }

    BinnedGabriel(
      unsigned int agent_count_,
      const Scalar cutoff_distance,
      const Vector& min_ = Vector(-20.0f),
      const Vector& max_ = Vector( 20.0f),
      const Scalar gabriel_coefficient = Scalar(0.8),
      const Scalar bin_size_scale = Scalar(1))
      : agent_count(agent_count_)
      , cutoff_distance_squared(cutoff_distance * cutoff_distance)
      , _min(min_)
      , _max(max_)
      , bin_size(bin_size_scale * cutoff_distance)
      , bin_extents(get_bin_extents(min_, max_, bin_size))
      , n_bins(get_bin_count(bin_extents))

      , gabriel_coefficient_squared(gabriel_coefficient * gabriel_coefficient)

      , particle_bins("gabriel_particle_bins", agent_count_)
      , sorter(particle_bins, 0, agent_count_, BinOp{n_bins, 0, n_bins}, true) { }
    
    static const constexpr Scalar epsilon = Scalar(1e-6);

    unsigned int agent_count;
    const Scalar cutoff_distance_squared;

    const Vector _min;
    const Vector _max;
    const Scalar bin_size;

    const VectorI bin_extents;
    const int n_bins;

    const Scalar gabriel_coefficient_squared;

    View<int> particle_bins;
    Kokkos::BinSort<View<int>, BinOp> sorter;

    int step_count = 0;
    int rebuild_every_n = 0;

    KOKKOS_INLINE_FUNCTION
    VectorI get_bin_coord_from_position(const Vector& position) const {
      VectorI result;
      for (int d = 0; d < dimensions; ++d)
        result[d] = Kokkos::max(0, Kokkos::min(static_cast<int>(
          Kokkos::floor((position[d] - _min[d]) / bin_size)), bin_extents[d] - 1)
        );
      return result;
    }

    KOKKOS_INLINE_FUNCTION
    int flatten_bin_index(const VectorI& coords) const {
      int idx = 0;
      int stride = 1;
      for (int d = 0; d < dimensions; ++d) {
        idx += coords[d] * stride;
        stride *= bin_extents[d];
      }
      return idx;
    }

    KOKKOS_INLINE_FUNCTION
    int get_n_bin_tasks(const int side) const {
      int n = 1;
      for (int d = 0; d < dimensions; ++d)
        n *= side;
      return n;
    }

    KOKKOS_INLINE_FUNCTION
    VectorI task_index_to_bin_offset(const int task_idx, const int side, const int radius_bins) const {
      VectorI result;
      int tmp = task_idx;
      for (int d = 0; d < dimensions; ++d) {
        result[d] = (tmp % side) - radius_bins;
        tmp /= side;
      }
      return result;
    }



    template<typename... Views>
    void build(
      detail::ViewPack<Views...>& in_view_pack,
      detail::ViewPack<Views...>& out_view_pack,
      PositionsView& old_velocities
    ) {
      const auto& input_positions = in_view_pack.first();

      Kokkos::parallel_for("gabriel_build", agent_count, KOKKOS_CLASS_LAMBDA(const unsigned int i) {
        const auto& position_i = input_positions(i);

        VectorI bin_coords = get_bin_coord_from_position(position_i);
        particle_bins(i) = flatten_bin_index(bin_coords);
      });

      sorter = Kokkos::BinSort<View<int>, BinOp>(particle_bins, 0, agent_count, BinOp{n_bins, 0, n_bins});
      sorter.create_permute_vector();
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
      const auto& permutation = sorter.get_permute_vector();
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
          const int i = team_member.league_rank();
          const auto& position_i = input_positions(i);

          auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar total_friction_i = 0.0;
          typename PositionsView::value_type total_velocity_i{0.0};

          // compute bin coords for particle i
          VectorI bin_coords = get_bin_coord_from_position(position_i);

          // number of bins to search in each direction based on cutoff
          const int radius_bins = static_cast<int>(Kokkos::ceil(cutoff_distance / bin_size));
          const int side = 2 * radius_bins + 1;
          const int n_bin_tasks = get_n_bin_tasks(side);

          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, n_bin_tasks),
            [&](const int task_idx, auto& local_delta, auto& local_friction, auto& local_velocity) {
              // map task_idx -> per-dimension offsets in [-radius_bins, radius_bins]
              const VectorI ni = bin_coords + task_index_to_bin_offset(task_idx, side, radius_bins);

              // exit if bin extents are exceeded
              for (int d = 0; d < dimensions; ++d) {
                if (ni[d] < 0 || ni[d] >= bin_extents[d]) {
                  return;
                }
              }

              const int b = flatten_bin_index(ni);
              const unsigned int start = bin_offsets(b);
              const unsigned int end = bin_offsets(b + 1);

              for (unsigned int idx = start; idx < end; ++idx) {
                const int j = static_cast<int>(permutation(idx));
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

                VectorI min_bin;
                VectorI max_bin;
                for (int d = 0; d < dimensions; ++d) {
                  min_bin[d] = Kokkos::max(0, Kokkos::min(
                    static_cast<int>(Kokkos::floor((midpoint[d] - radius - _min[d]) / bin_size)),
                    bin_extents[d] - 1)
                  );
                  max_bin[d] = Kokkos::max(0, Kokkos::min(
                    static_cast<int>(Kokkos::floor((midpoint[d] + radius - _min[d]) / bin_size)),
                    bin_extents[d] - 1)
                  );
                }

                // TODO: this
                bool blocked = false;
                for (int bx = min_bin[0]; bx <= max_bin[0] && !blocked; ++bx) {
                  for (int by = min_bin[1]; by <= max_bin[1] && !blocked; ++by) {
                    for (int bz = min_bin[2]; bz <= max_bin[2] && !blocked; ++bz) {

                      const int bb = flatten_bin_index(VectorI(bx, by, bz));
                      const unsigned int s2 = bin_offsets(bb);
                      const unsigned int e2 = bin_offsets(bb + 1);

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

#endif // KOCS_PAIR_FINDERS_BINNED_GABRIEL_HPP
