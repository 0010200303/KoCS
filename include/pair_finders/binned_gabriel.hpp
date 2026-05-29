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

    static inline constexpr VectorI get_bin_extents(const Vector& min_, const Vector& max_, const Scalar inv_bin_size) {
      VectorI result;
      for (int d = 0; d < dimensions; ++d)
        result[d] = static_cast<int>(Kokkos::ceil((max_[d] - min_[d]) * inv_bin_size));
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
      const Scalar cutoff_distance_,
      const Vector& min_ = Vector(-20.0f),
      const Vector& max_ = Vector( 20.0f),
      const Scalar gabriel_coefficient = Scalar(0.8),
      const Scalar bin_size_scale = Scalar(1))
      : agent_count(agent_count_)
      , cutoff_distance(cutoff_distance_)
      , cutoff_distance_squared(cutoff_distance_ * cutoff_distance_)
      , _min(min_)
      , _max(max_)
      , bin_size(bin_size_scale * cutoff_distance)
      , inv_bin_size(Scalar(1) / bin_size)
      , bin_extents(get_bin_extents(min_, max_, inv_bin_size))
      , n_bins(get_bin_count(bin_extents))

      , gabriel_radius_factor(gabriel_coefficient * gabriel_coefficient * Scalar(0.25))
      , search_radius(static_cast<int>(Kokkos::ceil(cutoff_distance_ * inv_bin_size)))
      , task_count(get_task_count(2 * search_radius + 1))

      , particle_bins("gabriel_particle_bins", agent_count_)
      , sorter(particle_bins, 0, agent_count_, BinOp{n_bins, 0, n_bins}, true) { }
    
    static const constexpr Scalar epsilon = Scalar(1e-6);

    unsigned int agent_count;
    const Scalar cutoff_distance;
    const Scalar cutoff_distance_squared;

    const Vector _min;
    const Vector _max;
    const Scalar bin_size;
    const Scalar inv_bin_size;

    const VectorI bin_extents;
    const int n_bins;

    const Scalar gabriel_radius_factor;
    const int search_radius;
    const int task_count;

    View<int> particle_bins;
    Kokkos::BinSort<View<int>, BinOp> sorter;

    int step_count = 0;
    int rebuild_every_n = 0;

    KOKKOS_INLINE_FUNCTION
    VectorI get_bin_coord_from_position(const Vector& position) const {
      VectorI result;
      for (int d = 0; d < dimensions; ++d)
        result[d] = Kokkos::max(0, Kokkos::min(static_cast<int>(
          Kokkos::floor((position[d] - _min[d]) * inv_bin_size)), bin_extents[d] - 1)
        );
      return result;
    }

    KOKKOS_INLINE_FUNCTION
    int flatten_bin_index(const VectorI& coords) const {
      if constexpr (dimensions == 1) {
        return coords[0];
      }
      else if constexpr (dimensions == 2) {
        return coords[0] + coords[1] * bin_extents[0];
      }
      else if constexpr (dimensions == 3) {
        return coords[0] + coords[1] * bin_extents[0] + coords[2] * (bin_extents[0] * bin_extents[1]);
      }
      else if constexpr (dimensions == 4) {
        return coords[0] + coords[1] * bin_extents[0] + coords[2] * (bin_extents[0] * bin_extents[1]) +
          coords[3] * (bin_extents[0] * bin_extents[1] * bin_extents[2]);
      }
      else {
        int idx = 0;
        int stride = 1;
        for (int d = 0; d < dimensions; ++d) {
          idx += coords[d] * stride;
          stride *= bin_extents[d];
        }
        return idx;
      }
    }

    KOKKOS_INLINE_FUNCTION
    int get_task_count(const int side) const {
      if constexpr (dimensions == 1) {
        return side;
      }
      else if constexpr (dimensions == 2) {
        return side * side;
      }
      else if constexpr (dimensions == 3) {
        return side * side * side;
      }
      else if constexpr (dimensions == 4) {
        return side * side * side * side;
      }
      else {
        int n = 1;
        for (int d = 0; d < dimensions; ++d)
          n *= side;
        return n;
      }
    }

    KOKKOS_INLINE_FUNCTION
    VectorI linear_index_to_offset(const int task_idx, const int side, const int radius_bins) const {
      VectorI result;
      if constexpr (dimensions == 1) {
        result[0] = task_idx - radius_bins;
      }
      else if constexpr (dimensions == 2) {
        result[0] = (task_idx % side) - radius_bins;
        result[1] = (task_idx / side) - radius_bins;
      }
      else if constexpr (dimensions == 3) {
        const int temp = task_idx / side;
        result[0] = (task_idx % side) - radius_bins;
        result[1] = (temp % side) - radius_bins;
        result[2] = (temp / side) - radius_bins;
      }
      else if constexpr (dimensions == 4) {
        const int temp1 = task_idx / side;
        result[0] = (task_idx % side) - radius_bins;
        result[1] = (temp1 % side) - radius_bins;

        const int temp2 = temp1 / side;
        result[2] = (temp2 % side) - radius_bins;
        result[3] = (temp2 / side) - radius_bins;
      }
      else {
        int temp = task_idx;
        for (int d = 0; d < dimensions; ++d) {
          result[d] = (temp % side) - radius_bins;
          temp /= side;
        }
      }
      return result;
    }

    KOKKOS_INLINE_FUNCTION
    bool is_bin_outside_extents(const VectorI& bin) const {
      if constexpr (dimensions == 1) {
        return bin[0] < 0 || bin[0] >= bin_extents[0];
      }
      else if constexpr (dimensions == 2) {
        return bin[0] < 0 || bin[0] >= bin_extents[0] ||
               bin[1] < 0 || bin[1] >= bin_extents[1];
      }
      else if constexpr (dimensions == 3) {
        return bin[0] < 0 || bin[0] >= bin_extents[0] ||
               bin[1] < 0 || bin[1] >= bin_extents[1] ||
               bin[2] < 0 || bin[2] >= bin_extents[2];
      }
      else if constexpr (dimensions == 4) {
        return bin[0] < 0 || bin[0] >= bin_extents[0] ||
               bin[1] < 0 || bin[1] >= bin_extents[1] ||
               bin[2] < 0 || bin[2] >= bin_extents[2] ||
               bin[3] < 0 || bin[3] >= bin_extents[3];
      }
      else {
        for (int d = 0; d < dimensions; ++d) {
          if (bin[d] < 0 || bin[d] >= bin_extents[d])
            return true;
        }
        return false;
      }
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

      // rebuild grid
      if (is_full_step == true) {
        if (step_count == 0 || step_count >= rebuild_every_n) {
          build(in_view_pack, out_view_pack, old_velocities);
          step_count = 0;
        }
        step_count++;
      }

      Kokkos::parallel_for(
        "naive_gabriel_apply_force",
        Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO()),
        KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team_member) {
          const int i = team_member.league_rank();
          const auto& position_i = input_positions(i);
          VectorI bin_coords = get_bin_coord_from_position(position_i);

          const int side = 2 * search_radius + 1;

          // setup data for accumulation
          auto total_delta_i = detail::make_accumulator_pack(out_view_pack);
          Scalar total_friction_i = 0.0;
          typename PositionsView::value_type total_velocity_i{0.0};

          Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, task_count),
            [&](const int task_idx, auto& local_delta, auto& local_friction, auto& local_velocity) {
              // map task_idx -> per-dimension offsets in [-search_radius, search_radius]
              const VectorI ni = bin_coords + linear_index_to_offset(task_idx, side, search_radius);
              if (is_bin_outside_extents(ni) == true)
                return;

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
                const auto radius_squared = distance_squared * gabriel_radius_factor;
                const Scalar radius = Kokkos::sqrt(radius_squared);

                // calculate exact overlapping bounds
                VectorI min_bin;
                VectorI span;
                int task_count_2 = 1;

                for (int d = 0; d < dimensions; ++d) {
                  min_bin[d] = Kokkos::max(0, Kokkos::min(
                    static_cast<int>(Kokkos::floor((midpoint[d] - radius - _min[d]) * inv_bin_size)),
                    bin_extents[d] - 1
                  ));
                  const int max_b = Kokkos::max(0, Kokkos::min(
                    static_cast<int>(Kokkos::floor((midpoint[d] + radius - _min[d]) * inv_bin_size)),
                    bin_extents[d] - 1
                  ));
                  span[d] = max_b - min_bin[d] + 1;
                  task_count_2 *= span[d];
                }

                bool blocked = false;
                for (int task_2 = 0; task_2 < task_count_2 && blocked == false; ++task_2) {
                  int b_2 = 0;
                  if constexpr (dimensions == 1) {
                    b_2 = min_bin[0] + task_2;
                  }
                  else if constexpr (dimensions == 2) {
                    const int b0 = min_bin[0] + (task_2 % span[0]);
                    const int b1 = min_bin[1] + (task_2 / span[0]);
                    b_2 = b0 + b1 * bin_extents[0];
                  }
                  else if constexpr (dimensions == 3) {
                    const int temp = task_2 / span[0];
                    const int b0 = min_bin[0] + (task_2 % span[0]);
                    const int b1 = min_bin[1] + (temp % span[1]);
                    const int b2 = min_bin[2] + (temp / span[1]);
                    b_2 = b0 + b1 * bin_extents[0] + b2 * (bin_extents[0] * bin_extents[1]);
                  }
                  else if constexpr (dimensions == 4) {
                    const int temp1 = task_2 / span[0];
                    const int b0 = min_bin[0] + (task_2 % span[0]);
                    const int b1 = min_bin[1] + (temp1 % span[1]);
                    const int temp2 = temp1 / span[1];
                    const int b2 = min_bin[2] + (temp2 % span[2]);
                    const int b3 = min_bin[3] + (temp2 / span[2]);
                    b_2 = b0 + b1 * bin_extents[0] + b2 * (bin_extents[0] * bin_extents[1]) + b3 * (bin_extents[0] * bin_extents[1] * bin_extents[2]);
                  }
                  else {
                    int temp = task_2;
                    int stride = 1;
                    for (int d = 0; d < dimensions; ++d) {
                      b_2 += (min_bin[d] + (temp % span[d])) * stride;
                      temp /= span[d];
                      stride *= bin_extents[d];
                    }
                  }

                  const unsigned int start_2 = bin_offsets(b_2);
                  const unsigned int end_2 = bin_offsets(b_2 + 1);

                  for (unsigned int idx_2 = start_2; idx_2 < end_2; ++idx_2) {
                    const int k = static_cast<int>(permutation(idx_2));
                    if (k == i || k == j)
                      continue;
                    
                    if (input_positions(k).distance_to_squared(midpoint) < radius_squared) {
                      blocked = true;
                      break;
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
