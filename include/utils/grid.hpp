#ifndef KOCS_UTILS_GRID_HPP
#define KOCS_UTILS_GRID_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_NumericTraits.hpp>

#include "functions.hpp"

namespace kocs::acceleration {
  template<typename Vector>
  class Grid {
    EXTRACT_VECTOR(Vector)
    using VectorI = VectorN<int, dimensions>;
    using BinOp = Kokkos::BinOp1D<View<int>>;
    using BinSort = Kokkos::BinSort<View<int>, BinOp>;

    static Scalar get_bin_size(
      const Vector& min_bounds_,
      const Vector& max_bounds_,
      const unsigned int agent_count_,
      const unsigned int agents_per_bin_ = 1
    ) {
      if (agent_count_ == 0)
        return Scalar(1);

      Scalar volume = 1;
      for (int d = 0; d < dimensions; ++d)
        volume *= (max_bounds_[d] - min_bounds_[d]);

      return static_cast<Scalar>(std::pow(
        (static_cast<Scalar>(agents_per_bin_) * volume) / static_cast<Scalar>(agent_count_), 1.0 / dimensions
      ));
    }

    public:
      Grid(
        const View<Vector> data_view_,
        const unsigned int agent_count_,
        const Vector& min_bounds_,
        const Vector& max_bounds_,
        const Scalar bin_size_
      ) : data_view(data_view_)
        , agent_count(agent_count_)
        , min_bounds(min_bounds_)
        , max_bounds(max_bounds_)
        , bin_size(bin_size_)
        , bin_extents(calc_bin_extents())
        , bin_count(calc_bin_count())
        , particle_bins(std::string("Grid") + std::to_string(dimensions) + "_particle_bins", agent_count_)
        , sorter(particle_bins, 0, agent_count_, BinOp{bin_count, 0, bin_count}) { }
      
      Grid(
        const View<Vector> data_view_,
        const unsigned int agent_count_,
        const utils::Bounds<Vector>& bounds,
        const unsigned int agents_per_bin_ = 1
      ) : Grid(
          data_view_,
          agent_count_,
          bounds.min,
          bounds.max,
          get_bin_size(bounds.min, bounds.max, agent_count_, agents_per_bin_)) { }

      Grid(
        const View<Vector> data_view_,
        const unsigned int agent_count_,
        const unsigned int agents_per_bin_ = 1
      ) : Grid(
          data_view_,
          agent_count_,
          utils::get_bounds(data_view_),
          agents_per_bin_) { }

      Grid(
        const View<Vector> data_view_,
        const unsigned int agents_per_bin_ = 1
      ) : Grid(data_view_, data_view_.extent(0), agents_per_bin_) { }

      Grid() : sorter(particle_bins, 0, 0, BinOp{0, 0, 0}) {}

    public:
      View<Vector> data_view;
      unsigned int agent_count = 0;

      Vector min_bounds;
      Vector max_bounds;
      Scalar bin_size = 0.0;

      VectorI bin_extents;
      int bin_count = 0;

      View<int> particle_bins;
      BinSort sorter;

    public:
      KOKKOS_INLINE_FUNCTION
      BinSort::offset_type get_permute_vector() const {
        return sorter.get_permute_vector();
      }

      KOKKOS_INLINE_FUNCTION
      BinSort::offset_type get_bin_offsets() const {
        return sorter.get_bin_offsets();
      }

      KOKKOS_INLINE_FUNCTION
      Vector get_min_bounds() const {
        return min_bounds;
      }

      KOKKOS_INLINE_FUNCTION
      Vector get_max_bounds() const {
        return max_bounds;
      }

      KOKKOS_INLINE_FUNCTION
      Scalar get_bin_size() const {
        return bin_size;
      }

      KOKKOS_INLINE_FUNCTION
      VectorI get_bin_extents() const {
        return bin_extents;
      }

      KOKKOS_INLINE_FUNCTION
      int get_bin_count() const {
        return bin_count;
      }

    public:
      KOKKOS_INLINE_FUNCTION
      constexpr VectorI calc_bin_extents() {
        VectorI result;
        for (int d = 0; d < dimensions; ++d)
          result[d] = static_cast<int>(Kokkos::ceil((max_bounds[d] - min_bounds[d]) / bin_size));
        return result;
      }

      KOKKOS_INLINE_FUNCTION
      constexpr int calc_bin_count() {
        int result = 1;
        for (int d = 0; d < dimensions; ++d)
          result *= bin_extents[d];
        return result;
      }

      KOKKOS_INLINE_FUNCTION
      VectorI calc_bin_coords_from_point(const Vector& point) const {
        VectorI result;
        for (int d = 0; d < dimensions; ++d)
          result[d] = Kokkos::max(0, Kokkos::min(
            static_cast<int>((point[d] - min_bounds[d]) / bin_size), bin_extents[d] - 1
          ));
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
      int calc_bin_index_from_point(const Vector& point) const {
        return flatten_bin_index(calc_bin_coords_from_point(point));
      }

      KOKKOS_INLINE_FUNCTION
      int calc_task_count(const int side) const {
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
    
      void rebuild() {
        Kokkos::parallel_for(
          std::string("Grid") + std::to_string(dimensions) + "rebuild",
          agent_count,
          KOKKOS_CLASS_LAMBDA(const unsigned int i) {
            VectorI bin_coords = calc_bin_coords_from_point(data_view(i));
            particle_bins(i) = flatten_bin_index(bin_coords);
          }
        );

        sorter = BinSort(particle_bins, 0, agent_count, BinOp{bin_count, 0, bin_count});
        sorter.create_permute_vector();
      }

      KOKKOS_INLINE_FUNCTION
      int get_nearest_point_index(const Vector& point, const int search_radius = 1) const {
        VectorI center_coords = calc_bin_coords_from_point(point);
        const int side = 2 * search_radius + 1;
        const int task_count = calc_task_count(side);

        int nearest_idx = -1;
        Scalar min_distance_squared = Kokkos::Experimental::finite_max_v<Scalar>;
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          VectorI ni = center_coords + linear_index_to_offset(task_idx, side, search_radius);
          if (is_bin_outside_extents(ni) == true)
            continue;
          
          const int b = flatten_bin_index(ni);
          const unsigned int start = sorter.get_bin_offsets()(b);
          const unsigned int end = sorter.get_bin_offsets()(b + 1);

          for (int idx = start; idx < end; ++idx) {
            const int j = static_cast<int>(sorter.get_permute_vector()(idx));

            const Scalar distance_squared = point.distance_to_squared(data_view(j));
            if (distance_squared < min_distance_squared) {
              nearest_idx = j;
              min_distance_squared = distance_squared;
            }
          }
        }
        return nearest_idx;
      }
  };
} // namespace kocs::acceleration

#endif // KOCS_UTILS_GRID_HPP
