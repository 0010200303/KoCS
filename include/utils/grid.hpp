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
        const Scalar bin_size_)
        : data_view(data_view_)
        , agent_count(agent_count_)
        , min_bounds(min_bounds_)
        , max_bounds(max_bounds_)
        , bin_size(bin_size_)
        , bin_extents(get_bin_extents())
        , bin_count(get_bin_count())
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

    private:
      const View<Vector> data_view;
      const unsigned int agent_count;

      const Vector min_bounds;
      const Vector max_bounds;
      const Scalar bin_size;

      const VectorI bin_extents;
      const int bin_count;

      View<int> particle_bins;
      BinSort sorter;
      BinSort::offset_type permute_vector;
      BinSort::offset_type bin_offsets;

    public:
      KOKKOS_INLINE_FUNCTION
      constexpr VectorI get_bin_extents() {
        VectorI result;
        for (int d = 0; d < dimensions; ++d)
          result[d] = static_cast<int>(Kokkos::ceil((max_bounds[d] - min_bounds[d]) / bin_size));
        return result;
      }

      KOKKOS_INLINE_FUNCTION
      constexpr int get_bin_count() {
        int result = 1;
        for (int d = 0; d < dimensions; ++d)
          result *= bin_extents[d];
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
      VectorI get_bin_coords_from_point(const Vector& point) const {
        VectorI result;
        for (int d = 0; d < dimensions; ++d)
          result[d] = Kokkos::max(0, Kokkos::min(
            static_cast<int>((point[d] - min_bounds[d]) / bin_size), bin_extents[d] - 1
          ));
        return result;
      }

      KOKKOS_INLINE_FUNCTION
      int get_bin_index_from_point(const Vector& point) const {
        return flatten_bin_index(get_bin_coords_from_point(point));
      }

      KOKKOS_INLINE_FUNCTION
      int get_task_count(const int side) const {
        int n = 1;
        for (int d = 0; d < dimensions; ++d)
          n *= side;
        return n;
      }

      KOKKOS_INLINE_FUNCTION
      VectorI liner_index_to_offset(const int task_idx, const int side, const int radius_bins) const {
        VectorI result;
        int tmp = task_idx;
        for (int d = 0; d < dimensions; ++d) {
          result[d] = (tmp % side) - radius_bins;
          tmp /= side;
        }
        return result;
      }
    
      void rebuild() {
        Kokkos::parallel_for(
          std::string("Grid") + std::to_string(dimensions) + "rebuild",
          agent_count,
          KOKKOS_CLASS_LAMBDA(const unsigned int i) {
            VectorI bin_coords = get_bin_coords_from_point(data_view(i));
            particle_bins(i) = flatten_bin_index(bin_coords);
          }
        );

        sorter = BinSort(particle_bins, 0, agent_count, BinOp{bin_count, 0, bin_count});
        sorter.create_permute_vector();
        permute_vector = sorter.get_permute_vector();
        bin_offsets = sorter.get_bin_offsets();
      }

      KOKKOS_INLINE_FUNCTION
      int get_nearest_point_index(const Vector& point, const int search_radius = 1) const {
        VectorI center_coords = get_bin_coords_from_point(point);
        const int side = 2 * search_radius + 1;
        const int task_count = get_task_count(side);

        int nearest_idx = -1;
        Scalar min_distance_squared = Kokkos::Experimental::finite_max_v<Scalar>;
        for (int task_idx = 0; task_idx < task_count; ++task_idx) {
          VectorI ni = center_coords + liner_index_to_offset(task_idx, side, search_radius);

          // skip bins outside extents
          bool out_of_bounds = false;
          for (int d = 0; d < dimensions; ++d) {
            if (ni[d] < 0 || ni[d] >= bin_extents[d]) {
              out_of_bounds = true;
              break;
            }
          }
          if (out_of_bounds == true)
            continue;
          
          const int b = flatten_bin_index(ni);
          const unsigned int start = bin_offsets(b);
          const unsigned int end = bin_offsets(b + 1);

          for (int idx = start; idx < end; ++idx) {
            const int j = static_cast<int>(permute_vector(idx));

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
