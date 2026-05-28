#ifndef KOCS_IO_VIEW_HPP
#define KOCS_IO_VIEW_HPP

#include <Kokkos_Core.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5DataType.hpp>

namespace HighFive {

namespace details {

// Specialize inspector to natively map Kokkos::View as an N-dimensional HighFive array.
template <typename DataType, typename... Properties>
struct inspector<Kokkos::View<DataType, Properties...>> {
    using type = Kokkos::View<DataType, Properties...>;
    
    static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename type::memory_space>::accessible, 
                  "HighFive only operate on Kokkos Views accessible from the host memory space.");

    using value_type = typename type::value_type; // Inner scalar/struct type
    using base_type = typename inspector<value_type>::base_type;
    using hdf5_type = typename inspector<value_type>::hdf5_type;

    static constexpr size_t ndim = inspector<value_type>::ndim + type::rank;
    static constexpr size_t min_ndim = inspector<value_type>::min_ndim + type::rank;
    static constexpr size_t max_ndim = inspector<value_type>::max_ndim + type::rank;
    
    // Contiguous Views are trivially nestable
    static constexpr bool is_trivially_nestable = inspector<value_type>::is_trivially_nestable;
    static constexpr bool is_trivially_copyable = inspector<value_type>::is_trivially_copyable;

    static size_t getRank(const type& val) {
        if (val.size() == 0) return type::rank + inspector<value_type>::ndim;
        return type::rank + inspector<value_type>::getRank(val.data()[0]);
    }

    static std::vector<size_t> getDimensions(const type& val) {
        std::vector<size_t> res;
        for (size_t i = 0; i < type::rank; ++i) {
            res.push_back(val.extent(i));
        }
        if (val.size() > 0) {
            auto inner_dim = inspector<value_type>::getDimensions(val.data()[0]);
            res.insert(res.end(), inner_dim.begin(), inner_dim.end());
        } else {
            std::vector<size_t> empty(inspector<value_type>::ndim, 0);
            res.insert(res.end(), empty.begin(), empty.end());
        }
        return res;
    }

    static void prepare(type& val, const std::vector<size_t>& next_dims) {
        // HighFive calls this on read. Assuming Views are pre-allocated by the user.
    }

    static hdf5_type* data(type& val) { 
        return inspector<value_type>::data(val.data()[0]); 
    }

    static const hdf5_type* data(const type& val) { 
        return inspector<value_type>::data(val.data()[0]); 
    }

    static void serialize(const type& val, const std::vector<size_t>& next_dims, hdf5_type* m) {}
    static void unserialize(const hdf5_type* vec_align, const std::vector<size_t>& next_dims, type& val) {}
};

} // namespace details
} // namespace HighFive

#endif // KOCS_IO_VIEW_HPP
