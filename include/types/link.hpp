// https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004952

#ifndef KOCS_TYPES_LINK_HPP
#define KOCS_TYPES_LINK_HPP

namespace kocs {
  struct Link {
    static constexpr unsigned int dimensions = 2;

    unsigned int a;
    unsigned int b;
  };
} // namespace kocs



namespace HighFive::details {
  template<>
  struct inspector<kocs::Link> {
    using type = kocs::Link;
    using base_type = unsigned int;
    using hdf5_type = typename inspector<unsigned int>::hdf5_type;

    static constexpr size_t ndim = 1;
    static constexpr size_t min_ndim = 1;
    static constexpr size_t max_ndim = 1;
    static constexpr bool is_trivially_nestable = false;
    static constexpr bool is_trivially_copyable = std::is_trivially_copyable<type>::value;

    static size_t getRank(const type&) { return ndim; }

    static std::vector<size_t> getDimensions(const type&) {
      return {2};
    }

    static void prepare(type&, const std::vector<size_t>&) { }

    static hdf5_type* data(type& value) {
      return &value.a;
    }

    static const hdf5_type* data(const type& value) {
      return &value.a;
    }

    static void serialize(const type& val, const std::vector<size_t>& m_dims, hdf5_type* m) {
      (void)m_dims;
      m[0] = val.a;
      m[1] = val.b;
    }

    static void unserialize(const hdf5_type* vec, const std::vector<size_t>& m_dims, type& val) {
      (void)m_dims;
      val.a = vec[0];
      val.b = vec[1];
    }
  };
} // namespace HighFive::details

#endif // KOCS_TYPES_LINK_HPP
