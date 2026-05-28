#ifndef KOCS_IO_HDF5_READER_HPP
#define KOCS_IO_HDF5_READER_HPP

#include <string>
#include <vector>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>

#include "../utils/utils.hpp"

namespace kocs::io {
  class HDF5_Reader {
    public:
      HDF5_Reader(const std::string& path) : file(path, HighFive::File::AccessMode::ReadOnly) { }

    private:
      HighFive::File file;

    public:
      std::vector<size_t> get_dataset_dimensions(const std::string& dataset_name) {
        HighFive::DataSet dataset = file.getDataSet(dataset_name);
        return dataset.getSpace().getDimensions();
      }

      template<typename T>
      void read_dataset(const std::string& dataset_name, View<T> out_view) {
        auto out_view_host = Kokkos::create_mirror_view(out_view);

        HighFive::DataSet dataset = file.getDataSet(dataset_name);
        dataset.read(out_view_host);

        if (out_view.data() != out_view_host.data()) {
          Kokkos::deep_copy(out_view, out_view_host);
        }
      }

      template<typename T, typename HostView>
      void read_dataset(const std::string& dataset_name, View<T> out_view, HostView out_view_host) {
        static_assert(std::is_same_v<HostView, typename View<T>::host_mirror_type>,
          "out_view_host must be the host mirror of out_view");

        HighFive::DataSet dataset = file.getDataSet(dataset_name);
        dataset.read(out_view_host);

        if (out_view.data() != out_view_host.data()) {
          Kokkos::deep_copy(out_view, out_view_host);
        }
      }
  };
} // namespace kocs::io

#endif // KOCS_IO_HDF5_READER_HPP
