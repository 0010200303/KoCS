#ifndef KOCS_IO_HDF5_READER_HPP
#define KOCS_IO_HDF5_READER_HPP

#include <string>

#include <highfive/highfive.hpp>

#include "../types/view.hpp"
#include "view.hpp"

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
        HighFive::DataSet dataset = file.getDataSet(dataset_name);
        dataset.read(const_cast<typename View<T>::t_host&>(out_view.view_host()));
        out_view.unconditional_sync_device();
      }
  };
} // namespace kocs::io

#endif // KOCS_IO_HDF5_READER_HPP
