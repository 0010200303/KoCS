#ifndef KOCS_IO_WRITER_HPP
#define KOCS_IO_WRITER_HPP

#include <type_traits>
#include <tuple>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>

#include <Kokkos_Core.hpp>
#include <highfive/highfive.hpp>

#include "../utils.hpp"

namespace kocs::writers {
  template<typename SimulationConfig>
  class HDF5_Writer {
    public:
      EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

      HDF5_Writer(
        const std::string& path,
        const unsigned int agent_count_,
        const std::size_t flush_threshold = 65536)
        : agent_count(agent_count_)
        , buffer_threshold(flush_threshold) {
        std::filesystem::path _path(path);
        std::filesystem::create_directories(_path.parent_path());

        filename = _path.filename().string();
        h5_file = std::make_unique<HighFive::File>(HighFive::File(path + ".h5", HighFive::File::AccessMode::Truncate));
        init_xmf(path);
      }

      ~HDF5_Writer() {
        finalize_xmf();
      }

      unsigned int agent_count;

    private:
      const std::size_t buffer_threshold;

      std::string filename;

      std::unique_ptr<HighFive::File> h5_file;
      std::ofstream xmf_file;
      std::string xmf_buffer;

      template<typename T, typename = void>
      struct has_to_array : std::false_type { };

      template<typename T>
      struct has_to_array<T, std::void_t<decltype(std::declval<const T&>().to_array())>> : std::true_type { };

      template<typename T, typename = void>
      struct has_get_dimensions : std::false_type { };

      template<typename T>
      struct has_get_dimensions<T, std::void_t<decltype(std::declval<const T&>().get_dimensions())>> : std::true_type { };

      template<typename T>
      static auto to_storage_value(const T& value) {
        if constexpr (has_to_array<T>::value)
          return value.to_array();
        else
          return value;
      }

      template<typename ViewType>
      auto view_to_vector(const ViewType& view) {
        using T = typename ViewType::value_type;
        using StorageT = decltype(to_storage_value(std::declval<const T&>()));

        auto sub_view = Kokkos::subview(view, std::make_pair(0u, agent_count));
        auto host_view = Kokkos::create_mirror_view(sub_view);
        Kokkos::deep_copy(host_view, sub_view);

        std::vector<StorageT> out(agent_count);
        for (std::size_t i = 0; i < agent_count; ++i)
          out[i] = to_storage_value(host_view(i));

        return out;
      }

      template<typename View>
      void write_single(HighFive::Group& group, const View& view) {
        auto vec = view_to_vector(view);
        
        group.createDataSet(view.label(), vec);
      }

      void init_xmf(const std::string& path) {
        xmf_file.open(path + ".xmf", std::ios::out | std::ios::trunc);
        if (xmf_file.is_open() == false)
          return;

        xmf_buffer += "<?xml version=\"1.0\"?>\n<Xdmf Version=\"3.0\">\n\t<Domain>\n\
\t\t<Grid Name=\"Agents\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
      }

      void finalize_xmf() {
        if (xmf_file.is_open() == false)
          return;
        
        xmf_buffer += "\t\t</Grid>\n\t</Domain>\n</Xdmf>\n";
        xmf_file.write(xmf_buffer.data(), xmf_buffer.size());
        xmf_file.close();
      }

      void write_xmf_grid_start(const unsigned int step) {
        xmf_buffer += "\t\t\t<Grid Name=\"t";
        xmf_buffer += std::to_string(step);
        xmf_buffer += "\" GridType=\"Uniform\">\n\t\t\t\t<Topology TopologyType=\"Polyvertex\" NumberOfElements=\"";
        xmf_buffer += std::to_string(agent_count);
        xmf_buffer += "\"/>\n\t\t\t\t<Geometry GeometryType=\"XYZ\">\n\t\t\t\t\t<DataItem Format=\"HDF\" Dimensions=\"";
        xmf_buffer += std::to_string(agent_count);
        xmf_buffer += " ";
        xmf_buffer += std::to_string(3);
        xmf_buffer += "\">\n\t\t\t\t\t\t";
        xmf_buffer += filename;
        xmf_buffer += ".h5:t";
        xmf_buffer += std::to_string(step);
        xmf_buffer += "/";
        xmf_buffer += "positions";
        xmf_buffer += "\n\t\t\t\t\t</DataItem>\n\t\t\t\t</Geometry>\n";
      }

      template<typename Tuple, std::size_t... Is>
      void write_xmf_grid_attributes_from_tuple(const unsigned int step, const Tuple& tpl, std::index_sequence<Is...>) {
        (void)std::initializer_list<int>{ (Is == 0 ? 0 : (write_xmf_grid_attribute(step, std::get<Is>(tpl)), 0))... };
      }

      template<typename View>
      void write_xmf_grid_attribute(const unsigned int step, const View& view) {
        // TODO: optimize this

        // extract dimensions
        std::vector<int> dimension_map;
        for (int i = 0; i < view.rank(); ++i)
          dimension_map.push_back(view.extent(i));

        using T = typename View::value_type;
        if constexpr (has_get_dimensions<T>::value)
          dimension_map.push_back(T{}.get_dimensions());

        // overwrite first dimension with agent_count
        dimension_map[0] = agent_count;

        std::string dimensions = "";
        for (const auto& d : dimension_map) {
          dimensions += std::to_string(d);
          dimensions += " ";
        }

        // get attribute type based on dimensions
        std::string attribute_type = "Matrix";
        if (dimension_map.size() == 1)
          attribute_type = "Scalar";
        else if (dimension_map.size() == 2 && dimension_map[1] == 3)
          attribute_type = "Vector";
        else if (dimension_map.size() == 2 && dimension_map[1] == 9)
          attribute_type = "Tensor";

        xmf_buffer += "\t\t\t\t<Attribute Name=\"";
        xmf_buffer += view.label();
        xmf_buffer += "\" AttributeType=\"";
        xmf_buffer += attribute_type;
        xmf_buffer += "\" Center=\"Node\">\n";
        xmf_buffer += "\t\t\t\t\t<DataItem Format=\"HDF\" Dimensions=\"";
        xmf_buffer += dimensions;
        xmf_buffer += "\">\n\t\t\t\t\t\t";
        xmf_buffer += filename;
        xmf_buffer += ".h5:t";
        xmf_buffer += std::to_string(step);
        xmf_buffer += "/";
        xmf_buffer += view.label();
        xmf_buffer += "\n\t\t\t\t\t</DataItem>\n\t\t\t\t</Attribute>\n";
      }

      void write_xmf_grid_end() {
        xmf_buffer += "\t\t\t</Grid>\n";

        if (xmf_buffer.size() > buffer_threshold) {
          xmf_file.write(xmf_buffer.data(), xmf_buffer.size());
          xmf_buffer.clear();
        }
      }

    public:
      // always expects the positions view to be passed first
      template<typename... Views>
      void write(unsigned int step, const Views&... views) {
        HighFive::Group group = h5_file->createGroup(std::string("t") + std::to_string(step));
        (write_single(group, views), ...);

        if (xmf_file.is_open() == false)
          return;

        auto all_tuple = std::tuple<const Views&...>(views...);
        const auto& first_view = std::get<0>(all_tuple);

        write_xmf_grid_start(step);
        write_xmf_grid_attributes_from_tuple(step, all_tuple, std::make_index_sequence<sizeof...(Views)>{});
        write_xmf_grid_end();
      }
  };
} // namespace kocs::writers

#endif // KOCS_IO_WRITER_HPP
