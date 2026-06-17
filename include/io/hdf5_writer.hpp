#ifndef KOCS_IO_HDF5_WRITER_HPP
#define KOCS_IO_HDF5_WRITER_HPP

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

#include "../utils/utils.hpp"
#include "../types/view.hpp"
#include "../types/link.hpp"
#include "view.hpp"

namespace kocs::io {
  template<typename SimulationConfig>
  class HDF5_Writer {
    EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

    public:
      struct Settings {
        bool write_xmf = true;
        std::size_t buffer_treshold = 65536;
      };

    public:
      HDF5_Writer(
        const std::string& path,
        const Settings& settings)
        : write_xmf(settings.write_xmf)
        , buffer_threshold(settings.buffer_treshold) {
        std::filesystem::path _path(path);
        std::filesystem::create_directories(_path.parent_path());

        filename = _path.filename().string();
        h5_file = std::make_unique<HighFive::File>(HighFive::File(path + ".h5", HighFive::File::AccessMode::Truncate));

        if (write_xmf == true)
          init_xmf(path);
      }

      ~HDF5_Writer() {
        finalize_xmf();
      }

    private:
      const std::size_t buffer_threshold;

      std::string filename;

      std::unique_ptr<HighFive::File> h5_file;
      std::ofstream xmf_file;
      std::string xmf_buffer;

      std::string xmf_static;

      bool write_xmf;

      template<typename T, typename = void>
      struct has_static_dimensions : std::false_type { };

      template<typename T>
      struct has_static_dimensions<T, std::void_t<decltype(T::dimensions)>> : std::true_type { };

      template<typename T, typename = void>
      struct has_get_dimensions : std::false_type { };

      template<typename T>
      struct has_get_dimensions<T, std::void_t<decltype(std::declval<const T&>().get_dimensions())>> : std::true_type { };

      template<typename T>
      void write_single(HighFive::Group& group, View<T>& view) {
        if (view.get_active_count() == 0)
          return;

        view.sync_host();

        auto sub_host = Kokkos::subview(view.view_host(), Kokkos::make_pair(0u, view.get_active_count()));
        group.createDataSet(view.label(), sub_host);
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

      template<typename T>
      void write_xmf_grid_start(const double time, const unsigned int step, const View<T>& first_view) {
        xmf_buffer += "\t\t\t<Grid Name=\"t";
        xmf_buffer += std::to_string(step);
        xmf_buffer += "\" GridType=\"Uniform\">\n\t\t\t\t<Time Value=\"";
        xmf_buffer += std::to_string(time);
        xmf_buffer += "\"/>\n\t\t\t\t<Topology TopologyType=\"Polyvertex\" NumberOfElements=\"";
        xmf_buffer += std::to_string(first_view.get_active_count());
        xmf_buffer += "\"/>\n\t\t\t\t<Geometry GeometryType=\"";
        
        if constexpr (dimensions == 1)
          xmf_buffer += "X";
        else if constexpr (dimensions == 2)
          xmf_buffer += "XY";
        else
          xmf_buffer += "XYZ";
        
        xmf_buffer += "\">\n\t\t\t\t\t<DataItem Format=\"HDF\" Dimensions=\"";
        xmf_buffer += std::to_string(first_view.get_active_count());
        xmf_buffer += " ";
        xmf_buffer += std::to_string(SimulationConfig::dimensions);
        xmf_buffer += "\">\n\t\t\t\t\t\t";
        xmf_buffer += filename;
        xmf_buffer += ".h5:t";
        xmf_buffer += std::to_string(step);
        xmf_buffer += "/";
        xmf_buffer += first_view.label();
        xmf_buffer += "\n\t\t\t\t\t</DataItem>\n\t\t\t\t</Geometry>\n";
      }

      template<typename T>
      void write_xmf_grid_attribute(std::string& buffer, const std::string& group, const View<T>& view) {
        // TODO: optimize this
        if (view.extent(0) == 0)
          return;

        // extract dimensions
        std::vector<int> dimension_map;
        for (int i = 0; i < view.rank(); ++i)
          dimension_map.push_back(view.extent(i));

        if constexpr (has_static_dimensions<T>::value)
          dimension_map.push_back(T::dimensions);
        else if constexpr (has_get_dimensions<T>::value)
          dimension_map.push_back(T{}.get_dimensions());

        // overwrite first dimension with views active count
        dimension_map[0] = view.get_active_count();

        std::string dimensions = "";
        for (const auto& d : dimension_map) {
          dimensions += std::to_string(d);
          dimensions += " ";
        }

        // get attribute type based on dimensions
        std::string attribute_type = "Matrix";
        if (dimension_map.size() == 1)
          attribute_type = "Scalar";
        else if (dimension_map.size() == 2 && (dimension_map[1] == 2 || dimension_map[1] == 3))
          attribute_type = "Vector";
        else if (dimension_map.size() == 2 && dimension_map[1] == 9)
          attribute_type = "Tensor";

        buffer += "\t\t\t\t<Attribute Name=\"";
        buffer += view.label();
        buffer += "\" AttributeType=\"";
        buffer += attribute_type;
        buffer += "\" Center=\"Node\">\n";
        buffer += "\t\t\t\t\t<DataItem Format=\"HDF\" Dimensions=\"";
        buffer += dimensions;
        buffer += "\">\n\t\t\t\t\t\t";
        buffer += filename;
        buffer += ".h5:";
        buffer += group;
        buffer += "/";
        buffer += view.label();
        buffer += "\n\t\t\t\t\t</DataItem>\n\t\t\t\t</Attribute>\n";
      }

      void write_xmf_grid_end() {
        xmf_buffer += xmf_static;
        xmf_buffer += "\t\t\t</Grid>\n";

        if (xmf_buffer.size() > buffer_threshold) {
          xmf_file.write(xmf_buffer.data(), xmf_buffer.size());
          xmf_buffer.clear();
        }
      }

    public:
      // always expects the position view to be passed first
      template<typename T0, typename... Ts>
      void write(const double time, const unsigned int step, View<T0>& first_view, View<Ts>&... rest_views) {
        HighFive::Group group = h5_file->createGroup(std::string("t") + std::to_string(step));
        write_single(group, first_view);
        (write_single(group, rest_views), ...);

        h5_file->flush();

        if (write_xmf == false || xmf_file.is_open() == false)
          return;

        write_xmf_grid_start(time, step, first_view);
        (write_xmf_grid_attribute(xmf_buffer, "t" + std::to_string(step), rest_views), ...);
        write_xmf_grid_end();
      }

      template<typename... Ts>
      void write_static(View<Ts>&... static_views) {
        if (xmf_static.empty() == false)
          throw std::runtime_error("Static data has already been written. You can only write to static data once.");

        HighFive::Group static_group = h5_file->createGroup("static");
        (write_single(static_group, static_views), ...);

        h5_file->flush();

        if (write_xmf == false || xmf_file.is_open() == false)
          return;

        (write_xmf_grid_attribute(xmf_static, "static", static_views), ...);
      }
  };
} // namespace kocs::io

#endif // KOCS_IO_HDF5_WRITER_HPP
