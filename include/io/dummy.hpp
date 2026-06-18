#ifndef KOCS_IO_DUMMY_HPP
#define KOCS_IO_DUMMY_HPP

#include <Kokkos_Core.hpp>

#include "../types/view.hpp"
#include "../types/link.hpp"

namespace kocs::io {
  template<typename SimulationConfig>
  class Dummy {
    public:
      struct Settings { };

      Dummy(const std::string& path, const Settings& settings) { }

      template<typename T0, typename... Views>
      void write(const double time, const unsigned int step, View<T0>& first_view, View<Ts>&... rest_views) { }

      template<typename... Ts>
      void write_static(View<Ts>&... static_views) { }
  };
} // namespace kocs::io

#endif // KOCS_IO_DUMMY_HPP
