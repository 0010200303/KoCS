#ifndef KOCS_TYPES_DEVICE_VAR_HPP
#define KOCS_TYPES_DEVICE_VAR_HPP

#include <Kokkos_Core.hpp>

namespace kocs {
  template<typename T>
  struct DeviceVar {
    DeviceVar(const std::string& label) : device_view(label) { }

    Kokkos::View<T> device_view;

    KOKKOS_INLINE_FUNCTION
    DeviceVar& operator=(const T& value) {
      KOKKOS_IF_ON_DEVICE((
        device_view() = value;
      ))
      KOKKOS_IF_ON_HOST((
        Kokkos::deep_copy(device_view, value);
      ))
      return *this;
    }

    KOKKOS_INLINE_FUNCTION
    operator T() const {
      KOKKOS_IF_ON_DEVICE((
        return device_view();
      ))
      KOKKOS_IF_ON_HOST((
        T host_value;
        Kokkos::deep_copy(host_value, device_view);
        return host_value;
      ))
    }
  };
} // namespace kocs

#endif // KOCS_TYPES_DEVICE_VAR_HPP
