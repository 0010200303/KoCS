#ifndef KOCS_TYPES_DEVICE_VAR_HPP
#define KOCS_TYPES_DEVICE_VAR_HPP

#include <Kokkos_Core.hpp>

namespace kocs {
  // only direct assignments are supported on the host
  // modifying fiels on host like "vector.x = 13" will not get copied to device!
  template<typename T>
  struct DeviceVar {
    DeviceVar() : device_view("DeviceVar (default constructed)") {
      Kokkos::deep_copy(device_view, T());
    }
    
    DeviceVar(const std::string& label) : device_view(label) {
      Kokkos::deep_copy(device_view, T());
    }

    DeviceVar(const std::string& label, const T& value) : device_view(label) {
      Kokkos::deep_copy(device_view, value);
    }

    DeviceVar(const T& value) : device_view("DeviceVar (directly assigned)") {
      Kokkos::deep_copy(device_view, value);
    }

    Kokkos::View<T> device_view;

    KOKKOS_INLINE_FUNCTION
    DeviceVar(const DeviceVar& other) : device_view(other.device_view) { }

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

    KOKKOS_INLINE_FUNCTION
    T* data() {
      return &device_view();
    }

    KOKKOS_INLINE_FUNCTION
    T* data() const {
      return &device_view();
    }
  };
} // namespace kocs

#endif // KOCS_TYPES_DEVICE_VAR_HPP
