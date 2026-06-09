#ifndef KOCS_TYPES_VIEW_HPP
#define KOCS_TYPES_VIEW_HPP

#include <cstdint>
#include <string>

#include <Kokkos_Core.hpp>

namespace kocs {
  template<typename T>
  struct View {
    View(const std::string& label, const unsigned int count)
      : device_view(label, count)
      , device_data_ptr(label + "::data_ptr")
    {
      update_device_ptr();

      // TODO: only create when needed
      host_view = Kokkos::create_mirror_view(device_view);
    }

    Kokkos::View<T*> device_view;
    Kokkos::View<T*>::host_mirror_type host_view;

    // using pointer indirection to easily support resizing
    Kokkos::View<uintptr_t> device_data_ptr;

    KOKKOS_INLINE_FUNCTION
    T& operator()(int i) const {
      KOKKOS_IF_ON_DEVICE((
        return reinterpret_cast<T*>(device_data_ptr())[i];
      ))

      KOKKOS_IF_ON_HOST((
        return host_view(i);
      ))
    }

    inline void update_device_ptr() {
      Kokkos::deep_copy(device_data_ptr, reinterpret_cast<uintptr_t>(device_view.data()));
    }

    inline void sync_host_to_device() {
      Kokkos::deep_copy(device_view, host_view);
    }

    inline void sync_device_to_host() {
      Kokkos::deep_copy(host_view, device_view);
    }

    inline unsigned int get_size() const {
      return device_view.extent(0);
    }

    inline void resize(const unsigned int value) {
      Kokkos::resize(device_view, value);
      update_device_ptr();

      host_view = Kokkos::create_mirror_view(device_view);
      Kokkos::deep_copy(host_view, device_view);
    }
  };
} // namespace kocs

#endif // KOCS_TYPES_VIEW_HPP
