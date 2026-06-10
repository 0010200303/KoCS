#ifndef KOCS_TYPES_VIEW_HPP
#define KOCS_TYPES_VIEW_HPP

#include <cstdint>
#include <string>

#include <Kokkos_Core.hpp>

namespace kocs {
  template<typename T>
  struct View {
    public:
      View(const std::string& label, const unsigned int count)
        : device_view(label, count) { }

    private:
      Kokkos::View<T*> device_view;
      Kokkos::View<T*>::host_mirror_type host_view;

    public:
      KOKKOS_INLINE_FUNCTION
      T& operator()(int i) const {
        KOKKOS_IF_ON_DEVICE((
          return device_view(i);
        ))

        KOKKOS_IF_ON_HOST((
          return host_view(i);
        ))
      }

      inline void sync_host_to_device() {
        Kokkos::deep_copy(device_view, host_view);
      }

      inline void sync_device_to_host() {
        if (host_view.extent(0) != device_view.extent(0))
          host_view = Kokkos::create_mirror_view(device_view);

        Kokkos::deep_copy(host_view, device_view);
      }

      KOKKOS_INLINE_FUNCTION
      unsigned int get_size() const {
        return device_view.extent(0);
      }

      inline void resize(const unsigned int value) {
        if (device_view.extent(0) == value)
          return;

        Kokkos::resize(device_view, value);

        sync_device_to_host();
      }
  };
} // namespace kocs

#endif // KOCS_TYPES_VIEW_HPP
