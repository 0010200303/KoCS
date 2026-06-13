#ifndef KOCS_TYPES_VIEW_HPP
#define KOCS_TYPES_VIEW_HPP

#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace kocs {
  template<typename T>
  struct View : public Kokkos::DualView<T*> {
    View(const std::string& label, const unsigned int count) : Kokkos::DualView<T*>(label, count) { }

    KOKKOS_INLINE_FUNCTION
    T& operator()(int i) const {
      KOKKOS_IF_ON_DEVICE((
        return this->view_device()(i);
      ))

      KOKKOS_IF_ON_HOST((
        return this->view_host()(i);
      ))
    }

    // hide base class methods
    inline void resize(const int value) {
      Kokkos::DualView<T*>::resize(value);
      this->sync_host();
    }

    // unconditional sync
    inline void sync_host_to_device() {
      this->modify_host();
      this->sync_device();
    }

    // unconditional sync
    inline void sync_device_to_host() {
      this->modify_device();
      this->sync_host();
    }
  };
} // namespace kocs

#endif // KOCS_TYPES_VIEW_HPP
