#ifndef KOCS_TYPES_VIEW_HPP
#define KOCS_TYPES_VIEW_HPP

#include <string>
#include <source_location>

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace kocs {
  template<typename T>
  struct View : public Kokkos::DualView<T*> {
    View()
      : Kokkos::DualView<T*>("kocs::View default", 0)
      , device_modified_flag("DualView:device_modified_flag")
      , active_count(0) { }

    View(const std::string& label, const unsigned int count)
      : Kokkos::DualView<T*>(label, count)
      , device_modified_flag("DualView:device_modified_flag")
      , active_count(count) { }

    Kokkos::View<bool> device_modified_flag;

    unsigned int active_count;

    inline std::string label() const {
      return this->view_device().label();
    }

    KOKKOS_INLINE_FUNCTION
    int rank() const {
      return 1;
    }

    KOKKOS_INLINE_FUNCTION
    T& operator()(const int i) const {
      KOKKOS_IF_ON_DEVICE((
        device_modified_flag() = true;
        return this->view_device()(i);
      ))
      KOKKOS_IF_ON_HOST((
        const_cast<View*>(this)->modify_host();
        return this->view_host()(i);
      ))
    }

    // does not set modified flags
    KOKKOS_INLINE_FUNCTION
    T& access(const int i) const {
      KOKKOS_IF_ON_DEVICE((
        return this->view_device()(i);
      ))
      KOKKOS_IF_ON_HOST((
        return this->view_host()(i);
      ))
    }

    KOKKOS_INLINE_FUNCTION
    T& read(const int i) const {
      return access(i);
    }

    // same as normal operator, just here for completeness
    KOKKOS_INLINE_FUNCTION
    T& write(const int i) const {
      return (*this)(i);
    }

    // shadow base class method
    inline void sync_host() {
      auto flag_host = Kokkos::create_mirror_view(device_modified_flag);
      Kokkos::deep_copy(flag_host, device_modified_flag);

      if (flag_host() == true) {
        this->modify_device();
        Kokkos::deep_copy(device_modified_flag, false);
      }
      Kokkos::DualView<T*>::sync_host();
    }

    inline void unconditional_sync_device() {
      this->modify_host();
      Kokkos::DualView<T*>::sync_device();
    }

    inline void unconditional_sync_host() {
      this->modify_device();
      Kokkos::deep_copy(device_modified_flag, false);
      Kokkos::DualView<T*>::sync_host();
    }

    // will do nothing if neither view was modified
    // will abort if both views were modified
    // eitherwise will sync correctly based on which view was modified
    inline void auto_sync() {
      bool flag_host;
      Kokkos::deep_copy(flag_host, device_modified_flag);
      if (flag_host == true)
        this->modify_device();

      if (this->need_sync_host())
        this->sync_host();
      else if (this->need_sync_device())
        this->sync_device();
    }

    inline void deep_copy(const View<T>& src) {
      Kokkos::deep_copy(static_cast<Kokkos::DualView<T*>&>(*this), static_cast<const Kokkos::DualView<T*>&>(src));
      Kokkos::deep_copy(device_modified_flag, false);
    }

    inline void deep_copy(const T& src) {
      Kokkos::deep_copy(this->view_device(), src);
      Kokkos::deep_copy(this->view_host(), src);
      Kokkos::deep_copy(device_modified_flag, false);
    }

    // shadow base class method
    inline void resize(const int value) {
      Kokkos::DualView<T*>::resize(value);
      Kokkos::deep_copy(device_modified_flag, false);
      Kokkos::DualView<T*>::sync_host();
    }

    KOKKOS_INLINE_FUNCTION
    unsigned int get_capacity() const {
      return this->extent(0);
    }

    inline  unsigned int get_active_count() const {
      return active_count;
    }

    inline void set_active_count(const unsigned int value) {
      active_count = value;
    }
  };
} // namespace kocs

#endif // KOCS_TYPES_VIEW_HPP
