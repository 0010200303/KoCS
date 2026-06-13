#ifndef KOCS_TYPES_VAR_HPP
#define KOCS_TYPES_VAR_HPP

#include <Kokkos_Core.hpp>

namespace kocs {
  template<typename T>
  struct Var : public Kokkos::DualView<T> {
    Var(const std::string& label)
      : Kokkos::DualView<T>(label)
      , device_modified_flag("Var:device_modified_flag") { }

    Kokkos::View<bool> device_modified_flag;

    KOKKOS_INLINE_FUNCTION
    T& operator()() const {
      KOKKOS_IF_ON_DEVICE((
        device_modified_flag() = true;
        return this->view_device()();
      ))
      KOKKOS_IF_ON_HOST((
        const_cast<Var*>(this)->modify_host();
        return this->view_host()();
      ))
    }

    KOKKOS_INLINE_FUNCTION
    const T& read() const {
      KOKKOS_IF_ON_DEVICE((
        return this->view_device()();
      ))
      KOKKOS_IF_ON_HOST((
        return this->view_host()();
      ))
    }

    // same as normal operator, just here for completeness
    KOKKOS_INLINE_FUNCTION
    T& write() const {
      return (*this)();
    }

    // 
    inline void sync_host() {
      auto flag_host = Kokkos::create_mirror_view(device_modified_flag);
      Kokkos::deep_copy(flag_host, device_modified_flag);

      if (flag_host() == true) {
        this->modify_device();
        Kokkos::deep_copy(device_modified_flag, false);
      }
      Kokkos::DualView<T*>::sync_host();
    }

    inline void uncontioninal_sync_device() {
      this->modify_host();
      Kokkos::DualView<T>::sync_device();
    }

    inline void unconditional_sync_host() {
      this->modify_device();
      Kokkos::deep_copy(device_modified_flag, false);
      Kokkos::DualView<T>::sync_host();
    }

    // will do nothing if neither view was modified
    // will abort if both views were modified
    // eitherwise will sync correctly based on which view was modified
    inline void auto_sync() {
      auto flag_host = Kokkos::create_mirror_view(device_modified_flag);
      Kokkos::deep_copy(flag_host, device_modified_flag);
      if (flag_host() == true)
        this->modify_device();
      
      this->sync_host();
      this->sync_device();
    }
  };
} // namespace kocs

// Partial specialization of Kokkos::deep_copy() for kocs::Var objects.
namespace Kokkos {
  template <class DT, class ST>
  void deep_copy(kocs::Var<DT>& dst, const kocs::Var<ST>& src) {
    if (src.need_sync_device()) {
      deep_copy(dst.view_host(), src.view_host());
      dst.modify_host();
    } else {
      deep_copy(dst.view_device(), src.view_device());
      dst.modify_device();
    }
  }

  template <class ExecutionSpace, class DT, class ST>
  void deep_copy(const ExecutionSpace& exec, kocs::Var<DT>& dst, const kocs::Var<ST>& src) {
    if (src.need_sync_device()) {
      deep_copy(exec, dst.view_host(), src.view_host());
      dst.modify_host();
    } else {
      deep_copy(exec, dst.view_device(), src.view_device());
      dst.modify_device();
    }
  }
}  // namespace Kokkos

#endif // KOCS_TYPES_VAR_HPP
