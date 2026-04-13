#ifndef KOCS_RUNTIME_GUARD_HPP
#define KOCS_RUNTIME_GUARD_HPP

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <mutex>

namespace kocs {
  class RuntimeGuard {
    public:
      RuntimeGuard() { acquire(); }
      ~RuntimeGuard() noexcept { release(); }

      RuntimeGuard(const RuntimeGuard&) = delete;
      RuntimeGuard& operator=(const RuntimeGuard&) = delete;
      RuntimeGuard(RuntimeGuard&&) = delete;
      RuntimeGuard& operator=(RuntimeGuard&&) = delete;

    private:
      static std::mutex& guard_mutex() {
        static std::mutex mtx;
        return mtx;
      }

      static std::size_t& ref_count() {
        static std::size_t count = 0;
        return count;
      }

      static bool& owns_runtime() {
        static bool owns = false;
        return owns;
      }

      static void acquire() {
        std::lock_guard<std::mutex> lock(guard_mutex());

        if (ref_count() == 0) {
          if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
            owns_runtime() = true;
          } else {
            owns_runtime() = false;
          }
        }

        ++ref_count();
      }

      static void release() noexcept {
        std::lock_guard<std::mutex> lock(guard_mutex());

        if (ref_count() == 0) return;
        --ref_count();

        if (ref_count() == 0 && owns_runtime()) {
          if (Kokkos::is_initialized()) {
            Kokkos::finalize();
          }
          owns_runtime() = false;
        }
      }
  };
}

#endif // KOCS_RUNTIME_GUARD_HPP
