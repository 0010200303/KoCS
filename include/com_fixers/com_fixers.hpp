#ifndef KOCS_COM_FIXERS_HPP
#define KOCS_COM_FIXERS_HPP

#include <Kokkos_Core.hpp>

namespace kocs::com_fixers {
  template<typename SimulationConfig>
  struct NoComFixer {
    template<typename DeltaView>
    auto fix(DeltaView positions_delta_view) {
      using DeltaType = typename DeltaView::non_const_value_type;
      return DeltaType{};
    }
  };

  template<typename SimulationConfig>
  struct GlobalComFixer {
    template<typename DeltaView>
    auto fix(DeltaView& position_delta_view) const {
      using DeltaType = typename DeltaView::non_const_value_type;
      DeltaType total_delta{};

      const std::size_t n = position_delta_view.extent(0);
      if (n == 0)
        return total_delta;

      Kokkos::parallel_reduce(
        "com_fix_sum",
        static_cast<int>(n),
        KOKKOS_LAMBDA(const int i, DeltaType& local_delta) {
          local_delta += position_delta_view(i);
        },
        total_delta
      );
      return total_delta / n;
    }
  };
}

#endif // KOCS_COM_FIXERS_HPP
