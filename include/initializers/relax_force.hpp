#ifndef KOCS_INITIALIZERS_RELAX_FORCE_HPP
#define KOCS_INITIALIZERS_RELAX_FORCE_HPP

#include "../forces/predefined.hpp"
#include "../utils/utils.hpp"

namespace kocs::details {
template<typename SimulationConfig>
  struct RelaxForce {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    Scalar min;
    Scalar max;

    KOKKOS_INLINE_FUNCTION
    RelaxForce(Scalar min_ = 0.8, Scalar max_ = 0.8) : min(min_), max(max_) { }

    // TODO: this should not be using hard coded names (ctx.position)
    PAIRWISE_FORCE_OP() {
      ctx.position.delta += forces::PiecewiseLinear(displacement, distance, min, max, Scalar(2), Scalar(1));
    }
  };
} // namespace kocs.:initializers

#endif // KOCS_INITIALIZERS_RELAX_FORCE_HPP
