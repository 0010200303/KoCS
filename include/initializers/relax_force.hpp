#ifndef KOCS_INITIALIZERS_RELAX_FORCE_HPP
#define KOCS_INITIALIZERS_RELAX_FORCE_HPP

#include "../forces/predefined.hpp"

namespace kocs::details {
template<typename SimulationConfig>
  struct RelaxForce {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    using tag = kocs::detail::PairwiseForceTag;

    Scalar min;
    Scalar max;

    KOKKOS_INLINE_FUNCTION
    RelaxForce(Scalar min_ = 0.8, Scalar max_ = 0.8) : min(min_), max(max_) { }

    template<typename... Rest>
    KOKKOS_INLINE_FUNCTION
    void operator()(
      const unsigned int i,
      const unsigned int j,
      const Vector& displacement,
      const Scalar distance,
      Random& rng,
      Scalar& friction,
      PAIRWISE_REF(Vector, position),
      Rest&&...
    ) const {
      position.delta += forces::PiecewiseLinear(displacement, distance, min, max);
    }
  };
} // namespace kocs.:initializers

#endif // KOCS_INITIALIZERS_RELAX_FORCE_HPP
