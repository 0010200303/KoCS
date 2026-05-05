#ifndef KOCS_INITIALIZERS_CUBOID_INIT_HPP
#define KOCS_INITIALIZERS_CUBOID_INIT_HPP

#include <numbers>
#include <utility> // added

#include <Kokkos_Core.hpp>

#include "../utils.hpp"

namespace kocs::initializers {
  template<typename SimulationConfig>
  struct RandomCuboid {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    VectorView positions_view;
    Vector min;
    Vector cuboid_dimensions;

    template<typename ViewType>
    RandomCuboid(
      ViewType positions,
      const Vector& minimum,
      const Vector& maximum
    ) : positions_view(positions)
      , min(minimum)
      , cuboid_dimensions(maximum - minimum) { }

    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned int i, Random& generator) const {
      for (int j = 0; j < dimensions; ++j)
        positions_view(i)[j] = min[j] + cuboid_dimensions[j] * generator.drand(0.0, 1.0);
    }
  };

  template<typename SimulationConfig>
  struct RelaxedCuboid : public RandomCuboid<SimulationConfig> {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)
    
    unsigned int steps;

    template<typename ViewType>
    RelaxedCuboid(
      ViewType positions,
      const Vector& minimum,
      const Vector& maximum,
      const unsigned int relaxation_steps = 2000
    ) : RandomCuboid<SimulationConfig>(positions, minimum, maximum)
      , steps(relaxation_steps) { }

    struct RelaxForce {
      // pure callable exposed as .force so kernel_fuser can extract it
      struct Impl {
        Scalar min;
        Scalar max;
        KOKKOS_INLINE_FUNCTION
        Impl(Scalar min_ = 0.8, Scalar max_ = 0.8) : min(min_), max(max_) { }

        template<typename... Rest>
        KOKKOS_INLINE_FUNCTION
        void operator()(
          const unsigned int i,
          const unsigned int j,
          const Vector& displacement,
          const Scalar& distance,
          Random& rng,
          Scalar& friction,
          PAIRWISE_REF(Vector, position),
          Rest&&...
        ) const {
          Scalar F = Kokkos::fmax(min - distance, 0) * 2.0 - Kokkos::fmax(distance - max, 0);
          position.delta += displacement * F / distance;
        }
      };

      Impl force;
      using tag = kocs::detail::PairwiseForceTag;

      KOKKOS_INLINE_FUNCTION
      RelaxForce(Scalar min_ = 0.8, Scalar max_ = 0.8) : force(min_, max_) { }

      template<typename Target>
      KOKKOS_INLINE_FUNCTION
      operator Target() const {
        return force | kocs::detail::pairwise_force;
      }
    };

    template<typename Simulation>
    void relax(Simulation& simulation) {
      for (int i = 0; i < steps; ++i)
        simulation.take_step(0.1, RelaxForce(0.8, 0.8));
    }
  };
} // namespace kocs::initializers

#endif // KOCS_INITIALIZERS_CUBOID_INIT_HPP
