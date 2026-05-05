#ifndef KOCS_INITIALIZERS_CUBOID_INIT_HPP
#define KOCS_INITIALIZERS_CUBOID_INIT_HPP

#include <numbers>

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

    struct RelaxForce {
      template<typename PositionType, typename DisplacementType, typename... Rest>
      KOKKOS_INLINE_FUNCTION
      void operator()(
        const Vector& displacement,
        const Scalar& distance,
        PAIRWISE_REF(Vector, position),
        Rest&&...
      ) const {
        Scalar F = Kokkos::fmax(0.8 - distance, 0) * 2.0 - Kokkos::fmax(distance - 0.8, 0);
        position.delta += displacement * F / distance;
      }
    };

    template<typename... Rest>
    KOKKOS_INLINE_FUNCTION
    static void relu(
      const Vector& displacement,
      const Scalar& distance,
      PAIRWISE_REF(Vector, position),
      Rest&&...
    ) {
      Scalar F = Kokkos::fmax(0.8 - distance, 0) * 2.0 - Kokkos::fmax(distance - 0.8, 0);
      position.delta += displacement * F / distance;
    }

    template<typename ViewType>
    RelaxedCuboid(
      ViewType positions,
      const Vector& minimum,
      const Vector& maximum,
      const unsigned int relaxation_steps = 2000
    ) : RandomCuboid<SimulationConfig>(positions, minimum, maximum)
      , steps(relaxation_steps) { }

    template<typename Simulation>
    void relax(Simulation& simulation) {
      auto relu_force = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position), auto&&...) {
        relu(displacement, distance, position);
      };

      for (int i = 0; i < steps; ++i)
        simulation.take_step(0.1, relu_force);
    }
  };
} // namespace kocs::initializers

#endif // KOCS_INITIALIZERS_CUBOID_INIT_HPP
