#ifndef KOCS_INITIALIZERS_RANDOM_SPHERE_INIT_HPP
#define KOCS_INITIALIZERS_RANDOM_SPHERE_INIT_HPP

#include <numbers>

#include "../utils/utils.hpp"
#include "relax_force.hpp"

namespace kocs::initializers {
  template<typename SimulationConfig>
  struct RandomHollowSphere {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)
    static_assert(dimensions == 3, "RandomHollowSphere requires 3-dimensional vectors");

    Scalar radius;
    VectorView positions_view;

    RandomHollowSphere(VectorView positions, Scalar radius_)
      : positions_view(positions), radius(radius_) { }

    KOKKOS_INLINE_FUNCTION
    void operator() (const unsigned int i, Random& generator) const {
      const Scalar u = static_cast<Scalar>(generator.drand());
      const Scalar v = static_cast<Scalar>(generator.drand());

      const Scalar z = Scalar(2.0) * u - Scalar(1.0);
      const Scalar theta = Scalar(2.0) * Scalar(std::numbers::pi) * v;
      const Scalar rxy = Kokkos::sqrt(Scalar(1.0) - z * z) * radius;

      positions_view(i)[0] = rxy * Kokkos::cos(theta);
      positions_view(i)[1] = rxy * Kokkos::sin(theta);
      positions_view(i)[2] = z * radius;
    }
  };

  template<typename SimulationConfig>
  struct RandomFilledSphere {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)
    static_assert(dimensions == 3, "RandomFilledSphere requires 3-dimensional vectors");

    Scalar radius;
    VectorView positions_view;

    RandomFilledSphere(VectorView positions, Scalar radius_)
      : positions_view(positions), radius(radius_) { }

    KOKKOS_INLINE_FUNCTION
    void operator() (const unsigned int i, Random& generator) const {
      const Scalar rt = static_cast<Scalar>(generator.drand());
      const Scalar r = radius * Kokkos::cbrt(rt);

      const Scalar u = static_cast<Scalar>(generator.drand());
      const Scalar v = static_cast<Scalar>(generator.drand());

      const Scalar z = Scalar(2.0) * u - Scalar(1.0);
      const Scalar theta = Scalar(2.0) * Scalar(std::numbers::pi) * v;
      const Scalar rxy = Kokkos::sqrt(Scalar(1.0) - z * z) * r;

      positions_view(i)[0] = rxy * Kokkos::cos(theta);
      positions_view(i)[1] = rxy * Kokkos::sin(theta);
      positions_view(i)[2] = z * r;
    }
  };

  template<typename SimulationConfig>
  struct RelaxedSphere : public RandomFilledSphere<SimulationConfig> {
    EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

    unsigned int steps;

    template<typename ViewType>
    RelaxedSphere(
      ViewType positions,
      const Scalar initial_radius,
      const unsigned int relaxation_steps = 2000
    ) : RandomFilledSphere<SimulationConfig>(positions, initial_radius)
      , steps(relaxation_steps) { }
    
    template<typename Simulation>
    void relax(Simulation& simulation) {
      for (int i = 0; i < steps; ++i)
        simulation.take_step(0.1, details::RelaxForce<SimulationConfig>(0.8, 0.8));
    }
  };
} // namespace kocs::initializers

#endif // KOCS_INITIALIZERS_RANDOM_SPHERE_INIT_HPP
