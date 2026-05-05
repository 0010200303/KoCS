// forces taken from "Recipes for Center-Based Modeling of Multicellular Systems"
// TODO: doi

#ifndef KOCS_FORCES_PREDEFINED_HPP
#define KOCS_FORCES_PREDEFINED_HPP

#include <Kokkos_Core.hpp>

#include "detail.hpp"

namespace kocs::forces {
  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar SpringPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar scale = Scalar(1.0)
  ) {;
    return scale * (homeostatic_radius - distance);
  }

  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector Spring(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar scale = Scalar(1.0)
  ) {
    if (distance == Scalar(0.0))
      return Vector{0.0};
    
    Scalar F = SpringPotential(distance, homeostatic_radius, scale);
    return displacement * F / distance;
  }

  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar PiecewiseLinearPotential(
    const Scalar distance,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(2.0),
    const Scalar adhesion_scale = Scalar(1.0)
  ) {
    return repulsion_scale * Kokkos::fmax(repulsion_radius - distance, 0) -
      adhesion_scale * Kokkos::fmax(distance - adhesion_radius, 0);
  }

  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector PiecewiseLinear(
    const Vector& displacement,
    const Scalar distance,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(2.0),
    const Scalar adhesion_scale = Scalar(1.0)
  ) {
    if (distance == Scalar(0.0))
      return Vector{0.0};

    Scalar F = PiecewiseLinearPotential(
      distance,
      repulsion_radius,
      adhesion_radius,
      repulsion_scale,
      adhesion_scale
    );
    return displacement * F / distance;
  }

  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar PiecewiseQuadraticPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(2.0),
    const Scalar adhesion_scale = Scalar(1.0)
  ) {
    Scalar repulsion = (Kokkos::fmin(distance, homeostatic_radius) - homeostatic_radius) *
      (distance - repulsion_radius);

    Scalar adhesion = -(Kokkos::fmax(distance, homeostatic_radius) - homeostatic_radius) *
      (Kokkos::fmin(distance, adhesion_radius) - adhesion_radius);

    return repulsion * repulsion_scale + adhesion * adhesion_scale;
  }

  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector PiecewiseQuadratic(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(2.0),
    const Scalar adhesion_scale = Scalar(1.0)
  ) {
    if (distance == Scalar(0.0))
      return Vector{0.0};

    Scalar F = PiecewiseQuadraticPotential(
      distance,
      homeostatic_radius,
      repulsion_radius,
      adhesion_radius,
      repulsion_scale,
      adhesion_scale
    );
    return displacement * F / distance;
  }

  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar CubicPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar cutoff_distance,
    const Scalar scale = Scalar(1.0)
  ) {
    return scale * Kokkos::pow(distance - cutoff_distance, 2.0) * (distance - homeostatic_radius);
  }

  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector Cubic(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar cutoff_distance,
    const Scalar scale = Scalar(1.0)
  ) {
    if (distance == Scalar(0.0))
      return Vector{0.0};

    Scalar F = CubicPotential(
      distance,
      homeostatic_radius,
      cutoff_distance,
      scale
    );
    return displacement * F / distance;
  }

  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar MorsePotential(
    const Scalar distance,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(2.0),
    const Scalar adhesion_scale = Scalar(1.0),
    const Scalar scale = Scalar(1.0)
  ) {
    Scalar repulsion = Kokkos::exp(-repulsion_scale * (distance - repulsion_radius));
    Scalar adhesion = Kokkos::exp(-adhesion_scale * (distance - adhesion_radius));

    return 2 * scale * (repulsion - adhesion);
  }
  
  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector Morse(
    const Vector& displacement,
    const Scalar distance,
    const Scalar repulsion_radius,
    const Scalar adhesion_radius,
    const Scalar repulsion_scale = Scalar(2.0),
    const Scalar adhesion_scale = Scalar(1.0),
    const Scalar scale = Scalar(1.0)
  ) {
    if (distance == Scalar(0.0))
      return Vector{0.0};

    Scalar F = MorsePotential(
      distance,
      repulsion_radius,
      adhesion_radius,
      repulsion_scale,
      adhesion_scale,
      scale
    );
    return displacement * F / distance;
  }

  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar LennardJonesPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar epsilon = Scalar(1.0)
  ) {
    if (distance == Scalar(0.0))
      return Scalar(0.0);
    
    return 4 * epsilon * (
      Kokkos::pow(homeostatic_radius / distance, 12.0) -
      Kokkos::pow(homeostatic_radius / distance, 6.0)
    );
  }

  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector LennardJones(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar epsilon = Scalar(1.0)
  ) {
    if (distance == Scalar(0.0))
      return Vector{0.0};

    Scalar F = LennardJonesPotential(
      distance,
      homeostatic_radius,
      epsilon
    );
    return displacement * F / distance;
  }

  template<typename Scalar>
  KOKKOS_INLINE_FUNCTION
  static Scalar HertzContactPotential(
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar young_modulus_i,
    const Scalar young_modulus_j,
    const Scalar poisson_ratio_i,
    const Scalar poisson_ratio_j,
    const Scalar radius_i,
    const Scalar radius_j
  ) {
    Scalar composite_youngs_modulus = young_modulus_i * young_modulus_j / 
      ((1.0 - Kokkos::pow(poisson_ratio_i, 2.0)) * young_modulus_j + 
       (1.0 - Kokkos::pow(poisson_ratio_j, 2.0)) * young_modulus_i);
    Scalar effective_radius = (radius_i * radius_j) / (radius_i + radius_j);

    return Scalar(4.0)/Scalar(3.0) * composite_youngs_modulus * Kokkos::sqrt(effective_radius) * 
      Kokkos::pow(Kokkos::fmax(homeostatic_radius - distance, 0), 3.0/2.0);
  }

  template<typename Scalar, typename Vector>
  KOKKOS_INLINE_FUNCTION
  static Vector HertzContact(
    const Vector& displacement,
    const Scalar distance,
    const Scalar homeostatic_radius,
    const Scalar young_modulus_i,
    const Scalar young_modulus_j,
    const Scalar poisson_ratio_i,
    const Scalar poisson_ratio_j,
    const Scalar radius_i,
    const Scalar radius_j
  ) {
    if (distance == Scalar(0.0))
      return Vector{0.0};

    Scalar F = HertzContactPotential(
      distance,
      homeostatic_radius,
      young_modulus_i,
      young_modulus_j,
      poisson_ratio_i,
      poisson_ratio_j,
      radius_i,
      radius_j
    );
    return displacement * F / distance;
  }
} // namespace kocs::forces

#endif // KOCS_FORCES_PREDEFINED_HPP
