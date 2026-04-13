#include <string>
#include <iostream>

#include "include/simulation.hpp"
#include "include/vector.hpp"

using namespace kocs;
struct SimulationConfig {
  using Scalar = float;
  static constexpr int dimensions = 3;

  using Vector = VectorN<Scalar, dimensions>;
  using VectorView = Kokkos::View<Vector*>;

  struct PositionField { };
  struct VelocityField { };

  using IntegrationFields = std::tuple<
    NamedField<PositionField, VectorView>,
    NamedField<VelocityField, VectorView>
  >;
};
using Vector = SimulationConfig::Vector;

int main() {
  // positions, velocities
  Simulation<SimulationConfig> sim(3);

  auto force = KOKKOS_LAMBDA(
    const int i,
    const int j,
    Simulation<SimulationConfig>::DeltaState& delta
  ) {
    delta[SimulationConfig::PositionField{}] += Vector(28.0f, 0.0f, 1.0f);
  };

  sim.take_step(force);

  return 0;
}
