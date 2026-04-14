#include <string>
#include <iostream>

#include "include/vector.hpp"
#include "include/simulation.hpp"
#include "include/initializers/line.hpp"

using namespace kocs;
struct SimulationConfig {
  using Scalar = float;
  static constexpr int dimensions = 3;

  using Vector = VectorN<Scalar, dimensions>;
  using VectorView = Kokkos::View<Vector*>;

  using IntegrationFields = std::tuple<
    VectorView // positions
    // VectorView  // velocities
  >;
};
using Vector = SimulationConfig::Vector;

int main() {
  Simulation<SimulationConfig> sim(3);
  auto positions = sim.get_field<0>();
  LineInitializer<SimulationConfig> init(positions);
  sim.init(init);

  auto force = KOKKOS_LAMBDA(
    const int i,
    const int j,
    Vector& position
  ) {
    position += Vector(28.0f, 0.0f, 7.0f);
  };

  sim.take_step(force);

  for (int i = 0; i < positions.extent(0); ++i) {
    std::cout << positions(i).x() << " "
      << positions(i).y() << " "
      << positions(i).z() << std::endl;
  }

  return 0;
}
