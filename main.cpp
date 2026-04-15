#include <string>
#include <iostream>

#include "include/utils.hpp"
#include "include/vector.hpp"
#include "include/simulation.hpp"
#include "include/initializers/line.hpp"
#include "include/io/writer.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  using IntegrationFields = std::tuple<
    VectorView, // positions
    VectorView  // velocities
  >;

  static constexpr const char* IntegrationFieldNames[] = {
    "positions",
    "velocities",
  };
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

#include <vector>
#include <highfive/highfive.hpp>

int main() {
  Simulation<SimulationConfig> sim(10);
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
  // sim.take_step(force);



  std::vector<Scalar[3]> pos(positions.extent(0));
  for (int i = 0; i < positions.extent(0); ++i) {
    pos[i][0] = positions(i).x();
    pos[i][1] = positions(i).y();
    pos[i][2] = positions(i).z();

    std::cout << positions(i).x() << " "
      << positions(i).y() << " "
      << positions(i).z() << std::endl;
  }

  Writer<SimulationConfig> writer("./output/tust");
  writer.write(0, sim);
  
  return 0;
}
