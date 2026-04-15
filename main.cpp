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

  static constexpr std::string_view IntegrationFieldNames[] = {
    "positions",
    "velocities",
  };
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

#include <vector>
#include <highfive/highfive.hpp>

int main() {
  Simulation<SimulationConfig> sim(3);

  // auto& positions = sim.get_field<0>();
  auto& positions = sim.get_field<Simulation<SimulationConfig>::index_of_field_name<"positions">()>();
  // auto& positions = sim.get_field("positions");

  return 0;

  LineInitializer<SimulationConfig> init(positions);
  sim.init(init);

  auto force = KOKKOS_LAMBDA(
    const int i,
    const int j,
    auto& rng,
    Vector& position,
    Vector& velocity
  ) {
    position += Vector(28.0f, 0.0f, 7.0f);
    position.x() = rng.drand(0.0f, 28.0f);
  };
  sim.take_step_rng(force);

  // Writer<SimulationConfig> writer("./output/tust");
  // writer.write(0, sim);
  
  return 0;
}
