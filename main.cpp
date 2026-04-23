#include <Kokkos_Core.hpp>

#include "include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, "positions"),
    FIELD(float, "masses")
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

int main() {
  Simulation<SimulationConfig> sim(128);
  auto& positions = sim.get_view<FIELD(Vector, "positions")>();
  auto& masses = sim.get_view<FIELD(float, "masses")>();

  initializers::RandomHollowSphere<SimulationConfig> init(2.0, positions);
  sim.init(init);

  Writer<SimulationConfig> writer("./output/tust");
  writer.write(0, sim);

  auto generic_force_mass = GENERIC_FORCE(unsigned int i, Vector& force, float& mass) {
    mass += 1.0;
  };

  auto generic_force_pos = GENERIC_FORCE(unsigned int i, Vector& force, float& mass) {
    force += Vector(100.0, 0.0, 0.0);
  };

  const float stiffness = 0.1f;
  auto pairwise_force_x = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force,
    float& mass
  ) {
    force.x() += displacement.x() * (stiffness - distance) / distance;
  };

  auto pairwise_force_y = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force,
    float& mass
  ) {
    force.y() += displacement.y() * (stiffness - distance) / distance;
  };

  auto pairwise_force_z = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force,
    float& mass
  ) {
    force.z() += displacement.z() * (stiffness - distance) / distance;
  };

  for (int i = 1; i <= 10; ++i) {
    sim.take_step(0.001,
      pairwise_force_x, pairwise_force_y, pairwise_force_z,
      generic_force_pos, generic_force_mass
    );

    writer.write(i, sim);
  }
  
  return 0;
}
