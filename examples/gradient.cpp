// example translated from https://github.com/germannp/yalla/blob/main/examples/gradient.cu

#include "../include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_FIELDS(
    FIELD(Vector, "positions"),
    FIELD(Scalar, "gradient")
  )
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(DefaultSimulationConfig)

const int n_cells = 61;
const int steps = 200;
const double dt = 0.005;
const Scalar r_max = 1.0;
const Scalar D = 10;

__device__ float4 diffusion(float4 Xi, float4 r, float dist, int i, int j)
{
    float4 dF{0};
    if (i == j) return dF;

    if (dist > r_max) return dF;

    dF.w = i == 11 ? 0 : -r.w * D;
    return dF;
}

int main() {
  Simulation<DefaultSimulationConfig> sim(n_cells, "./output/gradient");
  sim.init_random_filled_sphere(1.5);
  sim.write();

  auto diffusion = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Random& rng,
    Scalar& friction,
    Vector& force,
    Scalar& gradient
  ) {
    if (distance > r_max)
      return;
  };

  for (int i = 1; i <= steps; ++i) {
    sim.take_step(dt, diffusion);
    sim.write();
  }

  return 0;
}
