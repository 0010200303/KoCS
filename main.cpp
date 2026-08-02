#include "include/kocs.hpp"

using namespace kocs;
CREATE_SIMULATION_CONFIG(SimulationConfig,
  CONFIG_SCALAR(float)

  CONFIG_FIELDS(
    (Vector, position),
    (Vector, velocity),
    (Polarity, polarity),
    (int, type)
  )

  CONFIG_COM_FIXER(com_fixers::GlobalComFixer)
  CONFIG_PAIR_FINDER(pair_finders::NaiveAllPairs)

  CONFIG_WRITER(io::Dummy)
  CONFIG_INTEGRATOR(integrators::Euler)
)
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

const unsigned int n_cells = 100'000;
const double dt = 1.0;
const Scalar r_max = 1.0f;

int main() {
  Simulation<SimulationConfig>::Settings settings(n_cells, "./output/main");
  settings.cutoff_distance = r_max;

  Simulation<SimulationConfig> sim(settings);

  while (true) {

  }

  return 0;
}
