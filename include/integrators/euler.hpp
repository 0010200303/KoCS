#ifndef KOCS_INTEGRATORS_EULER_HPP
#define KOCS_INTEGRATORS_EULER_HPP

#include <Kokkos_Core.hpp>

#include "../utils.hpp"
#include "../simulation.hpp"

// namespace kocs::integrators {
//   template<typename SimulationConfig>
//   struct Euler {
//     EXTRACT_ALL_FROM_SIMULATION_CONFIG(SimulationConfig)

//     private:
//       Simulation<SimulationConfig>& simulation;

//     public:
//       Euler(Simulation<SimulationConfig>& simulation_) : simulation(simulation_) { }

//       template<typename ForceFn>
//       void take_step(ForceFn force, const double dt = 1.0) {
//         simulation.pair_finder.for_pair(KOKKOS_LAMBDA(const unsigned int i, const unsigned int j) {
//           force()
//         });
//       }
//   };
// }

#endif // KOCS_INTEGRATORS_EULER_HPP
