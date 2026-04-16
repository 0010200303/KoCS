#ifndef KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
#define KOCS_PAIR_FINDERS_ALL_PAIRS_HPP

#include <Kokkos_Core.hpp>

#include "../utils.hpp"
#include "../simulation.hpp"
#include "../simulation_config.hpp"

// namespace kocs::pair_finders {
//   template<SimulationConfig>
//   struct NaiveAllPairs {
//     EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

//     private:
//       Simulation<SimulationConfig>& simulation;

//     public:
//       NaiveAllPairs(Simulation<SimulationConfig>& simulation_) : simulation(simulation_) { }
    
//       template<typename Functor>
//       void for_pair(Functor functor, Storage local_storage) {
//         Kokkos::parallel_for(
//           "take_step",
//           Kokkos::TeamPolicy<>(agent_count, Kokkos::AUTO),
//           KOKKOS_CLASS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
//             const int i = team.league_rank();

//             LocalValues local_values{};

//             Kokkos::parallel_reduce(
//               Kokkos::TeamThreadRange(team, agent_count),
//               [&](const int j, LocalValues& local) {
//                 if (i == j)
//                   return;

//                 // functor(i, j);
//                 simulation.invoke_force(functor, i, j, local_data);
//               },
//               local_values
//             );

//             Kokkos::single(Kokkos::PerTeam(team), [&]() {
//               euler_update(state, local_values, i, dt);
//             });
//           }
//         );
//       }
//   };
// }

#endif // KOCS_PAIR_FINDERS_ALL_PAIRS_HPP
