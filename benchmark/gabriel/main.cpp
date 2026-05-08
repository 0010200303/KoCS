// benchmark using a simple pairwise force to test the 
// raw performance of various gabriel pair finders

#include "../../include/kocs.hpp"
#include "naive_gabriel.hpp"
#include "pre_calc_distance_gabriel.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_WRITER(writers::Dummy)
};
EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

struct NaiveConfig : public SimulationConfig {
  CONFIG_PAIR_FINDER(BenchmarkNaiveGabriel)
};

struct PreCalcConfig : public SimulationConfig {
  CONFIG_PAIR_FINDER(BenchmarkPreCalcGabriel)
};

enum class BenchmarkType { Naive, PreCalcDistance };

inline const char* benchmark_name(BenchmarkType b) {
  switch (b) {
    case BenchmarkType::Naive:
      return "Naive";
    case BenchmarkType::PreCalcDistance:
      return "PreCalculatedDistance";
  }
  return "UNKNOWN";
}

template<typename Config, typename Force>
double benchmark_gabriel(
  Force kernel,
  const int steps,
  const float dt,
  int n_agents,
  double& checksum_out
) {
  auto checksum = [&](const VectorView& positions) {
    auto host_pos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), positions);
    double sum = 0.0;
    for (int i = 0; i < host_pos.extent(0); ++i)
      sum += host_pos(i).x() + host_pos(i).y() + host_pos(i).z();
    return sum;
  };

  Simulation<Config> sim(n_agents, "");
  sim.init_random_filled_sphere(16.0);
  
  auto& positions = sim.template get_view<FIELD(Vector, positions)>();

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int i = 0; i < steps; ++i)
    sim.take_step(dt, kernel);

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

void run_benchmark_case(int n_agents, int n_steps, int n_reps, float dt_in, BenchmarkType bench) {
  auto kernel = PAIRWISE_FORCE(PAIRWISE_REF(Vector, position)) {
    position.delta += forces::Spring(displacement, distance, 0.5f) * 100.0;
  };

  double checksum = 0.0;
  double total_time = 0.0;
  for (int i = 0; i < n_reps; ++i) {
    if (bench == BenchmarkType::Naive) {
      total_time += benchmark_gabriel<NaiveConfig>(kernel, n_steps, dt_in, n_agents, checksum);
    } else if (bench == BenchmarkType::PreCalcDistance) {
      total_time += benchmark_gabriel<PreCalcConfig>(kernel, n_steps, dt_in, n_agents, checksum);
    }
  }
  double avg = total_time / static_cast<double>(n_reps);
  double time_per_step_ms = (avg / static_cast<double>(n_steps)) * 1e3;

  std::cout << benchmark_name(bench) << "," << n_agents << ","
            << n_steps << "," << n_reps << ","
            << std::setprecision(10) << dt_in << ","
            << std::setprecision(10) << time_per_step_ms << ","
            << std::setprecision(10) << checksum << "\n";
}

int main() {
  // const std::vector<int> agent_counts = {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
  const std::vector<int> agent_counts = {128, 256};
  const int steps = 100;
  const int repetitions = 10;
  const float dt = 0.1;

  Kokkos::initialize();
  {
    Kokkos::print_configuration(std::cout);

    std::cout << "benchmark,agents,steps,repetitions,dt,time_per_step_ms,checksum\n";

    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::Naive);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::PreCalcDistance);
  }
  Kokkos::finalize();

  return 0;
}
