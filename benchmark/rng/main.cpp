#include <Kokkos_Core.hpp>
#include "../../include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_WRITER(writers::Dummy)
};

EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

enum class BenchmarkType { Control, RNG };

inline const char* benchmark_name(BenchmarkType b) {
  switch (b) {
    case BenchmarkType::Control:
      return "Control";
    case BenchmarkType::RNG:
      return "RNG";
  }
  return "UNKNOWN";
}

template<typename Force>
double benchmark_kernel(
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

  Simulation<SimulationConfig> sim(n_agents, "");
  sim.init_random_hollow_sphere(16.0);
  
  auto& positions = sim.get_view<FIELD(Vector, "positions")>();

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int i = 0; i < steps; ++i) {
    sim.take_step_single(dt, kernel);
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

template<typename Force>
double benchmark_rng_kernel(
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

  Simulation<SimulationConfig> sim(n_agents, "");
  sim.init_random_hollow_sphere(16.0);
  
  auto& positions = sim.get_view<FIELD(Vector, "positions")>();

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int i = 0; i < steps; ++i) {
    sim.take_step(dt, kernel);
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

void run_benchmark_case(int n_agents, int n_steps, int n_reps, float dt_in, BenchmarkType bench) {
  auto control_kernel = GENERIC_FORCE(
    unsigned int i,
    Vector& force
  ) {
    force += Vector(1.0, 1.0, 1.0);
  };

  auto rng_kernel = GENERIC_FORCE(
    unsigned int i,
    // Random& rng,
    Vector& force
  ) {
    force += Vector(1.0, 1.0, 1.0);
  };

  double checksum = 0.0;
  double total_time = 0.0;
  for (int i = 0; i < n_reps; ++i) {
    if (bench == BenchmarkType::Control) {
      total_time += benchmark_kernel(control_kernel, n_steps, dt_in, n_agents, checksum);
    } else if (bench == BenchmarkType::RNG) {
      total_time += benchmark_rng_kernel(rng_kernel, n_steps, dt_in, n_agents, checksum);
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
  const std::vector<int> agent_counts = {65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608};
  const int steps = 100;
  const int repetitions = 10;
  const float dt = 0.000001;

  Kokkos::initialize();
  {
    Kokkos::print_configuration(std::cout);

    std::cout << "benchmark,agents,steps,repetitions,dt,time_per_step_ms,checksum\n";

    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::Control);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::RNG);
  }
  Kokkos::finalize();

  return 0;
}
