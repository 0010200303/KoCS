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

template<typename Force, typename ForceRNG>
double benchmark_kernel(
  Force kernel,
  ForceRNG kernel_rng,
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
    // sim.take_step(dt, kernel);
    sim.take_step(dt, kernel_rng);
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

template<typename Force, typename ForceRNG>
double benchmark_rng_kernel(
  Force kernel,
  ForceRNG kernel_rng,
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
    // sim.take_step_single(dt, kernel);
    sim.take_step_rng(dt, kernel_rng);
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

void run_benchmark_case(int n_agents, int n_steps, int n_reps, float dt_in, BenchmarkType bench) {
  auto kernel = GENERIC_FORCE(
    unsigned int i,
    Vector& force
  ) {
    const double idx = static_cast<double>(i);
    const double phase = static_cast<double>(i % 97);

    force += Vector(
      0.20 + 0.001 * phase,
      0.10 + 0.0005 * phase,
      0.30 + 0.0015 * phase
    );

    if ((i & 1u) == 0u) {
      force += Vector(0.05 * idx, -0.02 * idx, 0.01 * idx);
    } else {
      force += Vector(-0.03 * idx, 0.015 * idx, -0.005 * idx);
    }

    force += Vector(
      0.001 * static_cast<double>(i % 7),
      0.0005 * static_cast<double>(i % 11),
      0.00025 * static_cast<double>(i % 13)
    );
  };

  auto rng_kernel = GENERIC_FORCE(
    unsigned int i,
    Random& rng,
    Vector& force
  ) {
    const double r1 = rng.frand(-1.0, 1.0);
    const double r2 = rng.frand(-1.0, 1.0);
    const double r3 = rng.frand(-1.0, 1.0);
    const double r4 = rng.frand(0.0, 1.0);

    force += Vector(
      r1 + 0.25 * r2,
      r2 - 0.50 * r3,
      r3 + 0.10 * r1
    );

    if (r4 > 0.5) {
      force += Vector(r4, -0.5 * r4, 0.25 * r4);
    } else {
      force += Vector(-r4, 0.25 * r4, -0.5 * r4);
    }

    force += Vector(
      0.01 * static_cast<double>(i % 5),
      0.01 * static_cast<double>(i % 3),
      0.005 * static_cast<double>(i % 7)
    );
  };

  double checksum = 0.0;
  double total_time = 0.0;
  for (int i = 0; i < n_reps; ++i) {
    if (bench == BenchmarkType::Control) {
      total_time += benchmark_kernel(kernel, rng_kernel, n_steps, dt_in, n_agents, checksum);
    } else if (bench == BenchmarkType::RNG) {
      total_time += benchmark_rng_kernel(kernel, rng_kernel, n_steps, dt_in, n_agents, checksum);
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
  const std::vector<int> agent_counts = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
  const int steps = 1;
  const int repetitions = 1;
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
