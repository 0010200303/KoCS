#include <Kokkos_Core.hpp>
#include "../../include/kocs.hpp"

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_WRITER(writers::Dummy)
};

EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

enum class BenchmarkType { ControlKernel, SplitKernel, UserFusedKernel, AutoFusedKernel };

inline const char* benchmark_name(BenchmarkType b) {
  switch (b) {
    case BenchmarkType::ControlKernel:
      return "ControlKernel";
    case BenchmarkType::SplitKernel:
      return "SplitKernel";
    case BenchmarkType::UserFusedKernel:
      return "UserFusedKernel";
    case BenchmarkType::AutoFusedKernel:
      return "AutoFusedKernel";
  }
  return "UNKNOWN";
}

template<typename... Forces>
struct Fuser;

template<>
struct Fuser<> {
  Fuser() = default;

  template<typename... Args>
  KOKKOS_INLINE_FUNCTION
  void operator()(Args&&...) const { }
};

template<typename FirstForce, typename... RestForces>
struct Fuser<FirstForce, RestForces...> : Fuser<RestForces...> {
  using base_type = Fuser<RestForces...>;

  FirstForce force;

  KOKKOS_INLINE_FUNCTION
  Fuser() = default;

  KOKKOS_INLINE_FUNCTION
  Fuser(FirstForce first, RestForces... rest)
    : base_type(rest...)
    , force(first) { }

  template<typename... Args>
  KOKKOS_INLINE_FUNCTION
  void operator()(Args&&... args) const {
    force(static_cast<Args&&>(args)...);
    static_cast<const base_type&>(*this)(static_cast<Args&&>(args)...);
  }
};

template<typename... Forces>
Fuser(Forces...) -> Fuser<Forces...>;

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
  auto& positions = sim.get_view<FIELD(Vector, "positions")>();

  initializers::RandomHollowSphere<SimulationConfig> init(positions, 16.0);
  sim.init(init);

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

template<typename ForceX, typename ForceY, typename ForceZ>
double benchmark_split_kernel(
  ForceX kernel_x,
  ForceY kernel_y,
  ForceZ kernel_z,
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
  auto& positions = sim.get_view<FIELD(Vector, "positions")>();

  initializers::RandomHollowSphere<SimulationConfig> init(positions, 16.0);
  sim.init(init);

  Kokkos::fence();
  Kokkos::Timer timer;

  for (int i = 0; i < steps; ++i) {
    sim.take_step(dt, kernel_x);
    sim.take_step(dt, kernel_y);
    sim.take_step(dt, kernel_z);
  }

  Kokkos::fence();
  const double time = timer.seconds();
  checksum_out = checksum(positions);
  return time;
}

void run_benchmark_case(int n_agents, int n_steps, int n_reps, float dt_in, BenchmarkType bench) {
  const float stiffness = 0.1f;
  auto control_kernel = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force
  ) {
    force += displacement * (stiffness - distance) / distance;
  };

  auto split_kernel_x = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force
  ) {
    force.x() += displacement.x() * (stiffness - distance) / distance;
  };

  auto split_kernel_y = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force
  ) {
    force.y() += displacement.y() * (stiffness - distance) / distance;
  };

  auto split_kernel_z = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force
  ) {
    force.z() += displacement.z() * (stiffness - distance) / distance;
  };

  auto user_fused_kernel = PAIRWISE_FORCE(
    unsigned int i,
    unsigned int j,
    const Vector& displacement,
    const Scalar& distance,
    Vector& force
  ) {
    split_kernel_x(i, j, displacement, distance, force);
    split_kernel_y(i, j, displacement, distance, force);
    split_kernel_z(i, j, displacement, distance, force);
  };

  double checksum = 0.0;
  double total_time = 0.0;
  for (int i = 0; i < n_reps; ++i) {
    if (bench == BenchmarkType::ControlKernel) {
      total_time += benchmark_kernel(control_kernel, n_steps, dt_in, n_agents, checksum);
    } else if (bench == BenchmarkType::SplitKernel) {
      total_time += benchmark_split_kernel(split_kernel_x, split_kernel_y, split_kernel_z, n_steps, dt_in, n_agents, checksum);
    } else if (bench == BenchmarkType::UserFusedKernel) {
      total_time += benchmark_kernel(user_fused_kernel, n_steps, dt_in, n_agents, checksum);
    } else if (bench == BenchmarkType::AutoFusedKernel) {
      auto fused_kernel = Fuser{ split_kernel_x, split_kernel_y, split_kernel_z } | detail::pairwise_force;
      total_time += benchmark_kernel(user_fused_kernel, n_steps, dt_in, n_agents, checksum);
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
  const int steps = 100;
  const int repetitions = 10;
  const float dt = 0.000001;

  Kokkos::initialize();
  {
    Kokkos::print_configuration(std::cout);

    std::cout << "benchmark,agents,steps,repetitions,dt,time_per_step_ms,checksum\n";

    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::ControlKernel);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::SplitKernel);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::UserFusedKernel);
    for (int agents : agent_counts)
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::AutoFusedKernel);
  }
  Kokkos::finalize();

  return 0;
}
