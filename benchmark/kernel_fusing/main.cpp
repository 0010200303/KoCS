// Benchmark kernel fusion: Compare control (one PAIRWISE_FORCE with full-vector math),
// split (three separate take_step, each with one component), user-fused (one force
// that adds all three components), and auto-fused (three forces passed to one take_step
// — KernelFuser fuses them into a single kernel launch).

#include "../../include/kocs.hpp"
#include <iostream>

using namespace kocs;
struct SimulationConfig : public DefaultSimulationConfig {
  CONFIG_WRITER(io::Dummy)
  CONFIG_INTEGRATOR(integrators::Euler)
};

EXTRACT_TYPES_FROM_SIMULATION_CONFIG(SimulationConfig)

enum class BenchmarkType { ControlKernel, SplitKernel, UserFusedKernel, AutoFusedKernel };

inline const char* benchmark_name(BenchmarkType b) {
  switch (b) {
    case BenchmarkType::ControlKernel:   return "ControlKernel";
    case BenchmarkType::SplitKernel:     return "SplitKernel";
    case BenchmarkType::UserFusedKernel: return "UserFusedKernel";
    case BenchmarkType::AutoFusedKernel: return "AutoFusedKernel";
  }
  return "UNKNOWN";
}

static double compute_checksum(const VectorView& positions) {
  auto host_pos = Kokkos::create_mirror_view(positions.view_device());
  Kokkos::deep_copy(host_pos, positions.view_device());
  Kokkos::fence();
  double sum = 0.0;
  for (unsigned int i = 0; i < host_pos.extent(0); ++i)
    for (unsigned int d = 0; d < dimensions; ++d)
      sum += host_pos(i)[d];
  return sum;
}

void run_benchmark_case(
  int n_agents, int n_steps, int n_reps, float dt_in, BenchmarkType bench,
  const std::string& output_prefix
) {
  const float stiffness = 0.1f;

  // ---- Control: one force that does full-vector arithmetic per pair ----
  auto control_force = PAIRWISE_FORCE(
    ctx.position.delta += displacement * (stiffness - distance) / distance;
  );

  // ---- Component-wise split forces ----
  auto force_x = PAIRWISE_FORCE(
    ctx.position.delta.x() += displacement.x() * (stiffness - distance) / distance;
  );

  auto force_y = PAIRWISE_FORCE(
    ctx.position.delta.y() += displacement.y() * (stiffness - distance) / distance;
  );

  auto force_z = PAIRWISE_FORCE(
    ctx.position.delta.z() += displacement.z() * (stiffness - distance) / distance;
  );

  // ---- User-fused: one force that manually does all three components ----
  auto user_fused_force = PAIRWISE_FORCE(
    ctx.position.delta.x() += displacement.x() * (stiffness - distance) / distance;
    ctx.position.delta.y() += displacement.y() * (stiffness - distance) / distance;
    ctx.position.delta.z() += displacement.z() * (stiffness - distance) / distance;
  );

  double checksum = 0.0;
  double total_time = 0.0;

  for (int rep = 0; rep < n_reps; ++rep) {
    // Create a fresh simulation for each repetition to avoid state accumulation
    std::string out_path = output_prefix + "_" + std::to_string(rep);
    Simulation<SimulationConfig> sim(n_agents, out_path);
    sim.init(initializers::RandomHollowSphere<SimulationConfig>(
      sim.get_view<FIELD(Vector, position)>(), 16.0));

    switch (bench) {
      case BenchmarkType::SplitKernel: {
        Kokkos::fence();
        Kokkos::Timer timer;
        for (int i = 0; i < n_steps; ++i) {
          // Evaluate each split force into the shared delta buffer,
          // then apply the Euler update only once.
          // This is done to skip the kernel fuser
          sim.integrator.evaluate_forces(
            true, sim.random_pool,
            sim.integrator.stage_pack[0],
            sim.integrator.stage_pack[1],
            force_x());
          sim.integrator.evaluate_forces(
            true, sim.random_pool,
            sim.integrator.stage_pack[0],
            sim.integrator.stage_pack[1],
            force_y());
          sim.integrator.evaluate_forces(
            true, sim.random_pool,
            sim.integrator.stage_pack[0],
            sim.integrator.stage_pack[1],
            force_z());
          sim.integrator.apply_euler(dt_in);
        }
        Kokkos::fence();
        total_time += timer.seconds();
        break;
      }
      case BenchmarkType::ControlKernel: {
        Kokkos::fence();
        Kokkos::Timer timer;
        for (int i = 0; i < n_steps; ++i)
          sim.take_step(dt_in, control_force());
        Kokkos::fence();
        total_time += timer.seconds();
        break;
      }
      case BenchmarkType::UserFusedKernel: {
        Kokkos::fence();
        Kokkos::Timer timer;
        for (int i = 0; i < n_steps; ++i)
          sim.take_step(dt_in, user_fused_force());
        Kokkos::fence();
        total_time += timer.seconds();
        break;
      }
      case BenchmarkType::AutoFusedKernel: {
        Kokkos::fence();
        Kokkos::Timer timer;
        for (int i = 0; i < n_steps; ++i)
          sim.take_step(dt_in, force_x(), force_y(), force_z());
        Kokkos::fence();
        total_time += timer.seconds();
        break;
      }
    }
    checksum = compute_checksum(sim.get_view<FIELD(Vector, position)>());
  }

  double avg = total_time / static_cast<double>(n_reps);
  double time_per_step_ms = (avg / static_cast<double>(n_steps)) * 1e3;

  std::cout << benchmark_name(bench) << "," << n_agents << ","
            << n_steps << "," << n_reps << ","
            << std::setprecision(10) << dt_in << ","
            << std::setprecision(10) << time_per_step_ms << ","
            << std::setprecision(10) << checksum << "\n" << std::flush;
}

int main() {
  const std::vector<int> agent_counts = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
  const int steps = 100;
  const int repetitions = 10;
  const float dt = 0.000001f;

  Kokkos::initialize();
  {
    Kokkos::print_configuration(std::cout);
    std::cout << "benchmark,agents,steps,repetitions,dt,time_per_step_ms,checksum\n" << std::flush;

    for (int agents : agent_counts) {
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::ControlKernel, "bench_control");
    }
    for (int agents : agent_counts) {
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::SplitKernel, "bench_split");
    }
    for (int agents : agent_counts) {
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::UserFusedKernel, "bench_userfused");
    }
    for (int agents : agent_counts) {
      run_benchmark_case(agents, steps, repetitions, dt, BenchmarkType::AutoFusedKernel, "bench_autofused");
    }
  }
  Kokkos::finalize();

  return 0;
}
