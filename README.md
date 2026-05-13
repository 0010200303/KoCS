**Ko**kkos **C**ell **S**imulation

docs at: https://0010200303.github.io/KoCS/

# Requirements
- HDF5 (libhdf5-dev)

## optional
- Kokkos (pulled automatically if not installed)

# TODO
- add custom integrator example
- add custom pair finder example
- (add optimizations per PairFinder x Integrator pair where possible, disable per compiler flag)
- benchmark Kokkos::pow(x, 2) vs. x * x and Kokkos::pow(x, 3/2) vs. x * Kokkos::sqrt(x)
