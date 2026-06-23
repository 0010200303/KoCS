**Ko**kkos **C**ell **S**imulation

docs at: https://0010200303.github.io/KoCS/

# Requirements
- HDF5 (libhdf5-dev)

## optional
- Kokkos (pulled automatically if not installed)

# TODO
- rework some syntax (FIELDS, custom View definitions)
- implement delauney pair finder
- implement approximate gabriel pair finder (maybe use cones approach)
- add custom integrator example
- add custom pair finder example
- (add optimizations per PairFinder x Integrator pair where possible, disable per compiler flag)

# known issues
- older hdf5 versions cause small a memory leak when using the HDF5_writer on CPU (very observable when running branching on CPU)
