Quick Start
===========

This guide walks you through the basic steps required to set up, build and run a minimal KoCS project.

Prerequisites
-------------

Before building any KoCS project, ensure the following dependencies are met on your system:
- **C++ Standard**: A compiler with support for C++20 or newer
- **CMake**: Version 3.20 or higher
- **HDF5**: Required system library (on Debian/Ubuntu systems, install via ``apt install libhdf5-dev``)

See :doc:`requirements` for more details.

Setting up a project
--------------------

The simplest way to start a new project is to clone the framework repository from `GitHub <https://github.com/0010200303/KoCS>`__.

.. code-block:: bash

  git clone https://github.com/0010200303/KoCS

Compiling and running an executable
-----------------------------------

To quickly begin experimenting with the framework, try out the examples or edit the main.cpp file in the projects root and compile it using the provided build script:

.. code-block:: bash

  ./kocs.sh main.cpp

By default this produces a new executable named kocs, which can be launched with:

.. code-block:: bash

  ./kocs

The build script also provides several optional command-line arguments:

- -o / --output — Specify the name of the generated executable
- -e / --execute — Automatically execute the program after compilation
- -B / --backend — Select the Kokkos backend to use for compilation

Available backends:
^^^^^^^^^^^^^^^^^^^

**CPU Backends**

- SERIAL — Default CPU execution
- OPENMP — OpenMP-based parallel execution
- THREADS — Native threaded execution backend
- HPX — HPX runtime backend (experimental)

**GPU Backends**

- CUDA — NVIDIA GPU support
- HIP — AMD GPU support
- SYCL — Intel GPU support
- OPENMPTARGET — OpenMP target offloading backend
- OPENACC — OpenACC accelerator backend
