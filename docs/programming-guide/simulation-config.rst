Simulation Config
=====================

The behaviour and internal implementation details of a ``Simulation`` are defined through a ``SimulationConfig``.
This configuration type determines the ``Scalar`` datatype, dimensionality, fields registered for integration, integration scheme, neighbour search method, output writer and several additional core components used throughout the framework.

KoCS provides a ``DefaultSimulationConfig`` that defines a complete setup suitable as a base for most simulations:

.. code-block:: cpp

  struct DefaultSimulationConfig {
    CONFIG_SCALAR(float)
    CONFIG_DIMENSIONS(3)

    using Vector = kocs::VectorN<Scalar, dimensions>;
    using VectorView = Kokkos::View<Vector*>;
    using Polarity = kocs::Polarity_<Scalar>;

    CONFIG_FIELDS(
      FIELD(Vector, positions)
    ) 

    CONFIG_RANDOM_POOL(Kokkos::Random_XorShift64_Pool)
    CONFIG_PAIR_FINDER(kocs::pair_finders::NaiveAllPairs)
    CONFIG_COM_FIXER(kocs::com_fixers::GlobalComFixer)
    CONFIG_INTEGRATOR(kocs::integrators::Heun)

    CONFIG_WRITER(kocs::writers::HDF5_Writer)
  };

A custom configuration can easily be created by inheriting from ``DefaultSimulationConfig`` and overriding individual components as needed:

.. code-block:: cpp

  struct MySimulationConfig : public DefaultSimulationConfig {
    CONFIG_SCALAR(double)
    CONFIG_PAIR_FINDER(pair_finders::NaiveGabriel)
  };

Extracting Configured Types
---------------------------

Many framework types depend on the configured ``Scalar`` datatpye and the dimensionality. To simplify access to these derived types the ``EXTRACT_TYPES_FROM_SIMULATION_CONFIG`` macro is provided.
Placing the macro directly after a configuration definition exposes commonly used aliases such as ``Scalar``, ``Vector``, ``VectorView`` and ``Polarity`` with the correct underlying configuration parameters already applied.

.. code-block:: cpp

  struct MySimulationConfig : public DefaultSimulationConfig {
    CONFIG_SCALAR(double)
    CONFIG_PAIR_FINDER(pair_finders::NaiveGabriel)
  };
  EXTRACT_TYPES_FROM_SIMULATION_CONFIG(MySimulationConfig)

Configuration Components
------------------------

Scalar Type
^^^^^^^^^^^

The scalar type defines the underlying floating point precision used throughout the simulation. But experimentally also supports every other datatype.

.. code-block:: cpp

  CONFIG_SCALAR(float)

Typical choices are ``float`` for reduced memory usage or ``double`` for higher numerical precision.

Dimensions
^^^^^^^^^^

The dimensionality determines the size of all vector based types within the simulation and directly changes which ``Vector`` ``EXTRACT_TYPES_FROM_SIMULATION_CONFIG`` exports.

.. code-block:: cpp

  CONFIG_DIMENSIONS(3)

Integration Fields
^^^^^^^^^^^^^^^^^^

Simulation fields define the data stored for each agent that needs to be integrated. Fields are declared through ``CONFIG_FIELDS`` using the ``FIELD`` macro.

.. code-block:: cpp

  CONFIG_FIELDS(
    FIELD(Vector, positions),
    FIELD(Polarity, polarities)
  )

Each field defines both a type and an associated storage name (also used for output). Additional fields can be added to store custom simulation state such as custom velocities or gradients.

Random Pool
^^^^^^^^^^^

The ``RandomPool`` specifies the Kokkos random number generator backend used both internally and in user defined forces.

.. code-block:: cpp

  CONFIG_RANDOM_POOL(Kokkos::Random_XorShift64_Pool)

Pair Finder
^^^^^^^^^^^

``PairFinder`` determines how neighboring agents are detected for pairwise interactions.

NaiveAllPairs
"""""""""""""

Evaluates every possible agent pair and filters interactions using the configured cutoff distance. This approach is simple and robust but scales quadratically with the number of agents.

.. code-block:: cpp

  CONFIG_PAIR_FINDER(pair_finders::NaiveAllPairs)

NaiveGabriel
""""""""""""

Constructs neighborhoods using a Gabriel graph based approach. This can significantly reduce the number of evaluated interactions compared to exhaustive pairwise evaluation.

.. code-block:: cpp

  CONFIG_PAIR_FINDER(pair_finders::NaiveGabriel)

Integrator
^^^^^^^^^^

The ``Integrator`` defines how simulation state is advanced over time.

Euler
"""""

A first-order explicit integration scheme with minimal computational overhead.

.. code-block:: cpp

  CONFIG_INTEGRATOR(integrators::Euler)

Heun
""""

A second order, predictor-corrector integration scheme that generally provides improved stability and accuracy compared to Euler integration.

.. code-block:: cpp

  CONFIG_INTEGRATOR(integrators::Heun)

Center-of-Mass Fixer
^^^^^^^^^^^^^^^^^^^^

Center-of-mass fixers apply corrective forces to reduce unwanted drift during simulation runs.

NoComFixer
""""""""""

Disables center-of-mass correction entirely.

.. code-block:: cpp

  CONFIG_COM_FIXER(com_fixers::NoComFixer)

GlobalComFixer
""""""""""""""

Applies a global force based correction to stabilize the overall center of mass of the simulation.

.. code-block:: cpp

  CONFIG_COM_FIXER(com_fixers::GlobalComFixer)

Writer
^^^^^^

Writers are responsible for exporting simulation data to disk.

DummyWriter
"""""""""""

Disables output generation entirely. Primarily intended for benchmarking and testing purposes.

.. code-block:: cpp

  CONFIG_WRITER(writers::DummyWriter)

HDF5_Writer
"""""""""""

Default writer implementation using HDF5 for data storage together with XMF metadata generation for visualization workflows.

.. code-block:: cpp

  CONFIG_WRITER(writers::HDF5_Writer)

