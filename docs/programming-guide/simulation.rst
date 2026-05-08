The Simulation
==============

The ``Simulation`` class is the central component of the framework. It coordinates agent data management,
nieghbour discovery, integration and output generation within a unified interface. In practice, nearly all high-level workflows are driven through ``Simulation``.

Internally, the class manages the configured PairFinder, the selected integration scheme, the selected writer, com fixing and all registered simulation fields. While these components can also be used independently, ``Simulation`` provides the primary abstraction for building and running complete simulations.

Instantiating
-------------

A ``Simulation`` is parameterized through a ``SimulationConfig`` type, which defines the core behaviour and underlying implementation details of the simulation. This includes the ``Scalar`` datatype, neighbourhood discovery, integration backend and additional compiletime configuration options.

For convenience, KoCS provides a ``DefaultSimulationConfig`` that can either be used directly or extented to override specific settings:

.. code-block:: cpp

  struct MySimulationConfig : public DefaultSimulationConfig {
    CONFIG_SCALAR(double)
    CONFIG_PAIR_FINDER(pair_finders::NaiveGabriel)
  };

Additional details about configuration options are available in :doc:`simulation-config`.

Constructing a ``Simulation`` additionally requires several runtime parameters:

- ``agent_count`` — Total number of agents
- ``output_path`` — Base path and filename for generated output files
- ``cutoff_distance`` (default: 1000000) — Maximum interaction distance used for pairwise evaluations
- ``seed`` (default: 2807) — Seed used for all random number generation

.. attention::

  Determinism is not guaranteed across different hardware architectures or compiler implementations, even when using the same seed.

Once the configuration and runtime parameters are chosen, a new ``Simulation`` instance can be created directly:

.. code-block:: cpp

  Simulation<MySimulationConfig> sim(agent_count, "./my/path/my_file");

.. attention::
  
  Do **not** include a file extension in ``output_path``. Output files are automatically generated as ``.h5`` and ``.xmf`` files. 

Initializing
------------

``Simulation`` provides multiple built-in initialization routines for generating initial agent configurations, including random spheres, cuboids, hexagons and other common geometries.
Initialization routines can be invoked directly on the simulation instance:

.. code-block:: cpp

  sim.init_random_filled_sphere(radius);

In many cases initialization requires more than just assigning positions. To support this, KoCS provides the ``INIT_FUNC`` macro for defining custom initialization kernels that directly modify simulation fields immediately after position generation.

.. code-block:: cpp

  auto& polarities_view = sim.get_view<FIELD(Polarity, polarities)>();
  auto initial_conditions = INIT_FUNC() {
    polarities_view(i) = Polarity(positions_view(i));
  };
  sim.init_random_filled_sphere(radius, initial_conditions);

Further details on ``INIT_FUNC`` and field access patterns are available in :doc:`forces`.

For fully custom initialization workflows, ``Simulation`` also provides a generic ``init`` function that accepts arbitrary initialization kernels directly.

Taking a step
-------------

A simulation step is advanced through the ``take_step`` function by supplying the delta time together with all active forces for that iteration:

.. code-block::

  sim.take_step(dt, generic_force, pairwise_force_1, pairwise_force_2);

.. note::

  Supplying all forces in a single ``take_step`` call is recommended, as it allows the framework to apply internal optimizations and scheduling strategies more efficiently. 

Writing
-------

Simulation output can be written through the ``write`` function. The framework automatically handles data conversion, file generation and timestep bookkeeping internally.

.. code-block:: cpp

  sim.write();

Addtionally non integrated fields, such as static masses or agent types can also be included in the out by passing their ``Views`` to ``write``:

.. code-block:: cpp

  sim.write(masses, types);
