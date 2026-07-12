Initializers
============

Initializers define the initial configuration of agents in a simulation. They generate positions and optionally set up additional fields.

KoCS provides several built-in initializers for common geometries, accessible through the ``Simulation`` instance:

.. code-block:: cpp

  sim.init_random_filled_sphere(radius);
  sim.init_relaxed_sphere(radius);
  sim.init_random_hollow_sphere(radius);
  sim.init_random_cuboid(min, max);
  sim.init_relaxed_cuboid(min, max);
  sim.init_regular_hexagon(distance_to_neighbour);
  sim.init_regular_rectangle(distance_to_neighbour, nx);
  sim.init_random_disk(distance_to_neighbour);
  sim.init_line(distance);

Each initializer is also available as a standalone functor in the ``kocs::initializers`` namespace, allowing direct use outside the ``Simulation`` class.

Custom Initialization
---------------------

In many cases initialization requires more than just position assignment. KoCS supports custom field setup by passing ``INIT_FUNC`` lambdas as additional arguments to any built-in initializer:

.. code-block:: cpp

  auto my_init = INIT_FUNC(
    GENERIC_REF(Polarity, polarity)
  ) {
    // Access current agent index via `i`
    polarity.self = Polarity(positions_view(i));
  };
  sim.init_random_filled_sphere(radius, my_init);

The init function is executed immediately after position generation for each agent, allowing you to set up field values based on the generated position.

Multiple init functions can be chained:

.. code-block:: cpp

  sim.init_random_cuboid(min, max, my_init_1, my_init_2);

Available Initializers
----------------------

Random Filled Sphere
^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: kocs::initializers::RandomFilledSphere
   :project: KoCS
   :members:
   :undoc-members:

Random Hollow Sphere
^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: kocs::initializers::RandomHollowSphere
   :project: KoCS
   :members:
   :undoc-members:

Relaxed Sphere
^^^^^^^^^^^^^^

.. doxygenstruct:: kocs::initializers::RelaxedSphere
   :project: KoCS
   :members:
   :undoc-members:

Random Cuboid
^^^^^^^^^^^^^

.. doxygenstruct:: kocs::initializers::RandomCuboid
   :project: KoCS
   :members:
   :undoc-members:

Relaxed Cuboid
^^^^^^^^^^^^^^

.. doxygenstruct:: kocs::initializers::RelaxedCuboid
   :project: KoCS
   :members:
   :undoc-members:

Regular Hexagon
^^^^^^^^^^^^^^^

.. doxygenstruct:: kocs::initializers::RegularHexagon
   :project: KoCS
   :members:
   :undoc-members:

Regular Rectangle
^^^^^^^^^^^^^^^^^

.. doxygenstruct:: kocs::initializers::RegularRectangle
   :project: KoCS
   :members:
   :undoc-members:

Random Disk
^^^^^^^^^^^

.. doxygenstruct:: kocs::initializers::RandomDisk
   :project: KoCS
   :members:
   :undoc-members:

Line
^^^^

.. doxygenstruct:: kocs::initializers::Line
   :project: KoCS
   :members:
   :undoc-members:
