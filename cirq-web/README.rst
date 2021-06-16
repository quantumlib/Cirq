

`Cirq <https://quantumai.google/cirq>`__ is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

This module is **cirq-web**, which allows users to take advantage of Typescript-based 3d visualization tools
and features in Cirq. cirq-web also provides a development environment for contributors to create and add 
their own visualizations to the module.

Documentation
-------------
All current documentation for cirq-web can be found in the README files located in this module's subdirectories.

Below is a quick example of how to generate a portable 3d rendering of a bloch sphere using cirq-web:

.. code-block:: python

    from cirq_web import BlochSphere
    sphere = BlochSphere()
    sphere.generate_HTML_file()

This will create the file in the current working directory. There are additional options to specify the
output directory, open the visualization in a browser, etc. 

You can also view and interact with a bloch sphere in a Colab or Jupyter notebook setting
with the following:

.. code-block:: python

    from cirq_web import BlochSphere
    sphere = BlochSphere()
    display(sphere)

Note that you can pass a state vector into the :code:`BlochSphere()` constructor to view a particular
state.

See the example Jupyter notebook in this directory for more examples on how to use cirq-web.

Installation
------------

Cirq-web is currently in development, and therefore is only available via pre-release.

To install the pre-release version of only **cirq-web**, use `pip install cirq-web --pre`.

Note, that this will install both cirq-web and cirq-core.

To get all the optional modules installed as well, you'll have to use `pip install cirq` or `pip install cirq --pre` for the pre-release version.