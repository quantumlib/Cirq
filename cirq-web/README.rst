.. image:: https://raw.githubusercontent.com/quantumlib/Cirq/main/docs/images/Cirq_logo_color.png
  :target: https://github.com/quantumlib/cirq
  :alt: Cirq
  :width: 500px

`Cirq <https://quantumai.google/cirq>`__ is a Python library for writing, manipulating, and optimizing quantum
circuits and running them against quantum computers and simulators.

This module is **cirq-web**, which allows users to take advantage of browser based 3D visualization tools
and features in Cirq. cirq-web also provides a development environment for contributors to create and add 
their own visualizations to the module.

Documentation
-------------
Documentation for cirq-web can be found in the README files located in this module's subdirectories.

Below is a quick example of how to generate a portable 3D rendering of the Bloch sphere using cirq-web:

.. code-block:: python

    import cirq
    from cirq_web import BlochSphere

    # Prepare a state
    zero_state = [1+0j, 0+0j]
    state_vector = cirq.to_valid_state_vector(zero_state)

    # Create and display the Bloch sphere
    sphere = BlochSphere(state_vector=state_vector)
    sphere.generate_html_file()

This will create the file in the current working directory. There are additional options to specify the
output directory or to open the visualization in a browser for example.

You can also view and interact with a Bloch sphere in a Colab or Jupyter notebook setting
with the following:

.. code-block:: python

    import cirq
    from cirq_web import BlochSphere

    # Prepare a state
    zero_state = [1+0j, 0+0j]
    state_vector = cirq.to_valid_state_vector(zero_state)

    # Create and display the Bloch sphere
    sphere = BlochSphere(state_vector=state_vector)
    display(sphere)

See the example Jupyter notebook in this directory for more examples on how to use cirq-web.

Installation
------------

Cirq-web is currently in development, and therefore is only available via pre-release.

To install the pre-release version of only **cirq-web**, use `pip install cirq-web~=1.0.dev`.

Note, that this will install both cirq-web and cirq-core.

To get all the optional modules installed as well, you'll have to use `pip install cirq` or `pip install cirq~=1.0.dev` for the pre-release version.
