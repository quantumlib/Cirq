.. |cirqlogo| image:: https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg
   :alt: Cirq logo
   :target: https://github.com/quantumlib/cirq
   :height: 100px

.. |cirq| replace:: Cirq
.. _cirq: https://github.com/quantumlib/cirq

.. |cirq-docs| replace:: Cirq documentation site
.. _cirq-docs: https://quantumai.google/cirq

.. |cirq-github| replace:: Cirq GitHub repository
.. _cirq-github: https://github.com/quantumlib/Cirq

.. |cirq-releases| replace:: Cirq releases page
.. _cirq-releases: https://github.com/quantumlib/Cirq/releases

.. |cirq-web| replace:: ``cirq-web``
.. |cirq-core| replace:: ``cirq-core``

.. class:: centered

|cirqlogo|

|cirq|_ is a Python package for writing, manipulating, and running `quantum
circuits <https://en.wikipedia.org/wiki/Quantum_circuit>`__ on quantum
computers and simulators. Cirq provides useful abstractions for dealing with
today’s `noisy intermediate-scale quantum <https://arxiv.org/abs/1801.00862>`__
(NISQ) computers, where the details of quantum hardware are vital to achieving
state-of-the-art results. For more information about Cirq, please visit the
|cirq-docs|_.

This Python module is |cirq-web|, which allows users to take advantage of
browser-based 3D visualization tools and features in Cirq. |cirq-web| also
provides a development environment for contributors to create and add their own
visualizations to the module.


Installation
------------

|cirq-web| is currently in development, and therefore is only available in
pre-release form. To install it use ``pip install cirq-web~=1.0.dev``. (The
``~=`` has a special meaning to ``pip`` of selecting the latest version
compatible with the ``1.*`` and ``dev`` in the name. Despite apperances, this
will not install an old version 1.0 release!) This will also install
|cirq-core| automatically.

If you would like to install Cirq with all the optional modules, not just
|cirq-web|, then instead of the above commands, use ``pip install cirq`` for
the stable release or ``pip install cirq~=1.0.dev`` for the latest pre-release
version.


Documentation
-------------

Documentation for |cirq-web| can be found in the ``README`` files located in
the module's subdirectories in the |cirq-github|_. To get started with using
Cirq in general, please refer to the |cirq-docs|_.

Below is a quick example of using |cirq-web| to generate a portable 3D
rendering of the Bloch sphere:

.. code-block:: python

    import cirq
    from cirq_web import BlochSphere

    # Prepare a state
    zero_state = [1+0j, 0+0j]
    state_vector = cirq.to_valid_state_vector(zero_state)

    # Create and display the Bloch sphere
    sphere = BlochSphere(state_vector=state_vector)
    sphere.generate_html_file()

This will create an HTML file in the current working directory. There are
additional options to specify the output directory or to open the visualization
in a browser, for example.

You can also view and interact with a Bloch sphere in a `Google Colab
<https://colab.google.com>`_ notebook or Jupyter notebook. Here is an example:

.. code-block:: python

    import cirq
    from cirq_web import BlochSphere

    # Prepare a state
    zero_state = [1+0j, 0+0j]
    state_vector = cirq.to_valid_state_vector(zero_state)

    # Create and display the Bloch sphere
    sphere = BlochSphere(state_vector=state_vector)
    display(sphere)

You can find more example Jupyter notebooks in the |cirq-web| subdirectory of
the |cirq-github|_.




For more information about getting help, reporting bugs, and other matters
related to Cirq and the Cirq-Web integration module, please visit the
|cirq-github|_.


Disclaimer
----------

Cirq is not an official Google product. Copyright 2019 The Cirq Developers
