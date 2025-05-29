<div align="center">
<img width="200px" alt="Cirq logo"
src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# cirq-web

[Cirq] is a Python package for writing, manipulating, and running [quantum
circuits](https://en.wikipedia.org/wiki/Quantum_circuit) on quantum computers
and simulators. Cirq provides useful abstractions for dealing with today’s
[noisy intermediate-scale quantum](https://arxiv.org/abs/1801.00862) (NISQ)
computers, where the details of quantum hardware are vital to achieving
state-of-the-art results. For more information about Cirq, please visit the
[Cirq documentation site].

This Python module is `cirq-web`, which allows users to take advantage of
browser-based 3D visualization tools and features in Cirq. `cirq-web` also
provides a development environment for contributors to create and add their
own visualizations to the module.

[Cirq]: https://github.com/quantumlib/cirq
[Cirq documentation site]: https://quantumai.google/cirq

## Installation

This module is built on top of [Cirq]; installing this module will
automatically install the `cirq-core` module and other dependencies. There are
two installation options for the `cirq-web` module:

*   To install the stable version of `cirq-web`, use

    ```shell
    pip install cirq-web
    ```

*   To install the latest pre-release version of `cirq-web`, use

    ```shell
    pip install --upgrade cirq-web~=1.0.dev
    ```

    (The `~=` has a special meaning to `pip` of selecting the latest version
    compatible with the `1.*` and `dev` in the name. Despite appearances,
    this will not install an old version 1.0 release!)

If you would like to install Cirq with all the optional modules, not just
`cirq-web`, then instead of the above commands, use `pip install cirq` for the
stable release or `pip install --upgrade cirq~=1.0.dev` for the latest pre-release
version.

## Documentation

Documentation for `cirq-web` can be found in the `README` file located in the
module's subdirectory in the [Cirq repository on GitHub]. To get started
with using Cirq in general, please refer to the [Cirq documentation site].

Below is a quick example of using `cirq-web` to generate a portable 3D
rendering of the Bloch sphere:

```python
import cirq
from cirq_web import BlochSphere

# Prepare a state
zero_state = [1+0j, 0+0j]
state_vector = cirq.to_valid_state_vector(zero_state)

# Create and display the Bloch sphere
sphere = BlochSphere(state_vector=state_vector)
sphere.generate_html_file()
```

This will create an HTML file in the current working directory. There are
additional options to specify the output directory or to open the
visualization in a browser, for example.

You can also view and interact with a Bloch sphere in a [Google
Colab](https://colab.google.com) notebook or Jupyter notebook. Here is an
example:

```python
import cirq
from cirq_web import BlochSphere

# Prepare a state
zero_state = [1+0j, 0+0j]
state_vector = cirq.to_valid_state_vector(zero_state)

# Create and display the Bloch sphere
sphere = BlochSphere(state_vector=state_vector)
display(sphere)
```

You can find more example Jupyter notebooks in the `cirq-web` subdirectory of
the [Cirq repository on GitHub].

For more information about getting help, reporting bugs, and other matters
related to Cirq and the Cirq-Web integration module, please visit the [Cirq
repository on GitHub].

[Cirq repository on GitHub]: https://github.com/quantumlib/Cirq

## Disclaimer

Cirq is not an official Google product. Copyright 2019 The Cirq Developers.
