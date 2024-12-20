<p align="center">
    <a href="https://github.com/quantumlib/cirq">
        <img src="https://raw.githubusercontent.com/quantumlib/Cirq/main/docs/images/Cirq_logo_color.png"
                width="60%" alt="Cirq"/>
    </a>
</p>

Cirq is a Python library for writing, manipulating, and optimizing
quantum circuits and running them against quantum computers and
simulators.

[![Build Status](https://github.com/quantumlib/Cirq/actions/workflows/ci.yml/badge.svg?event=schedule)](https://github.com/quantumlib/Cirq)

[![image](https://codecov.io/gh/quantumlib/Cirq/branch/main/graph/badge.svg)](https://codecov.io/gh/quantumlib/Cirq)

[![image](https://badge.fury.io/py/cirq.svg)](https://badge.fury.io/py/cirq)

# Installation and Documentation

Cirq documentation is available at
[quantumai.google/cirq](https://quantumai.google/cirq).

Documentation for the latest **pre-release** version of Cirq (tracks the
repository's main branch; what you get if you
`pip install cirq~=1.0.dev`), is available
[here](https://quantumai.google/reference/python/cirq/all_symbols?version=nightly).

Documentation for the latest **stable** version of Cirq (what you get if
you `pip install cirq`) is available
[here](https://quantumai.google/reference/python/cirq/all_symbols).

- [Installation](https://quantumai.google/cirq/start/install)
- [Documentation](https://quantumai.google/cirq)
- [Tutorials](https://quantumai.google/cirq/build)

For a comprehensive list all of the interactive Jupyter Notebooks in our
repo (including the ones not yet published to the site) open our repo in
[Colab](https://colab.research.google.com/github/quantumlib/Cirq).

For the latest news regarding Cirq, sign up to the [Cirq-announce email
list](https://groups.google.com/forum/#!forum/cirq-announce)!

# Hello Qubit

A simple example to get you up and running:

``` python
import cirq

# Pick a qubit.
qubit = cirq.GridQubit(0, 0)

# Create a circuit
circuit = cirq.Circuit(
    cirq.X(qubit)**0.5,  # Square root of NOT.
    cirq.measure(qubit, key='m')  # Measurement.
)
print("Circuit:")
print(circuit)

# Simulate the circuit several times.
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=20)
print("Results:")
print(result)
```

Example output:

``` 
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
```

# Feature requests / Bugs / Questions

If you have feature requests or you found a bug, please [file them on
GitHub](https://github.com/quantumlib/Cirq/issues/new/choose).

For questions about how to use Cirq post to [Quantum Computing Stack
Exchange](https://quantumcomputing.stackexchange.com/) with the
[cirq](https://quantumcomputing.stackexchange.com/questions/tagged/cirq)
tag.

# How to cite Cirq

Cirq is uploaded to Zenodo automatically. Click on the badge below to
see all the citation formats for all versions.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4062499.svg)](https://doi.org/10.5281/zenodo.4062499)

# Cirq Contributors Community

We welcome contributions! Before opening your first PR, a good place to
start is to read our
[guidelines](https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md).

We are dedicated to cultivating an open and inclusive community to build
software for near term quantum computers. Please read our [code of
conduct](https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md)
for the rules of engagement within our community.

**Cirq Cynque** is our weekly meeting for contributors to discuss
upcoming features, designs, issues, community and status of different
efforts. To get an invitation please join the [cirq-dev email
list](https://groups.google.com/forum/#!forum/cirq-dev) which also
serves as yet another platform to discuss contributions and design
ideas.

# See Also

For those interested in using quantum computers to solve problems in
chemistry and materials science, we encourage exploring
[OpenFermion](https://github.com/quantumlib/openfermion) and its sister
library for compiling quantum simulation algorithms in Cirq,
[OpenFermion-Cirq](https://github.com/quantumlib/openfermion-cirq).

For machine learning enthusiasts, [Tensorflow
Quantum](https://github.com/tensorflow/quantum) is a great project to
check out!

For a powerful quantum circuit simulator that integrates well with Cirq,
we recommend looking at [qsim](https://github.com/quantumlib/qsim).

Finally, [ReCirq](https://github.com/quantumlib/ReCirq) contains real
world experiments using Cirq.

# Contact

For any questions or concerns not addressed here, please feel free to
reach out to <quantumai-oss-maintainers@googlegroups.com>.

Cirq is not an official Google product. Copyright 2019 The Cirq
Developers.
