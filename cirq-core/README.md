<div align="center">
<img width="220px" alt="Cirq logo"
src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">
</div>

# cirq-core

[Cirq] is a Python package for writing, manipulating, and running [quantum
circuits](https://en.wikipedia.org/wiki/Quantum_circuit) on quantum computers
and simulators. Cirq provides useful abstractions for dealing with today’s
[noisy intermediate-scale quantum](https://arxiv.org/abs/1801.00862) (NISQ)
computers, where the details of quantum hardware are vital to achieving
state-of-the-art results. For more information about Cirq, please visit the
[Cirq documentation site].

This Python module is `cirq-core`, which contains everything you'd need to
write quantum algorithms for NISQ devices and run them on the built-in Cirq
simulators.

To run algorithms on a given quantum computing platform, you will also need to
install an appropriate Cirq hardware interface module. Please visit the
[hardware section of the Cirq documentation
site](https://quantumai.google/cirq/hardware) for information about the
hardware interface modules currently available.

[Cirq]: https://github.com/quantumlib/cirq
[Cirq documentation site]: https://quantumai.google/cirq

## Installation

There are two installation options for the `cirq-core` module:

* To install the stable version of `cirq-core`, use `pip install cirq-core`.
* To install the pre-release version of `cirq-core`, use `pip install
  cirq-core~=1.0.dev`. (The `~=` has a special meaning to `pip` of
  selecting the latest version compatible with the `1.*` and `dev` in the
  name. Despite appearances, this will not install an old version 1.0 release!)

If you would like to install Cirq with all the optional modules, not just
`cirq-core`, then instead of the above commands, use `pip install cirq` for the
stable release or `pip install cirq~=1.0.dev` for the latest pre-release
version.

## Documentation

To get started with using Cirq, please refer to the [Cirq documentation site].

For more information about getting help, reporting bugs, and other matters
related to Cirq and the Cirq-Core integration module, please visit the [Cirq
repository on GitHub](https://github.com/quantumlib/Cirq).

## Disclaimer

Cirq is not an official Google product. Copyright 2019 The Cirq Developers.
