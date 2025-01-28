<div align="center">
<img width="190px" alt="Cirq logo"
src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg"><img width="60px" height="0"><img width="250px" alt="Pasqal logo" src="https://upload.wikimedia.org/wikipedia/commons/1/14/Pasqal-logo-green.svg">
</div>

# cirq-pasqal

This is the Python Cirq-Pasqal integration module, which provides everything
you'll need to run [Cirq] quantum algorithms on quantum computers made by
[Pasqal](https://pasqal.com/).

[Cirq] is a Python package for writing, manipulating, and running [quantum
circuits](https://en.wikipedia.org/wiki/Quantum_circuit) on quantum computers
and simulators. Cirq provides useful abstractions for dealing with today’s
[noisy intermediate-scale quantum](https://arxiv.org/abs/1801.00862) (NISQ)
computers, where the details of quantum hardware are vital to achieving
state-of-the-art results. For more information about Cirq, please visit the
[Cirq documentation site].

[Cirq]: https://github.com/quantumlib/cirq
[Cirq documentation site]: https://quantumai.google/cirq

## Installation

This module is built on top of [Cirq]; installing this module will
automatically install the `cirq-core` module and other dependencies. There are
two installation options for the `cirq-pasqal` module:

* To install the stable version of `cirq-pasqal`, use `pip install cirq-pasqal`.
* To install the latest pre-release version of |cirq-pasqal|, use `pip install
  cirq-pasqal~=1.0.dev`. (The `~=` has a special meaning to `pip` of selecting the
  latest version compatible with the `1.*` and `dev` in the name. Despite
  appearances, this will not install an old version 1.0 release!)

If you would like to install Cirq with all the optional modules, not just
`cirq-pasqal`, then instead of the above commands, use `pip install cirq` for the
stable release or `pip install cirq~=1.0.dev` for the latest pre-release
version.

## Documentation

To get started with using Pasqal quantum computers through Cirq, please refer to
the following documentation:

* [Access and authentication](https://quantumai.google/cirq/pasqal/access).
* [Getting started guide](https://quantumai.google/cirq/tutorials/pasqal/getting_started).

To get started with using Cirq in general, please refer to the [Cirq
documentation site].

For more information about getting help, reporting bugs, and other matters
related to Cirq and the Cirq-Pasqal integration module, please visit the [Cirq
repository on GitHub](https://github.com/quantumlib/Cirq).

## Disclaimer

Cirq is not an official Google product. Copyright 2019 The Cirq Developers.
