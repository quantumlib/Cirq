<div align="center">
<img width="160px" height="70px" alt="Cirq logo"
src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg"
><img width="50px" height="0" alt=""><img width="400px" alt="Google Quantum AI logo"
src="https://www.gstatic.com/devrel-devsite/prod/v0113b933d5c9ba4165415ef34b487d624de9fe7d51074fd538a31c5fc879d909/quantum/images/lockup.svg">
</div>

# cirq-google

This is the Cirq-Google integration module. It provides an interface to
Google's [Quantum Computing
Service](https://quantumai.google/cirq/google/concepts), and also contains
additional tools for calibration and characterization of Google's quantum
hardware devices.

| &#9432; Please note! |
|:--------------------:|
| Google's quantum hardware is currently available only to authorized partners. Access requires an application, usually with a Google sponsor.|

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
automatically install `cirq-core` and other dependencies. There are two
installation options for the `cirq-google` module:

*   To install the stable version of `cirq-google`, use

    ```shell
    pip install cirq-google
    ```

*   To install the latest pre-release version of `cirq-google`, use

    ```shell
    pip install cirq-google~=1.0.dev
    ```

    (The `~=` has a special meaning to `pip` of selecting the latest version
    compatible with the `1.*` and `dev` in the name. Despite appearances,
    this will not install an old version 1.0 release!)

If you would like to install Cirq with all the optional modules, not just
`cirq-google`, then instead of the above commands, use `pip install cirq` for
the stable release or `pip install cirq~=1.0.dev` for the latest pre-release
version.

## Documentation

To get started with using Google quantum computers through Cirq, please refer to
the following documentation:

*   [Access and authentication](https://quantumai.google/cirq/aqt/access).
*   [Getting started
    guide](https://quantumai.google/cirq/tutorials/aqt/getting_started).

To get started with using Cirq in general, please refer to the [Cirq
documentation site].

For more information about getting help, reporting bugs, and other matters
related to Cirq and the Cirq-Core integration module, please visit the [Cirq
repository on GitHub](https://github.com/quantumlib/Cirq).

## Disclaimer

Cirq is not an official Google product. Copyright 2019 The Cirq Developers.
