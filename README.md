# <!-- H1 title omitted because our logo acts as the title. -->

<div align="center">
<img width="280px" height="135px" alt="Cirq logo"
src="https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg">

Python package for writing, manipulating, and running [quantum
circuits](https://en.wikipedia.org/wiki/Quantum_circuit) on quantum computers
and simulators.

[![Licensed under the Apache 2.0
license](https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square)](https://github.com/quantumlib/Cirq/blob/main/LICENSE)
[![Compatible with Python versions 3.10 and
higher](https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Cirq project on
PyPI](https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c)](https://pypi.org/project/cirq)
[![Archived in
Zenodo](https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&style=flat-square&colorA=gray&colorB=3c60b1)](https://doi.org/10.5281/zenodo.4062499)

[Features](#features) &ndash;
[Installation](#installation) &ndash;
[Quick Start](#quick-start--hello-qubit-example) &ndash;
[Documentation](#cirq-documentation) &ndash;
[Integrations](#integrations) &ndash;
[Community](#community) &ndash;
[Citing Cirq](#citing-cirq) &ndash;
[Contact](#contact)

</div>

## Features

Cirq provides useful abstractions for dealing with today’s [noisy
intermediate-scale quantum](https://arxiv.org/abs/1801.00862) (NISQ) computers,
where the details of quantum hardware are vital to achieving state-of-the-art
results. Some of its features include:

*   Flexible gate definitions and custom gates
*   Parameterized circuits with symbolic variables
*   Circuit transformation, compilation and optimization
*   Hardware device modeling
*   Noise modeling
*   Multiple built-in quantum circuit simulators
*   Integration with [qsim](https://github.com/quantumlib/qsim) for
    high-performance simulation
*   Interoperability with [NumPy](https://numpy.org) and
    [SciPy](https://scipy.org)
*   Cross-platform compatibility

## Installation

Cirq supports Python version 3.10 and later, and can be used on Linux, MacOS,
and Windows, as well as [Google Colab](https://colab.google). For complete
installation instructions, please refer to the
[Install](https://quantumai.google/cirq/start/install) section of the online
Cirq documentation.

## Quick Start – “Hello Qubit” Example

Here is a simple example to get you up and running with Cirq after you have
installed it. Start a Python interpreter, and then type the following:

```python
import cirq

# Pick a qubit.
qubit = cirq.GridQubit(0, 0)

# Create a circuit.
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

Python should then print output similar to this:

```text
Circuit:
(0, 0): ───X^0.5───M('m')───
Results:
m=11000111111011001000
```

Congratulations! You have run your first quantum simulation in Cirq. You can
continue to learn more by exploring the [many Cirq tutorials](#tutorials)
described below.

## Cirq Documentation

The primary documentation site for Cirq is the [Cirq home page on the Quantum
AI website](https://quantumai.google/cirq). There and elsewhere, a variety of
documentation for Cirq is available.

### Tutorials

*   [Video tutorials] on YouTube are an engaging way to learn Cirq.
*   [Jupyter notebook-based tutorials] let you learn Cirq from your browser – no
    installation needed.
*   [Text-based tutorials] on the Cirq home page are great when combined with a
    local [installation] of Cirq on your computer. After starting with the
    [basics], you'll be ready to dive into tutorials on circuit building and
    circuit simulation under the [Build] and [Simulate] tabs, respectively. Check
    out the other tabs for more!

[Video tutorials]: https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4
[Jupyter notebook-based tutorials]: https://colab.research.google.com/github/quantumlib/Cirq
[Text-based tutorials]: https://quantumai.google/cirq
[installation]: https://quantumai.google/cirq/start/install
[basics]: https://quantumai.google/cirq/start/basics
[Build]: https://quantumai.google/cirq/build
[Simulate]: https://quantumai.google/cirq/simula

### Reference Documentation

*   Docs for the [current stable release] correspond to what you get with
    `pip install cirq`.
*   Docs for the [pre-release] correspond to what you get with
    `pip install cirq~=1.0.dev`.

[current stable release]: https://quantumai.google/reference/python/cirq/all_symbols
[pre-release]: https://quantumai.google/reference/python/cirq/all_symbols?version=nightly

### Examples

*   The [examples subdirectory](./examples/) of the Cirq GitHub repo has many
    programs illustrating the application of Cirq to everything from common
    textbook algorithms to more advanced methods.
*   The [Experiments page](https://quantumai.google/cirq/experiments/) on the
    Cirq documentation site has yet more examples, from simple to advanced.

### Change log

*   The [Cirq releases](https://github.com/quantumlib/cirq/releases) page on
    GitHub lists the changes in each release.

## Integrations

Google Quantum AI has a suite of open-source software that lets you do more
with Cirq. From high-performance simulators, to novel tools for expressing and
analyzing fault-tolerant quantum algorithms, our software stack lets you
develop quantum programs for a variety of applications.

<div align="center">

| Your interests                                  | Software to explore  |
|-------------------------------------------------|----------------------|
| Quantum algorithms?<br>Fault-tolerant quantum computing (FTQC)? | [Qualtran] |
| Large circuits and/or a lot of simulations?     | [qsim] |
| Circuits with thousands of qubits and millions of Clifford operations? | [Stim] |
| Quantum error correction (QEC)?                 | [Stim] |
| Chemistry and/or material science?              | [OpenFermion]<br>[OpenFermion-FQE]<br>[OpenFermion-PySCF]<br>[OpenFermion-Psi4] |
| Quantum machine learning (QML)?                 | [TensorFlow Quantum] |
| Real experiments using Cirq?                    | [ReCirq] |

</div>

[Qualtran]: https://github.com/quantumlib/qualtran
[qsim]: https://github.com/quantumlib/qsim
[Stim]: https://github.com/quantumlib/ssim
[OpenFermion]: https://github.com/quantumlib/openfermion
[OpenFermion-FQE]: https://github.com/quantumlib/OpenFermion-FQE
[OpenFermion-PySCF]: https://github.com/quantumlib/OpenFermion-PySCF
[OpenFermion-Psi4]: https://github.com/quantumlib/OpenFermion-Psi4
[TensorFlow Quantum]: https://github.com/tensorflow/quantum
[ReCirq]: https://github.com/quantumlib/ReCirq

## Community

<a href="https://github.com/quantumlib/Cirq/graphs/contributors"><img
width="160em" alt="Total number of contributors to Cirq"
src="https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&color=ccc&style=flat-square"/></a>

Cirq has benefited from [open-source contributions] by over 200 people and
counting. We are dedicated to cultivating an open and inclusive community to
build software for quantum computers, and have a [code of conduct] for our
community.

[open-source contributions]: https://github.com/quantumlib/Cirq/graphs/contributors
[code of conduct]: https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md

### Announcements

Stay on top of Cirq developments using the approach that best suits your needs:

*   For releases and major announcements: sign up to the low-volume mailing list
    [`cirq-announce`].
*   For releases only:
    *   Via GitHub notifications: configure [repository notifications] for Cirq.
    *   Via Atom/RSS from GitHub: subscribe to the GitHub [Cirq releases Atom feed].
    *   Via RSS from PyPI: subscribe to the [PyPI releases RSS feed] for Cirq.

Cirq releases take place approximately every quarter.

[`cirq-announce`]: https://groups.google.com/forum/#!forum/cirq-announce
[repository notifications]: https://docs.github.com/github/managing-subscriptions-and-notifications-on-github/configuring-notifications
[Cirq releases Atom feed]: https://github.com/quantumlib/Cirq/releases.atom
[PyPI releases RSS feed]: https://pypi.org/rss/project/cirq/releases.xml

### Questions and Discussions

*   Do you have questions about using Cirq? Post them to the [Quantum Computing
    Stack Exchange] and tag them with the [`cirq`] tag. You can also search past
    questions using that tag – it's a great way to learn!
*   Would you like to get more involved in Cirq development? _Cirq Cynq_ is our
    biweekly virtual meeting of contributors to discuss everything from issues to
    ongoing efforts, as well as to ask questions. Become a member of
    [_cirq-dev_](https://groups.google.com/forum/#!forum/cirq-dev) to get an
    automatic meeting invitation!

[Quantum Computing Stack Exchange]: https://quantumcomputing.stackexchange.com
[`cirq`]: https://quantumcomputing.stackexchange.com/questions/tagged/cirq

### Issues and Pull Requests

*   Do you have a feature request or want to report a bug? [Open an issue on
    GitHub] to report it!
*   Do you have a code contribution? Read our [contribution guidelines], then
    open a [pull request]!

[Open an issue on GitHub]: https://github.com/quantumlib/Cirq/issues/new/choose
[contribution guidelines]: https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md
[pull request]: https://help.github.com/articles/about-pull-requests

## Citing Cirq<a name="how-to-cite-cirq"></a>

When publishing articles or otherwise writing about Cirq, please cite the Cirq
version you use – it will help others reproduce your results. We use Zenodo to
preserve releases. The following links let you download the bibliographic
record for the latest stable release of Cirq in some popular formats:

<div align="center">

[![Download BibTeX bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e)](https://citation.doi.org/format?doi=10.5281/zenodo.4062499&style=bibtex)&nbsp;&nbsp;
[![Download CSL JSON bibliography record for latest Cirq
release](https://img.shields.io/badge/Download%20record-e0e0e0.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=json)](https://citation.doi.org/metadata?doi=10.5281/zenodo.4062499)

</div>

For formatted citations and records in other formats, as well as records for
all releases of Cirq past and present, please visit the [Cirq page on
Zenodo](https://doi.org/10.5281/zenodo.4062499).

## Contact

For any questions or concerns not addressed here, please email
<quantum-oss-maintainers@google.com>.

## Disclaimer

Cirq is not an official Google product. Copyright 2019 The Cirq Developers.
