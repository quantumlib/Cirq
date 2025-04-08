# Research libraries and tools

The following document provides an ecosystem overview of various open-source tools compatible with Cirq and that can work together to enable quantum computing research.

![image alt text](../images/ecosystem.png)

* **Research Libraries and Tools**: Libraries and tools used for researching new quantum algorithms and designing and preparing experiments on quantum devices.

* **Cirq**: A framework specifically for programming noisy intermediate-scale quantum computers.

* **Quantum Cloud Services**: Cirq can connect to a variety of quantum cloud services. Behind each cloud service, quantum algorithms run on either a quantum processor or simulator.

* **Quantum Circuit Simulators**: Cirq is compatible with a number of quantum circuit simulators that can run either locally or in a distributed fashion.

## Algorithm libraries and experiments

|Name|Main sponsor|Description|
|--- |--- |--- |
|[Cirq](https://github.com/quantumlib/Cirq)|Google|A framework for creating, editing, and invoking Noisy Intermediate-Scale Quantum (NISQ) circuits.|
|[OpenFermion](https://github.com/quantumlib/OpenFermion)|Google|An algorithms library for developing new quantum chemistry and materials simulation algorithms|
|[Qualtran](https://github.com/quantumlib/qualtran)|Google|A library for expressing and analyzing fault-tolerant quantum algorithms|
|[ReCirq](https://github.com/quantumlib/ReCirq)|Google|A repository of example experiments, tools, and tutorials in quantum computing|
|[Stim](https://github.com/quantumlib/stim)|Google|A library for high-speed simulation of Clifford circuits and quantum error correction|
|[TensorFlow Quantum](https://tensorflow.org/quantum)|Google|A library for developing new quantum machine learning algorithms|
|[unitary](https://github.com/quantumlib/unitary)|Google|An API library providing common operations for adding quantum behaviors to games|
|[PennyLane](https://pennylane.ai/)|Xanadu|A library for quantum machine learning with TensorFlow, PyTorch, or NumPy|

## Development tools

|Name|Main sponsor|Description|
|--- |--- |--- |
|[BQSKit](https://bqskit.lbl.gov/)|Lawrence Berkeley Labs|A portable quantum compiler framework with circuit optimization, synthesis, and gate set transpilation|
|[Mitiq](https://github.com/unitaryfund/mitiq)|Unitary Foundation|A library for error mitigation|
|[pyGSTi](https://www.pygsti.info/)|Sandia National Labs|A library for modeling and characterizing noisy quantum information processors|
|[Qristal](https://github.com/qbrilliance/qristal)|Quantum Brilliance|A library for designing, optimizing, simulating and running hybrid quantum programs|
|[Quantum Programming Studio](https://quantum-circuit.com/)|Quantastica|Web system for constructing and simulating quantum algorithms|
|[QUEKO](https://github.com/UCLA-VAST/QUEKO-benchmark)|UCLA|A tool for generating benchmarks with known optimal solutions|
|[QuTiP](https://github.com/qutip)|QuTiP|Toolbox for user-friendly and efficient numerical simulations of a wide variety of Hamiltonians|
|[staq](https://github.com/softwareQinc/staq)|softwareQ Inc|C++ library for the synthesis, transformation, optimization, and compilation of quantum circuits|
|[Superstaq](https://github.com/Infleqtion/client-superstaq/tree/main)|Infleqtion|An SDK that optimizes the execution of quantum programs by tailoring to underlying hardware primitives|
|[tket](https://docs.quantinuum.com/tket/index.html)|Quantinuum|A platform-agnostic SDK for circuit optimization, compilation and noise mitigation|
|[XACC](https://github.com/ORNL-QCI/xacc)|Oak Ridge National Labs|Extensible compilation framework using a novel, polymorphic quantum intermediate representation|

## Quantum computing cloud services

|Company|Type of Quantum Computer|
|--- |--- |
|[Alpine Quantum Technologies](https://quantumai.google/cirq/hardware/aqt/getting_started)|Trapped ions|
|[IonQ](https://quantumai.google/cirq/hardware/ionq/getting_started)|Trapped ions|
|[IQM](https://iqm-finland.github.io/cirq-on-iqm/)|Superconducting qubits|
|[Microsoft Azure Quantum](https://quantumai.google/cirq/hardware/azure-quantum/getting_started_ionq)|Trapped ions (Honeywell and IonQ)|
|[Pasqal](https://quantumai.google/cirq/hardware/pasqal/getting_started)|Neutral atoms|

For more information for vendors about integrating with Cirq,
see our [RFC page](../dev/rfc_process.md#new_hardware_integrations).

## High performance quantum circuit simulators

|Name|Main sponsor|Description|
|--- |--- |--- |
|[Qibo](https://qibo.science/)|Technology Innovation Institute|API library for hardware-accelerated quantum simulation and quantum hardware control|
|[qsim](https://github.com/quantumlib/qsim)|Google|A high-performance circuit simulator for Schrödinger simulations|
|[quimb](https://github.com/jcmgray/quimb)|Johnnie Gray|A high-performance circuit simulator using tensor-networks|
|[qulacs](https://github.com/qulacs/cirq-qulacs)|Quansys|A high-performance circuit simulator for Schrödinger simulations|
|[Stim](https://github.com/quantumlib/stim)|Google|A library for high-speed simulation of Clifford circuits and quantum error correction|
|[cuQuantum](https://github.com/NVIDIA/cuQuantum)|NVIDIA|API libraries for speeding up quantum simulation on NVIDIA GPUs|
