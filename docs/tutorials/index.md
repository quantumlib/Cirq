
# Textbook examples

Cirq comes with a collection of example implementations of well-known quantum
algorithms that demonstrate the main features of the library.
Most of these can be found in the
[examples directory](https://github.com/quantumlib/Cirq/tree/master/examples)
in the Github repository.

In addition to these examples, you can also find the notebooks that generated
the cirq docs found here in the
[docs directory](https://github.com/quantumlib/Cirq/tree/master/docs) as well
as all the more advanced tutorials listed further down on this page.

Below is a listing of the examples in the repository:

## Introductory algorithms

*    [Hello Qubit](https://github.com/quantumlib/Cirq/blob/master/examples/hello_qubit.py)
Simple first program showing how to create a quantum circuit.

*    [Deutsch Algorithm](https://github.com/quantumlib/Cirq/blob/master/examples/deutsch.py)
Textbook example of the simplest quantum advantage.

*    [Bernstein-Vazirani](https://github.com/quantumlib/Cirq/blob/master/examples/bernstein_vazirani.py)
Textbook algorithm determining a global property of a function with surprisingly few calls to it.

*    [Bell Inequality](https://github.com/quantumlib/Cirq/blob/master/examples/bell_inequality.py)
Demonstration of a Bell inequality which shows impossibility of local hidden variable theories.

*    [BB84](https://github.com/quantumlib/Cirq/blob/master/examples/bb84.py)
Textbook algorithm for Quantum Key Distribution.

*    [Noisy simulation](https://github.com/quantumlib/Cirq/blob/master/examples/noisy_simulation_example.py)
How to use a noisy simulator to generate results with amplitude damping.

*    [Line placement](https://github.com/quantumlib/Cirq/blob/master/examples/place_on_bristlecone.py)
How to find a line of adjacent qubits on a device.

*    [Quantum Teleportation](https://github.com/quantumlib/Cirq/blob/master/examples/quantum_teleportation.py)
A demonstration of using 2 classical bits to transport a quantum state from one
qubit to another.

*    [Super dense coding](https://github.com/quantumlib/Cirq/blob/master/examples/superdense_coding.py)
Transmit 2 classical bits using one quantum bit.

## Introductory error correction

*   [Shor's Code](https://github.com/quantumlib/Cirq/blob/master/examples/shors_code.py)
Quantum error correction with Shor's Code 

## Intermediate textbook algorithms

*    [Grover Algorithm](https://github.com/quantumlib/Cirq/blob/master/examples/grover.py)
Textbook algorithm for finding a single element hidden within a oracle function.

*    [Quantum Fourier Transform](https://github.com/quantumlib/Cirq/blob/master/examples/quantum_fourier_transform.py)
A demonstration of a 4-qubit quantum fourier transform (QFT).

*    [Basic Arithmetic](https://github.com/quantumlib/Cirq/blob/master/examples/basic_arithmetic.py)
Algorithms for adding and multiplying numbers as represented by quantum states.

*    [Phase estimation](https://github.com/quantumlib/Cirq/blob/master/examples/phase_estimator.py)
Textbook algorithm for phase estimation, i.e. for finding an eigenvalue of a unitary operator.

*    [Shor](https://github.com/quantumlib/Cirq/blob/master/examples/shor.py)
Quantum algorithm for integer factoring.

*    [QAOA](https://github.com/quantumlib/Cirq/blob/master/examples/qaoa.py)
Demonstration of the quantum approximation optimization algorithm (QAOA) on a
max-cut problem.


## Intermediate NISQ techniques

*    [XEB](https://github.com/quantumlib/Cirq/blob/master/examples/cross_entropy_benchmarking_example.py)
Fidelity estimation using cross-entropy benchmarking (XEB).

*    [Direct fidelity](https://github.com/quantumlib/Cirq/blob/master/examples/direct_fidelity_estimation.py)
Direct fidelity estimation to distinguish a desired state from the actual state
using few Pauli measurements.

*    [Qubit Characterization](https://github.com/quantumlib/Cirq/blob/master/examples/qubit_characterizations_example.py)
Qubit characterizations using Rabi oscillations, randomized
benchmarking, and tomography.


*    [Swap networks](https://github.com/quantumlib/Cirq/blob/master/examples/swap_networks.py)
Algorithm for efficiently emulating full connectivity on a limited connectivity grid of qubits.


## Advanced algorithms

*    [HHL](https://github.com/quantumlib/Cirq/blob/master/examples/hhl.py)
Algorithm for solving linear systems using quantum phase estimation.

*    [BCS Mean Field](https://github.com/quantumlib/Cirq/blob/master/examples/bcs_mean_field.py)
Quantum circuit to prepare the BCS ground states for superconductors/superfluids.


## Advanced tutorials

*    [Variational Algorithm](./variational_algorithm.ipynb)
Case study demonstrating construction of an ansatz for a two-dimensional Ising
model and how to simulate and optimize it.

*    [QAOA](qaoa.ipynb)
Demonstration of optimizing cost of a max-cut problem using quantum
approximation optimization algorithm (QAOA)

*    [Hidden Linear Function](./hidden_linear_function.ipynb)
Demonstration of a problem similar to Bernstein-Vazirani that uses a hidden
function rather than using an Oracle.

*    [Quantum Walk](quantum_walks.ipynb)
Demonstration of both classical and quantum random walks that shows their
similarities and differences.

*    [Rabi Oscillations](rabi_oscillations.ipynb)
Example of using sweeps and symbols to show rotation of a qubit by different
angles.
