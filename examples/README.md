# Cirq Examples

The files in this directory (`cirq/examples/`) are examples of various quantum
programs, algorithms, and demonstrations of using Cirq. This directory is not
included in the packaged Cirq releases.

<div align="center">
<p align="center">
<img alt="Everyone is free to copy, use, modify, and publish this example code"
        align="center" height="30px"
        src="https://img.shields.io/badge/Everyone%20is%20free%20to%20copy,%20modify,%20use,%20and/or%20publish%20these%20example%20programs-ffe288.svg?style=for-the-badge">
</p>
</div>

## Summary of examples

Here is a summary of the examples found in this directory:

* [`basic_arithmetic.py`](basic_arithmetic.py): examples of various basic
  arithmetic circuits.
* [`bb84.py`](bb84.py): demonstration of the BB84 QKD Protocol.
* [`bcs_mean_field.py`](bcs_mean_field.py): example of a quantum circuit
  to prepare the BCS ground states for superconductors/superfluids.
* [`bell_inequality.py`](bell_inequality.py): demonstration of Bell's theorem.
* [`bernstein_vazirani.py`](bernstein_vazirani.py): demonstration of the
  Bernstein-Vazirani algorithm.
* [`deutsch.py`](deutsch.py): demonstration of Deutsch's algorithm.
* [`direct_fidelity_estimation.ipynb`](direct_fidelity_estimation.ipynb):
  an example that walks through the steps of running the direct fidelity
  estimation (DFE) algorithm
* [`grover.py`](grover.py): demonstration of Grover's algorithm.
* [`heatmaps.py`](heatmaps.py): demonstration of how `cirq.Heatmap` can
  be used to generate a heatmap of qubit fidelities.
* [`hello_qubit.py`](hello_qubit.py): example of a simple quantum circuit.
* [`hhl.py`](hhl.py): demonstration of the algorithm for solving linear systems
  by Harrow, Hassidim, and Lloyd.
* [`hidden_shift_algorithm.py`](hidden_shift_algorithm.py):
  demonstration of a Hidden Shift algorithm.
* [`noisy_simulation_example.py`](noisy_simulation_example.py): example
  of a noisy circuit using the `cirq.ConstantQubitNoiseModel` class.
* [`phase_estimator.py`](phase_estimator.py): example of a phase
  estimator circuit.
* [`qaoa.py`](qaoa.py): example of the Quantum Approximate Optimization
  Algorithm applied to the Max-Cut problem.
* [`quantum_fourier_transform.py`](quantum_fourier_transform.py):
  example of a circuit for Quantum Fourier Transform (QFT) on 4 qubits.
* [`quantum_teleportation.py`](quantum_teleportation.py): demonstration
  of a (simulation of) quantum teleportation.
* [`qubit_characterizations_example.py`](qubit_characterizations_example.py):
  examples of how to run various qubit characterizations.
* [`shor.py`](shor.py): demonstration of Shor's algorithm.
* [`shors_code.py`](shors_code.py): demonstration of Shor's code, a
  stabilizer code for quantum error correction.
* [`simon_algorithm.py`](simon_algorithm.py): demonstration of Simon's
  algorithm.
* [`stabilizer_code.ipynb`](stabilizer_code.ipynb): example of quantum
  error correction using a stabilizer code.
* [`superdense_coding.py`](superdense_coding.py): example of superdense coding.
* [`swap_networks.py`](swap_networks.py): demontration of swap networks.
* [`two_qubit_gate_compilation.py`](two_qubit_gate_compilation.py):
  example application of the two-qubit gate compilation algorithm.

## Tips

To learn more about the examples in this directory, look inside the Python
`.py` files for a [docstring](https://en.wikipedia.org/wiki/Docstring) comment
near the top of the file. (You can ignore the `_test.py` files; those are [unit
tests](https://en.wikipedia.org/wiki/Unit_testing) for the example files.)

The [_Experiments_ page on the Cirq documentation
site](https://quantumai.google/cirq/experiments/qcqmc/high_level) has more
examples 
