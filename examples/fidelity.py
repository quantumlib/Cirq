"""Simulates the fidelity estimation.

Direct Fidelity Estimation from Few Pauli Measurements
https://arxiv.org/abs/1104.4695

Practical characterization of quantum devices without tomography
https://arxiv.org/abs/1104.3835
"""

import cirq
import heapq
import itertools
import math
import numpy


def build_circuit():
  # Builds an arbitrary circuit to test. The circuit is non Clifford to show the
  # use of simulators.
  qubits = cirq.LineQubit.range(3)
  circuit = cirq.Circuit(
      cirq.Z(qubits[0])**0.25,  # T-Gate, non Clifford.
      cirq.X(qubits[1])**0.123,
      cirq.X(qubits[2])**0.456)
  return circuit, qubits


def simulate_trace(circuit, pauli_gates, qubits, noise):
  simulator = cirq.DensityMatrixSimulator(noise=noise)

  n = len(pauli_gates)
  d = 2**n

  trace = 0
  for x in range(d):
    xbin = numpy.binary_repr(x, width=n)

    rotated_initial_state = cirq.final_wavefunction(
        cirq.DensePauliString(pauli_gates),
        qubit_order=qubits,
        initial_state=x)

    y = simulator.simulate(circuit, initial_state=rotated_initial_state).measurements['y']
    trace += sum([int(xbin[i]) == y[i] for i in range(n)])

  prob = trace * trace / d

  return trace, prob


def main():
  circuit, qubits = build_circuit()
  circuit.append(cirq.measure(*qubits, key='y'))

  noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))

  n = len(qubits)

  pauli_traces = [
      simulate_trace(circuit, pauli_gates, qubits, noise=None) + (pauli_gates,)
      for pauli_gates in itertools.product([cirq.I, cirq.X, cirq.Y, cirq.Z],
                                           repeat=n)
  ]
  highest_probs = heapq.nlargest(n, pauli_traces, key=lambda e: e[1])

  fidelity = 0.0
  for prob_tuple in highest_probs:
    rho_i, Pr_i, pauli_gates = prob_tuple
    if Pr_i == 0:
      break

    sigma_i, _ = simulate_trace(circuit, pauli_gates, qubits, noise)

    fidelity += Pr_i * sigma_i / rho_i

  print(fidelity)


if __name__ == '__main__':
  main()
