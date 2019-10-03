"""Simulates the fidelity estimation.

Direct Fidelity Estimation from Few Pauli Measurements
https://arxiv.org/abs/1104.4695

Practical characterization of quantum devices without tomography
https://arxiv.org/abs/1104.3835
"""

import cirq
import heapq
import itertools
import numpy


def build_circuit():
  qubits = [cirq.LineQubit(i) for i in range(3)]

  rot0 = cirq.ZPowGate(exponent=0.250)  # T-Gate, non Clifford.
  rot1 = cirq.XPowGate(exponent=0.123)
  rot2 = cirq.XPowGate(exponent=0.456)

  circuit = cirq.Circuit()
  circuit.append(rot0(cirq.LineQubit(0)))
  circuit.append(rot1(cirq.LineQubit(1)))
  circuit.append(rot2(cirq.LineQubit(2)))

  return circuit, qubits


def build_noisy_circuit(circuit, qubits):
  noisy_circuit = circuit.copy()

  noisy_gate = cirq.amplitude_damp(0.01)
  for qubit in qubits:
    noisy_circuit.append(noisy_gate(qubit))

  return noisy_circuit

def average_measurements(results):
  x = results.measurements['x']
  return numpy.mean([numpy.prod([2 * y - 1 for y in row]) for row in x])

def main():
  simulator = cirq.DensityMatrixSimulator()

  circuit, qubits = build_circuit()
  wavefunction = circuit.final_wavefunction()

  n = len(qubits)
  d = 2**n

  highest_probs = []

  for i, pauli_ops in enumerate(
      itertools.product({cirq.I, cirq.X, cirq.Y, cirq.Z}, repeat=n)):
    pauli_string = cirq.PauliString(dict(zip(qubits, pauli_ops)))
    display = cirq.approx_pauli_string_expectation(pauli_string, num_samples=n)

    circuit_copy = circuit.copy()
    circuit_copy.append(display.measurement_basis_change())
    circuit_copy.append(cirq.measure(*qubits, key='x'))

    results = simulator.run(circuit_copy, repetitions=n)
    rho_i = average_measurements(results)
    Pr_i = rho_i * rho_i / d

    if Pr_i > 0:
      heapq.heappush(highest_probs, (Pr_i, rho_i, i, pauli_ops))
    if len(highest_probs) > n:
      heapq.heappop(highest_probs)

  noisy_circuit = build_noisy_circuit(circuit, qubits)

  fidelity = 0.0
  for prob_tuple in highest_probs:
    Pr_i = prob_tuple[0]
    rho_i = prob_tuple[1]
    pauli_ops = prob_tuple[3]

    pauli_string = cirq.PauliString(dict(zip(qubits, pauli_ops)))

    circuit_copy = noisy_circuit.copy()
    circuit_copy.append(display.measurement_basis_change())
    circuit_copy.append(cirq.measure(*qubits, key='x'))

    results = simulator.run(circuit_copy, repetitions=n)
    sigma_i = average_measurements(results)

    fidelity += Pr_i * sigma_i / rho_i

  print(fidelity)


if __name__ == '__main__':
  main()
