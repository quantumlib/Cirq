"""Simulates the fidelity estimation.

Direct Fidelity Estimation from Few Pauli Measurements
https://arxiv.org/abs/1104.4695

Practical characterization of quantum devices without tomography
https://arxiv.org/abs/1104.3835
"""

import cirq
import heapq
import itertools


def build_circuit():
  qubits = [cirq.LineQubit(i) for i in range(3)]

  rot1 = cirq.XPowGate(exponent=0.123)
  rot2 = cirq.XPowGate(exponent=0.456)

  circuit = cirq.Circuit()
  circuit.append(rot1(cirq.LineQubit(1)))
  circuit.append(rot2(cirq.LineQubit(2)))

  return circuit, qubits


def rotated_trace(mat):
  return abs(sum([mat[i][i] for i in range(len(mat))]))


def main():
  circuit, qubits = build_circuit()

  n = len(qubits)
  d = 2**n

  highest_rho_s = []

  for i, pauli_ops in enumerate(
      itertools.product({cirq.I, cirq.X, cirq.Y, cirq.Z}, repeat=n)):
    op_dict = dict(zip(qubits, pauli_ops))

    circuit_copy = circuit.copy()
    circuit_copy.append(cirq.PauliString(op_dict))

    rho_i = rotated_trace(circuit_copy.unitary())

    if rho_i > 0:
      heapq.heappush(highest_rho_s, (rho_i, i, pauli_ops))
    if len(highest_rho_s) > n:
      heapq.heappop(highest_rho_s)

  noisy_gate = cirq.amplitude_damp(0.01)

  fidelity = 0.0
  for rho_tuple in highest_rho_s:
    rho_i = rho_tuple[0]
    Pr_i = rho_i * rho_i / d
    op_dict = dict(zip(qubits, rho_tuple[2]))

    circuit_copy = circuit.copy()
    for qubit in qubits:
      circuit_copy.append(noisy_gate(qubit))
    circuit_copy.append(cirq.PauliString(op_dict))

    sigma_i = rotated_trace(circuit_copy.unitary())
    fidelity += Pr_i * sigma_i / rho_i

  print(fidelity)


if __name__ == '__main__':
  main()
