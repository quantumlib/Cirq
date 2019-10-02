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

def main():
  circuit, qubits = build_circuit()
  wavefunction = circuit.final_wavefunction()

  n = len(qubits)
  d = 2**n

  highest_rho_s = []

  for i, pauli_ops in enumerate(
      itertools.product({cirq.I, cirq.X, cirq.Y, cirq.Z}, repeat=n)):
    pauli_string = cirq.PauliString(dict(zip(qubits, pauli_ops)))
    q_map = dict(zip(qubits, range(n)))

    rho_i = pauli_string.expectation_from_wavefunction(wavefunction, q_map).real

    if rho_i > 0:
      heapq.heappush(highest_rho_s, (rho_i, i, pauli_ops))
    if len(highest_rho_s) > n:
      heapq.heappop(highest_rho_s)

  noisy_circuit = build_noisy_circuit(circuit, qubits)
  noisy_wavefunction = noisy_circuit.final_wavefunction()

  fidelity = 0.0
  for rho_tuple in highest_rho_s:
    rho_i = rho_tuple[0]
    pauli_ops = rho_tuple[2]

    pauli_string = cirq.PauliString(dict(zip(qubits, pauli_ops)))

    Pr_i = rho_i * rho_i / d

    sigma_i = pauli_string.expectation_from_wavefunction(noisy_wavefunction, q_map).real
    fidelity += Pr_i * sigma_i / rho_i

  print(fidelity)


if __name__ == '__main__':
  main()
