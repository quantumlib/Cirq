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
  qubits = cirq.LineQubit.range(3)
  circuit = cirq.Circuit(
      cirq.Z(qubits[0])**0.25,  # T-Gate, non Clifford.
      cirq.X(qubits[1])**0.123,
      cirq.X(qubits[2])**0.456)
  return circuit, qubits


def simulate_trace(circuit, pauli_gates):
  simulator = cirq.DensityMatrixSimulator()

  n = len(pauli_gates)
  d = 2**n

  trace = 0
  for x in range(d):
    xbin = numpy.binary_repr(x, width=n)

    rotated_initial_state = cirq.final_wavefunction(
        cirq.DensePauliString(pauli_gates),
        initial_state=x)

    print('TONYBOOM rotated_initial_state=%s pauli_gates=%s' % (rotated_initial_state, list(pauli_gates)))  # DO NOT SUBMIT
    return 1.0  # DO NOT SUBMIT

    y = simulator.simulate(circuit, initial_state=rotated_initial_state).measurements['y']
    trace += sum([int(xbin[i]) == y[i] for i in range(n)])

  return trace


def main():
  circuit, qubits = build_circuit()
  noisy_circuit = circuit.with_noise(cirq.amplitude_damp(0.01))

  circuit.append(cirq.measure(*qubits, key='y'))
  noisy_circuit.append(cirq.measure(*qubits, key='y'))

  n = len(qubits)
  d = 2**n

  highest_probs = []
  for i, pauli_gates in enumerate(itertools.product({cirq.I, cirq.X, cirq.Y, cirq.Z}, repeat=n)):
    rho_i = simulate_trace(circuit, pauli_gates)

    Pr_i = rho_i * rho_i / d

    if Pr_i > 0:
      heapq.heappush(highest_probs, (Pr_i, rho_i, i, pauli_gates))
    if len(highest_probs) > n:
      heapq.heappop(highest_probs)
  return  # DO NOT SUBMIT
  fidelity = 0.0
  for prob_tuple in highest_probs:
    Pr_i = prob_tuple[0]
    rho_i = prob_tuple[1]
    pauli_gates = prob_tuple[3]

    sigma_i = simulate_trace(noisy_circuit, pauli_gates)

    fidelity += Pr_i * sigma_i / rho_i

  print(fidelity)


if __name__ == '__main__':
  main()
