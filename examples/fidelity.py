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


def simulate_trace(circuit, Pi):
  simulator = cirq.DensityMatrixSimulator()

  n = len(Pi)
  d = 2**n

  rot = numpy.asarray([1], numpy.complex64)
  for op in Pi:
    if op == 'I':
      rot = numpy.kron(rot, numpy.asarray([[1, 0], [0, 1]], numpy.complex64))
    elif op == 'X':
      rot = numpy.kron(rot, numpy.asarray([[0, 1], [1, 0]], numpy.complex64))
    elif op == 'Y':
      rot = numpy.kron(rot, numpy.asarray([[0, -1j], [1j, 0]], numpy.complex64))
    elif op == 'Z':
      rot = numpy.kron(rot, numpy.asarray([[1, 0], [0, -1]], numpy.complex64))

  trace = 0
  for x in range(d):
    xbin = numpy.binary_repr(x, width=n)

    xvec = numpy.zeros([d], numpy.complex64)
    xvec[x] = 1

    initial_state = numpy.matmul(rot, xvec)

    y = simulator.simulate(circuit, initial_state=xvec).measurements['y']
    trace += sum([int(xbin[i]) == y[i] for i in range(n)])

  return trace


def main():
  circuit, qubits = build_circuit()
  noisy_circuit = build_noisy_circuit(circuit, qubits)

  circuit.append(cirq.measure(*qubits, key='y'))
  noisy_circuit.append(cirq.measure(*qubits, key='y'))

  n = len(qubits)
  d = 2**n

  highest_probs = []
  for i, Pi in enumerate(itertools.product({'I', 'X', 'Y', 'Z'}, repeat=n)):
    rho_i = simulate_trace(circuit, Pi)

    Pr_i = rho_i * rho_i / d

    if Pr_i > 0:
      heapq.heappush(highest_probs, (Pr_i, rho_i, i, Pi))
    if len(highest_probs) > n:
      heapq.heappop(highest_probs)

  fidelity = 0.0
  for prob_tuple in highest_probs:
    Pr_i = prob_tuple[0]
    rho_i = prob_tuple[1]
    Pi = prob_tuple[3]

    sigma_i = simulate_trace(noisy_circuit, Pi)

    fidelity += Pr_i * sigma_i / rho_i

  print(fidelity)


if __name__ == '__main__':
  main()
