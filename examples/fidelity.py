"""Implements direct fidelity estimation.

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


def compute_characteristic_function(circuit, P_i, qubits, noise):
  n = len(P_i)
  d = 2**n

  trace = 0  # rho_i / sigma_i in https://arxiv.org/pdf/1104.3835.pdf
  for x in range(d):
    xbin = numpy.binary_repr(x, width=n)

    pauli_string = cirq.PauliString(dict(zip(qubits, P_i)))
    display = cirq.approx_pauli_string_expectation(pauli_string, num_samples=1)
    y = cirq.sample(cirq.Circuit(
        pauli_string,
        circuit)).measurements['y'][0]

    trace += sum([int(xbin[i]) == y[i] for i in range(n)])

  prob = trace * trace / d  # Pr(i) in https://arxiv.org/pdf/1104.3835.pdf

  return trace, prob


def main():
  circuit, qubits = build_circuit()
  circuit.append(cirq.measure(*qubits, key='y'))

  noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(0.1))

  n = len(qubits)

  # Computes for every \hat{P_i} of https://arxiv.org/pdf/1104.3835.pdf,
  # estimate rho_i and Pr(i). We then collect tuples (rho_i, Pr(i), \hat{Pi})
  # inside the variable 'pauli_traces'.
  pauli_traces = [
      compute_characteristic_function(circuit, P_i, qubits, noise=None) + (P_i,)
      for P_i in itertools.product([cirq.I, cirq.X, cirq.Y, cirq.Z],
                                           repeat=n)
  ]
  highest_probs = heapq.nlargest(n, pauli_traces, key=lambda e: e[1])

  fidelity = 0.0
  N = 0
  for rho_i, Pr_i, P_i in highest_probs:
    if Pr_i == 0:
      break

    sigma_i, _ = compute_characteristic_function(circuit, P_i, qubits, noise)

    fidelity += Pr_i * sigma_i / rho_i
    N += 1

  print(fidelity / N)


if __name__ == '__main__':
  main()
