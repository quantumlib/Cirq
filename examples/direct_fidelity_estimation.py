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
from typing import List
from typing import Tuple


def build_circuit():
  # Builds an arbitrary circuit to test. The circuit is non Clifford to show the
  # use of simulators.
  qubits = cirq.LineQubit.range(3)
  circuit = cirq.Circuit(
      cirq.Z(qubits[0])**0.25,  # T-Gate, non Clifford.
      cirq.X(qubits[1])**0.123,
      cirq.X(qubits[2])**0.456)
  return circuit, qubits


def compute_characteristic_function(circuit: cirq.Circuit,
                                    P_i: Tuple[cirq.Gate, ...],
                                    qubits: List[cirq.Qid],
                                    noise: cirq.NoiseModel):
  n = len(P_i)
  d = 2**n

  simulator = cirq.DensityMatrixSimulator()
  # rho or sigma in https://arxiv.org/pdf/1104.3835.pdf
  density_matrix = simulator.simulate(circuit).final_density_matrix

  pauli_string = cirq.PauliString(dict(zip(qubits, P_i)))
  qubit_map = dict(zip(qubits, range(n)))
  # rho_i or sigma_i in https://arxiv.org/pdf/1104.3835.pdf
  trace = pauli_string.expectation_from_density_matrix(density_matrix,
                                                       qubit_map).real

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
  pauli_traces = []
  for P_i in itertools.product([cirq.I, cirq.X, cirq.Y, cirq.Z], repeat=n):
    rho_i, Pr_i = compute_characteristic_function(
        circuit, P_i, qubits, noise=None)
    pauli_traces.append({'P_i': P_i, 'rho_i': rho_i, 'Pr_i': Pr_i})

  assert len(pauli_traces) == 4**n

  p = [x['Pr_i'] for x in pauli_traces]
  assert numpy.isclose(sum(p), 1.0, atol=1e-6)

  fidelity = 0.0
  for _ in range(n):
    # Randomly sample as per probability.
    i = numpy.random.choice(range(4**n), p=p)

    Pr_i = pauli_traces[i]['Pr_i']
    P_i = pauli_traces[i]['P_i']
    rho_i = pauli_traces[i]['rho_i']

    sigma_i, _ = compute_characteristic_function(circuit, P_i, qubits, noise)

    fidelity += Pr_i * sigma_i / rho_i

  print(fidelity / n)


if __name__ == '__main__':
  main()
