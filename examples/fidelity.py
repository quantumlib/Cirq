"""
Simulates the fidelity estimation.

Direct Fidelity Estimation from Few Pauli Measurements
https://arxiv.org/abs/1104.4695

Practical characterization of quantum devices without tomography
https://arxiv.org/abs/1104.3835
"""

import cirq
import itertools

def build_circuit():
  qubits = [cirq.LineQubit(i) for i in range(3)]

  rot1 = cirq.XPowGate(exponent=0.123)
  rot2 = cirq.XPowGate(exponent=0.456)

  circuit = cirq.Circuit()
  circuit.append(rot1(cirq.LineQubit(1)))
  circuit.append(rot2(cirq.LineQubit(2)))

  return circuit, qubits

def main():
  circuit, qubits = build_circuit()

  n = len(qubits)

  for pauli_ops in itertools.product({cirq.I, cirq.X, cirq.Y, cirq.Z}, repeat=n):
    op_dict = dict(zip(qubits, pauli_ops))

    circuit_copy = circuit.copy()
    circuit_copy.append(cirq.PauliString(op_dict))
    circuit_copy.append(cirq.measure(*qubits, key='measure'))

if __name__ == '__main__':
  main()
