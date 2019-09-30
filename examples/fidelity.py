"""
Simulates the fidelity estimation.

Direct Fidelity Estimation from Few Pauli Measurements
https://arxiv.org/abs/1104.4695

Practical characterization of quantum devices without tomography
https://arxiv.org/abs/1104.3835
"""

import cirq

def build_circuit():
  qubits = [cirq.LineQubit(i) for i in range(3)]

  rot1 = cirq.XPowGate(exponent=0.123)
  rot2 = cirq.XPowGate(exponent=0.456)

  circuit = cirq.Circuit()
  circuit.append(rot1(cirq.LineQubit(1)))
  circuit.append(rot2(cirq.LineQubit(2)))

  circuit.append(cirq.measure(*qubits, key='measure'))

  return circuit, qubits

def main():
  circuit, qubits = build_circuit()

  # results = cirq.sample(program=circuit,
  #                       # noise=cirq.ConstantQubitNoiseModel(
  #                       #     cirq.amplitude_damp(1e-3)),
  #                       repetitions=100)
  # print(results.histogram(key='measure'))

  simulator = cirq.Simulator()
  print(simulator.simulate(program=circuit))

if __name__ == '__main__':
  main()
