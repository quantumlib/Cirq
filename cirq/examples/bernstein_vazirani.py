# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import random
from typing import List

from absl import app
from absl import flags

import cirq


FLAGS = flags.FLAGS
NUM_SHOTS = 10
NUM_QUBITS = 16


def bitstring(bits):
    return ''.join(str(int(b)) for b in bits)


def bv_circuit(qubits: List[cirq.QubitId], a: int) -> cirq.Circuit:
    """Creates a circuit for the Bernstein-Vazirani algorithm.

    Args:
        qubits: list of qubits in the circuit.
        a: integer < 2**len(qubits), representing the unknown bit string.

    Returns:
        Result object, containing measurement data after the circuit has run.
    """
    # 1. Create a circuit (qubits start in the |0> state).
    circuit = cirq.Circuit()
    # 2. Apply Hadamard gates to the inputs.
    H_layer = [cirq.H(qubit) for qubit in qubits]
    circuit.append(H_layer)
    # 3. Apply the inner-product oracle
    O_layer = [cirq.Z(qubit) for i, qubit in enumerate(qubits) if a & (1 << i)]
    circuit.append(O_layer)
    # 4. Apply Hadamard gates to the outputs
    circuit.append(H_layer)
    # 5. Apply measurement layer
    circuit.append(cirq.ops.MeasurementGate('result').on(qubit)
                   for qubit in qubits)
    return circuit


def bv(n_qubits: int, a: int, shots: int = NUM_SHOTS) -> collections.Counter:
    """Creates and executes the circuit for Bernstein-Vazirani algorithm.

    Args:
        n_qubits: integer < 30, number of qubits in the simulated circuit.
        a: integer < 2**n_qubits, representing the unknown bit string.
        shots: number of times the circuit has been executed.

    Returns:
        Result object, containing measurement data after the circuit has run.
    """
    # Generate the circuit.
    qubits = [cirq.google.XmonQubit(0, x) for x in range(n_qubits)]
    circuit = bv_circuit(qubits, a)

    print(circuit)

    # Simulate and return results.
    simulator = cirq.google.Simulator()
    result = simulator.run(circuit, repetitions=NUM_SHOTS)
    result_bits = result.measurements['result']  # 2D array of (rep, qubit)
    result_strs = [bitstring(bits) for bits in result_bits]
    return collections.Counter(result_strs)



def main(argv):
    """Demonstrates Bernstein-Vazirani algorithm.

    Generates random number which could be represented with the given number of
    qubits and uses it as the argument for BV circuit.
    Shows that the returned measurement corresponds to the bit representation
    of the generated number.

    Args:
        argv: unused.
    """
    del argv  # Unused.
    n_qubits = NUM_QUBITS
    a = random.randrange(2**n_qubits - 1)
    a_bits = [bool(a & (1 << i)) for i in range(n_qubits)]
    a_bitstring = bitstring(a_bits)
    print('Expected bitstring: {}'.format(a_bitstring))
    results = bv(n_qubits, a)
    print('Results: {}'.format(results))
    print('Returned bitstring: ', results.most_common(1)[0])


if __name__ == '__main__':
    app.run(main)
