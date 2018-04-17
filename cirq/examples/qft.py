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

from absl import app

import cirq
import numpy as np

GRID_LENGTH = 2
GRID_WIDTH = 2

def cz_and_swap(q0, q1, rot):
    yield cirq.CZ(q0, q1)**rot
    yield cirq.SWAP(q0,q1)


def qft_circuit():

    # Define a sequence of qubits.

    a,b,c,d = [cirq.google.XmonQubit(x, y) for x in range(GRID_LENGTH) for y in range(GRID_WIDTH)]

    # Create a qft circuit for 2*2 planar qubit architecture.
    circuit = cirq.Circuit.from_ops(
        cirq.H(a),
        cz_and_swap(a, b, 0.5),
        cz_and_swap(b, c, 0.25),
        cirq.H(a),
        cz_and_swap(a, b, 0.5),
        cz_and_swap(c, d, 0.125),
        cz_and_swap(b, c, 0.25),
        cirq.H(a),
        cz_and_swap(a, b, 0.5),
        cirq.H(a),
    )

    # Debug step
    print(circuit)

    return circuit


def main(argv):
    """Demonstrates Quantum Fourier transform.
    Args:
      argv: unused.
    """
    del argv  # Unused.

    qft_circuit = qft_circuit()

    # Run and collect results
    simulator = cirq.google.Simulator()
    result = simulator.run(qft_circuit)
    print(np.around(result.final_states[0], 3))


if __name__ == '__main__':
    app.run(main)
