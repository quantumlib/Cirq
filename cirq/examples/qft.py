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
from absl import flags

import cirq
import numpy as np

FLAGS = flags.FLAGS
GRID_LENGTH = 2
GRID_WIDTH = 2

def cz_and_swap(q0, q1, rot):
  yield (cirq.ops.CZ**(rot)).on(q0, q1)
  yield cirq.ops.SWAP(q0,q1)


def qft_circuit(grid_length, grid_width):

  # Define a sequence of qubits.
  qubits = [cirq.google.XmonQubit(x, y) for x in range(grid_length) for y in range(grid_width)]

  # Create a qft circuit for 2*2 planar qubit architecture.
  circuit = cirq.circuits.Circuit()
  circuit.append((cirq.ops.H).on(qubits[0]))
  circuit.append(cz_and_swap(qubits[0],qubits[1], 0.5))
  circuit.append(cz_and_swap(qubits[1],qubits[2], 0.25))
  circuit.append((cirq.ops.H).on(qubits[0]))
  circuit.append(cz_and_swap(qubits[0],qubits[1], 0.5))
  circuit.append(cz_and_swap(qubits[2],qubits[3], 0.125))
  circuit.append(cz_and_swap(qubits[1],qubits[2], 0.25))
  circuit.append((cirq.ops.H).on(qubits[0]))
  circuit.append(cz_and_swap(qubits[0],qubits[1], 0.5))
  circuit.append((cirq.ops.H).on(qubits[0]))

  # Debug step
  print(circuit.to_text_diagram())

  # Run and collect results
  simulator = cirq.google.Simulator()
  result = simulator.run(circuit)
  print(np.around(result.final_states[0], 3))


def main(argv):
  """Demonstrates Quantum Fourier transform.
  Args:
    argv: unused.
  """
  del argv  # Unused.
  qft_circuit(GRID_LENGTH, GRID_WIDTH)


if __name__ == '__main__':
  app.run(main)
