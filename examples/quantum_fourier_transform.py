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
"""
Creates and simulates a circuit for Quantum Fourier Transform(QFT) 
on a 4 qubit system.

In this example we demonstrate Fourier Transform on   
(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) vector. To do the same, we prepare
the input state of the qubit's as |0000>.
=== EXAMPLE OUTPUT ===

Circuit:
(0, 0): ─H───@^0.5───×───H────────────@^0.5─────×───H──────────@^0.5──×─H
             │       │                │         │               │     │
(0, 1): ─────@───────×───@^0.25───×───@─────────×───@^0.25───×──@─────×──
                         │        │                 │        │
(1, 0): ─────────────────┼────────┼───@^0.125───×───┼────────┼───────────
                         │        │   │         │   │        │
(1, 1): ─────────────────@────────×───@─────────×───@────────×───────────

FinalState
[0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j
 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j]
"""

import numpy as np

import cirq

def main():
    """Demonstrates Quantum Fourier transform.
    """
    # Create circuit
    qft_circuit = generate_2x2_grid_qft_circuit()

    print('Circuit:')
    print(qft_circuit)

    # Simulate and collect final_state
    simulator = cirq.google.XmonSimulator()
    result = simulator.simulate(qft_circuit)
    
    print()
    print('FinalState')
    print(np.around(result.final_state, 3))
    
def _cz_and_swap(q0, q1, rot):
    yield cirq.CZ(q0, q1)**rot
    yield cirq.SWAP(q0,q1)

# Create a quantum fourier transform circuit for 2*2 planar qubit architecture.
# Circuit is adopted from https://arxiv.org/pdf/quant-ph/0402196.pdf
def generate_2x2_grid_qft_circuit():
    # Define a 2*2 square grid of qubits.

    a,b,c,d = [cirq.google.GridQubit(0, 0), cirq.google.GridQubit(0, 1),
               cirq.google.GridQubit(1, 1), cirq.google.GridQubit(1, 0)]

    circuit = cirq.Circuit.from_ops(
        cirq.H(a),
        _cz_and_swap(a, b, 0.5),
        _cz_and_swap(b, c, 0.25),
        _cz_and_swap(c, d, 0.125),
        cirq.H(a),
        _cz_and_swap(a, b, 0.5),
        _cz_and_swap(b, c, 0.25),
        cirq.H(a),
        _cz_and_swap(a, b, 0.5),
        cirq.H(a),
        strategy=cirq.InsertStrategy.EARLIEST
    )
    return circuit

if __name__ == '__main__':
    main()
