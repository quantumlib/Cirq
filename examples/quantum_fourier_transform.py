# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Creates and simulates a circuit for Quantum Fourier Transform(QFT) on 4 qubits.

In this example we demonstrate Fourier Transform on
(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) vector. To do the same, we prepare the input state of the
qubits as |0000>.
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

from __future__ import annotations

import numpy as np

import cirq


def main():
    """Demonstrates Quantum Fourier transform."""
    # Create circuit
    qft_circuit = generate_2x2_grid_qft_circuit()
    print('Circuit:')
    print(qft_circuit)
    # Simulate and collect final_state
    simulator = cirq.Simulator()
    result = simulator.simulate(qft_circuit)
    print()
    print('FinalState')
    print(np.around(result.final_state_vector, 3))


def _cz_and_swap(q0, q1, rot):
    yield cirq.CZ(q0, q1) ** rot
    yield cirq.SWAP(q0, q1)


# Create a quantum fourier transform circuit for 2*2 planar qubit architecture.
# Circuit is adopted from https://arxiv.org/pdf/quant-ph/0402196.pdf
def generate_2x2_grid_qft_circuit():
    # Define a 2*2 square grid of qubits.
    a, b, c, d = [
        cirq.GridQubit(0, 0),
        cirq.GridQubit(0, 1),
        cirq.GridQubit(1, 1),
        cirq.GridQubit(1, 0),
    ]

    circuit = cirq.Circuit(
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
        strategy=cirq.InsertStrategy.EARLIEST,
    )
    return circuit


if __name__ == '__main__':
    main()
