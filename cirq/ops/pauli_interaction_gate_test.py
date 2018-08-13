# Copyright 2018 The Cirq Developers
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

import itertools
import pytest
import numpy as np
from cirq.testing import (
    EqualsTester,
    assert_allclose_up_to_global_phase,
)

import cirq


_bools = (False, True)

def _all_interaction_gates(half_turns_list=(1,)):
    for pauli0, invert0, pauli1, invert1, half_turns in itertools.product(
                                                        cirq.Pauli.XYZ, _bools,
                                                        cirq.Pauli.XYZ, _bools,
                                                        half_turns_list):
        yield cirq.PauliInteractionGate(pauli0, invert0,
                                        pauli1, invert1,
                                        half_turns=half_turns)


def test_eq_ne_and_hash():
    eq = EqualsTester()
    for pauli0, invert0, pauli1, invert1, half_turns in itertools.product(
                                                        cirq.Pauli.XYZ, _bools,
                                                        cirq.Pauli.XYZ, _bools,
                                                        (0.1, -0.25, 1)):
        gate_gen = lambda offset: cirq.PauliInteractionGate(
                                        pauli0, invert0,
                                        pauli1, invert1,
                                        half_turns=half_turns + offset)
        eq.add_equality_group(gate_gen(0), gate_gen(0), gate_gen(2),
                              gate_gen(-4), gate_gen(-2))

@pytest.mark.parametrize('gate',
    _all_interaction_gates(half_turns_list=(0.1, -0.25, 0.5, 1)))
def test_interchangable_qubits(gate):
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    op0 = gate(q0, q1)
    op1 = gate(q1, q0)
    mat0 = cirq.Circuit.from_ops(
                    op0,
                ).to_unitary_matrix()
    mat1 = cirq.Circuit.from_ops(
                    op1,
                ).to_unitary_matrix()
    same = op0 == op1
    same_check = cirq.allclose_up_to_global_phase(mat0, mat1)
    assert same == same_check

@pytest.mark.parametrize('gate',
    _all_interaction_gates(half_turns_list=(0.1, -0.25, 0.5, 1)))
def test_decompose(gate):
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    circuit = cirq.Circuit.from_ops(
                    gate(q0, q1))
    cirq.ExpandComposite().optimize_circuit(circuit)
    decompose_mat = circuit.to_unitary_matrix()
    gate_mat = gate.matrix()
    assert_allclose_up_to_global_phase(decompose_mat, gate_mat,
                                       rtol=1e-7, atol=1e-7)

def test_exponent():
    cnot = cirq.PauliInteractionGate(cirq.Pauli.Z, False, cirq.Pauli.X, False)
    np.testing.assert_almost_equal(
        (cnot**0.5).matrix(),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5+0.5j, 0.5-0.5j],
            [0, 0, 0.5-0.5j, 0.5+0.5j],
        ]))

    # Matrix must be consistent with decomposition.
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    g = cnot**0.25
    cirq.testing.assert_allclose_up_to_global_phase(
        g.matrix(),
        cirq.Circuit.from_ops(g.default_decompose([q0, q1])
                              ).to_unitary_matrix(),
        rtol=1e-7, atol=1e-7)

def test_decomposes_despite_symbol():
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    gate = cirq.PauliInteractionGate(cirq.Pauli.Z, False, cirq.Pauli.X, False,
                                     half_turns=cirq.Symbol('x'))
    op_tree = gate.default_decompose([q0, q1])
    ops = tuple(cirq.flatten_op_tree(op_tree))
    assert ops

def test_text_diagrams():
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    circuit = cirq.Circuit.from_ops(
        cirq.PauliInteractionGate(cirq.Pauli.X, False,
                                  cirq.Pauli.X, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.Pauli.X, True,
                                  cirq.Pauli.X, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.Pauli.X, False,
                                  cirq.Pauli.X, True)(q0, q1),
        cirq.PauliInteractionGate(cirq.Pauli.X, True,
                                  cirq.Pauli.X, True)(q0, q1),
        cirq.PauliInteractionGate(cirq.Pauli.X, False,
                                  cirq.Pauli.Y, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.Pauli.Y, False,
                                  cirq.Pauli.Z, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.Pauli.Z, False,
                                  cirq.Pauli.Y, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.Pauli.Y, True,
                                  cirq.Pauli.Z, True)(q0, q1),
        cirq.PauliInteractionGate(cirq.Pauli.Z, True,
                                  cirq.Pauli.Y, True)(q0, q1))
    assert circuit.to_text_diagram().strip() == """
q0: ───X───(-X)───X──────(-X)───X───Y───@───(-Y)───(-@)───
       │   │      │      │      │   │   │   │      │
q1: ───X───X──────(-X)───(-X)───Y───@───Y───(-@)───(-Y)───
    """.strip()

@pytest.mark.parametrize('gate,gate_repr', (
    (cirq.PauliInteractionGate(cirq.Pauli.X, False, cirq.Pauli.X, False),
     'cirq.PauliInteractionGate(+X, +X)'),
    (cirq.PauliInteractionGate(cirq.Pauli.X, True,  cirq.Pauli.X, False),
     'cirq.PauliInteractionGate(-X, +X)'),
    (cirq.PauliInteractionGate(cirq.Pauli.X, False, cirq.Pauli.X, True ),
     'cirq.PauliInteractionGate(+X, -X)'),
    (cirq.PauliInteractionGate(cirq.Pauli.X, True,  cirq.Pauli.X, True ),
     'cirq.PauliInteractionGate(-X, -X)'),
    (cirq.PauliInteractionGate(cirq.Pauli.X, False, cirq.Pauli.Y, False),
     'cirq.PauliInteractionGate(+X, +Y)'),
    (cirq.PauliInteractionGate(cirq.Pauli.Y, False, cirq.Pauli.Z, False),
     'cirq.PauliInteractionGate(+Y, +Z)'),
    (cirq.PauliInteractionGate(cirq.Pauli.Z, False, cirq.Pauli.Y, False),
     'cirq.PauliInteractionGate(+Z, +Y)'),
    (cirq.PauliInteractionGate(cirq.Pauli.Y, True,  cirq.Pauli.Z, True ),
     'cirq.PauliInteractionGate(-Y, -Z)'),
    (cirq.PauliInteractionGate(cirq.Pauli.Z, True,  cirq.Pauli.Y, True ),
     'cirq.PauliInteractionGate(-Z, -Y)')))
def test_repr(gate, gate_repr):
    assert repr(gate) == gate_repr
