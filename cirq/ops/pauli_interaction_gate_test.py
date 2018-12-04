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

import cirq


_bools = (False, True)


def _all_interaction_gates(exponents=(1,)):
    for pauli0, invert0, pauli1, invert1, e in itertools.product(
            cirq.Pauli.XYZ, _bools,
            cirq.Pauli.XYZ, _bools,
            exponents):
        yield cirq.PauliInteractionGate(pauli0, invert0,
                                        pauli1, invert1,
                                        exponent=e)


@pytest.mark.parametrize('gate',
                         _all_interaction_gates())
def test_pauli_interaction_gates_consistent_protocols(gate):
    cirq.testing.assert_implements_consistent_protocols(gate)


def test_eq_ne_and_hash():
    eq = cirq.testing.EqualsTester()
    for pauli0, invert0, pauli1, invert1, e in itertools.product(
            cirq.Pauli.XYZ, _bools,
            cirq.Pauli.XYZ, _bools,
            (0.125, -0.25, 1)):
        def gate_gen(offset):
            return cirq.PauliInteractionGate(
                pauli0, invert0,
                pauli1, invert1,
                exponent=e + offset)
        eq.add_equality_group(gate_gen(0), gate_gen(0), gate_gen(2),
                              gate_gen(-4), gate_gen(-2))


@pytest.mark.parametrize('gate',
                         _all_interaction_gates(exponents=(0.1, -0.25, 0.5, 1)))
def test_interchangeable_qubits(gate):
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


def test_exponent():
    cnot = cirq.PauliInteractionGate(cirq.Pauli.Z, False, cirq.Pauli.X, False)
    np.testing.assert_almost_equal(
        cirq.unitary(cnot**0.5),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5+0.5j, 0.5-0.5j],
            [0, 0, 0.5-0.5j, 0.5+0.5j],
        ]))


def test_decomposes_despite_symbol():
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    gate = cirq.PauliInteractionGate(cirq.Pauli.Z, False, cirq.Pauli.X, False,
                                     exponent=cirq.Symbol('x'))
    assert cirq.decompose_once_with_qubits(gate, [q0, q1])


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
