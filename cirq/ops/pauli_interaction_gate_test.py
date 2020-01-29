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
import sympy


import cirq


_bools = (False, True)
_paulis = (cirq.X, cirq.Y, cirq.Z)


def _all_interaction_gates(exponents=(1,)):
    for pauli0, invert0, pauli1, invert1, e in itertools.product(
            _paulis, _bools,
            _paulis, _bools,
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
            _paulis, _bools,
            _paulis, _bools,
            (0.125, -0.25, 1)):
        eq.add_equality_group(
            cirq.PauliInteractionGate(pauli0,
                                      invert0,
                                      pauli1,
                                      invert1,
                                      exponent=e))


def test_exponent_shifts_are_equal():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.PauliInteractionGate(cirq.X, False, cirq.X, False, exponent=e)
        for e in [0.1, 0.1, 2.1, -1.9, 4.1])
    eq.add_equality_group(
        cirq.PauliInteractionGate(cirq.X, True, cirq.X, False, exponent=e)
        for e in [0.1, 0.1, 2.1, -1.9, 4.1])
    eq.add_equality_group(
        cirq.PauliInteractionGate(cirq.Y, False, cirq.Z, False, exponent=e)
        for e in [0.1, 0.1, 2.1, -1.9, 4.1])
    eq.add_equality_group(
        cirq.PauliInteractionGate(cirq.Z, False, cirq.Y, True, exponent=e)
        for e in [0.1, 0.1, 2.1, -1.9, 4.1])


@pytest.mark.parametrize('gate',
                         _all_interaction_gates(exponents=(0.1, -0.25, 0.5, 1)))
def test_interchangeable_qubits(gate):
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    op0 = gate(q0, q1)
    op1 = gate(q1, q0)
    mat0 = cirq.Circuit(op0).unitary()
    mat1 = cirq.Circuit(op1).unitary()
    same = op0 == op1
    same_check = cirq.allclose_up_to_global_phase(mat0, mat1)
    assert same == same_check


def test_exponent():
    cnot = cirq.PauliInteractionGate(cirq.Z, False, cirq.X, False)
    np.testing.assert_almost_equal(
        cirq.unitary(cnot**0.5),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5+0.5j, 0.5-0.5j],
            [0, 0, 0.5-0.5j, 0.5+0.5j],
        ]))


def test_repr():
    cnot = cirq.PauliInteractionGate(cirq.Z, False, cirq.X, False)
    cirq.testing.assert_equivalent_repr(cnot)


def test_decomposes_despite_symbol():
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    gate = cirq.PauliInteractionGate(cirq.Z, False, cirq.X, False,
                                     exponent=sympy.Symbol('x'))
    assert cirq.decompose_once_with_qubits(gate, [q0, q1])


def test_text_diagrams():
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    circuit = cirq.Circuit(
        cirq.PauliInteractionGate(cirq.X, False, cirq.X, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.X, True, cirq.X, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.X, False, cirq.X, True)(q0, q1),
        cirq.PauliInteractionGate(cirq.X, True, cirq.X, True)(q0, q1),
        cirq.PauliInteractionGate(cirq.X, False, cirq.Y, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.Y, False, cirq.Z, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.Z, False, cirq.Y, False)(q0, q1),
        cirq.PauliInteractionGate(cirq.Y, True, cirq.Z, True)(q0, q1),
        cirq.PauliInteractionGate(cirq.Z, True, cirq.Y, True)(q0, q1))
    assert circuit.to_text_diagram().strip() == """
q0: ───X───(-X)───X──────(-X)───X───Y───@───(-Y)───(-@)───
       │   │      │      │      │   │   │   │      │
q1: ───X───X──────(-X)───(-X)───Y───@───Y───(-@)───(-Y)───
    """.strip()
