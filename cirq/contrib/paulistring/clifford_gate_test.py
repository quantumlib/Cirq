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
from cirq.testing import (
    assert_allclose_up_to_global_phase,
    allclose_up_to_global_phase
)

import cirq

from cirq.contrib.paulistring import PAULI_X, PAULI_Y, PAULI_Z
from cirq.contrib.paulistring import (
    CliffordGate,
    PauliTransform,
    CLIFFORD_I,
    CLIFFORD_H,
    CLIFFORD_X,
    CLIFFORD_Y,
    CLIFFORD_Z,
    CLIFFORD_X_sqrt,
    CLIFFORD_X_nsqrt,
    CLIFFORD_Y_sqrt,
    CLIFFORD_Y_nsqrt,
    CLIFFORD_Z_sqrt,
    CLIFFORD_Z_nsqrt,
)


_paulis = (PAULI_X, PAULI_Y, PAULI_Z)
_bools = (False, True)


def _assert_not_mirror(gate) -> None:
    trans_x = gate.transform(PAULI_X)
    trans_y = gate.transform(PAULI_Y)
    trans_z = gate.transform(PAULI_Z)
    right_handed = (trans_x.flip ^ trans_y.flip ^ trans_z.flip ^
                   (trans_x.to - trans_y.to != 1))
    assert right_handed, 'Mirrors'

def _assert_no_collision(gate) -> None:
    trans_x = gate.transform(PAULI_X)
    trans_y = gate.transform(PAULI_Y)
    trans_z = gate.transform(PAULI_Z)
    assert trans_x.to != trans_y.to, 'Collision'
    assert trans_y.to != trans_z.to, 'Collision'
    assert trans_z.to != trans_x.to, 'Collision'


def _all_rotations_xz():
    for px, flip_x, pz, flip_z in itertools.product(_paulis, _bools,
                                                    _paulis, _bools):
        if px == pz:
            continue
        yield PauliTransform(px, flip_x), PauliTransform(pz, flip_z)

def _all_clifford_gates():
    for trans_x, trans_z in _all_rotations_xz():
        yield CliffordGate(trans_x, trans_z)


@pytest.mark.parametrize('pauli,flip_x,flip_z',
    itertools.product(_paulis, _bools, _bools))
def test_init_value_error(pauli, flip_x, flip_z):
    with pytest.raises(ValueError):
        CliffordGate((pauli, flip_x), (pauli, flip_z))

@pytest.mark.parametrize('trans_x,trans_z', _all_rotations_xz())
def test_init_success(trans_x, trans_z):
    gate = CliffordGate(trans_x, trans_z)
    assert gate.transform(PAULI_X) == trans_x
    assert gate.transform(PAULI_Z) == trans_z
    assert gate.rotates_pauli_to(PAULI_X) == trans_x.to
    assert gate.rotates_pauli_to(PAULI_Z) == trans_z.to
    assert gate.flips_pauli(PAULI_X) == trans_x.flip
    assert gate.flips_pauli(PAULI_Z) == trans_z.flip
    _assert_not_mirror(gate)
    _assert_no_collision(gate)

@pytest.mark.parametrize('trans_x,trans_z', _all_rotations_xz())
def test_eq_ne_and_hash(trans_x, trans_z):
    gate = CliffordGate(trans_x, trans_z)
    gate_eq = CliffordGate(trans_x, trans_z)
    assert gate == gate_eq
    assert not gate != gate_eq
    assert hash(gate) == hash(gate_eq)

    for trans_x2, trans_z2 in _all_rotations_xz():
        if trans_x == trans_x2 and trans_z == trans_z2:
            continue
        gate2 = CliffordGate(trans_x2, trans_z2)
        assert not gate == gate2
        assert gate != gate2
        assert hash(gate) != hash(gate2)

def test_eq_other_type():
    assert not CLIFFORD_X == object()

@pytest.mark.parametrize('gate,rep', (
    (CLIFFORD_I,       'CliffordGate(X:+X, Y:+Y, Z:+Z)'),
    (CLIFFORD_H,       'CliffordGate(X:+Z, Y:-Y, Z:+X)'),
    (CLIFFORD_X,       'CliffordGate(X:+X, Y:-Y, Z:-Z)'),
    (CLIFFORD_X_sqrt,  'CliffordGate(X:+X, Y:+Z, Z:-Y)')))
def test_repr(gate, rep):
    assert repr(gate) == rep

@pytest.mark.parametrize('gate,trans_y', (
    (CLIFFORD_I,       (PAULI_Y, False)),
    (CLIFFORD_H,       (PAULI_Y, True)),
    (CLIFFORD_X,       (PAULI_Y, True)),
    (CLIFFORD_Y,       (PAULI_Y, False)),
    (CLIFFORD_Z,       (PAULI_Y, True)),
    (CLIFFORD_X_sqrt,  (PAULI_Z, False)),
    (CLIFFORD_X_nsqrt, (PAULI_Z, True)),
    (CLIFFORD_Y_sqrt,  (PAULI_Y, False)),
    (CLIFFORD_Y_nsqrt, (PAULI_Y, False)),
    (CLIFFORD_Z_sqrt,  (PAULI_X, True)),
    (CLIFFORD_Z_nsqrt, (PAULI_X, False))))
def test_y_rotation(gate, trans_y):
    assert gate.transform(PAULI_Y) == trans_y

@pytest.mark.parametrize('gate,gate_equiv', (
    (CLIFFORD_I,       cirq.X ** 0),
    (CLIFFORD_H,       cirq.H),
    (CLIFFORD_X,       cirq.X),
    (CLIFFORD_Y,       cirq.Y),
    (CLIFFORD_Z,       cirq.Z),
    (CLIFFORD_X_sqrt,  cirq.X ** 0.5),
    (CLIFFORD_X_nsqrt, cirq.X ** -0.5),
    (CLIFFORD_Y_sqrt,  cirq.Y ** 0.5),
    (CLIFFORD_Y_nsqrt, cirq.Y ** -0.5),
    (CLIFFORD_Z_sqrt,  cirq.Z ** 0.5),
    (CLIFFORD_Z_nsqrt, cirq.Z ** -0.5)))
def test_decompose(gate, gate_equiv):
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit.from_ops(
                    gate(q0),
                ).to_unitary_matrix()
    mat_check = cirq.Circuit.from_ops(
                    gate_equiv(q0),
                ).to_unitary_matrix()
    assert_allclose_up_to_global_phase(mat, mat_check)

@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_inverse(gate):
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit.from_ops(gate(q0)).to_unitary_matrix()
    mat_inv = cirq.Circuit.from_ops(gate.inverse()(q0)).to_unitary_matrix()
    assert_allclose_up_to_global_phase(mat, mat_inv.T.conj())

@pytest.mark.parametrize('gate,other',
    itertools.product(_all_clifford_gates(),
                      _all_clifford_gates()))
def test_commutes_with_single_qubit_gate(gate, other):
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit.from_ops(
                    gate(q0),
                    other(q0),
                ).to_unitary_matrix()
    mat_swap = cirq.Circuit.from_ops(
                    other(q0),
                    gate(q0),
                ).to_unitary_matrix()
    commutes = gate.commutes_with_single_qubit_gate(other)
    commutes_check = allclose_up_to_global_phase(mat, mat_swap)
    assert commutes == commutes_check

@pytest.mark.parametrize('gate,pauli,half_turns',
    itertools.product(_all_clifford_gates(),
                      _paulis,
                      (0.1, 0.25, 0.5, 1, -0.5)))
def test_commutes_with_pauli(gate, pauli, half_turns):
    pauli_gates = {PAULI_X: cirq.X,
                   PAULI_Y: cirq.Y,
                   PAULI_Z: cirq.Z}
    pauli_gate = pauli_gates[pauli] ** half_turns
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit.from_ops(
                    gate(q0),
                    pauli_gate(q0),
                ).to_unitary_matrix()
    mat_swap = cirq.Circuit.from_ops(
                    pauli_gate(q0),
                    gate(q0),
                ).to_unitary_matrix()
    commutes = gate.commutes_with_pauli(pauli, whole=half_turns == 1)
    commutes_check = allclose_up_to_global_phase(mat, mat_swap)
    assert commutes == commutes_check

@pytest.mark.parametrize('gate,other',
    itertools.product(_all_clifford_gates(),
                      _all_clifford_gates()))
def test_single_qubit_gate_after_switching_order(gate, other):
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit.from_ops(
                    gate(q0),
                    other(q0),
                ).to_unitary_matrix()
    mat_swap = cirq.Circuit.from_ops(
                    gate.single_qubit_gate_after_switching_order(other)(q0),
                    gate(q0),
                ).to_unitary_matrix()
    assert_allclose_up_to_global_phase(mat, mat_swap)

@pytest.mark.parametrize('gate,sym', (
    (CLIFFORD_I,       'I'),
    (CLIFFORD_H,       'H'),
    (CLIFFORD_X,       'X'),
    (CLIFFORD_X_sqrt,  'X'),
    (CLIFFORD_X_nsqrt, 'X'),
    (CliffordGate((PAULI_Y, False), (PAULI_X, True)), 'X^-0.5-Z^0.5')))
def test_text_diagram_wire_symbols(gate, sym):
    assert gate.text_diagram_wire_symbols() == (sym,)

@pytest.mark.parametrize('gate,exp', (
    (CLIFFORD_I,       1),
    (CLIFFORD_H,       1),
    (CLIFFORD_X,       1),
    (CLIFFORD_X_sqrt,  0.5),
    (CLIFFORD_X_nsqrt, -0.5),
    (CliffordGate((PAULI_Y, False), (PAULI_X, True)), 1)))
def test_text_diagram_exponent(gate, exp):
    assert gate.text_diagram_exponent() == exp

