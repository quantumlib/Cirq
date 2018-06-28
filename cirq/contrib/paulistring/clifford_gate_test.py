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
    EqualsTester,
    assert_allclose_up_to_global_phase,
    allclose_up_to_global_phase
)

import cirq

from cirq.contrib.paulistring import Pauli
from cirq.contrib.paulistring import (
    CliffordGate,
    PauliTransform,
)


_bools = (False, True)


def _assert_not_mirror(gate) -> None:
    trans_x = gate.transform(Pauli.X)
    trans_y = gate.transform(Pauli.Y)
    trans_z = gate.transform(Pauli.Z)
    right_handed = (trans_x.flip ^ trans_y.flip ^ trans_z.flip ^
                   (trans_x.to - trans_y.to != 1))
    assert right_handed, 'Mirrors'

def _assert_no_collision(gate) -> None:
    trans_x = gate.transform(Pauli.X)
    trans_y = gate.transform(Pauli.Y)
    trans_z = gate.transform(Pauli.Z)
    assert trans_x.to != trans_y.to, 'Collision'
    assert trans_y.to != trans_z.to, 'Collision'
    assert trans_z.to != trans_x.to, 'Collision'


def _all_rotations_xz():
    for px, flip_x, pz, flip_z in itertools.product(Pauli.XYZ, _bools,
                                                    Pauli.XYZ, _bools):
        if px == pz:
            continue
        yield PauliTransform(px, flip_x), PauliTransform(pz, flip_z)

def _all_clifford_gates():
    for trans_x, trans_z in _all_rotations_xz():
        yield CliffordGate(trans_x, trans_z)


@pytest.mark.parametrize('pauli,flip_x,flip_z',
    itertools.product(Pauli.XYZ, _bools, _bools))
def test_init_value_error(pauli, flip_x, flip_z):
    with pytest.raises(ValueError):
        CliffordGate((pauli, flip_x), (pauli, flip_z))

@pytest.mark.parametrize('trans_x,trans_z', _all_rotations_xz())
def test_init_success(trans_x, trans_z):
    gate = CliffordGate(trans_x, trans_z)
    assert gate.transform(Pauli.X) == trans_x
    assert gate.transform(Pauli.Z) == trans_z
    assert gate.rotates_pauli_to(Pauli.X) == trans_x.to
    assert gate.rotates_pauli_to(Pauli.Z) == trans_z.to
    assert gate.flips_pauli(Pauli.X) == trans_x.flip
    assert gate.flips_pauli(Pauli.Z) == trans_z.flip
    _assert_not_mirror(gate)
    _assert_no_collision(gate)

def test_eq_ne_and_hash():
    eq = EqualsTester()
    eq.add_equality_group(Pauli.X)
    for trans_x, trans_z in _all_rotations_xz():
        gate1 = CliffordGate(trans_x, trans_z)
        gate2 = CliffordGate(trans_x, trans_z)
        eq.add_equality_group(gate1, gate2)

@pytest.mark.parametrize('gate,rep', (
    (CliffordGate.I,       'CliffordGate(X:+X, Y:+Y, Z:+Z)'),
    (CliffordGate.H,       'CliffordGate(X:+Z, Y:-Y, Z:+X)'),
    (CliffordGate.X,       'CliffordGate(X:+X, Y:-Y, Z:-Z)'),
    (CliffordGate.X_sqrt,  'CliffordGate(X:+X, Y:+Z, Z:-Y)')))
def test_repr(gate, rep):
    assert repr(gate) == rep

@pytest.mark.parametrize('gate,trans_y', (
    (CliffordGate.I,       (Pauli.Y, False)),
    (CliffordGate.H,       (Pauli.Y, True)),
    (CliffordGate.X,       (Pauli.Y, True)),
    (CliffordGate.Y,       (Pauli.Y, False)),
    (CliffordGate.Z,       (Pauli.Y, True)),
    (CliffordGate.X_sqrt,  (Pauli.Z, False)),
    (CliffordGate.X_nsqrt, (Pauli.Z, True)),
    (CliffordGate.Y_sqrt,  (Pauli.Y, False)),
    (CliffordGate.Y_nsqrt, (Pauli.Y, False)),
    (CliffordGate.Z_sqrt,  (Pauli.X, True)),
    (CliffordGate.Z_nsqrt, (Pauli.X, False))))
def test_y_rotation(gate, trans_y):
    assert gate.transform(Pauli.Y) == trans_y

@pytest.mark.parametrize('gate,gate_equiv', (
    (CliffordGate.I,       cirq.X ** 0),
    (CliffordGate.H,       cirq.H),
    (CliffordGate.X,       cirq.X),
    (CliffordGate.Y,       cirq.Y),
    (CliffordGate.Z,       cirq.Z),
    (CliffordGate.X_sqrt,  cirq.X ** 0.5),
    (CliffordGate.X_nsqrt, cirq.X ** -0.5),
    (CliffordGate.Y_sqrt,  cirq.Y ** 0.5),
    (CliffordGate.Y_nsqrt, cirq.Y ** -0.5),
    (CliffordGate.Z_sqrt,  cirq.Z ** 0.5),
    (CliffordGate.Z_nsqrt, cirq.Z ** -0.5)))
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
    assert gate == gate.inverse().inverse()

@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_inverse_matrix(gate):
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
                      Pauli.XYZ,
                      (0.1, 0.25, 0.5, 1, -0.5)))
def test_commutes_with_pauli(gate, pauli, half_turns):
    pauli_gates = {Pauli.X: cirq.X,
                   Pauli.Y: cirq.Y,
                   Pauli.Z: cirq.Z}
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
    (CliffordGate.I,       'I'),
    (CliffordGate.H,       'H'),
    (CliffordGate.X,       'X'),
    (CliffordGate.X_sqrt,  'X'),
    (CliffordGate.X_nsqrt, 'X'),
    (CliffordGate((Pauli.Y, False), (Pauli.X, True)), 'X^-0.5-Z^0.5')))
def test_text_diagram_wire_symbols(gate, sym):
    assert gate.text_diagram_wire_symbols() == (sym,)

@pytest.mark.parametrize('gate,exp', (
    (CliffordGate.I,       1),
    (CliffordGate.H,       1),
    (CliffordGate.X,       1),
    (CliffordGate.X_sqrt,  0.5),
    (CliffordGate.X_nsqrt, -0.5),
    (CliffordGate((Pauli.Y, False), (Pauli.X, True)), 1)))
def test_text_diagram_exponent(gate, exp):
    assert gate.text_diagram_exponent() == exp

