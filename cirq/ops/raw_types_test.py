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

import pytest

import cirq


def test_gate_calls_validate():
    class ValiGate(cirq.Gate):
        def validate_args(self, qubits):
            if len(qubits) == 3:
                raise ValueError()

    g = ValiGate()
    q00 = cirq.QubitId()
    q01 = cirq.QubitId()
    q10 = cirq.QubitId()

    _ = g.on(q00)
    _ = g.on(q01)
    _ = g.on(q00, q10)
    with pytest.raises(ValueError):
        _ = g.on(q00, q10, q01)

    _ = g(q00)
    _ = g(q00, q10)
    with pytest.raises(ValueError):
        _ = g(q10, q01, q00)


def test_named_qubit_str():
    q = cirq.NamedQubit('a')
    assert q.name == 'a'
    assert str(q) == 'a'


# Python 2 gives a different repr due to unicode strings being prefixed with u.
@cirq.testing.only_test_in_python3
def test_named_qubit_repr():
    q = cirq.NamedQubit('a')
    assert repr(q) == "NamedQubit('a')"


def test_operation_init():
    q = cirq.QubitId()
    g = cirq.Gate()
    v = cirq.Operation(g, (q,))
    assert v.gate == g
    assert v.qubits == (q,)


def test_operation_eq():
    g1 = cirq.Gate()
    g2 = cirq.Gate()
    r1 = [cirq.QubitId()]
    r2 = [cirq.QubitId()]
    r12 = r1 + r2
    r21 = r2 + r1

    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.Operation(g1, r1))
    eq.make_equality_group(lambda: cirq.Operation(g2, r1))
    eq.make_equality_group(lambda: cirq.Operation(g1, r2))
    eq.make_equality_group(lambda: cirq.Operation(g1, r12))
    eq.make_equality_group(lambda: cirq.Operation(g1, r21))
    eq.add_equality_group(cirq.Operation(cirq.CZ, r21),
                          cirq.Operation(cirq.CZ, r12))

    # Interchangeable subsets.

    class PairGate(cirq.Gate, cirq.InterchangeableQubitsGate):
        def qubit_index_to_equivalence_group_key(self, index: int):
            return index // 2

    p = PairGate()
    a0, a1, b0, b1, c0 = cirq.LineQubit.range(5)
    eq.add_equality_group(p(a0, a1, b0, b1), p(a1, a0, b1, b0))
    eq.add_equality_group(p(b0, b1, a0, a1))
    eq.add_equality_group(p(a0, a1, b0, b1, c0), p(a1, a0, b1, b0, c0))
    eq.add_equality_group(p(a0, b0, a1, b1, c0))
    eq.add_equality_group(p(a0, c0, b0, b1, a1))
    eq.add_equality_group(p(b0, a1, a0, b1, c0))


def test_operation_pow():
    Y = cirq.Y
    qubit = cirq.QubitId()
    assert (Y ** 0.5)(qubit) == Y(qubit) ** 0.5
