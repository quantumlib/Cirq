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

"""Tests for the expand composite optimization pass."""

from cirq.circuits import (Circuit, DropEmptyMoments, ExpandComposite,
                           InsertStrategy, Moment)
from cirq.extension import Extensions
from cirq.ops import CNOT, CNotGate, CompositeGate, CZ, QubitId, SWAP, X, Y, Z


def assert_equal_mod_empty(expected, actual):
    drop_empty = DropEmptyMoments()
    drop_empty.optimize_circuit(actual)
    if expected != actual:
        print("expected: ", expected)
        print("actual: ", actual)
    assert expected == actual


def test_empty_circuit():
    circuit = Circuit()
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    assert_equal_mod_empty(Circuit(), circuit)


def test_empty_moment():
    circuit = Circuit([])
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    assert_equal_mod_empty(Circuit([]), circuit)


def test_ignore_non_composite():
    q0, q1 = QubitId(), QubitId()
    circuit = Circuit()
    circuit.append([X(q0), Y(q1), CZ(q0, q1), Z(q0)])
    expected = Circuit(circuit.moments)
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    assert_equal_mod_empty(expected, circuit)


def test_composite_default():
    q0, q1 = QubitId(), QubitId()
    cnot = CNOT(q0, q1)
    circuit = Circuit()
    circuit.append(cnot)
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = Circuit()
    expected.append([(Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1)])
    assert_equal_mod_empty(expected, circuit)


def test_multiple_composite_default():
    q0, q1 = QubitId(), QubitId()
    cnot = CNOT(q0, q1)
    circuit = Circuit()
    circuit.append([cnot, cnot])
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = Circuit()
    decomp = [(Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1)]
    expected.append([decomp, decomp])
    assert_equal_mod_empty(expected, circuit)


def test_mix_composite_non_composite():
    q0, q1 = QubitId(), QubitId()
    cnot = CNOT(q0, q1)
    circuit = Circuit()
    circuit.append([X(q0), cnot, X(q1)])
    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = Circuit()
    expected.append(
        [X(q0), (Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1),
         X(q1)])
    assert_equal_mod_empty(expected, circuit)


def test_recursive_composite():
    q0, q1 = QubitId(), QubitId()
    swap = SWAP(q0, q1)
    circuit = Circuit()
    circuit.append(swap)

    opt = ExpandComposite()
    opt.optimize_circuit(circuit)
    expected = Circuit()
    expected.append([(Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1)])
    expected.append([(Y ** -0.5)(q0), CZ(q1, q0), (Y ** 0.5)(q0)],
                    strategy=InsertStrategy.INLINE)
    expected.append([(Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1)],
                    strategy=InsertStrategy.INLINE)
    assert_equal_mod_empty(expected, circuit)


class OtherCNot(CNotGate):
    
    def default_decompose(self, qubits):
        c, t = qubits
        yield Z(c)
        yield (Y**-0.5)(t)
        yield CZ(c, t)
        yield (Y**0.5)(t)
        yield Z(c)


def test_composite_extension_overrides():
    q0, q1 = QubitId(), QubitId()
    cnot = CNOT(q0, q1)
    circuit = Circuit()
    circuit.append(cnot)
    opt = ExpandComposite(composite_gate_extension=Extensions({
        CompositeGate: {CNotGate: lambda e: OtherCNot()}
    }))
    opt.optimize_circuit(circuit)
    expected = Circuit()
    expected.append([Z(q0), (Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1), Z(q0)])
    assert_equal_mod_empty(expected, circuit)


def test_recursive_composite_extension_overrides():
    q0, q1 = QubitId(), QubitId()
    swap = SWAP(q0, q1)
    circuit = Circuit()
    circuit.append(swap)
    opt = ExpandComposite(composite_gate_extension=Extensions({
        CompositeGate: {CNotGate: lambda e: OtherCNot()}
    }))
    opt.optimize_circuit(circuit)
    expected = Circuit()
    expected.append([Z(q0), (Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1), Z(q0)])
    expected.append([Z(q1), (Y ** -0.5)(q0), CZ(q1, q0), (Y ** 0.5)(q0), Z(q1)],
                    strategy=InsertStrategy.INLINE)
    expected.append([Z(q0), (Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1), Z(q0)],
                    strategy=InsertStrategy.INLINE)
    assert_equal_mod_empty(expected, circuit)


def test_earliest_insert_strategy():
    # Fills empty moment.
    q0, q1 = QubitId(), QubitId()
    circuit = Circuit([Moment()])
    cnot = CNOT(q0, q1)
    circuit.append(cnot)

    opt = ExpandComposite(insert_strategy=InsertStrategy.EARLIEST)
    opt.optimize_circuit(circuit)
    expected = Circuit()
    expected.append([(Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1)])
    assert_equal_mod_empty(expected, circuit)


def test_new_insert_strategy():
    q0, q1 = QubitId(), QubitId()
    circuit = Circuit([Moment()])
    cnot = CNOT(q0, q1)
    circuit.append(cnot)

    opt = ExpandComposite(insert_strategy=InsertStrategy.NEW)
    opt.optimize_circuit(circuit)
    expected = Circuit([Moment([(Y ** -0.5)(q1)]),
                        Moment([CZ(q0, q1)]),
                        Moment([(Y ** 0.5)(q1)])])
    assert_equal_mod_empty(expected, circuit)


def test_inline_insert_strategy():
    q0, q1 = QubitId(), QubitId()
    circuit = Circuit([Moment()])
    cnot = CNOT(q0, q1)
    circuit.append(cnot)

    opt = ExpandComposite(insert_strategy=InsertStrategy.INLINE)
    opt.optimize_circuit(circuit)
    expected = Circuit([Moment()])
    expected.append([(Y ** -0.5)(q1), CZ(q0, q1), (Y ** 0.5)(q1)],
                    strategy=InsertStrategy.EARLIEST)
    expected.moments.append(Moment())
    # Preserves empty moment.
    assert expected == circuit
