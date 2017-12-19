# Copyright 2017 Google LLC
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

from cirq import extension
from cirq import ops
from cirq.circuits import Circuit, from_ascii, Moment, to_ascii


def test_from_ascii_empty():
    assert from_ascii('') == Circuit()

    assert from_ascii('(0, 0): ------') == Circuit()

    assert from_ascii("""
(0, 0): ------
    """) == Circuit()

    assert from_ascii("""
(0, 0): ------

(0, 1): ------
    """) == Circuit()


def test_from_ascii_single_qubit_ops():
    q00 = ops.QubitId(0, 0)
    q12 = ops.QubitId(1, 2)
    assert from_ascii('(0, 0): --X--') == Circuit([Moment([ops.X(q00)])])

    assert from_ascii('(0, 0): --X^0.5--') == Circuit(
        [Moment([(ops.X**0.5)(q00)])])

    assert from_ascii('(1, 2): --Z--') == Circuit([Moment([ops.Z(q12)])])

    assert from_ascii("""
(0, 0): --Z--
(1, 2): --X--
        """) == Circuit([Moment([ops.Z(q00),
                                 ops.X(q12)])])


def test_from_ascii_two_qubit_ops():
    q00 = ops.QubitId(0, 0)
    q10 = ops.QubitId(1, 0)

    assert from_ascii("""
(0, 0): --.--
(1, 0): --X--
        """) == Circuit([Moment([ops.CNOT(q00, q10)])])

    assert from_ascii("""
(0, 0): --x--
(1, 0): --.--
        """) == Circuit([Moment([ops.CNOT(q10, q00)])])

    assert from_ascii("""
(0, 0): --Z--
          |
(1, 0): --X--
        """) == Circuit([Moment([ops.CNOT(q00, q10)])])

    assert from_ascii("""
(0, 0): --Z--
          |
(2, 0): --|--
(1, 0): --Z--
        """) == Circuit([Moment([ops.CZ(q00, q10)])])

    assert from_ascii("""
(0, 0): --Z-----
          |^0.5
(1, 0): --Z-----
        """) == Circuit([Moment([(ops.CZ**0.5)(q00, q10)])])

    assert from_ascii("""
(0, 0): --@-----
          |^0.5
(2, 0): --+-----
          |^0.5
(1, 0): --Z^0.5-
        """) == Circuit([Moment([(ops.CZ**0.125)(q00, q10)])])


def test_from_ascii_teleportation_from_diagram():
    ali = ops.QubitId(0, 0)
    bob = ops.QubitId(0, 1)
    msg = ops.QubitId(1, 0)
    tmp = ops.QubitId(1, 1)

    assert from_ascii("""
(1, 0): ------X^0.5--@-H-M----@---
                     |        |
(0, 0): --H-@--------X---M-@--|---
            |              |  |
(0, 1): ----X--------------X--|-Z-
(1, 1): ----------------------X-@-
        """) == Circuit([
            Moment([ops.H(ali)]),
            Moment([ops.CNOT(ali, bob)]),
            Moment([(ops.X**0.5)(msg)]),
            Moment([ops.CNOT(msg, ali)]),
            Moment([ops.H(msg)]),
            Moment(
                [ops.MeasurementGate()(msg),
                 ops.MeasurementGate()(ali)]),
            Moment([ops.CNOT(ali, bob)]),
            Moment([ops.CNOT(msg, tmp)]),
            Moment([ops.CZ(bob, tmp)]),
        ])


def test_from_ascii_fail_on_duplicate_qubit():
    with pytest.raises(ValueError):
        _ = from_ascii("""
(0, 0): -X---
(0, 0): ---X-
        """)


def test_fail_on_double_colon():
    with pytest.raises(ValueError):
        _ = from_ascii("""
(0, 0): -X-:-
        """)


def test_fail_on_unknown_operation():
    with pytest.raises(ValueError):
        _ = from_ascii("""
(0, 0): --unknown--
        """)


def test_fail_on_adjacent_operations():
    with pytest.raises(ValueError):
        _ = from_ascii("""
(0, 0): --XY--
        """)


def test_to_ascii_teleportation_to_diagram():
    ali = ops.QubitId(0, 0)
    bob = ops.QubitId(0, 1)
    msg = ops.QubitId(1, 0)
    tmp = ops.QubitId(1, 1)

    c = Circuit([
        Moment([ops.H(ali)]),
        Moment([ops.CNOT(ali, bob)]),
        Moment([(ops.X**0.5)(msg)]),
        Moment([ops.CNOT(msg, ali)]),
        Moment([ops.H(msg)]),
        Moment(
            [ops.MeasurementGate()(msg),
             ops.MeasurementGate()(ali)]),
        Moment([ops.CNOT(ali, bob)]),
        Moment([ops.CNOT(msg, tmp)]),
        Moment([ops.CZ(bob, tmp)]),
    ])
    assert """
(0, 0): ---H---@-----------X-------M---@-----------
               |           |           |
(0, 1): -------X-----------|-----------X-------Z---
                           |                   |
(1, 0): -----------X^0.5---@---H---M-------@---|---
                                           |   |
(1, 1): -----------------------------------X---Z---
        """.strip() == to_ascii(c).strip()


def test_to_ascii_extended_gate():
    q = ops.QubitId(0, 0)

    class FGate(ops.Gate):
        pass

    f = FGate()
    c = Circuit([
        Moment([f.on(q)]),
    ])

    # Fails without extension.
    with pytest.raises(TypeError):
        _ = to_ascii(c)

    # Succeeds with extension.
    class FGateAsAscii(ops.AsciiDiagrammableGate):
        def __init__(self, f_gate):
            self.f_gate = f_gate

        def ascii_wire_symbols(self):
            return 'F'

    diagram = to_ascii(c,
                       extension.Extensions({
                           ops.AsciiDiagrammableGate: {
                               FGate: FGateAsAscii
                           }
                       }))

    assert diagram.strip() == """
(0, 0): ---F---
        """.strip()
