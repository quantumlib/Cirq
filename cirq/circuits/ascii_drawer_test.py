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

import pytest

from cirq import extension
from cirq import ops
from cirq.google import XmonQubit
from cirq.circuits import Circuit, from_ascii, Moment, to_ascii


def test_to_ascii_teleportation_to_diagram():
    ali = XmonQubit(0, 0)
    bob = XmonQubit(0, 1)
    msg = XmonQubit(1, 0)
    tmp = XmonQubit(1, 1)

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

    assert to_ascii(c).strip() == """
(0, 0): ---H---@-----------X-------M---@-----------
               |           |           |
(0, 1): -------X-----------|-----------X-------Z---
                           |                   |
(1, 0): -----------X^0.5---@---H---M-------@---|---
                                           |   |
(1, 1): -----------------------------------X---Z---
        """.strip()

    assert to_ascii(c, transpose=True).strip() == """
(0, 0) (0, 1) (1, 0) (1, 1)
|      |      |      |
H      |      |      |
|      |      |      |
@------X      |      |
|      |      |      |
|      |      X^0.5  |
|      |      |      |
X-------------@      |
|      |      |      |
|      |      H      |
|      |      |      |
M      |      M      |
|      |      |      |
@------X      |      |
|      |      |      |
|      |      @------X
|      |      |      |
|      Z-------------Z
|      |      |      |
        """.strip()


def test_to_ascii_extended_gate():
    q = XmonQubit(0, 0)

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
