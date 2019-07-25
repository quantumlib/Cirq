# Copyright 2019 The Cirq Developers
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
import cirq.protocols
import io


def test_line_qubit_roundtrip():
    q1 = cirq.LineQubit(12)

    buffer = io.StringIO()
    cirq.protocols.to_json(q1, buffer)

    buffer.seek(0)
    text = buffer.read()

    print()
    print(text)

    assert text == """{
  "cirq_type": "LineQubit",
  "x": 12
}"""

    buffer.seek(0)
    q2 = cirq.protocols.read_json(buffer)
    assert q1 == q2


def test_op_roundtrip():
    q = cirq.LineQubit(5)
    op1 = cirq.Rx(.123).on(q)

    buffer = io.StringIO()
    cirq.protocols.to_json(op1, buffer)

    buffer.seek(0)
    text = buffer.read()
    assert text == """{
  "cirq_type": "GateOperation",
  "gate": {
    "cirq_type": "XPowGate",
    "exponent": 0.03915211600060625,
    "global_shift": -0.5
  },
  "qubits": [
    {
      "cirq_type": "LineQubit",
      "x": 5
    }
  ]
}"""

    print()
    print(text)

    buffer.seek(0)
    op2 = cirq.protocols.read_json(buffer)
    assert op1 == op2
