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


def assert_roundtrip(obj, text_should_be=None):
    buffer = io.StringIO()
    cirq.protocols.to_json(obj, buffer)

    if text_should_be is not None:
        buffer.seek(0)
        text = buffer.read()

        print()
        print(text)

        assert text == text_should_be

    buffer.seek(0)
    obj2 = cirq.protocols.read_json(buffer)
    assert obj == obj2


def test_line_qubit_roundtrip():
    q1 = cirq.LineQubit(12)
    assert_roundtrip(q1, text_should_be="""{
  "cirq_type": "LineQubit",
  "x": 12
}""")


def test_gridqubit_roundtrip():
    q = cirq.GridQubit(15, 18)
    assert_roundtrip(q, text_should_be="""{
  "cirq_type": "GridQubit",
  "row": 15,
  "col": 18
}""")


def test_op_roundtrip():
    q = cirq.LineQubit(5)
    op1 = cirq.Rx(.123).on(q)
    assert_roundtrip(op1, text_should_be="""{
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
}""")


PREP_FUNCTIONS = {
    'CCXPowGate': lambda: cirq.CCNOT(*cirq.LineQubit.range(3)),
    'CCZPowGate': lambda: cirq.CCZ(*cirq.LineQubit.range(3)),
    'CNotPowGate': lambda: cirq.CNOT(*cirq.LineQubit.range(2)) ** 2,
    'CZPowGate': lambda: cirq.CZ(*cirq.LineQubit.range(2)),
    'EigenGate': lambda: None,
    'GateOperation': lambda: None,
    'GridQubit': lambda: cirq.GridQubit(10, 11),
    'HPowGate': lambda: cirq.H(cirq.GridQubit(10, 11)),
    'ISwapPowGate': lambda: cirq.ISWAP(*cirq.LineQubit.range(2)) ** 0.5,
    'LineQubit': lambda: cirq.LineQubit(0),
    'PauliInteractionGate': lambda: None,
    'SingleQubitPauliStringGateOperation': lambda: None,
    'SwapPowGate': lambda: cirq.SWAP(*cirq.LineQubit.range(2)),
    'XPowGate': lambda: cirq.X(cirq.LineQubit(5)) ** 123,
    'XXPowGate': lambda: cirq.XX(*cirq.LineQubit.range(2)),
    'YPowGate': lambda: cirq.Y(cirq.LineQubit(10)) ** 0.123,
    'YYPowGate': lambda: cirq.YY(*cirq.LineQubit.range(2)),
    'ZPowGate': lambda: cirq.Z(cirq.LineQubit(0)) ** 0.5,
    'ZZPowGate': lambda: cirq.ZZ(*cirq.LineQubit.range(2)),
}


@pytest.mark.parametrize('cirq_type',
                         cirq.protocols.json.RESOLVER_CACHE
                         .cirq_class_resolver_dictionary.keys())
def test_all_roundtrip(cirq_type: str):
    prep_func = PREP_FUNCTIONS[cirq_type]
    obj = prep_func()
    assert_roundtrip(obj)
