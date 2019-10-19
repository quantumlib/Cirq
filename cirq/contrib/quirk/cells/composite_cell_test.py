# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

import cirq
from cirq.contrib.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.contrib.quirk import quirk_json_to_circuit


def test_custom_matrix_gate():
    a, b = cirq.LineQubit.range(2)

    # Without name.
    assert_url_to_circuit_returns(
        '{"cols":[["~cv0d"]],'
        '"gates":[{"id":"~cv0d","matrix":"{{0,1},{1,0}}"}]}',
        cirq.Circuit(cirq.MatrixGate(np.array([
            [0, 1],
            [1, 0],
        ])).on(a),))

    # With name.
    assert_url_to_circuit_returns(
        '{"cols":[["~cv0d"]],'
        '"gates":[{"id":"~cv0d","name":"test","matrix":"{{0,i},{1,0}}"}]}',
        cirq.Circuit(cirq.MatrixGate(np.array([
            [0, 1j],
            [1, 0],
        ])).on(a),))

    # Multi-qubit. Reversed qubit order to account for endian-ness difference.
    assert_url_to_circuit_returns(
        '{"cols":[["X"],["~2hj0"]],'
        '"gates":[{"id":"~2hj0",'
        '"matrix":"{{-1,0,0,0},{0,i,0,0},{0,0,1,0},{0,0,0,-i}}"}]}',
        cirq.Circuit(
            cirq.X(a),
            cirq.MatrixGate(np.diag([-1, 1j, 1, -1j])).on(b, a),
        ),
        output_amplitudes_from_quirk=[{
            "r": 0,
            "i": 0
        }, {
            "r": 0,
            "i": 1
        }, {
            "r": 0,
            "i": 0
        }, {
            "r": 0,
            "i": 0
        }])


def test_custom_circuit_gate():
    a, b, c, d, e = cirq.LineQubit.range(5)

    # Without name.
    assert_url_to_circuit_returns(
        '{"cols":[["~d3pq"],["Y"]],'
        '"gates":[{"id":"~d3pq","circuit":{"cols":[["H"],["•","X"]]}}]}',
        cirq.Circuit(
            cirq.H(a),
            cirq.X(b).controlled_by(a),
            cirq.Y(a),
        ))

    # With name.
    assert_url_to_circuit_returns(
        '{"cols":[["~d3pq"],["Y"]],'
        '"gates":[{"id":"~d3pq","name":"test",'
        '"circuit":{"cols":[["H"],["•","X"]]}}]}',
        cirq.Circuit(
            cirq.H(a),
            cirq.X(b).controlled_by(a),
            cirq.Y(a),
        ))

    # With internal input.
    assert_url_to_circuit_returns(
        '{"cols":[["~a5ls"]],"gates":[{"id":"~a5ls",'
        '"circuit":{"cols":[["inputA1","+=A1"]]}}]}',
        cirq.Circuit(
            cirq.contrib.quirk.QuirkArithmeticOperation('+=A1',
                                                        target=[b],
                                                        inputs=[[a]])))

    # With external input.
    assert_url_to_circuit_returns(
        '{"cols":[["inputA1","~r79k"]],"gates":[{"id":"~r79k",'
        '"circuit":{"cols":[["+=A1"]]}}]}',
        cirq.Circuit(
            cirq.contrib.quirk.QuirkArithmeticOperation('+=A1',
                                                        target=[b],
                                                        inputs=[[a]])))

    # Broadcast input.
    assert_url_to_circuit_returns(
        '{"cols":[["~q1fh",1,1,"inputA2"]],"gates":[{"id":"~q1fh",'
        '"circuit":{"cols":[["+=A2"],[1,"+=A2"],[1,"+=A2"]]}}]}',
        cirq.Circuit(
            cirq.contrib.quirk.QuirkArithmeticOperation('+=A2',
                                                        target=[a, b],
                                                        inputs=[[d, e]]),
            cirq.contrib.quirk.QuirkArithmeticOperation('+=A2',
                                                        target=[b, c],
                                                        inputs=[[d, e]]),
            cirq.contrib.quirk.QuirkArithmeticOperation('+=A2',
                                                        target=[b, c],
                                                        inputs=[[d, e]]),
        ))

    # Nested custom gate.
    assert_url_to_circuit_returns(
        '{"cols":[["~gtnd"]],"gates":[{"id":"~ct36",'
        '"circuit":{"cols":[["X"],["X"]]}},{"id":"~gtnd",'
        '"circuit":{"cols":[["~ct36"],["~ct36"]]}}]}',
        cirq.Circuit(cirq.X(a)) * 4)

    # Nested custom gate wrong order.
    with pytest.raises(ValueError, match='Unrecognized column entry'):
        _ = quirk_json_to_circuit({
            "cols": [["~gtnd"]],
            "gates": [
                {
                    "id": "~gtnd",
                    "circuit": {
                        "cols": [["~ct36"], ["~ct36"]]
                    }
                },
                {
                    "id": "~ct36",
                    "circuit": {
                        "cols": [["X"], ["X"]]
                    }
                },
            ]
        })
