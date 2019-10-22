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


def test_survives_a_billion_laughs():
    # If this test is timing out, it means you made a change that accidentally
    # iterated over the circuit contents before they were counted and checked
    # against the maximum. It is not possible to test for the billion laughs
    # attack by doing a small case, because iterating-then-counting instead of
    # counting-then-iterating misleadingly succeeds on small cases.
    with pytest.raises(ValueError, match=f'{4**26} operations'):
        cirq.contrib.quirk.quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={'
            '"cols":[["~z"]],'
            '"gates":['
            '{"id":"~a","circuit":{"cols":[["H"],["H"],["H"],["H"]]}},'
            '{"id":"~b","circuit":{"cols":[["~a"],["~a"],["~a"],["~a"]]}},'
            '{"id":"~c","circuit":{"cols":[["~b"],["~b"],["~b"],["~b"]]}},'
            '{"id":"~d","circuit":{"cols":[["~c"],["~c"],["~c"],["~c"]]}},'
            '{"id":"~e","circuit":{"cols":[["~d"],["~d"],["~d"],["~d"]]}},'
            '{"id":"~f","circuit":{"cols":[["~e"],["~e"],["~e"],["~e"]]}},'
            '{"id":"~g","circuit":{"cols":[["~f"],["~f"],["~f"],["~f"]]}},'
            '{"id":"~h","circuit":{"cols":[["~g"],["~g"],["~g"],["~g"]]}},'
            '{"id":"~i","circuit":{"cols":[["~h"],["~h"],["~h"],["~h"]]}},'
            '{"id":"~j","circuit":{"cols":[["~i"],["~i"],["~i"],["~i"]]}},'
            '{"id":"~k","circuit":{"cols":[["~j"],["~j"],["~j"],["~j"]]}},'
            '{"id":"~l","circuit":{"cols":[["~k"],["~k"],["~k"],["~k"]]}},'
            '{"id":"~m","circuit":{"cols":[["~l"],["~l"],["~l"],["~l"]]}},'
            '{"id":"~n","circuit":{"cols":[["~m"],["~m"],["~m"],["~m"]]}},'
            '{"id":"~o","circuit":{"cols":[["~n"],["~n"],["~n"],["~n"]]}},'
            '{"id":"~p","circuit":{"cols":[["~o"],["~o"],["~o"],["~o"]]}},'
            '{"id":"~q","circuit":{"cols":[["~p"],["~p"],["~p"],["~p"]]}},'
            '{"id":"~r","circuit":{"cols":[["~q"],["~q"],["~q"],["~q"]]}},'
            '{"id":"~s","circuit":{"cols":[["~r"],["~r"],["~r"],["~r"]]}},'
            '{"id":"~t","circuit":{"cols":[["~s"],["~s"],["~s"],["~s"]]}},'
            '{"id":"~u","circuit":{"cols":[["~t"],["~t"],["~t"],["~t"]]}},'
            '{"id":"~v","circuit":{"cols":[["~u"],["~u"],["~u"],["~u"]]}},'
            '{"id":"~w","circuit":{"cols":[["~v"],["~v"],["~v"],["~v"]]}},'
            '{"id":"~x","circuit":{"cols":[["~w"],["~w"],["~w"],["~w"]]}},'
            '{"id":"~y","circuit":{"cols":[["~x"],["~x"],["~x"],["~x"]]}},'
            '{"id":"~z","circuit":{"cols":[["~y"],["~y"],["~y"],["~y"]]}}'
            ']}',
            max_operation_count=10**6)


def test_completes_weight_zero_billion_laughs():
    circuit = cirq.contrib.quirk.quirk_url_to_circuit(
        'https://algassert.com/quirk#circuit={'
        '"cols":[["~z"]],'
        '"gates":['
        '{"id":"~a","circuit":{"cols":['
        '["•"],["inputA2"],["setA"],["⊗","xpar"]]}},'
        '{"id":"~b","circuit":{"cols":[["~a"],["~a"],["~a"],["~a"]]}},'
        '{"id":"~c","circuit":{"cols":[["~b"],["~b"],["~b"],["~b"]]}},'
        '{"id":"~d","circuit":{"cols":[["~c"],["~c"],["~c"],["~c"]]}},'
        '{"id":"~e","circuit":{"cols":[["~d"],["~d"],["~d"],["~d"]]}},'
        '{"id":"~f","circuit":{"cols":[["~e"],["~e"],["~e"],["~e"]]}},'
        '{"id":"~g","circuit":{"cols":[["~f"],["~f"],["~f"],["~f"]]}},'
        '{"id":"~h","circuit":{"cols":[["~g"],["~g"],["~g"],["~g"]]}},'
        '{"id":"~i","circuit":{"cols":[["~h"],["~h"],["~h"],["~h"]]}},'
        '{"id":"~j","circuit":{"cols":[["~i"],["~i"],["~i"],["~i"]]}},'
        '{"id":"~k","circuit":{"cols":[["~j"],["~j"],["~j"],["~j"]]}},'
        '{"id":"~l","circuit":{"cols":[["~k"],["~k"],["~k"],["~k"]]}},'
        '{"id":"~m","circuit":{"cols":[["~l"],["~l"],["~l"],["~l"]]}},'
        '{"id":"~n","circuit":{"cols":[["~m"],["~m"],["~m"],["~m"]]}},'
        '{"id":"~o","circuit":{"cols":[["~n"],["~n"],["~n"],["~n"]]}},'
        '{"id":"~p","circuit":{"cols":[["~o"],["~o"],["~o"],["~o"]]}},'
        '{"id":"~q","circuit":{"cols":[["~p"],["~p"],["~p"],["~p"]]}},'
        '{"id":"~r","circuit":{"cols":[["~q"],["~q"],["~q"],["~q"]]}},'
        '{"id":"~s","circuit":{"cols":[["~r"],["~r"],["~r"],["~r"]]}},'
        '{"id":"~t","circuit":{"cols":[["~s"],["~s"],["~s"],["~s"]]}},'
        '{"id":"~u","circuit":{"cols":[["~t"],["~t"],["~t"],["~t"]]}},'
        '{"id":"~v","circuit":{"cols":[["~u"],["~u"],["~u"],["~u"]]}},'
        '{"id":"~w","circuit":{"cols":[["~v"],["~v"],["~v"],["~v"]]}},'
        '{"id":"~x","circuit":{"cols":[["~w"],["~w"],["~w"],["~w"]]}},'
        '{"id":"~y","circuit":{"cols":[["~x"],["~x"],["~x"],["~x"]]}},'
        '{"id":"~z","circuit":{"cols":[["~y"],["~y"],["~y"],["~y"]]}}'
        ']}',
        max_operation_count=0)
    assert circuit == cirq.Circuit()
