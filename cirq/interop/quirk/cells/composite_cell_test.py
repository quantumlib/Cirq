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

import pytest

import cirq
from cirq import quirk_json_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.interop.quirk.cells.composite_cell import _iterator_to_iterable


def test_iterator_to_iterable():
    k = 0

    def counter():
        nonlocal k
        k += 1
        return k - 1

    # Normal iterator usage.
    k = 0
    generator = (counter() for _ in range(10))
    assert k == 0
    assert list(generator) == list(range(10))
    assert k == 10
    assert list(generator) == []
    assert k == 10

    # Converted to iterable usage.
    k = 0
    generator = _iterator_to_iterable(counter() for _ in range(10))
    assert k == 0  # Does not immediately iterate.
    assert list(generator) == list(range(10))
    assert k == 10
    assert list(generator) == list(range(10))
    assert k == 10

    # Simultaneous converted to iterable usage with gradual iteration.
    k = 0
    generator = _iterator_to_iterable(counter() for _ in range(10))
    iter1 = iter(generator)
    iter2 = iter(generator)
    assert k == 0
    # iter1 pulls ahead.
    assert next(iter1) == 0
    assert k == 1
    assert next(iter1) == 1
    assert k == 2
    # iter2 catches up and pulls ahead.
    assert next(iter2) == 0
    assert k == 2
    assert next(iter2) == 1
    assert k == 2
    assert next(iter2) == 2
    assert k == 3
    # Both finish
    assert list(iter1) == list(range(2, 10))
    assert k == 10
    assert list(iter2) == list(range(3, 10))
    assert k == 10


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
            cirq.interop.quirk.QuirkArithmeticOperation('+=A1',
                                                        target=[b],
                                                        inputs=[[a]])))

    # With external input.
    assert_url_to_circuit_returns(
        '{"cols":[["inputA1","~r79k"]],"gates":[{"id":"~r79k",'
        '"circuit":{"cols":[["+=A1"]]}}]}',
        cirq.Circuit(
            cirq.interop.quirk.QuirkArithmeticOperation('+=A1',
                                                        target=[b],
                                                        inputs=[[a]])))

    # With external control.
    assert_url_to_circuit_returns(
        '{"cols":[["•",1,"~r79k"]],"gates":[{"id":"~r79k",'
        '"circuit":{"cols":[["X"],["Y","Z"]]}}]}',
        cirq.Circuit(
            cirq.X(c).controlled_by(a),
            cirq.Y(c).controlled_by(a),
            cirq.Z(d).controlled_by(a)))

    # With external and internal control.
    assert_url_to_circuit_returns(
        '{"cols":[["•",1,"~r79k"]],"gates":[{"id":"~r79k",'
        '"circuit":{"cols":[["X"],["⊕","Z"]]}}]}',
        cirq.Circuit(
            cirq.X(c).controlled_by(a),
            cirq.Y(c)**0.5,
            cirq.Z(d).controlled_by(a, c),
            cirq.Y(c)**-0.5))

    # Broadcast input.
    assert_url_to_circuit_returns(
        '{"cols":[["~q1fh",1,1,"inputA2"]],"gates":[{"id":"~q1fh",'
        '"circuit":{"cols":[["+=A2"],[1,"+=A2"],[1,"+=A2"]]}}]}',
        cirq.Circuit(
            cirq.interop.quirk.QuirkArithmeticOperation('+=A2',
                                                        target=[a, b],
                                                        inputs=[[d, e]]),
            cirq.interop.quirk.QuirkArithmeticOperation('+=A2',
                                                        target=[b, c],
                                                        inputs=[[d, e]]),
            cirq.interop.quirk.QuirkArithmeticOperation('+=A2',
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
