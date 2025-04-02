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
import json
import urllib

import numpy as np
import pytest

import cirq
from cirq import quirk_json_to_circuit, quirk_url_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns


def test_parse_simple_cases():
    a, b = cirq.LineQubit.range(2)

    assert quirk_url_to_circuit('http://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit('https://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit('https://algassert.com/quirk#') == cirq.Circuit()
    assert quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[]}') == cirq.Circuit()

    assert quirk_url_to_circuit(
        'https://algassert.com/quirk#circuit={'
        '%22cols%22:[[%22H%22],[%22%E2%80%A2%22,%22X%22]]'
        '}'
    ) == cirq.Circuit(cirq.H(a), cirq.X(b).controlled_by(a))


@pytest.mark.parametrize(
    'url,error_cls,msg',
    [
        ('http://algassert.com/quirk#bad', ValueError, 'must start with "circuit="'),
        ('http://algassert.com/quirk#circuit=', json.JSONDecodeError, None),
    ],
)
def test_parse_url_failures(url, error_cls, msg):
    with pytest.raises(error_cls, match=msg):
        _ = quirk_url_to_circuit(url)


@pytest.mark.parametrize(
    'url,msg',
    [
        ('http://algassert.com/quirk#circuit=[]', 'top-level dictionary'),
        ('http://algassert.com/quirk#circuit={}', '"cols" entry'),
        ('http://algassert.com/quirk#circuit={"cols": 1}', 'cols must be a list'),
        ('http://algassert.com/quirk#circuit={"cols": [0]}', 'col must be a list'),
        ('http://algassert.com/quirk#circuit={"cols": [[0]]}', 'Unrecognized column entry: 0'),
        (
            'http://algassert.com/quirk#circuit={"cols": [["not a real"]]}',
            'Unrecognized column entry: ',
        ),
        (
            'http://algassert.com/quirk#circuit={"cols": [[]], "other": 1}',
            'Unrecognized Circuit JSON keys',
        ),
    ],
)
def test_parse_failures(url, msg):
    parsed_url = urllib.parse.urlparse(url)
    data = json.loads(parsed_url.fragment[len('circuit=') :])

    with pytest.raises(ValueError, match=msg):
        _ = quirk_url_to_circuit(url)

    with pytest.raises(ValueError, match=msg):
        _ = quirk_json_to_circuit(data)


def test_parse_with_qubits():
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(0, 2)

    assert quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}',
        qubits=cirq.GridQubit.rect(4, 4),
    ) == cirq.Circuit(cirq.H(a), cirq.X(b).controlled_by(a))

    assert quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["H"],["•",1,"X"]]}',
        qubits=cirq.GridQubit.rect(4, 4),
    ) == cirq.Circuit(cirq.H(a), cirq.X(c).controlled_by(a))

    with pytest.raises(IndexError, match="qubits specified"):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}',
            qubits=[cirq.GridQubit(0, 0)],
        )


def test_extra_cell_makers():
    assert cirq.quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
        extra_cell_makers=[
            cirq.interop.quirk.cells.CellMaker(
                identifier='iswap', size=2, maker=lambda args: cirq.ISWAP(*args.qubits)
            )
        ],
    ) == cirq.Circuit(cirq.ISWAP(*cirq.LineQubit.range(2)))

    assert cirq.quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
        extra_cell_makers={'iswap': cirq.ISWAP},
    ) == cirq.Circuit(cirq.ISWAP(*cirq.LineQubit.range(2)))

    assert cirq.quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["iswap"], ["toffoli"]]}',
        extra_cell_makers=[
            cirq.interop.quirk.cells.CellMaker(
                identifier='iswap', size=2, maker=lambda args: cirq.ISWAP(*args.qubits)
            ),
            cirq.interop.quirk.cells.CellMaker(
                identifier='toffoli', size=3, maker=lambda args: cirq.TOFFOLI(*args.qubits)
            ),
        ],
    ) == cirq.Circuit(
        [cirq.ISWAP(*cirq.LineQubit.range(2)), cirq.TOFFOLI(*cirq.LineQubit.range(3))]
    )

    assert cirq.quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["iswap"], ["toffoli"]]}',
        extra_cell_makers={'iswap': cirq.ISWAP, 'toffoli': cirq.TOFFOLI},
    ) == cirq.Circuit(
        [cirq.ISWAP(*cirq.LineQubit.range(2)), cirq.TOFFOLI(*cirq.LineQubit.range(3))]
    )


def test_init():
    b, c, d, e, f = cirq.LineQubit.range(1, 6)
    assert_url_to_circuit_returns(
        '{"cols":[],"init":[0,1,"+","-","i","-i"]}',
        cirq.Circuit(
            cirq.X(b),
            cirq.ry(np.pi / 2).on(c),
            cirq.ry(-np.pi / 2).on(d),
            cirq.rx(-np.pi / 2).on(e),
            cirq.rx(np.pi / 2).on(f),
        ),
    )

    assert_url_to_circuit_returns(
        '{"cols":[],"init":["+"]}',
        output_amplitudes_from_quirk=[
            {"r": 0.7071067690849304, "i": 0},
            {"r": 0.7071067690849304, "i": 0},
        ],
    )

    assert_url_to_circuit_returns(
        '{"cols":[],"init":["i"]}',
        output_amplitudes_from_quirk=[
            {"r": 0.7071067690849304, "i": 0},
            {"r": 0, "i": 0.7071067690849304},
        ],
    )

    with pytest.raises(ValueError, match="init must be a list"):
        _ = cirq.quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[],"init":0}')

    with pytest.raises(ValueError, match="Unrecognized init state"):
        _ = cirq.quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[],"init":[2]}')


def test_custom_gate_parse_failures():
    with pytest.raises(ValueError, match='must be a list'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":5}')

    with pytest.raises(ValueError, match='gate json must be a dict'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[],"gates":[5]}')

    with pytest.raises(ValueError, match='Circuit JSON must be a dict'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[],"gates":[{"id":"~a","circuit":5}]}'
        )

    with pytest.raises(ValueError, match='matrix json must be a string'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[],"gates":[{"id":"~a","matrix":5}]}'
        )

    with pytest.raises(ValueError, match='Not surrounded by {{}}'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[],'
            '"gates":[{"id":"~a","matrix":"abc"}]}'
        )

    with pytest.raises(ValueError, match='must have an id'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[],'
            '"gates":['
            '{"matrix":"{{1,0},{0,1}}"}'
            ']}'
        )

    with pytest.raises(ValueError, match='both a matrix and a circuit'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[],'
            '"gates":['
            '{"id":"~a","circuit":{"cols":[]},"matrix":"{{1,0},{0,1}}"}'
            ']}'
        )

    with pytest.raises(ValueError, match='matrix or a circuit'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[],"gates":[{"id":"~a"}]}'
        )

    with pytest.raises(ValueError, match='duplicate identifier'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[],'
            '"gates":['
            '{"id":"~a","matrix":"{{1,0},{0,1}}"},'
            '{"id":"~a","matrix":"{{1,0},{0,1}}"}]}'
        )


def test_custom_matrix_gate():
    a, b = cirq.LineQubit.range(2)

    # Without name.
    assert_url_to_circuit_returns(
        '{"cols":[["~cv0d"]],"gates":[{"id":"~cv0d","matrix":"{{0,1},{1,0}}"}]}',
        cirq.Circuit(cirq.MatrixGate(np.array([[0, 1], [1, 0]])).on(a)),
    )

    # With name.
    assert_url_to_circuit_returns(
        '{"cols":[["~cv0d"]],"gates":[{"id":"~cv0d","name":"test","matrix":"{{0,i},{1,0}}"}]}',
        cirq.Circuit(cirq.MatrixGate(np.array([[0, 1j], [1, 0]])).on(a)),
    )

    # Multi-qubit. Reversed qubit order to account for endian-ness difference.
    assert_url_to_circuit_returns(
        '{"cols":[["X"],["~2hj0"]],'
        '"gates":[{"id":"~2hj0",'
        '"matrix":"{{-1,0,0,0},{0,i,0,0},{0,0,1,0},{0,0,0,-i}}"}]}',
        cirq.Circuit(cirq.X(a), cirq.MatrixGate(np.diag([-1, 1j, 1, -1j])).on(b, a)),
        output_amplitudes_from_quirk=[
            {"r": 0, "i": 0},
            {"r": 0, "i": 1},
            {"r": 0, "i": 0},
            {"r": 0, "i": 0},
        ],
    )


def test_survives_a_billion_laughs():
    # If this test is timing out, it means you made a change that accidentally
    # iterated over the circuit contents before they were counted and checked
    # against the maximum. It is not possible to test for the billion laughs
    # attack by doing a small case, because iterating-then-counting instead of
    # counting-then-iterating misleadingly succeeds on small cases.
    with pytest.raises(ValueError, match=f'{4**26} operations'):
        cirq.quirk_url_to_circuit(
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
            max_operation_count=10**6,
        )


def test_completes_weight_zero_billion_laughs():
    circuit = cirq.quirk_url_to_circuit(
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
        max_operation_count=0,
    )
    assert circuit == cirq.Circuit()


def test_example_qft_circuit():
    qft_example_diagram = """
0: ───×───────────────H───@───────────@────────────────────@──────────────────────────────@─────────────────────────────────────────@───────────────────────────────────────────────────@─────────────────────────────────────────────────────────────@───────────────────────────────────────────────────────────────────────
      │                   │           │                    │                              │                                         │                                                   │                                                             │
1: ───┼───×───────────────@^0.5───H───┼────────@───────────┼─────────@────────────────────┼──────────@──────────────────────────────┼─────────@─────────────────────────────────────────┼─────────@───────────────────────────────────────────────────┼─────────@─────────────────────────────────────────────────────────────
      │   │                           │        │           │         │                    │          │                              │         │                                         │         │                                                   │         │
2: ───┼───┼───×───────────────────────@^0.25───@^0.5───H───┼─────────┼────────@───────────┼──────────┼─────────@────────────────────┼─────────┼──────────@──────────────────────────────┼─────────┼─────────@─────────────────────────────────────────┼─────────┼─────────@───────────────────────────────────────────────────
      │   │   │                                            │         │        │           │          │         │                    │         │          │                              │         │         │                                         │         │         │
3: ───┼───┼───┼───×────────────────────────────────────────@^(1/8)───@^0.25───@^0.5───H───┼──────────┼─────────┼────────@───────────┼─────────┼──────────┼─────────@────────────────────┼─────────┼─────────┼──────────@──────────────────────────────┼─────────┼─────────┼─────────@─────────────────────────────────────────
      │   │   │   │                                                                       │          │         │        │           │         │          │         │                    │         │         │          │                              │         │         │         │
4: ───┼───┼───┼───×───────────────────────────────────────────────────────────────────────@^(1/16)───@^(1/8)───@^0.25───@^0.5───H───┼─────────┼──────────┼─────────┼────────@───────────┼─────────┼─────────┼──────────┼─────────@────────────────────┼─────────┼─────────┼─────────┼──────────@──────────────────────────────
      │   │   │                                                                                                                     │         │          │         │        │           │         │         │          │         │                    │         │         │         │          │
5: ───┼───┼───×─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────@^0.031───@^(1/16)───@^(1/8)───@^0.25───@^0.5───H───┼─────────┼─────────┼──────────┼─────────┼────────@───────────┼─────────┼─────────┼─────────┼──────────┼─────────@────────────────────
      │   │                                                                                                                                                                             │         │         │          │         │        │           │         │         │         │          │         │
6: ───┼───×─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────@^0.016───@^0.031───@^(1/16)───@^(1/8)───@^0.25───@^0.5───H───┼─────────┼─────────┼─────────┼──────────┼─────────┼────────@───────────
      │                                                                                                                                                                                                                                               │         │         │         │          │         │        │
7: ───×───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────@^0.008───@^0.016───@^0.031───@^(1/16)───@^(1/8)───@^0.25───@^0.5───H───
    """

    qft_example_json = (
        '{"cols":['
        # '["Counting8"],'
        '["Chance8"],'
        '["…","…","…","…","…","…","…","…"],'
        '["Swap",1,1,1,1,1,1,"Swap"],'
        '[1,"Swap",1,1,1,1,"Swap"],'
        '[1,1,"Swap",1,1,"Swap"],'
        '[1,1,1,"Swap","Swap"],'
        '["H"],'
        '["Z^½","•"],'
        '[1,"H"],'
        '["Z^¼","Z^½","•"],'
        '[1,1,"H"],'
        '["Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,"H"],'
        '["Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,"H"],'
        '["Z^⅟₃₂","Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,1,"H"],'
        '["Z^⅟₆₄","Z^⅟₃₂","Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,1,1,"H"],'
        '["Z^⅟₁₂₈","Z^⅟₆₄","Z^⅟₃₂","Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,1,1,1,"H"]]}'
    )
    qft_example_json_uri_escaped = (
        '{%22cols%22:['
        # '[%22Counting8%22],'
        '[%22Chance8%22],'
        '[%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,'
        '%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22],'
        '[%22Swap%22,1,1,1,1,1,1,%22Swap%22],'
        '[1,%22Swap%22,1,1,1,1,%22Swap%22],'
        '[1,1,%22Swap%22,1,1,%22Swap%22],'
        '[1,1,1,%22Swap%22,%22Swap%22],'
        '[%22H%22],'
        '[%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,%22H%22],'
        '[%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,%22H%22],'
        '[%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,'
        '%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%86%E2%82%84%22,'
        '%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,'
        '%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%81%E2%82%82%E2%82%88%22,'
        '%22Z^%E2%85%9F%E2%82%86%E2%82%84%22,'
        '%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,'
        '%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,1,1,1,%22H%22]]}'
    )
    assert_url_to_circuit_returns(qft_example_json, diagram=qft_example_diagram)
    assert_url_to_circuit_returns(qft_example_json_uri_escaped, diagram=qft_example_diagram)
