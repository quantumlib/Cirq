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
import json
import urllib

import numpy as np
import pytest

import cirq
from cirq.contrib.quirk import quirk_url_to_circuit, quirk_json_to_circuit
from cirq.contrib.quirk.cells.testing import assert_url_to_circuit_returns


def test_parse_simple_cases():
    a, b = cirq.LineQubit.range(2)

    assert quirk_url_to_circuit('http://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit('https://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit(
        'https://algassert.com/quirk#') == cirq.Circuit()
    assert quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[]}') == cirq.Circuit()

    assert quirk_url_to_circuit(
        'https://algassert.com/quirk#circuit={'
        '%22cols%22:[[%22H%22],[%22%E2%80%A2%22,%22X%22]]'
        '}') == cirq.Circuit(cirq.H(a),
                             cirq.X(b).controlled_by(a))


@pytest.mark.parametrize('url,error_cls,msg', [
    ('http://algassert.com/quirk#bad', ValueError,
     'must start with "circuit="'),
    ('http://algassert.com/quirk#circuit=', json.JSONDecodeError, None),
])
def test_parse_url_failures(url, error_cls, msg):
    with pytest.raises(error_cls, match=msg):
        _ = quirk_url_to_circuit(url)


@pytest.mark.parametrize(
    'url,msg',
    [('http://algassert.com/quirk#circuit=[]', 'top-level dictionary'),
     ('http://algassert.com/quirk#circuit={}', '"cols" entry'),
     ('http://algassert.com/quirk#circuit={"cols": 1}', 'cols must be a list'),
     ('http://algassert.com/quirk#circuit={"cols": [0]}', 'col must be a list'),
     ('http://algassert.com/quirk#circuit={"cols": [[0]]}',
      'Unrecognized column entry: 0'),
     ('http://algassert.com/quirk#circuit={"cols": [["not a real"]]}',
      'Unrecognized column entry: '),
     ('http://algassert.com/quirk#circuit={"cols": [[]], "other": 1}',
      'Unrecognized Circuit JSON keys')])
def test_parse_failures(url, msg):
    parsed_url = urllib.parse.urlparse(url)
    data = json.loads(parsed_url.fragment[len('circuit='):])

    with pytest.raises(ValueError, match=msg):
        _ = quirk_url_to_circuit(url)

    with pytest.raises(ValueError, match=msg):
        _ = quirk_json_to_circuit(data)


def test_parse_not_supported_yet():
    with pytest.raises(NotImplementedError,
                       match='Custom gates not supported yet'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [[]], "gates": []}')


def test_parse_with_qubits():
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(0, 2)

    assert quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}',
        qubits=cirq.GridQubit.rect(4, 4)) == cirq.Circuit(
            cirq.H(a),
            cirq.X(b).controlled_by(a))

    assert quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["H"],["•",1,"X"]]}',
        qubits=cirq.GridQubit.rect(4, 4)) == cirq.Circuit(
            cirq.H(a),
            cirq.X(c).controlled_by(a))

    with pytest.raises(IndexError, match="qubits specified"):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}',
            qubits=[cirq.GridQubit(0, 0)])


def test_extra_cell_makers():
    assert cirq.contrib.quirk.quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
        extra_cell_makers=[
            cirq.contrib.quirk.cells.CellMaker(
                identifier='iswap',
                size=2,
                maker=lambda args: cirq.ISWAP(*args.qubits))
        ]) == cirq.Circuit(cirq.ISWAP(*cirq.LineQubit.range(2)))

    assert cirq.contrib.quirk.quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[["iswap"]]}',
        extra_cell_makers={'iswap': cirq.ISWAP}) == cirq.Circuit(
            cirq.ISWAP(*cirq.LineQubit.range(2)))


def test_init():
    b, c, d, e, f = cirq.LineQubit.range(1, 6)
    assert_url_to_circuit_returns(
        '{"cols":[],"init":[0,1,"+","-","i","-i"]}',
        cirq.Circuit(cirq.X(b),
                     cirq.Ry(np.pi / 2).on(c),
                     cirq.Ry(-np.pi / 2).on(d),
                     cirq.Rx(-np.pi / 2).on(e),
                     cirq.Rx(np.pi / 2).on(f)))

    assert_url_to_circuit_returns('{"cols":[],"init":["+"]}',
                                  output_amplitudes_from_quirk=[{
                                      "r": 0.7071067690849304,
                                      "i": 0
                                  }, {
                                      "r": 0.7071067690849304,
                                      "i": 0
                                  }])

    assert_url_to_circuit_returns('{"cols":[],"init":["i"]}',
                                  output_amplitudes_from_quirk=[{
                                      "r": 0.7071067690849304,
                                      "i": 0
                                  }, {
                                      "r":
                                      0,
                                      "i":
                                      0.7071067690849304
                                  }])

    with pytest.raises(ValueError, match="init must be a list"):
        _ = cirq.contrib.quirk.quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols":[],"init":0}')

    with pytest.raises(ValueError, match="Unrecognized init state"):
        _ = cirq.contrib.quirk.quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols":[],"init":[2]}')
