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

import pytest

import cirq
from cirq.contrib.quirk import quirk_url_to_circuit


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


def test_parse_failures():
    with pytest.raises(ValueError, match='must start with "circuit="'):
        _ = quirk_url_to_circuit('http://algassert.com/quirk#bad')

    with pytest.raises(json.JSONDecodeError):
        _ = quirk_url_to_circuit('http://algassert.com/quirk#circuit=')

    with pytest.raises(ValueError, match='top-level dictionary'):
        _ = quirk_url_to_circuit('http://algassert.com/quirk#circuit=[]')

    with pytest.raises(ValueError, match='"cols" entry'):
        _ = quirk_url_to_circuit('http://algassert.com/quirk#circuit={}')

    with pytest.raises(ValueError, match='cols must be a list'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": 1}')

    with pytest.raises(ValueError, match='col must be a list'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [0]}')

    with pytest.raises(ValueError, match='Unrecognized column entry: 0'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [[0]]}')

    with pytest.raises(ValueError, match='Unrecognized column entry: '):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [["not a real"]]}')

    with pytest.raises(ValueError, match='Unrecognized Circuit JSON keys'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [[]], "other": 1}')


def test_parse_not_supported_yet():
    with pytest.raises(NotImplementedError,
                       match='Custom gates not supported yet'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [[]], "gates": []}')

    with pytest.raises(NotImplementedError,
                       match='initial states not supported yet'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [[]], "init": []}')
