# Copyright 2020 The Cirq Developers
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
from unittest.mock import patch
import copy
import numpy as np
import sympy
import pytest

import cirq
import cirq_pasqal


class MockGet:
    def __init__(self, json):
        self.counter = 0
        self.json = json

    def raise_for_status(self):
        pass

    @property
    def text(self):
        self.counter += 1
        if self.counter > 1:
            return self.json


def _make_sampler(device) -> cirq_pasqal.PasqalSampler:

    sampler = cirq_pasqal.PasqalSampler(
        remote_host='http://00.00.00/', access_token='N/A', device=device
    )
    return sampler


def test_pasqal_circuit_init():
    qs = cirq.NamedQubit.range(3, prefix='q')
    ex_circuit = cirq.Circuit()
    ex_circuit.append([[cirq.CZ(qs[i], qs[i + 1]), cirq.X(qs[i + 1])] for i in range(len(qs) - 1)])
    test_circuit = cirq.Circuit()
    test_circuit.append(
        [[cirq.CZ(qs[i], qs[i + 1]), cirq.X(qs[i + 1])] for i in range(len(qs) - 1)]
    )

    for moment1, moment2 in zip(test_circuit, ex_circuit):
        assert moment1 == moment2


@patch('cirq_pasqal.pasqal_sampler.requests.get')
@patch('cirq_pasqal.pasqal_sampler.requests.post')
def test_run_sweep(mock_post, mock_get):
    """Test running a sweep.

    Encodes a random binary number in the qubits, sweeps between odd and even
    without noise and checks if the results match.
    """

    qs = [cirq_pasqal.ThreeDQubit(i, j, 0) for i in range(3) for j in range(3)]

    par = sympy.Symbol('par')
    sweep = cirq.Linspace(key='par', start=0.0, stop=1.0, length=2)

    num = np.random.randint(0, 2 ** 9)
    binary = bin(num)[2:].zfill(9)

    device = cirq_pasqal.PasqalVirtualDevice(control_radius=1, qubits=qs)
    ex_circuit = cirq.Circuit()

    for i, b in enumerate(binary[:-1]):
        if b == '1':
            ex_circuit.append(cirq.X(qs[-i - 1]), strategy=cirq.InsertStrategy.NEW)

    ex_circuit_odd = copy.deepcopy(ex_circuit)
    ex_circuit_odd.append(cirq.X(qs[0]), strategy=cirq.InsertStrategy.NEW)
    ex_circuit_odd.append(cirq.measure(*qs), strategy=cirq.InsertStrategy.NEW)

    xpow = cirq.XPowGate(exponent=par)
    ex_circuit.append([xpow(qs[0])], strategy=cirq.InsertStrategy.NEW)
    ex_circuit.append(cirq.measure(*qs), strategy=cirq.InsertStrategy.NEW)

    mock_get.return_value = MockGet(cirq.to_json(ex_circuit_odd))
    sampler = _make_sampler(device)

    with pytest.raises(ValueError, match="Non-empty moment after measurement"):
        wrong_circuit = copy.deepcopy(ex_circuit)
        wrong_circuit.append(cirq.X(qs[0]))
        sampler.run_sweep(program=wrong_circuit, params=sweep, repetitions=1)

    data = sampler.run_sweep(program=ex_circuit, params=sweep, repetitions=1)

    submitted_json = mock_post.call_args[1]['data']
    assert cirq.read_json(json_text=submitted_json) == ex_circuit_odd
    assert mock_post.call_count == 2
    assert data[1] == ex_circuit_odd


@patch('cirq_pasqal.pasqal_sampler.requests.get')
@patch('cirq_pasqal.pasqal_sampler.requests.post')
def test_run_sweep_device_deprecated(mock_post, mock_get):
    """Test running a sweep.

    Encodes a random binary number in the qubits, sweeps between odd and even
    without noise and checks if the results match.
    """

    qs = [cirq_pasqal.ThreeDQubit(i, j, 0) for i in range(3) for j in range(3)]

    par = sympy.Symbol('par')
    sweep = cirq.Linspace(key='par', start=0.0, stop=1.0, length=2)

    num = np.random.randint(0, 2 ** 9)
    binary = bin(num)[2:].zfill(9)

    device = cirq_pasqal.PasqalVirtualDevice(control_radius=1, qubits=qs)
    ex_circuit = cirq.Circuit()

    for i, b in enumerate(binary[:-1]):
        if b == '1':
            ex_circuit.append(cirq.X(qs[-i - 1]), strategy=cirq.InsertStrategy.NEW)

    ex_circuit_odd = copy.deepcopy(ex_circuit)
    ex_circuit_odd.append(cirq.X(qs[0]), strategy=cirq.InsertStrategy.NEW)
    ex_circuit_odd.append(cirq.measure(*qs), strategy=cirq.InsertStrategy.NEW)

    xpow = cirq.XPowGate(exponent=par)
    ex_circuit.append([xpow(qs[0])], strategy=cirq.InsertStrategy.NEW)
    ex_circuit.append(cirq.measure(*qs), strategy=cirq.InsertStrategy.NEW)

    mock_get.return_value = MockGet(cirq.to_json(ex_circuit_odd))
    sampler = _make_sampler(device)
    ex_circuit._device = device
    with cirq.testing.assert_deprecated('The program.device component', deadline='v0.15'):
        _ = sampler.run_sweep(program=ex_circuit, params=sweep, repetitions=1)
