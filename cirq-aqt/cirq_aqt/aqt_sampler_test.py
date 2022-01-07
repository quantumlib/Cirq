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

from unittest import mock
import numpy as np
import pytest
import sympy

import cirq
from cirq_aqt import AQTSampler, AQTSamplerLocalSimulator
from cirq_aqt.aqt_device import get_aqt_device, get_op_string


class EngineReturn:
    """A put mock class for testing the REST interface"""

    def __init__(self):
        self.test_dict = {
            'status': 'queued',
            'id': '2131da',
            'samples': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
        self.counter = 0

    def json(self):
        self.counter += 1
        return self.test_dict

    def update(self, *args, **kwargs):
        if self.counter >= 2:
            self.test_dict['status'] = 'finished'
        return self


class EngineError(EngineReturn):
    """A put mock class for testing error responses"""

    def __init__(self):
        self.test_dict = {'status': 'error', 'id': '2131da', 'samples': "Error message"}
        self.counter = 0


class EngineNoid(EngineReturn):
    """A put mock class for testing error responses
    This will not return an id at the first call"""

    def __init__(self):
        self.test_dict = {'status': 'queued'}
        self.counter = 0


class EngineNoStatus(EngineReturn):
    """A put mock class for testing error responses
    This will not return a status in the second call"""

    def update(self, *args, **kwargs):
        del self.test_dict['status']
        return self


class EngineNoStatus2(EngineReturn):
    """A put mock class for testing error responses
    This will not return a status in the second call"""

    def update(self, *args, **kwargs):
        if self.counter >= 1:
            del self.test_dict['status']
        return self


class EngineErrorSecond(EngineReturn):
    """A put mock class for testing error responses
    This will return an error on the second put call"""

    def update(self, *args, **kwargs):
        if self.counter >= 1:
            self.test_dict['status'] = 'error'
        return self


def test_aqt_sampler_error_handling():
    for e_return in [
        EngineError(),
        EngineErrorSecond(),
        EngineNoStatus(),
        EngineNoStatus2(),
        EngineNoid(),
    ]:
        with mock.patch(
            'cirq_aqt.aqt_sampler.put', return_value=e_return, side_effect=e_return.update
        ) as _mock_method:
            theta = sympy.Symbol('theta')
            num_points = 1
            max_angle = np.pi
            repetitions = 10
            sampler = AQTSampler(remote_host="http://localhost:5000", access_token='testkey')
            _, qubits = get_aqt_device(1)
            circuit = cirq.Circuit(cirq.X(qubits[0]) ** theta)
            sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
            with pytest.raises(RuntimeError):
                _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)


def test_aqt_sampler_empty_circuit():
    num_points = 10
    max_angle = np.pi
    repetitions = 1000
    num_qubits = 4
    _, _qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal = True
    circuit = cirq.Circuit()
    sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
    with pytest.raises(RuntimeError):
        _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)


def test_aqt_sampler():
    put_call_args0 = {
        'access_token': 'testkey',
        'id': '2131da',
    }

    e_return = EngineReturn()
    with mock.patch(
        'cirq_aqt.aqt_sampler.put', return_value=e_return, side_effect=e_return.update
    ) as mock_method:
        theta = sympy.Symbol('theta')
        num_points = 1
        max_angle = np.pi
        repetitions = 10
        sampler = AQTSampler(remote_host="http://localhost:5000", access_token='testkey')
        _, qubits = get_aqt_device(1)
        circuit = cirq.Circuit(cirq.X(qubits[0]) ** theta)
        sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
        results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)
        excited_state_probs = np.zeros(num_points)
        for i in range(num_points):
            excited_state_probs[i] = np.mean(results[i].measurements['m'])
    callargs = mock_method.call_args[1]['data']
    for keys in put_call_args0:
        assert callargs[keys] == put_call_args0[keys]
    assert mock_method.call_count == 3


def test_aqt_sampler_sim():
    theta = sympy.Symbol('theta')
    num_points = 10
    max_angle = np.pi
    repetitions = 1000
    num_qubits = 4
    _, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal = True
    circuit = cirq.Circuit(
        cirq.X(qubits[3]) ** theta,
        cirq.X(qubits[0]),
        cirq.X(qubits[0]),
        cirq.X(qubits[1]),
        cirq.X(qubits[1]),
        cirq.X(qubits[2]),
        cirq.X(qubits[2]),
    )
    circuit.append(cirq.PhasedXPowGate(phase_exponent=0.5, exponent=-0.5).on(qubits[0]))
    circuit.append(cirq.PhasedXPowGate(phase_exponent=0.5, exponent=0.5).on(qubits[0]))
    sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
    results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)
    excited_state_probs = np.zeros(num_points)
    for i in range(num_points):
        excited_state_probs[i] = np.mean(results[i].measurements['m'])
    assert excited_state_probs[-1] == 0.25


def test_aqt_sampler_sim_xtalk():
    num_points = 10
    max_angle = np.pi
    repetitions = 100
    num_qubits = 4
    _, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    sampler.simulate_ideal = False
    circuit = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.X(qubits[1]),
        cirq.X(qubits[1]),
        cirq.X(qubits[3]),
        cirq.X(qubits[2]),
    )
    sweep = cirq.Linspace(key='theta', start=0.1, stop=max_angle / np.pi, length=num_points)
    _results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)


def test_aqt_sampler_ms():
    repetitions = 1000
    num_qubits = 4
    _, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerLocalSimulator()
    circuit = cirq.Circuit(cirq.Z.on_each(*qubits), cirq.Z.on_each(*qubits))
    for _dummy in range(9):
        circuit.append(cirq.XX(qubits[0], qubits[1]) ** 0.5)
    circuit.append(cirq.Z(qubits[0]) ** 0.5)
    results = sampler.run(circuit, repetitions=repetitions)
    hist = results.histogram(key='m')
    assert hist[12] > repetitions / 3
    assert hist[0] > repetitions / 3


def test_aqt_device_wrong_op_str():
    circuit = cirq.Circuit()
    q0, q1 = cirq.LineQubit.range(2)
    circuit.append(cirq.CNOT(q0, q1) ** 1.0)
    for op in circuit.all_operations():
        with pytest.raises(ValueError):
            _result = get_op_string(op)
