import mock
import numpy as np
import pytest
import sympy

from cirq import X, Y, Z, XX, Circuit, study
from cirq.aqt import AQTSampler, AQTRemoteSimulator
from cirq.aqt.aqt_device import get_aqt_device


class EngineReturn:
    """A put mock class for testing the REST interface"""

    def __init__(self):
        self.test_dict = {
            'status': 'queued',
            'id': '2131da',
            'samples': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        self.counter = 0

    def json(self):
        self.counter += 1
        return self.test_dict

    def update(self, *args, **kwargs):
        if self.counter >= 1:
            self.test_dict['status'] = 'finished'
        return self


def test_aqt_sampler():

    put_call_args0 = {
        'access_token': 'testkey',
        'id': '2131da',
    }
    # put_call_args = {
    #     'data': '[["X", 0.1, [0]]]',
    #     'access_token': 'testkey',
    #     'repetitions': 10,
    #     'id': '2131da',
    #     'num_qubits': 1
# }x

    e_return = EngineReturn()
    with mock.patch('cirq.aqt.aqt_sampler.put',
                    return_value=e_return,
                    side_effect=e_return.update) as mock_method:
        theta = sympy.Symbol('theta')
        num_points = 1
        max_angle = np.pi
        repetitions = 10
        sampler = AQTSampler(remote_host="http://localhost:5000",
                             access_token='testkey')
        device, qubits = get_aqt_device(1)
        circuit = Circuit.from_ops(X(qubits[0])**theta, device=device)
        sweep = study.Linspace(key='theta',
                               start=0.1,
                               stop=max_angle / np.pi,
                               length=num_points)
        results = sampler.run_sweep(circuit,
                                    params=sweep,
                                    repetitions=repetitions)
        excited_state_probs = np.zeros(num_points)
        for i in range(num_points):
            excited_state_probs[i] = np.mean(results[i].measurements['m'])
    callargs = mock_method.call_args[1]['data']
    for keys in put_call_args0:
        assert callargs[keys] == put_call_args0[keys]
    assert mock_method.call_count == 2


def test_aqt_sampler_sim():
    theta = sympy.Symbol('theta')
    num_points = 10
    max_angle = np.pi
    repetitions = 1000
    num_qubits = 4
    device, qubits = get_aqt_device(num_qubits)
    sampler = AQTRemoteSimulator()
    sampler.simulate_ideal = True
    circuit = Circuit.from_ops(X(qubits[3])**theta, device=device)
    sweep = study.Linspace(key='theta',
                           start=0.1,
                           stop=max_angle / np.pi,
                           length=num_points)
    results = sampler.run_sweep(circuit, params=sweep, repetitions=repetitions)
    excited_state_probs = np.zeros(num_points)
    # print(results)
    for i in range(num_points):
        excited_state_probs[i] = np.mean(results[i].measurements['m'])
    assert excited_state_probs[-1] == 0.25


def test_aqt_sampler_sim_xtalk():
    num_points = 10
    max_angle = np.pi
    repetitions = 100
    num_qubits = 4
    device, qubits = get_aqt_device(num_qubits)
    sampler = AQTRemoteSimulator()
    sampler.simulate_ideal = False
    circuit = Circuit.from_ops(X(qubits[0]),
                               X(qubits[3]),
                               X(qubits[2]),
                               device=device)
    sweep = study.Linspace(key='theta',
                           start=0.1,
                           stop=max_angle / np.pi,
                           length=num_points)
    _results = sampler.run_sweep(circuit,
                                 params=sweep,
                                 repetitions=repetitions)


def test_aqt_sampler_ms():
    repetitions = 1000
    num_qubits = 4
    device, qubits = get_aqt_device(num_qubits)
    sampler = AQTRemoteSimulator()
    circuit = Circuit(device=device)
    for _dummy in range(9):
        circuit.append(XX(qubits[0], qubits[1])**0.5)
    results = sampler.run(circuit, repetitions=repetitions)
    hist = (results.histogram(key='m'))
    print(hist)
    assert hist[12] > repetitions / 3
    assert hist[0] > repetitions / 3


def test_aqt_sampler_wrong_gate():
    repetitions = 100
    num_qubits = 4
    device, qubits = get_aqt_device(num_qubits)
    sampler = AQTRemoteSimulator()
    circuit = Circuit(device=device)
    circuit.append(Y(qubits[0])**0.5)
    circuit.append(Z(qubits[0])**0.5)
    with pytest.raises(ValueError):
        _results = sampler.run(circuit, repetitions=repetitions)
