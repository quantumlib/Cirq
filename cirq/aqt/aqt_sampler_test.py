import mock
import numpy as np
import pytest
import sympy

from cirq import LineQubit, X, Y, Z, XX, Circuit, study
from cirq.aqt import AQTSampler, AQTSamplerSim
from cirq.aqt.aqt_device import get_aqt_device


class EngineReturn:
    """A put mock class for testing the REST interface"""

    def __init__(self):
        self.test_dict = {
            'status': 'queued',
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
    put_call_args = {
        'data': '[["X", 0.1, [0]]]',
        'acccess_token': 'testkey',
        'repetitions': 10,
        'num_qubits': 1
    }
    e_return = EngineReturn()
    with mock.patch('cirq.aqt.aqt_sampler.put',
                    return_value=e_return,
                    side_effect=e_return.update) as mock_method:
        theta = sympy.Symbol('theta')
        num_points = 1
        max_angle = np.pi
        repetitions = 10
        sampler = AQTSampler()
        qubit = LineQubit(0)
        circuit = Circuit.from_ops(X(qubit)**theta)
        sweep = study.Linspace(key='theta',
                               start=0.1,
                               stop=max_angle / np.pi,
                               length=num_points)
        results = sampler.run_sweep(circuit,
                                    params=sweep,
                                    repetitions=repetitions,
                                    num_qubits=1,
                                    remote_host="http://localhost:5000",
                                    access_token='testkey')
        excited_state_probs = np.zeros(num_points)
        for i in range(num_points):
            excited_state_probs[i] = np.mean(results[i].measurements['m'])
    callargs = mock_method.call_args[1]['data']
    for keys in put_call_args:
        assert callargs[keys] == put_call_args[keys]
    assert mock_method.call_count == 2


def test_aqt_sampler_sim():
    theta = sympy.Symbol('theta')
    num_points = 10
    max_angle = np.pi
    repetitions = 100
    num_qubits = 4
    _device, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerSim()
    sampler.simulate_ideal = True
    circuit = Circuit.from_ops(X(qubits[3])**theta)
    sweep = study.Linspace(key='theta',
                           start=0.1,
                           stop=max_angle / np.pi,
                           length=num_points)
    results = sampler.run_sweep(circuit,
                                params=sweep,
                                repetitions=repetitions,
                                num_qubits=num_qubits)
    excited_state_probs = np.zeros(num_points)
    # print(results)
    for i in range(num_points):
        excited_state_probs[i] = np.mean(results[i].measurements['m'])
    print(excited_state_probs[-1])
    assert excited_state_probs[-1] == 0.25


def test_aqt_sampler_sim_xtalk():
    num_points = 10
    max_angle = np.pi
    repetitions = 100
    num_qubits = 4
    _device, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerSim()
    sampler.simulate_ideal = False
    circuit = Circuit.from_ops(X(qubits[0]), X(qubits[3]), X(qubits[2]))
    sweep = study.Linspace(key='theta',
                           start=0.1,
                           stop=max_angle / np.pi,
                           length=num_points)
    _results = sampler.run_sweep(circuit,
                                 params=sweep,
                                 repetitions=repetitions,
                                 num_qubits=num_qubits)


def test_aqt_sampler_ms():
    # TODO: Check big/little endian of result
    repetitions = 100
    num_qubits = 4
    device, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerSim()
    circuit = Circuit(device=device)
    for _dummy in range(9):
        circuit.append(XX(qubits[0], qubits[1])**0.5)
    results = sampler.run(circuit,
                          repetitions=repetitions,
                          num_qubits=num_qubits)
    hist = (results.histogram(key='m'))
    print(hist)
    assert hist[12] > repetitions / 3
    assert hist[0] > repetitions / 3


def test_aqt_sampler_wrong_gate():
    repetitions = 100
    num_qubits = 4
    device, qubits = get_aqt_device(num_qubits)
    sampler = AQTSamplerSim()
    circuit = Circuit(device=device)
    circuit.append(Y(qubits[0])**0.5)
    circuit.append(Z(qubits[0])**0.5)
    with pytest.raises(RuntimeError):
        _results = sampler.run(circuit,
                               repetitions=repetitions,
                               num_qubits=num_qubits)
