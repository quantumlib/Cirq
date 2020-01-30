"""Tests for pasqal_circuit."""
from os import getenv

import pytest
import numpy as np
import sympy
import cirq

from . import pasqal_circuit, pasqal_qubits, pasqal_device


def _make_sampler() -> pasqal_circuit.PasqalSampler:
    # Retrieve API access token from environment variable to avoid storing it in the source
    sampler = pasqal_circuit.PasqalSampler(
        remote_host='http://34.98.71.118/v0/pasqal',
        access_token=getenv("PASQAL_API_ACCESS_TOKEN")
    )
    return sampler


def test_pasqal_circuit_init():
    qs = pasqal_qubits.ThreeDGridQubit.square(3)

    ex_circuit = cirq.Circuit()
    ex_circuit.append([[cirq.CZ(qs[i], qs[i + 1]), cirq.X(qs[i + 1])]
                       for i in range(len(qs) - 1)])

    device = pasqal_device.PasqalDevice(control_radius=3, qubits=qs)

    # test_circuit = pasqal_circuit.PasqalCircuit(ex_circuit, device = device)

    test_circuit = cirq.Circuit(device=device)
    test_circuit.append([[cirq.CZ(qs[i], qs[i + 1]), cirq.X(qs[i + 1])]
                         for i in range(len(qs) - 1)])

    # with pytest.raises(ValueError,
    #                    match = "PasqalDevice necessary for constructor!"):
    #     pasqal_circuit.PasqalCircuit(ex_circuit,
    #                                  device = cirq.UNCONSTRAINED_DEVICE)
    #
    # with pytest.raises(ValueError,
    #                    match = "PasqalDevice necessary for constructor!"):
    #     pasqal_circuit.PasqalCircuit(ex_circuit, device = None)

    for moment1, moment2 in zip(test_circuit, ex_circuit):
        assert moment1 == moment2

# def test_simulate_samples():
#     '''
#     Encodes a random binary number in the qubits,
#     samples once without noise and checks if the result
#     matches the original number.
#     '''
#
#     qs = pasqal_qubits.ThreeDGridQubit.square(3)
#
#     num = np.random.randint(0,2**9)
#     binary = bin(num)[2:].zfill(9)
#
#     ex_circuit = cirq.Circuit()
#     for i,b in enumerate(binary):
#         if b == '1':
#             ex_circuit.append(cirq.X(qs[-i-1]))
#     ex_circuit.append([cirq.measure(q) for q in qs])
#
#     data = pasqal_circuit.PasqalSampler().simulate_samples(
#         program=ex_circuit,
#         simulate_ideal=True,
#         repetitions = 1
#     ).data.to_dict()
#     for i, q in enumerate(qs):
#         assert data['({}, {}, {})'.format(q.row, q.col, q.lay)][0] ==
#         int(binary[-i-1])


def test_run():
    '''
    Encodes a random binary number in the qubits, samples once without noise and
    checks if the result matches the original number.
    '''

    qs = [pasqal_qubits.ThreeDGridQubit(i, j, 0) for i in range(3)
          for j in range(3)]

    num = np.random.randint(0, 2**9)
    binary = bin(num)[2:].zfill(9)

    device = pasqal_device.PasqalDevice(control_radius=1, qubits=qs)

    ex_circuit = cirq.Circuit(device=device)
    for i, b in enumerate(binary):
        if b == '1':
            ex_circuit.append(cirq.X(qs[-i - 1]))
    ex_circuit.append([cirq.measure(q) for q in qs])

    sampler = _make_sampler()
    data = sampler.run(program=ex_circuit,
                       simulate_ideal=True,
                       repetitions=1).data.to_dict()
    for i, q in enumerate(qs):
        assert data['({}, {}, {})'.format(q.row, q.col, q.lay)][0] == int(binary[-i - 1])


def test_run_sweep():
    '''
    Encodes a random binary number in the qubits, sweeps between odd and even
    without noise and checks if the results match.
    '''

    qs = [pasqal_qubits.ThreeDGridQubit(i, j, 0) for i in range(3) for j in range(3)]

    par = sympy.Symbol('par')
    sweep = cirq.Linspace(key='par', start=0.0, stop=1.0, length=2)

    num = np.random.randint(0, 2**9)
    binary = bin(num)[2:].zfill(9)

    device = pasqal_device.PasqalDevice(control_radius=1, qubits=qs)
    ex_circuit = cirq.Circuit(device=device)

    xpow = cirq.XPowGate(exponent=par)
    ex_circuit.append([xpow(qs[0])])
    for i, b in enumerate(binary[:-1]):
        if b == '1':
            ex_circuit.append(cirq.X(qs[-i - 1]))
    ex_circuit.append([cirq.measure(q) for q in qs])

    sampler = _make_sampler()
    data_raw = sampler.run_sweep(program=ex_circuit,
                                 params=sweep,
                                 simulate_ideal=True,
                                 repetitions=1)

    data0 = data_raw[0].data.to_dict()
    data1 = data_raw[1].data.to_dict()

    assert data0['(0, 0, 0)'][0] == 0
    assert data1['(0, 0, 0)'][0] == 1

    for i, q in enumerate(qs[1:], 1):
        assert data0['({}, {}, {})'.format(q.row, q.col, q.lay)][0] == int(binary[-i - 1])
