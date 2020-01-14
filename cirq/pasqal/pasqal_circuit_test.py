"""Tests for pasqal_circuit."""

import pytest
import numpy as np
import cirq
from cirq.circuits import InsertStrategy

from . import pasqal_circuit, pasqal_qubits, pasqal_device

def test_pasqal_circuit_init():
    qs = pasqal_qubits.ThreeDGridQubit.square(3)

    ex_circuit = cirq.Circuit()
    ex_circuit.append([[cirq.CZ(qs[i],qs[i+1]), cirq.X(qs[i+1])]
                        for i in range(len(qs)-1)])

    device = pasqal_device.PasqalDevice(control_radius = 3, qubits = qs )

    test_circuit = pasqal_circuit.PasqalCircuit(ex_circuit, device = device)

    with pytest.raises(ValueError, match = "PasqalDevice necessary for constructor!"):
        pasqal_circuit.PasqalCircuit(ex_circuit, device = cirq.UNCONSTRAINED_DEVICE)

    with pytest.raises(ValueError, match = "PasqalDevice necessary for constructor!"):
        pasqal_circuit.PasqalCircuit(ex_circuit, device = None)

    for moment1, moment2 in zip(test_circuit, ex_circuit):
        assert moment1 == moment2

# def test_simulate_samples():
#     '''
#     Encodes a random binary number in the qubits, samples once without noise and
#     checks if the result matches the original number.
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
#     data = pasqal_circuit.PasqalSampler().simulate_samples(program=ex_circuit,
#                                                            simulate_ideal=True,
#                                                            repetitions = 1).data.to_dict()
#     for i, q in enumerate(qs):
#         assert data['({}, {}, {})'.format(q.row, q.col, q.lay)][0] == int(binary[-i-1])

def test_run():
    '''
    Encodes a random binary number in the qubits, samples once without noise and
    checks if the result matches the original number.
    '''

    qs = [cirq.GridQubit(i,j) for i in range(3) for j in range(3)]

    num = np.random.randint(0,2**9)
    binary = bin(num)[2:].zfill(9)

    ex_circuit = cirq.Circuit()
    for i,b in enumerate(binary):
        if b == '1':
            ex_circuit.append(cirq.X(qs[-i-1]))
    ex_circuit.append([cirq.measure(q) for q in qs])

    sampler = pasqal_circuit.PasqalSampler(remote_host= 'http://34.98.71.118/v0/pasqal')
    data = sampler.run(program=ex_circuit,
                       simulate_ideal=True,
                       repetitions = 1).data.to_dict()
    for i, q in enumerate(qs):
        assert data['({}, {})'.format(q.row, q.col)][0] == int(binary[-i-1])
