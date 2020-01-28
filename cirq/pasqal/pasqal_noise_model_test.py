"""Tests for pasqal_qubits."""

import pytest
import cirq

from cirq import ops, Circuit
from cirq import CZ, Z
from cirq.pasqal import ThreeDGridQubit, PasqalNoiseModel, PasqalDevice
from cirq.pasqal.pasqal_noise_model import get_op_string

def test_NoiseModel_init():
    noise_model = PasqalNoiseModel()
    assert noise_model.noise_op_dict=={
                                       'X': ops.depolarize(1e-2),
                                       'Y': ops.depolarize(1e-2),
                                       'Z': ops.depolarize(1e-2),
                                       'CX': ops.depolarize(3e-2),
                                       'CZ': ops.depolarize(3e-2),
                                       'CCX': ops.depolarize(8e-2),
                                       'CCZ': ops.depolarize(8e-2),
                                       }


def test_noisy_moments():
    noise_model = PasqalNoiseModel()
    p_qubits= ThreeDGridQubit.cube(4)
    p_device = PasqalDevice(control_radius=2,qubits=p_qubits)
    circuit = Circuit()
    circuit.append(CZ(p_qubits[0], p_qubits[1]))
    circuit.append(Z(p_qubits[1]))

    # p_circuit = PasqalCircuit(circuit,device=p_device)
    p_circuit = Circuit(circuit, device=p_device)

    n_mts=[]
    for moment in p_circuit._moments:
        n_mts.append(noise_model.noisy_moment(moment, p_qubits))
    assert n_mts == [[CZ.on(ThreeDGridQubit(0, 0, 0),
                                 ThreeDGridQubit(0, 0, 1)),
                      cirq.depolarize(p=0.03).on(ThreeDGridQubit(0, 0, 0)),
                      cirq.depolarize(p=0.03).on(ThreeDGridQubit(0, 0, 1))],
                     [Z.on(ThreeDGridQubit(0, 0, 1)),
                      cirq.depolarize(p=0.01).on(ThreeDGridQubit(0, 0, 1))]]


def test_get_op_string():
    noise_model = PasqalNoiseModel()
    p_qubits= ThreeDGridQubit.cube(4)
    circuit = Circuit()
    circuit.append(ops.H(p_qubits[0]))
    with pytest.raises(ValueError, match='Got unknown gate:'):
        for moment in circuit._moments:
            _=noise_model.noisy_moment(moment, p_qubits)
