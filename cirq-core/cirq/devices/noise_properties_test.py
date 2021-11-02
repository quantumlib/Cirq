# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import List, Tuple
import numpy as np
import pytest
import cirq, cirq_google

# from cirq.testing import assert_equivalent_op_tree
from cirq.devices.noise_properties import (
    FSimEntanglingAngles,
    FSimPhaseAngles,
    NoiseProperties,
    NoiseModelFromNoiseProperties,
    Z_GATE,
    MEASURE_GATE,
    RESET_GATE,
    MW_GATE,
    FSIM_GATE,
    ISWAP_GATE,
    CZ_GATE,
    SYMMETRIC_TWO_QUBIT_GATES,
    SINGLE_QUBIT_GATES,
    TWO_QUBIT_GATES,
)


DEFAULT_GATE_NS = {
    Z_GATE: 25.0,
    MEASURE_GATE: 4000.0,
    RESET_GATE: 250.0,
    MW_GATE: 25.0,
    FSIM_GATE: 32.0,
    ISWAP_GATE: 32.0,
    CZ_GATE: 32.0,
    # WAIT_GATE is a special case.
}

# These properties are for testing purposes only - they are not representative
# of device behavior for any existing hardware.
def sample_noise_properties(
    system_qubits: List[cirq.Qid], qubit_pairs: List[Tuple[cirq.Qid, cirq.Qid]]
):
    return NoiseProperties(
        gate_times_ns=DEFAULT_GATE_NS,
        T1_ns={q: 1e5 for q in system_qubits},
        Tphi_ns={q: 2e5 for q in system_qubits},
        ro_fidelities={q: np.array([0.001, 0.01]) for q in system_qubits},
        entangler_errors={
            (g, q0, q1): FSimEntanglingAngles(0.01, 0.02)
            for g in SYMMETRIC_TWO_QUBIT_GATES
            for q0, q1 in qubit_pairs
        },
        gate_pauli_errors={
            **{(g, q): 0.001 for g in SINGLE_QUBIT_GATES for q in system_qubits},
            **{(g, q0, q1): 0.01 for g in TWO_QUBIT_GATES for q0, q1 in qubit_pairs},
        },
        z_phase_errors={
            (g, q0, q1): FSimPhaseAngles(0.03, 0.04, 0.05)
            for g in SYMMETRIC_TWO_QUBIT_GATES
            for q0, q1 in qubit_pairs
        },
    )


def test_model_from_props():
    system_qubits = cirq.GridQubit.rect(2, 2)
    qubit_pairs = [
        (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
        (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
        (cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)),
        (cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
        (cirq.GridQubit(1, 1), cirq.GridQubit(0, 1)),
        (cirq.GridQubit(1, 1), cirq.GridQubit(1, 0)),
        (cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)),
        (cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
    ]
    props = sample_noise_properties(system_qubits, qubit_pairs)
    model = NoiseModelFromNoiseProperties(props)

    circuit = cirq.Circuit(
        cirq.H(system_qubits[0]),
        cirq.H(system_qubits[2]),
        cirq.CNOT(*system_qubits[0:2]),
        cirq.CNOT(*system_qubits[2:4]),
        cirq.measure(*system_qubits, key='m'),
    )
    print('circuit')
    print(circuit)
    syc_circuit = cirq_google.optimized_for_sycamore(circuit)
    print('syc_circuit')
    print(syc_circuit)
    noisy_circuit = syc_circuit.with_noise(model)
    print('noisy_circuit')
    print(noisy_circuit.moments)
    print(cirq.kraus_to_choi(cirq.kraus(noisy_circuit.moments[4].operations[0])))

    # assert False


# TODO: rewrite tests
# Things to test:
# - each possible native gate is converted properly, including:
#   Z phase, microwave, [sqrt_]iswap, sycamore (cirq_google), CZ, measure
# - Z's are virtual unless tagged as physical (cirq_google.ops.PhysicalZTag)


def test_zphase_gates():
    # TODO: Includes cirq.Z, cirq.Z ** t
    q0 = cirq.LineQubit(0)
    props = sample_noise_properties([q0], [])
    model = NoiseModelFromNoiseProperties(props)
    circuit = cirq.Circuit(cirq.Z(q0))
    noisy_circuit = circuit.with_noise(model)
    print(noisy_circuit.moments)
    assert len(noisy_circuit.moments) == 2
    assert len(noisy_circuit.moments[1].operations) == 1
    noise_op = noisy_circuit.moments[1].operations[0]
    assert isinstance(noise_op.gate, cirq.KrausChannel)
    noise_choi = cirq.kraus_to_choi(cirq.kraus(noise_op))
    assert np.allclose(
        noise_choi,
        [
            [1, 0, 0, 9.99750031e-01],
            [0, 2.49968753e-04, 0, 0],
            [0, 0, 0, 0],
            [9.99750031e-01, 0, 0, 9.99750031e-01],
        ],
    )
    assert False


def test_physical_zphase_gates():
    # TODO: Includes cirq.Z, cirq.Z ** t with PhysicalZTag
    assert False


def test_microwave_gates():
    # TODO: Includes cirq.X, cirq.Y
    assert False


def test_iswap_gates():
    # TODO: Includes cirq.ISWAP, cirq.ISWAP ** 0.5
    assert False


def test_cz_gates():
    # TODO: includes cirq.CZ
    assert False


def test_measure_gates():
    # TODO: includes single- and multi-qubit measure
    assert False
