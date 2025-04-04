# Copyright 2022 The Cirq Developers
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

from typing import Dict, List, Tuple

import numpy as np
import pytest

import cirq
import cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq.ops.fsim_gate import PhasedFSimGate
from cirq_google.devices.google_noise_properties import (
    GoogleNoiseProperties,
    NoiseModelFromGoogleNoiseProperties,
)

DEFAULT_GATE_NS: Dict[type, float] = {
    cirq.ZPowGate: 25.0,
    cirq.MeasurementGate: 4000.0,
    cirq.ResetChannel: 250.0,
    cirq.PhasedXZGate: 25.0,
    cirq.FSimGate: 32.0,
    # SYC is normally 12ns, but setting it equal to other two-qubit gates
    # simplifies the tests.
    cirq_google.SycamoreGate: 32.0,
    cirq.PhasedFSimGate: 32.0,
    cirq.ISwapPowGate: 32.0,
    cirq.CZPowGate: 32.0,
    # cirq.WaitGate is a special case.
}

# Mock pauli error rates for 1- and 2-qubit gates.
SINGLE_QUBIT_ERROR = 0.001
TWO_QUBIT_ERROR = 0.01


# These properties are for testing purposes only - they are not representative
# of device behavior for any existing hardware.
def sample_noise_properties(
    system_qubits: List[cirq.Qid], qubit_pairs: List[Tuple[cirq.Qid, cirq.Qid]]
):
    # Known false positive: https://github.com/PyCQA/pylint/issues/5857
    return GoogleNoiseProperties(  # pylint: disable=unexpected-keyword-arg
        gate_times_ns=DEFAULT_GATE_NS,
        t1_ns={q: 1e5 for q in system_qubits},
        tphi_ns={q: 2e5 for q in system_qubits},
        readout_errors={q: [SINGLE_QUBIT_ERROR, TWO_QUBIT_ERROR] for q in system_qubits},
        gate_pauli_errors={
            **{
                OpIdentifier(g, q): 0.001
                for g in GoogleNoiseProperties.single_qubit_gates()
                for q in system_qubits
            },
            **{
                OpIdentifier(g, q0, q1): 0.01
                for g in GoogleNoiseProperties.symmetric_two_qubit_gates()
                for q0, q1 in qubit_pairs
            },
        },
        fsim_errors={
            OpIdentifier(g, q0, q1): cirq.PhasedFSimGate(0.01, 0.03, 0.04, 0.05, 0.02)
            for g in GoogleNoiseProperties.symmetric_two_qubit_gates()
            for q0, q1 in qubit_pairs
        },
    )


def test_consistent_repr():
    q0, q1 = cirq.LineQubit.range(2)
    test_props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    cirq.testing.assert_equivalent_repr(
        test_props, setup_code="import cirq, cirq_google\nimport numpy as np"
    )


def test_equals():
    q0, q1, q2 = cirq.LineQubit.range(3)
    test_props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    assert test_props != "mismatched_type"
    test_props_v2 = test_props.with_params(readout_errors={q2: [0.01, 0.02]})
    assert test_props != test_props_v2


def test_zphase_gates():
    q0 = cirq.LineQubit(0)
    props = sample_noise_properties([q0], [])
    model = NoiseModelFromGoogleNoiseProperties(props)
    circuit = cirq.Circuit(cirq.Z(q0) ** 0.3)
    noisy_circuit = circuit.with_noise(model)
    assert noisy_circuit == circuit


def test_with_params_fill():
    # Test single-value with_params.
    q0, q1 = cirq.LineQubit.range(2)
    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    expected_vals = {
        'gate_times_ns': 91,
        't1_ns': 92,
        'tphi_ns': 93,
        'readout_errors': [0.094, 0.095],
        'gate_pauli_errors': 0.096,
        'fsim_errors': cirq.PhasedFSimGate(0.0971, 0.0972, 0.0973, 0.0974, 0.0975),
    }
    props_v2 = props.with_params(
        gate_times_ns=expected_vals['gate_times_ns'],
        t1_ns=expected_vals['t1_ns'],
        tphi_ns=expected_vals['tphi_ns'],
        readout_errors=expected_vals['readout_errors'],
        gate_pauli_errors=expected_vals['gate_pauli_errors'],
        fsim_errors=expected_vals['fsim_errors'],
    )
    assert props_v2 != props
    for key in props.gate_times_ns:
        assert key in props_v2.gate_times_ns
        assert props_v2.gate_times_ns[key] == expected_vals['gate_times_ns']
    for key in props.t1_ns:
        assert key in props_v2.t1_ns
        assert props_v2.t1_ns[key] == expected_vals['t1_ns']
    for key in props.tphi_ns:
        assert key in props_v2.tphi_ns
        assert props_v2.tphi_ns[key] == expected_vals['tphi_ns']
    for key in props.readout_errors:
        assert key in props_v2.readout_errors
        assert np.allclose(props_v2.readout_errors[key], expected_vals['readout_errors'])
    for key in props.gate_pauli_errors:
        assert key in props_v2.gate_pauli_errors
        assert props_v2.gate_pauli_errors[key] == expected_vals['gate_pauli_errors']
    for key in props.fsim_errors:
        assert key in props_v2.fsim_errors
        assert props_v2.fsim_errors[key] == expected_vals['fsim_errors']


def test_with_params_target():
    # Test targeted-value with_params.
    q0, q1 = cirq.LineQubit.range(2)
    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    expected_vals = {
        'gate_times_ns': {cirq.ZPowGate: 91},
        't1_ns': {q0: 92},
        'tphi_ns': {q1: 93},
        'readout_errors': {q0: [0.094, 0.095]},
        'gate_pauli_errors': {cirq.OpIdentifier(cirq.PhasedXZGate, q1): 0.096},
        'fsim_errors': {
            cirq.OpIdentifier(cirq.CZPowGate, q0, q1): cirq.PhasedFSimGate(
                0.0971, 0.0972, 0.0973, 0.0974, 0.0975
            )
        },
    }
    props_v2 = props.with_params(
        gate_times_ns=expected_vals['gate_times_ns'],
        t1_ns=expected_vals['t1_ns'],
        tphi_ns=expected_vals['tphi_ns'],
        readout_errors=expected_vals['readout_errors'],
        gate_pauli_errors=expected_vals['gate_pauli_errors'],
        fsim_errors=expected_vals['fsim_errors'],
    )
    assert props_v2 != props
    for field_name, expected in expected_vals.items():
        target_dict = getattr(props_v2, field_name)
        for key, val in expected.items():
            if isinstance(target_dict[key], np.ndarray):
                assert np.allclose(target_dict[key], val)
            else:
                assert target_dict[key] == val


def test_with_params_opid_with_gate():
    # Test gate-based opid with_params.
    q0, q1 = cirq.LineQubit.range(2)
    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    expected_vals = {
        'gate_pauli_errors': 0.096,
        'fsim_errors': cirq.PhasedFSimGate(0.0971, 0.0972, 0.0973, 0.0974, 0.0975),
    }
    props_v2 = props.with_params(
        gate_pauli_errors={cirq.PhasedXZGate: expected_vals['gate_pauli_errors']},
        fsim_errors={cirq.CZPowGate: expected_vals['fsim_errors']},
    )
    assert props_v2 != props
    gpe_op_id_0 = cirq.OpIdentifier(cirq.PhasedXZGate, q0)
    gpe_op_id_1 = cirq.OpIdentifier(cirq.PhasedXZGate, q1)
    assert props_v2.gate_pauli_errors[gpe_op_id_0] == expected_vals['gate_pauli_errors']
    assert props_v2.gate_pauli_errors[gpe_op_id_1] == expected_vals['gate_pauli_errors']

    fsim_op_id_0 = cirq.OpIdentifier(cirq.CZPowGate, q0, q1)
    fsim_op_id_1 = cirq.OpIdentifier(cirq.CZPowGate, q1, q0)
    assert props_v2.fsim_errors[fsim_op_id_0] == expected_vals['fsim_errors']
    assert props_v2.fsim_errors[fsim_op_id_1] == expected_vals['fsim_errors']


@pytest.mark.parametrize(
    'op',
    [
        (cirq.Z(cirq.LineQubit(0)) ** 0.3).with_tags(cirq_google.PhysicalZTag()),
        cirq.PhasedXZGate(x_exponent=0.8, z_exponent=0.2, axis_phase_exponent=0.1).on(
            cirq.LineQubit(0)
        ),
    ],
)
def test_single_qubit_gates(op):
    q0 = cirq.LineQubit(0)
    props = sample_noise_properties([q0], [])
    model = NoiseModelFromGoogleNoiseProperties(props)
    circuit = cirq.Circuit(op)
    noisy_circuit = circuit.with_noise(model)
    assert len(noisy_circuit.moments) == 3
    assert len(noisy_circuit.moments[0].operations) == 1
    assert noisy_circuit.moments[0].operations[0] == op.with_tags(PHYSICAL_GATE_TAG)

    # Depolarizing noise
    assert len(noisy_circuit.moments[1].operations) == 1
    depol_op = noisy_circuit.moments[1].operations[0]
    assert isinstance(depol_op.gate, cirq.DepolarizingChannel)
    assert np.isclose(depol_op.gate.p, 0.00081252)

    # Thermal noise
    assert len(noisy_circuit.moments[2].operations) == 1
    thermal_op = noisy_circuit.moments[2].operations[0]
    assert isinstance(thermal_op.gate, cirq.KrausChannel)
    thermal_choi = cirq.kraus_to_choi(cirq.kraus(thermal_op))
    assert np.allclose(
        thermal_choi,
        [
            [1, 0, 0, 9.99750031e-01],
            [0, 2.49968753e-04, 0, 0],
            [0, 0, 0, 0],
            [9.99750031e-01, 0, 0, 9.99750031e-01],
        ],
    )

    # Pauli error for depol_op + thermal_op == total (0.001)
    depol_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(depol_op)
    thermal_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(thermal_op)
    total_err = depol_pauli_err + thermal_pauli_err
    assert np.isclose(total_err, SINGLE_QUBIT_ERROR)


@pytest.mark.parametrize(
    'op',
    [
        cirq.ISWAP(*cirq.LineQubit.range(2)) ** 0.6,
        cirq.CZ(*cirq.LineQubit.range(2)) ** 0.3,
        cirq_google.SYC(*cirq.LineQubit.range(2)),
    ],
)
def test_two_qubit_gates(op):
    q0, q1 = cirq.LineQubit.range(2)
    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    model = NoiseModelFromGoogleNoiseProperties(props)
    circuit = cirq.Circuit(op)
    noisy_circuit = circuit.with_noise(model)
    assert len(noisy_circuit.moments) == 4
    assert len(noisy_circuit.moments[0].operations) == 1
    assert noisy_circuit.moments[0].operations[0] == op.with_tags(PHYSICAL_GATE_TAG)

    # Depolarizing noise
    assert len(noisy_circuit.moments[1].operations) == 1
    depol_op = noisy_circuit.moments[1].operations[0]
    assert isinstance(depol_op.gate, cirq.DepolarizingChannel)
    assert np.isclose(depol_op.gate.p, 0.00719705)

    # FSim angle corrections
    assert len(noisy_circuit.moments[2].operations) == 1
    fsim_op = noisy_circuit.moments[2].operations[0]
    assert isinstance(fsim_op.gate, cirq.PhasedFSimGate)
    assert fsim_op == PhasedFSimGate(theta=0.01, zeta=0.03, chi=0.04, gamma=0.05, phi=0.02).on(
        q0, q1
    )

    # Thermal noise
    assert len(noisy_circuit.moments[3].operations) == 2
    thermal_op_0 = noisy_circuit.moments[3].operation_at(q0)
    thermal_op_1 = noisy_circuit.moments[3].operation_at(q1)
    assert isinstance(thermal_op_0.gate, cirq.KrausChannel)
    assert isinstance(thermal_op_1.gate, cirq.KrausChannel)
    thermal_choi_0 = cirq.kraus_to_choi(cirq.kraus(thermal_op_0))
    thermal_choi_1 = cirq.kraus_to_choi(cirq.kraus(thermal_op_1))
    expected_thermal_choi = np.array(
        [
            [1, 0, 0, 9.99680051e-01],
            [0, 3.19948805e-04, 0, 0],
            [0, 0, 0, 0],
            [9.99680051e-01, 0, 0, 9.99680051e-01],
        ]
    )
    assert np.allclose(thermal_choi_0, expected_thermal_choi)
    assert np.allclose(thermal_choi_1, expected_thermal_choi)

    # Pauli error for depol_op + fsim_op + thermal_op_(0|1) == total (0.01)
    depol_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(depol_op)
    fsim_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(fsim_op)
    thermal0_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(thermal_op_0)
    thermal1_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(thermal_op_1)
    total_err = depol_pauli_err + thermal0_pauli_err + thermal1_pauli_err + fsim_pauli_err
    assert np.isclose(total_err, TWO_QUBIT_ERROR)


def test_supertype_match():
    # Verifies that ops in gate_pauli_errors which only appear as their
    # supertypes in fsim_errors are properly accounted for.
    q0, q1 = cirq.LineQubit.range(2)
    op_id = OpIdentifier(cirq_google.SycamoreGate, q0, q1)
    test_props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    expected_err = test_props._depolarizing_error[op_id]

    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    props.fsim_errors = {
        k: cirq.PhasedFSimGate(0.5, 0.4, 0.3, 0.2, 0.1)
        for k in [OpIdentifier(cirq.FSimGate, q0, q1), OpIdentifier(cirq.FSimGate, q1, q0)]
    }
    assert props._depolarizing_error[op_id] != expected_err


def test_measure_gates():
    q00, q01, q10, q11 = cirq.GridQubit.rect(2, 2)
    qubits = [q00, q01, q10, q11]
    props = sample_noise_properties(
        qubits,
        [
            (q00, q01),
            (q01, q00),
            (q10, q11),
            (q11, q10),
            (q00, q10),
            (q10, q00),
            (q01, q11),
            (q11, q01),
        ],
    )
    model = NoiseModelFromGoogleNoiseProperties(props)
    op = cirq.measure(*qubits, key='m')
    circuit = cirq.Circuit(cirq.measure(*qubits, key='m'))
    noisy_circuit = circuit.with_noise(model)
    # Measurement gates are prepended by amplitude damping, and nothing else.
    assert len(noisy_circuit.moments) == 2

    # Amplitude damping before measurement
    assert len(noisy_circuit.moments[0].operations) == 4
    for q in qubits:
        op = noisy_circuit.moments[0].operation_at(q)
        assert isinstance(op.gate, cirq.GeneralizedAmplitudeDampingChannel), q
        assert np.isclose(op.gate.p, 0.90909090), q
        assert np.isclose(op.gate.gamma, 0.011), q

    # Original measurement is after the noise.
    assert len(noisy_circuit.moments[1].operations) == 1
    # Measurements are untagged during reconstruction.
    assert noisy_circuit.moments[1] == circuit.moments[0]


def test_wait_gates():
    q0 = cirq.LineQubit(0)
    props = sample_noise_properties([q0], [])
    model = NoiseModelFromGoogleNoiseProperties(props)
    op = cirq.wait(q0, nanos=100)
    circuit = cirq.Circuit(op)
    noisy_circuit = circuit.with_noise(model)
    assert len(noisy_circuit.moments) == 2
    assert noisy_circuit.moments[0].operations[0] == op.with_tags(PHYSICAL_GATE_TAG)

    # No depolarizing noise because WaitGate has none.

    assert len(noisy_circuit.moments[1].operations) == 1
    thermal_op = noisy_circuit.moments[1].operations[0]
    assert isinstance(thermal_op.gate, cirq.KrausChannel)
    thermal_choi = cirq.kraus_to_choi(cirq.kraus(thermal_op))
    assert np.allclose(
        thermal_choi,
        [
            [1, 0, 0, 9.990005e-01],
            [0, 9.99500167e-04, 0, 0],
            [0, 0, 0, 0],
            [9.990005e-01, 0, 0, 9.990005e-01],
        ],
    )
