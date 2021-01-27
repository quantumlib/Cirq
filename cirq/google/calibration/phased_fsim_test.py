# Copyright 2021 The Cirq Developers
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
import pytest
import numpy as np

import cirq
from cirq.google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    PhasedFSimCharacterization,
    PhasedFSimCalibrationResult,
    try_convert_sqrt_iswap_to_fsim,
)


def test_asdict():
    characterization_angles = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    characterization = PhasedFSimCharacterization(**characterization_angles)
    assert characterization.asdict() == characterization_angles


def test_all_none():
    assert PhasedFSimCharacterization().all_none()

    characterization_angles = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    for angle, value in characterization_angles.items():
        assert not PhasedFSimCharacterization(**{angle: value}).all_none()


def test_any_none():
    characterization_angles = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    assert not PhasedFSimCharacterization(**characterization_angles).any_none()

    for angle in characterization_angles:
        none_angles = dict(characterization_angles)
        del none_angles[angle]
        assert PhasedFSimCharacterization(**none_angles).any_none()


def test_parameters_for_qubits_swapped():
    characterization = PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5)
    assert characterization.parameters_for_qubits_swapped() == PhasedFSimCharacterization(
        theta=0.1, zeta=-0.2, chi=-0.3, gamma=0.4, phi=0.5
    )


def test_merge_with():
    characterization = PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=0.3)
    other = PhasedFSimCharacterization(gamma=0.4, phi=0.5, theta=0.6)
    assert characterization.merge_with(other) == PhasedFSimCharacterization(
        theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5
    )


def test_override_by():
    characterization = PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=0.3)
    other = PhasedFSimCharacterization(gamma=0.4, phi=0.5, theta=0.6)
    assert characterization.override_by(other) == PhasedFSimCharacterization(
        theta=0.6, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5
    )


def test_floquet_to_calibration_layer():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = FloquetPhasedFSimCalibrationRequest(
        gate=gate,
        pairs=((q_00, q_01), (q_02, q_03)),
        options=FloquetPhasedFSimCalibrationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=True,
        ),
    )

    assert request.to_calibration_layer() == cirq.google.CalibrationLayer(
        calibration_type='floquet_phased_fsim_characterization',
        program=cirq.Circuit([gate.on(q_00, q_01), gate.on(q_02, q_03)]),
        args={
            'est_theta': True,
            'est_zeta': True,
            'est_chi': False,
            'est_gamma': False,
            'est_phi': True,
            'readout_corrections': True,
        },
    )


def test_floquet_parse_result():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = FloquetPhasedFSimCalibrationRequest(
        gate=gate,
        pairs=((q_00, q_01), (q_02, q_03)),
        options=FloquetPhasedFSimCalibrationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=True,
        ),
    )

    result = cirq.google.CalibrationResult(
        code=cirq.google.api.v2.calibration_pb2.SUCCESS,
        error_message=None,
        token=None,
        valid_until=None,
        metrics=cirq.google.Calibration(
            cirq.google.api.v2.metrics_pb2.MetricsSnapshot(
                metrics=[
                    cirq.google.api.v2.metrics_pb2.Metric(
                        name='angles',
                        targets=[
                            '0_qubit_a',
                            '0_qubit_b',
                            '0_theta_est',
                            '0_zeta_est',
                            '0_phi_est',
                            '1_qubit_a',
                            '1_qubit_b',
                            '1_theta_est',
                            '1_zeta_est',
                            '1_phi_est',
                        ],
                        values=[
                            cirq.google.api.v2.metrics_pb2.Value(str_val='0_0'),
                            cirq.google.api.v2.metrics_pb2.Value(str_val='0_1'),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.1),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.2),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.3),
                            cirq.google.api.v2.metrics_pb2.Value(str_val='0_2'),
                            cirq.google.api.v2.metrics_pb2.Value(str_val='0_3'),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.4),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.5),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.6),
                        ],
                    )
                ]
            )
        ),
    )

    assert request.parse_result(result) == PhasedFSimCalibrationResult(
        parameters={
            (q_00, q_01): PhasedFSimCharacterization(
                theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3
            ),
            (q_02, q_03): PhasedFSimCharacterization(
                theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6
            ),
        },
        gate=gate,
        options=FloquetPhasedFSimCalibrationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=True,
        ),
    )


def test_floquet_parse_result_bad_metric():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = FloquetPhasedFSimCalibrationRequest(
        gate=gate,
        pairs=((q_00, q_01), (q_02, q_03)),
        options=FloquetPhasedFSimCalibrationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=True,
        ),
    )
    result = cirq.google.CalibrationResult(
        code=cirq.google.api.v2.calibration_pb2.SUCCESS,
        error_message=None,
        token=None,
        valid_until=None,
        metrics=cirq.google.Calibration(
            cirq.google.api.v2.metrics_pb2.MetricsSnapshot(
                metrics=[
                    cirq.google.api.v2.metrics_pb2.Metric(
                        name='angles',
                        targets=[
                            '1000gerbils',
                        ],
                        values=[
                            cirq.google.api.v2.metrics_pb2.Value(str_val='100_10'),
                        ],
                    )
                ]
            )
        ),
    )
    with pytest.raises(ValueError, match='Unknown metric name 1000gerbils'):
        _ = request.parse_result(result)


def test_get_parameters():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    result = PhasedFSimCalibrationResult(
        parameters={
            (q_00, q_01): PhasedFSimCharacterization(
                theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3
            ),
            (q_02, q_03): PhasedFSimCharacterization(
                theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6
            ),
        },
        gate=gate,
        options=FloquetPhasedFSimCalibrationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=True,
        ),
    )
    assert result.get_parameters(q_00, q_01) == PhasedFSimCharacterization(
        theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3
    )
    assert result.get_parameters(q_01, q_00) == PhasedFSimCharacterization(
        theta=0.1, zeta=-0.2, chi=None, gamma=None, phi=0.3
    )
    assert result.get_parameters(q_02, q_03) == PhasedFSimCharacterization(
        theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6
    )
    assert result.get_parameters(q_00, q_03) == None


def test_try_convert_sqrt_iswap_to_fsim_converts_correctly():
    expected = cirq.FSimGate(theta=np.pi / 4, phi=0)
    expected_unitary = cirq.unitary(expected)

    fsim = cirq.FSimGate(theta=np.pi / 4, phi=0)
    assert np.allclose(cirq.unitary(fsim), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(fsim) == expected
    assert try_convert_sqrt_iswap_to_fsim(cirq.FSimGate(theta=np.pi / 4, phi=0.1)) is None
    assert try_convert_sqrt_iswap_to_fsim(cirq.FSimGate(theta=np.pi / 3, phi=0)) is None

    phased_fsim = cirq.PhasedFSimGate(theta=np.pi / 4, phi=0)
    assert np.allclose(cirq.unitary(phased_fsim), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(phased_fsim) == expected
    assert (
        try_convert_sqrt_iswap_to_fsim(cirq.PhasedFSimGate(theta=np.pi / 4, zeta=0.1, phi=0))
        is None
    )

    iswap_pow = cirq.ISwapPowGate(exponent=-0.5)
    assert np.allclose(cirq.unitary(iswap_pow), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(iswap_pow) == expected
    assert try_convert_sqrt_iswap_to_fsim(cirq.ISwapPowGate(exponent=-0.4)) is None

    phased_iswap_pow = cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=-0.5)
    assert np.allclose(cirq.unitary(phased_iswap_pow), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(phased_iswap_pow) == expected
    assert (
        try_convert_sqrt_iswap_to_fsim(cirq.PhasedISwapPowGate(exponent=-0.5, phase_exponent=0.1))
        is None
    )

    assert try_convert_sqrt_iswap_to_fsim(cirq.CZ) is None
