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
    ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    PhaseCalibratedFSimGate,
    PhasedFSimCharacterization,
    PhasedFSimCalibrationResult,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    merge_matching_results,
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


def test_from_moment():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    m = cirq.Moment(cirq.ISWAP(q_00, q_01) ** 0.5, cirq.ISWAP(q_02, q_03) ** 0.5)
    options = FloquetPhasedFSimCalibrationOptions(
        characterize_theta=True,
        characterize_zeta=True,
        characterize_chi=False,
        characterize_gamma=False,
        characterize_phi=True,
    )
    request = FloquetPhasedFSimCalibrationRequest.from_moment(m, options)
    assert request == FloquetPhasedFSimCalibrationRequest(
        gate=cirq.ISWAP ** 0.5, pairs=((q_00, q_01), (q_02, q_03)), options=options
    )

    non_identical = cirq.Moment(cirq.ISWAP(q_00, q_01) ** 0.5, cirq.ISWAP(q_02, q_03))
    with pytest.raises(ValueError, match='must be identical'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(non_identical, options)

    sq = cirq.Moment(cirq.X(q_00))
    with pytest.raises(ValueError, match='must be two qubit gates'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(sq, options)

    threeq = cirq.Moment(cirq.TOFFOLI(q_00, q_01, q_02))
    with pytest.raises(ValueError, match='must be two qubit gates'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(threeq, options)

    not_gate = cirq.Moment(cirq.CircuitOperation(cirq.FrozenCircuit()))
    with pytest.raises(ValueError, match='must be two qubit gates'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(not_gate, options)

    empty = cirq.Moment()
    with pytest.raises(ValueError, match='No gates found'):
        _ = FloquetPhasedFSimCalibrationRequest.from_moment(empty, options)


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
    assert result.get_parameters(q_00, q_03) is None


def test_merge_matching_results():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    parameters_1 = {
        (q_00, q_01): PhasedFSimCharacterization(
            theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3
        ),
    }
    parameters_2 = {
        (q_02, q_03): PhasedFSimCharacterization(
            theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6
        ),
    }

    results = [
        PhasedFSimCalibrationResult(
            parameters=parameters_1,
            gate=gate,
            options=options,
        ),
        PhasedFSimCalibrationResult(
            parameters=parameters_2,
            gate=gate,
            options=options,
        ),
    ]

    assert merge_matching_results(results) == PhasedFSimCalibrationResult(
        parameters={**parameters_1, **parameters_2},
        gate=gate,
        options=options,
    )


def test_merge_matching_results_when_empty_none():
    assert merge_matching_results([]) is None


def test_merge_matching_results_when_incompatible_fails():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    parameters_1 = {
        (q_00, q_01): PhasedFSimCharacterization(
            theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3
        ),
    }
    parameters_2 = {
        (q_02, q_03): PhasedFSimCharacterization(
            theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6
        ),
    }

    with pytest.raises(ValueError):
        results = [
            PhasedFSimCalibrationResult(
                parameters=parameters_1,
                gate=gate,
                options=options,
            ),
            PhasedFSimCalibrationResult(
                parameters=parameters_1,
                gate=gate,
                options=options,
            ),
        ]
        assert merge_matching_results(results)

    with pytest.raises(ValueError):
        results = [
            PhasedFSimCalibrationResult(
                parameters=parameters_1,
                gate=gate,
                options=options,
            ),
            PhasedFSimCalibrationResult(
                parameters=parameters_2,
                gate=cirq.CZ,
                options=options,
            ),
        ]
        assert merge_matching_results(results)

    with pytest.raises(ValueError):
        results = [
            PhasedFSimCalibrationResult(
                parameters=parameters_1,
                gate=gate,
                options=options,
            ),
            PhasedFSimCalibrationResult(
                parameters=parameters_2,
                gate=gate,
                options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            ),
        ]
        assert merge_matching_results(results)


@pytest.mark.parametrize('phase_exponent', np.linspace(0, 1, 5))
def test_phase_calibrated_fsim_gate_as_characterized_phased_fsim_gate(phase_exponent: float):
    a, b = cirq.LineQubit.range(2)
    ideal_gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    characterized_gate = cirq.PhasedFSimGate(
        theta=ideal_gate.theta, zeta=0.1, chi=0.2, gamma=0.3, phi=ideal_gate.phi
    )
    parameters = PhasedFSimCharacterization(
        theta=ideal_gate.theta,
        zeta=characterized_gate.zeta,
        chi=characterized_gate.chi,
        gamma=characterized_gate.gamma,
        phi=ideal_gate.phi,
    )

    calibrated = PhaseCalibratedFSimGate(ideal_gate, phase_exponent=phase_exponent)
    phased_gate = calibrated.as_characterized_phased_fsim_gate(parameters).on(a, b)

    assert np.allclose(
        cirq.unitary(phased_gate),
        cirq.unitary(
            cirq.Circuit(
                [
                    [cirq.Z(a) ** -phase_exponent, cirq.Z(b) ** phase_exponent],
                    characterized_gate.on(a, b),
                    [cirq.Z(a) ** phase_exponent, cirq.Z(b) ** -phase_exponent],
                ]
            )
        ),
    )


@pytest.mark.parametrize('phase_exponent', np.linspace(0, 1, 5))
def test_phase_calibrated_fsim_gate_compensated(phase_exponent: float):
    a, b = cirq.LineQubit.range(2)
    ideal_gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    characterized_gate = cirq.PhasedFSimGate(
        theta=ideal_gate.theta, zeta=0.1, chi=0.2, gamma=0.3, phi=ideal_gate.phi
    )
    parameters = PhasedFSimCharacterization(
        theta=ideal_gate.theta,
        zeta=characterized_gate.zeta,
        chi=characterized_gate.chi,
        gamma=characterized_gate.gamma,
        phi=ideal_gate.phi,
    )

    calibrated = PhaseCalibratedFSimGate(ideal_gate, phase_exponent=phase_exponent)

    # Passing characterized_gate as engine_gate simulates the hardware execution.
    operations = calibrated.with_zeta_chi_gamma_compensated(
        (a, b), parameters, engine_gate=characterized_gate
    )

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.Circuit(operations)),
        cirq.unitary(
            cirq.Circuit(
                [
                    [cirq.Z(a) ** -phase_exponent, cirq.Z(b) ** phase_exponent],
                    ideal_gate.on(a, b),
                    [cirq.Z(a) ** phase_exponent, cirq.Z(b) ** -phase_exponent],
                ]
            )
        ),
        atol=1e-8,
    )


def test_try_convert_sqrt_iswap_to_fsim_converts_correctly():
    expected = cirq.FSimGate(theta=np.pi / 4, phi=0)
    expected_unitary = cirq.unitary(expected)

    fsim = cirq.FSimGate(theta=np.pi / 4, phi=0)
    assert np.allclose(cirq.unitary(fsim), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(fsim) == PhaseCalibratedFSimGate(expected, 0.0)
    assert try_convert_sqrt_iswap_to_fsim(cirq.FSimGate(theta=np.pi / 4, phi=0.1)) is None
    assert try_convert_sqrt_iswap_to_fsim(cirq.FSimGate(theta=np.pi / 3, phi=0)) is None

    phased_fsim = cirq.PhasedFSimGate(theta=np.pi / 4, phi=0)
    assert np.allclose(cirq.unitary(phased_fsim), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(phased_fsim) == PhaseCalibratedFSimGate(expected, 0.0)
    assert (
        try_convert_sqrt_iswap_to_fsim(cirq.PhasedFSimGate(theta=np.pi / 4, zeta=0.1, phi=0))
        is None
    )

    iswap_pow = cirq.ISwapPowGate(exponent=-0.5)
    assert np.allclose(cirq.unitary(iswap_pow), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(iswap_pow) == PhaseCalibratedFSimGate(expected, 0.0)
    assert try_convert_sqrt_iswap_to_fsim(cirq.ISwapPowGate(exponent=-0.4)) is None

    phased_iswap_pow = cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=-0.5)
    assert np.allclose(cirq.unitary(phased_iswap_pow), expected_unitary)
    assert try_convert_sqrt_iswap_to_fsim(phased_iswap_pow) == PhaseCalibratedFSimGate(
        expected, 0.0
    )
    assert (
        try_convert_sqrt_iswap_to_fsim(cirq.PhasedISwapPowGate(exponent=-0.5, phase_exponent=0.1))
        is None
    )

    assert try_convert_sqrt_iswap_to_fsim(cirq.CZ) is None


def test_result_override():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    result = PhasedFSimCalibrationResult(
        parameters={
            (q_00, q_01): PhasedFSimCharacterization(
                theta=0.1, zeta=0.2, chi=None, gamma=0.4, phi=0.5
            ),
            (q_02, q_03): PhasedFSimCharacterization(
                theta=0.6, zeta=0.7, chi=None, gamma=0.9, phi=1.0
            ),
        },
        gate=gate,
        options=options,
    )

    overridden = result.override(options.zeta_chi_gamma_correction_override())

    assert overridden == PhasedFSimCalibrationResult(
        parameters={
            (q_00, q_01): PhasedFSimCharacterization(
                theta=0.1, zeta=0.0, chi=None, gamma=0.0, phi=0.5
            ),
            (q_02, q_03): PhasedFSimCharacterization(
                theta=0.6, zeta=0.0, chi=None, gamma=0.0, phi=1.0
            ),
        },
        gate=gate,
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    )


def test_options_phase_corrected_override():
    assert (
        ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION.zeta_chi_gamma_correction_override()
        == PhasedFSimCharacterization(zeta=0.0, chi=0.0, gamma=0.0)
    )

    assert (
        FloquetPhasedFSimCalibrationOptions(
            characterize_theta=False,
            characterize_zeta=False,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=False,
        ).zeta_chi_gamma_correction_override()
        == PhasedFSimCharacterization()
    )
