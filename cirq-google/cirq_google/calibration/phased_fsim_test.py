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
import os

from typing import cast
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import sympy
from google.protobuf import text_format

import cirq
import cirq_google
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.calibration.phased_fsim import (
    ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    PhaseCalibratedFSimGate,
    PhasedFSimCalibrationError,
    PhasedFSimCharacterization,
    PhasedFSimCalibrationResult,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    merge_matching_results,
    to_zphase_data,
    try_convert_gate_to_fsim,
    try_convert_syc_or_sqrt_iswap_to_fsim,
    try_convert_sqrt_iswap_to_fsim,
    XEBPhasedFSimCalibrationRequest,
    XEBPhasedFSimCalibrationOptions,
    _parse_xeb_fidelities_df,
    _parse_characterized_angles,
)
from cirq_google.serialization.arg_func_langs import arg_to_proto


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

    assert request.to_calibration_layer() == cirq_google.CalibrationLayer(
        calibration_type='floquet_phased_fsim_characterization',
        program=cirq.Circuit([gate.on(q_00, q_01), gate.on(q_02, q_03)]),
        args={
            'est_theta': True,
            'est_zeta': True,
            'est_chi': False,
            'est_gamma': False,
            'est_phi': True,
            'readout_corrections': True,
            'version': 2,
        },
    )


def test_floquet_to_calibration_layer_with_version_override():
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
            version=3,
        ),
    )

    assert request.to_calibration_layer() == cirq_google.CalibrationLayer(
        calibration_type='floquet_phased_fsim_characterization',
        program=cirq.Circuit([gate.on(q_00, q_01), gate.on(q_02, q_03)]),
        args={
            'est_theta': True,
            'est_zeta': True,
            'est_chi': False,
            'est_gamma': False,
            'est_phi': True,
            'readout_corrections': True,
            'version': 3,
        },
    )


def test_floquet_to_calibration_layer_readout_thresholds():
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
            readout_error_tolerance=0.4,
        ),
    )

    assert request.to_calibration_layer() == cirq_google.CalibrationLayer(
        calibration_type='floquet_phased_fsim_characterization',
        program=cirq.Circuit([gate.on(q_00, q_01), gate.on(q_02, q_03)]),
        args={
            'est_theta': True,
            'est_zeta': True,
            'est_chi': False,
            'est_gamma': False,
            'est_phi': True,
            'readout_corrections': True,
            'readout_error_tolerance': 0.4,
            'correlated_readout_error_tolerance': 7 / 6 * 0.4 - 1 / 6,
            'version': 2,
        },
    )


def test_floquet_to_calibration_layer_with_measure_qubits():
    qubits = tuple(cirq.GridQubit(0, index) for index in range(5))
    q_00, q_01, q_02, q_03, _ = qubits
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
            measure_qubits=qubits,
        ),
    )

    assert request.to_calibration_layer() == cirq_google.CalibrationLayer(
        calibration_type='floquet_phased_fsim_characterization',
        program=cirq.Circuit([gate.on(q_00, q_01), gate.on(q_02, q_03), cirq.measure(*qubits)]),
        args={
            'est_theta': True,
            'est_zeta': True,
            'est_chi': False,
            'est_gamma': False,
            'est_phi': True,
            'readout_corrections': True,
            'version': 2,
        },
    )


def test_xeb_to_calibration_layer():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=0.75, phi=0.0)
    request = XEBPhasedFSimCalibrationRequest(
        gate=gate,
        pairs=((q_00, q_01), (q_02, q_03)),
        options=XEBPhasedFSimCalibrationOptions(
            n_library_circuits=22,
            fatol=0.0078125,
            xatol=0.0078125,
            fsim_options=XEBPhasedFSimCharacterizationOptions(
                characterize_theta=True,
                characterize_zeta=True,
                characterize_chi=False,
                characterize_gamma=False,
                characterize_phi=True,
            ),
        ),
    )
    layer = request.to_calibration_layer()
    assert layer == cirq_google.CalibrationLayer(
        calibration_type='xeb_phased_fsim_characterization',
        program=cirq.Circuit([gate.on(q_00, q_01), gate.on(q_02, q_03)]),
        args={
            'n_library_circuits': 22,
            'n_combinations': 10,
            'cycle_depths': '5_25_50_100_200_300',
            'fatol': 0.0078125,
            'xatol': 0.0078125,
            'characterize_theta': True,
            'characterize_zeta': True,
            'characterize_chi': False,
            'characterize_gamma': False,
            'characterize_phi': True,
        },
    )

    # Serialize to proto
    calibration = v2.calibration_pb2.FocusedCalibration()
    new_layer = calibration.layers.add()
    new_layer.calibration_type = layer.calibration_type
    for arg in layer.args:
        arg_to_proto(layer.args[arg], out=new_layer.args[arg])
    cirq_google.CIRCUIT_SERIALIZER.serialize(layer.program, msg=new_layer.layer)
    with open(os.path.dirname(__file__) + '/test_data/xeb_calibration_layer.textproto') as f:
        desired_textproto = f.read()

    layer_str = str(new_layer)
    assert layer_str == desired_textproto


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
        gate=cirq.ISWAP**0.5, pairs=((q_00, q_01), (q_02, q_03)), options=options
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

    result = cirq_google.CalibrationResult(
        code=cirq_google.api.v2.calibration_pb2.SUCCESS,
        error_message=None,
        token=None,
        valid_until=None,
        metrics=cirq_google.Calibration(
            cirq_google.api.v2.metrics_pb2.MetricsSnapshot(
                metrics=[
                    cirq_google.api.v2.metrics_pb2.Metric(
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
                            cirq_google.api.v2.metrics_pb2.Value(str_val='0_0'),
                            cirq_google.api.v2.metrics_pb2.Value(str_val='0_1'),
                            cirq_google.api.v2.metrics_pb2.Value(double_val=0.1),
                            cirq_google.api.v2.metrics_pb2.Value(double_val=0.2),
                            cirq_google.api.v2.metrics_pb2.Value(double_val=0.3),
                            cirq_google.api.v2.metrics_pb2.Value(str_val='0_2'),
                            cirq_google.api.v2.metrics_pb2.Value(str_val='0_3'),
                            cirq_google.api.v2.metrics_pb2.Value(double_val=0.4),
                            cirq_google.api.v2.metrics_pb2.Value(double_val=0.5),
                            cirq_google.api.v2.metrics_pb2.Value(double_val=0.6),
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


def test_floquet_parse_result_failure():
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = FloquetPhasedFSimCalibrationRequest(
        gate=gate,
        pairs=(),
        options=FloquetPhasedFSimCalibrationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=True,
        ),
    )

    result = cirq_google.CalibrationResult(
        code=cirq_google.api.v2.calibration_pb2.ERROR_CALIBRATION_FAILED,
        error_message="Test message",
        token=None,
        valid_until=None,
        metrics=cirq_google.Calibration(),
    )

    with pytest.raises(PhasedFSimCalibrationError, match='Test message'):
        request.parse_result(result)


def _load_xeb_results_textproto() -> cirq_google.CalibrationResult:
    with open(os.path.dirname(__file__) + '/test_data/xeb_results.textproto') as f:
        metrics_snapshot = text_format.Parse(
            f.read(), cirq_google.api.v2.metrics_pb2.MetricsSnapshot()
        )
    return cirq_google.CalibrationResult(
        code=cirq_google.api.v2.calibration_pb2.SUCCESS,  # type: ignore
        error_message=None,
        token=None,
        valid_until=None,
        metrics=cirq_google.Calibration(metrics_snapshot),
    )


def test_xeb_parse_fidelities():
    q0, q1, q2, q3 = [cirq.GridQubit(0, index) for index in range(4)]
    pair0 = (q0, q1)
    pair1 = (q2, q3)
    result = _load_xeb_results_textproto()
    metrics = result.metrics
    df = _parse_xeb_fidelities_df(metrics, 'initial_fidelities')
    should_be = pd.DataFrame(
        {
            'cycle_depth': [5, 5, 25, 25],
            'layer_i': [0, 0, 0, 0],
            'pair_i': [0, 1, 0, 1],
            'fidelity': [0.99, 0.99, 0.88, 0.88],
            'pair': [pair0, pair1, pair0, pair1],
        }
    )
    pd.testing.assert_frame_equal(df, should_be)

    df = _parse_xeb_fidelities_df(metrics, 'final_fidelities')
    should_be = pd.DataFrame(
        {
            'cycle_depth': [5, 5, 25, 25],
            'layer_i': [0, 0, 0, 0],
            'pair_i': [0, 1, 0, 1],
            'fidelity': [0.99, 0.99, 0.98, 0.98],
            'pair': [pair0, pair1, pair0, pair1],
        }
    )
    pd.testing.assert_frame_equal(df, should_be)


def test_xeb_parse_bad_fidelities():
    metrics = cirq_google.Calibration(
        metrics={
            'initial_fidelities_depth_5': {
                ('layer_0', 'pair_0', cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)): [1.0]
            }
        }
    )
    df = _parse_xeb_fidelities_df(metrics, 'initial_fidelities')
    pd.testing.assert_frame_equal(
        df,
        pd.DataFrame(
            {
                'cycle_depth': [5],
                'layer_i': [0],
                'pair_i': [0],
                'fidelity': [1.0],
                'pair': [(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1))],
            }
        ),
    )

    metrics = cirq_google.Calibration(
        metrics={'initial_fidelities_depth_5x': {('layer_0', 'pair_0', '0_0', '1_1'): [1.0]}}
    )
    df = _parse_xeb_fidelities_df(metrics, 'initial_fidelities')
    assert len(df) == 0, 'bad metric name ignored'

    metrics = cirq_google.Calibration(
        metrics={'initial_fidelities_depth_5': {('bad_name_0', 'pair_0', '0_0', '1_1'): [1.0]}}
    )
    with pytest.raises(ValueError, match=r'Could not parse layer value for bad_name_0'):
        _parse_xeb_fidelities_df(metrics, 'initial_fidelities')


def test_xeb_parse_angles():
    q0, q1, q2, q3 = [cirq.GridQubit(0, index) for index in range(4)]
    result = _load_xeb_results_textproto()
    metrics = result.metrics
    angles = _parse_characterized_angles(metrics, 'characterized_angles')
    assert angles == {
        (q0, q1): {'theta': -0.7853981, 'phi': 0.0},
        (q2, q3): {'theta': -0.7853981, 'phi': 0.0},
    }


def test_xeb_parse_result_failure():
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = XEBPhasedFSimCalibrationRequest(
        gate=gate,
        pairs=(),
        options=XEBPhasedFSimCalibrationOptions(
            fsim_options=XEBPhasedFSimCharacterizationOptions(
                characterize_theta=False,
                characterize_zeta=False,
                characterize_chi=False,
                characterize_gamma=False,
                characterize_phi=True,
            )
        ),
    )

    result = cirq_google.CalibrationResult(
        code=cirq_google.api.v2.calibration_pb2.ERROR_CALIBRATION_FAILED,
        error_message="Test message",
        token=None,
        valid_until=None,
        metrics=cirq_google.Calibration(),
    )

    with pytest.raises(PhasedFSimCalibrationError, match='Test message'):
        request.parse_result(result)


def test_xeb_parse_result():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = XEBPhasedFSimCalibrationRequest(
        gate=gate,
        pairs=((q_00, q_01), (q_02, q_03)),
        options=XEBPhasedFSimCalibrationOptions(
            fsim_options=XEBPhasedFSimCharacterizationOptions(
                characterize_theta=False,
                characterize_zeta=False,
                characterize_chi=False,
                characterize_gamma=False,
                characterize_phi=True,
            )
        ),
    )

    result = _load_xeb_results_textproto()
    assert request.parse_result(result) == PhasedFSimCalibrationResult(
        parameters={
            (q_00, q_01): PhasedFSimCharacterization(phi=0.0, theta=-0.7853981),
            (q_02, q_03): PhasedFSimCharacterization(phi=0.0, theta=-0.7853981),
        },
        gate=gate,
        options=request.options,
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
    result = cirq_google.CalibrationResult(
        code=cirq_google.api.v2.calibration_pb2.SUCCESS,
        error_message=None,
        token=None,
        valid_until=None,
        metrics=cirq_google.Calibration(
            cirq_google.api.v2.metrics_pb2.MetricsSnapshot(
                metrics=[
                    cirq_google.api.v2.metrics_pb2.Metric(
                        name='angles',
                        targets=['1000gerbils'],
                        values=[cirq_google.api.v2.metrics_pb2.Value(str_val='100_10')],
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


def test_to_zphase_data():
    q0, q1, q2 = cirq.GridQubit.rect(1, 3)
    result_1 = PhasedFSimCalibrationResult(
        {
            (q0, q1): PhasedFSimCharacterization(zeta=0.1, gamma=0.2),
            (q1, q2): PhasedFSimCharacterization(zeta=0.3, gamma=0.4),
        },
        gate=cirq_google.SycamoreGate(),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    )
    result_2 = PhasedFSimCalibrationResult(
        {
            (q0, q1): PhasedFSimCharacterization(zeta=0.5, gamma=0.6),
            (q1, q2): PhasedFSimCharacterization(zeta=0.7, gamma=0.8),
        },
        gate=cirq.ISwapPowGate(),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    )
    assert to_zphase_data([result_1, result_2]) == {
        'syc': {'zeta': {(q0, q1): 0.1, (q1, q2): 0.3}, 'gamma': {(q0, q1): 0.2, (q1, q2): 0.4}},
        'sqrt_iswap': {
            'zeta': {(q0, q1): 0.5, (q1, q2): 0.7},
            'gamma': {(q0, q1): 0.6, (q1, q2): 0.8},
        },
    }
    # Test update and override
    result_3 = PhasedFSimCalibrationResult(
        {
            (q0, q1): PhasedFSimCharacterization(theta=0.01),
            (q1, q2): PhasedFSimCharacterization(zeta=0.02),
            (q2, q0): PhasedFSimCharacterization(zeta=0.03, gamma=0.04, theta=0.05),
        },
        gate=cirq_google.SycamoreGate(),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    )
    assert to_zphase_data([result_1, result_3]) == {
        'syc': {
            'zeta': {(q0, q1): 0.1, (q1, q2): 0.02, (q2, q0): 0.03},
            'gamma': {(q0, q1): 0.2, (q1, q2): 0.4, (q2, q0): 0.04},
            'theta': {(q0, q1): 0.01, (q2, q0): 0.05},
        }
    }


def test_to_zphase_unknown_gate_raises_error():
    q0, q1, q2 = cirq.GridQubit.rect(1, 3)
    result_1 = PhasedFSimCalibrationResult(
        {
            (q0, q1): PhasedFSimCharacterization(zeta=0.1, gamma=0.2),
            (q1, q2): PhasedFSimCharacterization(zeta=0.3, gamma=0.4),
        },
        gate=cirq.CZPowGate(),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    )
    with pytest.raises(ValueError, match="Only 'SycamoreGate' and 'ISwapPowGate' are supported"):
        _ = to_zphase_data([result_1])


def test_merge_matching_results():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    parameters_1 = {
        (q_00, q_01): PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3)
    }
    parameters_2 = {
        (q_02, q_03): PhasedFSimCharacterization(theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6)
    }

    results = [
        PhasedFSimCalibrationResult(parameters=parameters_1, gate=gate, options=options),
        PhasedFSimCalibrationResult(parameters=parameters_2, gate=gate, options=options),
    ]

    assert merge_matching_results(results) == PhasedFSimCalibrationResult(
        parameters={**parameters_1, **parameters_2}, gate=gate, options=options
    )


def test_merge_matching_results_when_empty_none():
    assert merge_matching_results([]) is None


def test_merge_matching_results_when_incompatible_fails():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    parameters_1 = {
        (q_00, q_01): PhasedFSimCharacterization(theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3)
    }
    parameters_2 = {
        (q_02, q_03): PhasedFSimCharacterization(theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6)
    }

    with pytest.raises(ValueError):
        results = [
            PhasedFSimCalibrationResult(parameters=parameters_1, gate=gate, options=options),
            PhasedFSimCalibrationResult(parameters=parameters_1, gate=gate, options=options),
        ]
        assert merge_matching_results(results)

    with pytest.raises(ValueError):
        results = [
            PhasedFSimCalibrationResult(parameters=parameters_1, gate=gate, options=options),
            PhasedFSimCalibrationResult(parameters=parameters_2, gate=cirq.CZ, options=options),
        ]
        assert merge_matching_results(results)

    with pytest.raises(ValueError):
        results = [
            PhasedFSimCalibrationResult(parameters=parameters_1, gate=gate, options=options),
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
        theta=cast(float, ideal_gate.theta),
        zeta=cast(float, characterized_gate.zeta),
        chi=cast(float, characterized_gate.chi),
        gamma=cast(float, characterized_gate.gamma),
        phi=cast(float, ideal_gate.phi),
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
        theta=cast(float, ideal_gate.theta),
        zeta=cast(float, characterized_gate.zeta),
        chi=cast(float, characterized_gate.chi),
        gamma=cast(float, characterized_gate.gamma),
        phi=cast(float, ideal_gate.phi),
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
        try_convert_sqrt_iswap_to_fsim(cirq.PhasedISwapPowGate(exponent=-0.6, phase_exponent=0.1))
        is None
    )

    assert try_convert_sqrt_iswap_to_fsim(cirq.CZ) is None

    assert (
        try_convert_sqrt_iswap_to_fsim(
            cirq.FSimGate(theta=sympy.Symbol('t'), phi=sympy.Symbol('p'))
        )
        is None
    )


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


@mock.patch('cirq_google.engine.engine_client.EngineClient')
def test_result_engine_job(_client):
    result = PhasedFSimCalibrationResult(
        parameters={},
        gate=cirq.FSimGate(theta=np.pi / 4, phi=0.0),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
        project_id='project_id',
        program_id='program_id',
        job_id='job_id',
    )

    assert result.engine_job.project_id == 'project_id'
    assert result.engine_job.program_id == 'program_id'
    assert result.engine_job.job_id == 'job_id'


def test_result_engine_job_none():
    result = PhasedFSimCalibrationResult(
        parameters={},
        gate=cirq.FSimGate(theta=np.pi / 4, phi=0.0),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    )

    assert result.engine_job is None


@mock.patch('cirq_google.engine.engine_client.EngineClient')
def test_result_engine_calibration(_client):
    result = PhasedFSimCalibrationResult(
        parameters={},
        gate=cirq.FSimGate(theta=np.pi / 4, phi=0.0),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
        project_id='project_id',
        program_id='program_id',
        job_id='job_id',
    )

    test_calibration = cirq_google.Calibration()
    result.engine_job.get_calibration = lambda: test_calibration

    assert result.engine_calibration == test_calibration


def test_result_engine_calibration_none():
    result = PhasedFSimCalibrationResult(
        parameters={},
        gate=cirq.FSimGate(theta=np.pi / 4, phi=0.0),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    )

    assert result.engine_calibration is None


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


def test_try_convert_gate_to_fsim():
    def check(gate: cirq.Gate, expected: PhaseCalibratedFSimGate):
        assert np.allclose(cirq.unitary(gate), cirq.unitary(expected))
        assert try_convert_gate_to_fsim(gate) == expected

    check(
        cirq.FSimGate(theta=0.3, phi=0.5),
        PhaseCalibratedFSimGate(cirq.FSimGate(theta=0.3, phi=0.5), 0.0),
    )

    check(
        cirq.FSimGate(7 * np.pi / 4, 0.0),
        PhaseCalibratedFSimGate(cirq.FSimGate(np.pi / 4, 0.0), 0.5),
    )

    check(
        cirq.ISwapPowGate(exponent=-0.5),
        PhaseCalibratedFSimGate(cirq.FSimGate(theta=0.25 * np.pi, phi=0.0), 0.0),
    )

    check(
        cirq.ops.ISwapPowGate(exponent=0.8, global_shift=2.5),
        PhaseCalibratedFSimGate(cirq.FSimGate(theta=0.4 * np.pi, phi=0.0), 0.5),
    )

    gate = cirq.ops.ISwapPowGate(exponent=0.3, global_shift=0.4)
    assert try_convert_gate_to_fsim(gate) is None

    check(
        cirq.PhasedFSimGate(theta=0.2, phi=0.5, chi=1.5 * np.pi),
        PhaseCalibratedFSimGate(cirq.FSimGate(theta=0.2, phi=0.5), 0.25),
    )

    gate = cirq.PhasedFSimGate(theta=0.2, phi=0.5, zeta=1.5 * np.pi)
    assert try_convert_gate_to_fsim(gate) is None

    check(
        cirq.PhasedISwapPowGate(exponent=-0.5, phase_exponent=0.75),
        PhaseCalibratedFSimGate(cirq.FSimGate(theta=0.25 * np.pi, phi=0.0), 0.25),
    )

    check(cirq.CZ, PhaseCalibratedFSimGate(cirq.FSimGate(theta=0.0, phi=np.pi), 0.0))

    check(
        cirq.ops.CZPowGate(exponent=0.3),
        PhaseCalibratedFSimGate(cirq.FSimGate(theta=0.0, phi=-0.3 * np.pi), 0.0),
    )

    check(
        cirq.ops.CZPowGate(exponent=0.8, global_shift=2.5),
        PhaseCalibratedFSimGate(cirq.FSimGate(theta=0.0, phi=-0.8 * np.pi), 0.0),
    )

    gate = cirq.ops.CZPowGate(exponent=0.3, global_shift=0.4)
    assert try_convert_gate_to_fsim(gate) is None

    check(
        cirq_google.ops.SYC,
        PhaseCalibratedFSimGate(cirq.FSimGate(phi=np.pi / 6, theta=np.pi / 2), 0.0),
    )

    assert try_convert_gate_to_fsim(cirq.CX) is None

    # Parameterized gates are not supported.
    x = sympy.Symbol('x')
    assert try_convert_gate_to_fsim(cirq.ops.ISwapPowGate(exponent=x)) is None
    assert try_convert_gate_to_fsim(cirq.PhasedFSimGate(theta=x)) is None
    assert try_convert_gate_to_fsim(cirq.FSimGate(theta=x, phi=x)) is None
    assert try_convert_gate_to_fsim(cirq.PhasedISwapPowGate(exponent=x)) is None
    assert try_convert_gate_to_fsim(cirq.CZPowGate(exponent=x)) is None


def test_try_convert_syc_or_sqrt_iswap_to_fsim():
    def check_converts(gate: cirq.Gate):
        result = try_convert_syc_or_sqrt_iswap_to_fsim(gate)
        assert np.allclose(cirq.unitary(gate), cirq.unitary(result))

    def check_none(gate: cirq.Gate):
        assert try_convert_syc_or_sqrt_iswap_to_fsim(gate) is None

    check_converts(cirq_google.ops.SYC)
    check_converts(cirq.FSimGate(np.pi / 2, np.pi / 6))
    check_none(cirq.FSimGate(0, np.pi))
    check_converts(cirq.FSimGate(np.pi / 4, 0.0))
    check_none(cirq.FSimGate(0.2, 0.3))
    check_converts(cirq.ISwapPowGate(exponent=0.5))
    check_converts(cirq.ISwapPowGate(exponent=-0.5))
    check_none(cirq.ISwapPowGate(exponent=0.3))
    check_converts(cirq.PhasedFSimGate(theta=np.pi / 4, phi=0.0, chi=0.7))
    check_none(cirq.PhasedFSimGate(theta=0.3, phi=0.4))
    check_converts(cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=0.75))
    check_none(cirq.PhasedISwapPowGate(exponent=0.4, phase_exponent=0.75))
    check_none(cirq.ops.CZPowGate(exponent=1.0))
    check_none(cirq.CX)


# Test that try_convert_gate_to_fsim is extension of try_convert_sqrt_iswap_to_fsim.
# In other words, that both function return the same result for all gates on which
# try_convert_sqrt_iswap_to_fsim is defined.
def test_gate_translators_are_consistent():
    def check(gate):
        result1 = try_convert_gate_to_fsim(gate)
        result2 = try_convert_sqrt_iswap_to_fsim(gate)
        assert result1 == result2
        assert result1 is not None

    check(cirq.FSimGate(theta=np.pi / 4, phi=0))
    check(cirq.FSimGate(theta=-np.pi / 4, phi=0))
    check(cirq.FSimGate(theta=7 * np.pi / 4, phi=0))
    check(cirq.PhasedFSimGate(theta=np.pi / 4, phi=0))
    check(cirq.ISwapPowGate(exponent=0.5))
    check(cirq.ISwapPowGate(exponent=-0.5))
    check(cirq.PhasedISwapPowGate(exponent=0.5, phase_exponent=-0.5))
