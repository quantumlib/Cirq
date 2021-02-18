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
from typing import Optional

from unittest import mock
import itertools
import numpy as np
import pytest

import cirq
import cirq.google.calibration.workflow as workflow

from cirq.google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq.google.calibration.phased_fsim import (
    ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    PhaseCalibratedFSimGate,
    PhasedFSimCharacterization,
    PhasedFSimCalibrationResult,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
)


SQRT_ISWAP_PARAMETERS = cirq.google.PhasedFSimCharacterization(
    theta=np.pi / 4, zeta=0.0, chi=0.0, gamma=0.0, phi=0.0
)
SQRT_ISWAP_GATE = cirq.FSimGate(np.pi / 4, 0.0)


def _fsim_identity_converter(gate: cirq.Gate) -> Optional[PhaseCalibratedFSimGate]:
    if isinstance(gate, cirq.FSimGate):
        return PhaseCalibratedFSimGate(gate, 0.0)
    return None


def test_make_floquet_request_for_moment_none_for_measurements() -> None:
    a, b, c, d = cirq.LineQubit.range(4)
    moment = cirq.Moment(cirq.measure(a, b, c, d))
    assert (
        workflow.prepare_floquet_characterization_for_moment(
            moment, WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
        )
        is None
    )


def test_make_floquet_request_for_moment_fails_for_non_gate_operation() -> None:
    moment = cirq.Moment(cirq.GlobalPhaseOperation(coefficient=1.0))
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_floquet_characterization_for_moment(
            moment, WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
        )


def test_make_floquet_request_for_moment_fails_for_unsupported_gate() -> None:
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment(cirq.CZ(a, b))
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_floquet_characterization_for_moment(
            moment,
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            gates_translator=_fsim_identity_converter,
        )


def test_make_floquet_request_for_moment_fails_for_mixed_gates() -> None:
    a, b, c, d = cirq.LineQubit.range(4)
    moment = cirq.Moment(
        [
            cirq.FSimGate(theta=np.pi / 4, phi=0.0).on(a, b),
            cirq.FSimGate(theta=np.pi / 8, phi=0.0).on(c, d),
        ]
    )
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_floquet_characterization_for_moment(
            moment,
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            gates_translator=_fsim_identity_converter,
        )


def test_make_floquet_request_for_moment_fails_for_mixed_moment() -> None:
    a, b, c = cirq.LineQubit.range(3)
    moment = cirq.Moment([cirq.FSimGate(theta=np.pi / 4, phi=0.0).on(a, b), cirq.Z.on(c)])
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_floquet_characterization_for_moment(
            moment, WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
        )


def test_make_floquet_request_for_moments() -> None:
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_GATE.on(a, b), SQRT_ISWAP_GATE.on(c, d)],
            [SQRT_ISWAP_GATE.on(b, c)],
            [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(b)],
        ]
    )
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(
        circuit, options=options
    )

    assert requests == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c),), gate=SQRT_ISWAP_GATE, options=options
        ),
    ]

    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, None]


def test_make_floquet_request_for_moments_merges_sub_sets() -> None:
    a, b, c, d, e = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_GATE.on(a, b), SQRT_ISWAP_GATE.on(c, d)],
            [SQRT_ISWAP_GATE.on(b, c)],
            [SQRT_ISWAP_GATE.on(a, b)],
        ]
    )
    circuit += cirq.Moment([SQRT_ISWAP_GATE.on(b, c), SQRT_ISWAP_GATE.on(d, e)])
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(
        circuit, options=options
    )

    assert requests == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c), (d, e)), gate=SQRT_ISWAP_GATE, options=options
        ),
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 0, 1]


def test_make_floquet_request_for_moments_merges_many_circuits() -> None:
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    a, b, c, d, e = cirq.LineQubit.range(5)

    circuit_1 = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_GATE.on(a, b), SQRT_ISWAP_GATE.on(c, d)],
            [SQRT_ISWAP_GATE.on(b, c)],
            [SQRT_ISWAP_GATE.on(a, b)],
        ]
    )

    circuit_with_calibration_1, requests_1 = workflow.prepare_floquet_characterization_for_moments(
        circuit_1, options=options
    )

    assert requests_1 == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c),), gate=SQRT_ISWAP_GATE, options=options
        ),
    ]
    assert circuit_with_calibration_1.circuit == circuit_1
    assert circuit_with_calibration_1.moment_to_calibration == [None, 0, 1, 0]

    circuit_2 = cirq.Circuit([SQRT_ISWAP_GATE.on(b, c), SQRT_ISWAP_GATE.on(d, e)])

    circuit_with_calibration_2, requests_2 = workflow.prepare_floquet_characterization_for_moments(
        circuit_2, options=options, initial=requests_1
    )

    assert requests_2 == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c), (d, e)), gate=SQRT_ISWAP_GATE, options=options
        ),
    ]
    assert circuit_with_calibration_2.circuit == circuit_2
    assert circuit_with_calibration_2.moment_to_calibration == [1]


def test_make_floquet_request_for_moments_does_not_merge_sub_sets_when_disabled() -> None:
    a, b, c, d, e = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_GATE.on(a, b), SQRT_ISWAP_GATE.on(c, d)],
            [SQRT_ISWAP_GATE.on(b, c)],
            [SQRT_ISWAP_GATE.on(a, b)],
        ]
    )
    circuit += cirq.Circuit(
        [SQRT_ISWAP_GATE.on(b, c), SQRT_ISWAP_GATE.on(d, e)],
        [SQRT_ISWAP_GATE.on(b, c)],
    )
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(
        circuit, options=options, merge_subsets=False
    )

    assert requests == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c),), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b),), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c), (d, e)), gate=SQRT_ISWAP_GATE, options=options
        ),
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 2, 3, 1]


def test_make_floquet_request_for_moments_merges_compatible_sets() -> None:
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    circuit = cirq.Circuit([cirq.X(a), cirq.Y(c)])
    circuit += cirq.Moment([SQRT_ISWAP_GATE.on(a, b)])
    circuit += cirq.Moment([SQRT_ISWAP_GATE.on(b, c), SQRT_ISWAP_GATE.on(d, e)])
    circuit += cirq.Moment([SQRT_ISWAP_GATE.on(c, d)])
    circuit += cirq.Moment([SQRT_ISWAP_GATE.on(a, f), SQRT_ISWAP_GATE.on(d, e)])
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(
        circuit, options=options
    )

    assert requests == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, f), (b, c), (d, e)), gate=SQRT_ISWAP_GATE, options=options
        ),
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 0, 1]


def test_make_floquet_request_for_operations() -> None:
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)

    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    # Prepare characterizations for a single circuit.
    circuit_1 = cirq.Circuit(
        [
            [cirq.X(q00), cirq.Y(q11)],
            [SQRT_ISWAP_GATE.on(q00, q01), SQRT_ISWAP_GATE.on(q10, q11)],
            [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(q01)],
        ]
    )

    requests_1 = workflow.prepare_floquet_characterization_for_operations(
        circuit_1, options=options
    )

    assert requests_1 == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q10, q11),), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q00, q01),), gate=SQRT_ISWAP_GATE, options=options
        ),
    ]

    # Prepare characterizations for a list of circuits.
    circuit_2 = cirq.Circuit(
        [
            [SQRT_ISWAP_GATE.on(q00, q01), SQRT_ISWAP_GATE.on(q10, q11)],
            [SQRT_ISWAP_GATE.on(q00, q10), SQRT_ISWAP_GATE.on(q01, q11)],
            [SQRT_ISWAP_GATE.on(q10, q20), SQRT_ISWAP_GATE.on(q11, q21)],
        ]
    )

    requests_2 = workflow.prepare_floquet_characterization_for_operations(
        [circuit_1, circuit_2], options=options
    )

    # The order of moments originates from HALF_GRID_STAGGERED_PATTERN.
    assert requests_2 == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q00, q10), (q11, q21)), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q01, q11), (q10, q20)), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q10, q11),), gate=SQRT_ISWAP_GATE, options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q00, q01),), gate=SQRT_ISWAP_GATE, options=options
        ),
    ]


def test_make_floquet_request_for_operations_when_no_interactions() -> None:
    q00 = cirq.GridQubit(0, 0)
    q11 = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit([cirq.X(q00), cirq.X(q11)])

    assert workflow.prepare_floquet_characterization_for_operations(circuit) == []


def test_make_floquet_request_for_operations_when_non_grid_fails() -> None:
    q00 = cirq.GridQubit(0, 0)
    q11 = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit(SQRT_ISWAP_GATE.on(q00, q11))

    with pytest.raises(ValueError):
        workflow.prepare_floquet_characterization_for_operations(circuit)


def test_make_floquet_request_for_operations_when_multiple_gates_fails() -> None:
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(
        [SQRT_ISWAP_GATE.on(q00, q01), cirq.FSimGate(theta=0.0, phi=np.pi).on(q00, q01)]
    )

    with pytest.raises(ValueError):
        workflow.prepare_floquet_characterization_for_operations(
            circuit, gates_translator=_fsim_identity_converter
        )


def test_make_zeta_chi_gamma_compensation_for_operations():
    a, b, c, d = cirq.LineQubit.range(4)
    parameters_ab = cirq.google.PhasedFSimCharacterization(zeta=0.5, chi=0.4, gamma=0.3)
    parameters_bc = cirq.google.PhasedFSimCharacterization(zeta=-0.5, chi=-0.4, gamma=-0.3)
    parameters_cd = cirq.google.PhasedFSimCharacterization(zeta=0.2, chi=0.3, gamma=0.4)

    parameters_dict = {(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd}

    engine_simulator = cirq.google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={
            pair: parameters.merge_with(SQRT_ISWAP_PARAMETERS)
            for pair, parameters in parameters_dict.items()
        }
    )

    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_GATE.on(a, b), SQRT_ISWAP_GATE.on(c, d)],
            [SQRT_ISWAP_GATE.on(b, c)],
        ]
    )

    options = cirq.google.FloquetPhasedFSimCalibrationOptions(
        characterize_theta=False,
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_phi=False,
    )

    characterizations = [
        PhasedFSimCalibrationResult(
            parameters={pair: parameters}, gate=SQRT_ISWAP_GATE, options=options
        )
        for pair, parameters in parameters_dict.items()
    ]

    calibrated_circuit = workflow.make_zeta_chi_gamma_compensation_for_operations(
        circuit,
        characterizations,
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated_circuit),
        cirq.final_state_vector(circuit),
    )


def test_make_zeta_chi_gamma_compensation_for_operations_permit_mixed_moments():
    with pytest.raises(NotImplementedError):
        workflow.make_zeta_chi_gamma_compensation_for_operations(
            cirq.Circuit(), [], permit_mixed_moments=True
        )


def test_run_characterization():
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

    job = cirq.google.engine.EngineJob('', '', '', None)
    job._calibration_results = [result]

    engine = mock.MagicMock(spec=cirq.google.Engine)
    engine.run_calibration.return_value = job

    progress_calls = []

    def progress(step: int, steps: int) -> None:
        progress_calls.append((step, steps))

    actual = workflow.run_calibrations(
        [request], engine, 'qproc', cirq.google.FSIM_GATESET, progress_func=progress
    )

    expected = [
        PhasedFSimCalibrationResult(
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
    ]

    assert actual == expected
    assert progress_calls == [(1, 1)]


def test_run_characterization_empty():
    assert workflow.run_calibrations([], None, 'qproc', cirq.google.FSIM_GATESET) == []


def test_run_characterization_fails_when_invalid_arguments():
    with pytest.raises(ValueError):
        assert workflow.run_calibrations(
            [], None, 'qproc', cirq.google.FSIM_GATESET, max_layers_per_request=0
        )

    request = FloquetPhasedFSimCalibrationRequest(
        gate=SQRT_ISWAP_GATE,
        pairs=(),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    )
    engine = mock.MagicMock(spec=cirq.google.Engine)

    with pytest.raises(ValueError):
        assert workflow.run_calibrations([request], engine, None, cirq.google.FSIM_GATESET)

    with pytest.raises(ValueError):
        assert workflow.run_calibrations([request], engine, 'qproc', None)

    with pytest.raises(ValueError):
        assert workflow.run_calibrations([request], 0, 'qproc', cirq.google.FSIM_GATESET)


def test_run_characterization_with_simulator():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = SQRT_ISWAP_GATE

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

    simulator = PhasedFSimEngineSimulator.create_with_ideal_sqrt_iswap()

    actual = workflow.run_calibrations([request], simulator)

    assert actual == [
        PhasedFSimCalibrationResult(
            parameters={
                (q_00, q_01): PhasedFSimCharacterization(
                    theta=np.pi / 4, zeta=0.0, chi=None, gamma=None, phi=0.0
                ),
                (q_02, q_03): PhasedFSimCharacterization(
                    theta=np.pi / 4, zeta=0.0, chi=None, gamma=None, phi=0.0
                ),
            },
            gate=SQRT_ISWAP_GATE,
            options=FloquetPhasedFSimCalibrationOptions(
                characterize_theta=True,
                characterize_zeta=True,
                characterize_chi=False,
                characterize_gamma=False,
                characterize_phi=True,
            ),
        )
    ]


def test_run_floquet_characterization_for_moments():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)

    circuit = cirq.Circuit([gate.on(q_00, q_01), gate.on(q_02, q_03)])

    options = FloquetPhasedFSimCalibrationOptions(
        characterize_theta=True,
        characterize_zeta=True,
        characterize_chi=False,
        characterize_gamma=False,
        characterize_phi=True,
    )

    job = cirq.google.engine.EngineJob('', '', '', None)
    job._calibration_results = [
        cirq.google.CalibrationResult(
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
    ]

    engine = mock.MagicMock(spec=cirq.google.Engine)
    engine.run_calibration.return_value = job

    circuit_with_calibration, requests = workflow.run_floquet_characterization_for_moments(
        circuit, engine, 'qproc', cirq.google.FSIM_GATESET, options=options
    )

    assert requests == [
        PhasedFSimCalibrationResult(
            parameters={
                (q_00, q_01): PhasedFSimCharacterization(
                    theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3
                ),
                (q_02, q_03): PhasedFSimCharacterization(
                    theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6
                ),
            },
            gate=gate,
            options=options,
        )
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [0]


@pytest.mark.parametrize(
    'theta,zeta,chi,gamma,phi',
    itertools.product([0.1, 0.7], [-0.3, 0.1, 0.5], [-0.3, 0.2, 0.4], [-0.6, 0.1, 0.6], [0.2, 0.6]),
)
def test_fsim_phase_corrections(
    theta: float, zeta: float, chi: float, gamma: float, phi: float
) -> None:
    a, b = cirq.LineQubit.range(2)

    expected_gate = cirq.PhasedFSimGate(theta=theta, zeta=-zeta, chi=-chi, gamma=-gamma, phi=phi)
    expected = cirq.unitary(expected_gate)

    corrected = workflow.FSimPhaseCorrections.from_characterization(
        (a, b),
        PhaseCalibratedFSimGate(cirq.FSimGate(theta=theta, phi=phi), 0.0),
        cirq.google.PhasedFSimCharacterization(
            theta=theta, zeta=zeta, chi=chi, gamma=gamma, phi=phi
        ),
        characterization_index=5,
    )
    actual = cirq.unitary(corrected.as_circuit())

    assert cirq.equal_up_to_global_phase(actual, expected)
    assert corrected.moment_to_calibration == [None, 5, None]


@pytest.mark.parametrize(
    'theta,zeta,chi,gamma,phi',
    itertools.product(
        [np.pi / 4, -0.2], [-0.3, 0.1, 0.5], [-0.3, 0.2, 0.4], [-0.6, 0.1, 0.6], [0.2, 0.6]
    ),
)
def test_phase_corrected_fsim_operations_with_phase_exponent(
    theta: float, zeta: float, chi: float, gamma: float, phi: float
) -> None:
    a, b = cirq.LineQubit.range(2)

    phase_exponent = 0.5

    # Theta is negated to match the phase exponent of 0.5.
    expected_gate = cirq.PhasedFSimGate(theta=-theta, zeta=-zeta, chi=-chi, gamma=-gamma, phi=phi)
    expected = cirq.unitary(expected_gate)

    corrected = workflow.FSimPhaseCorrections.from_characterization(
        (a, b),
        PhaseCalibratedFSimGate(cirq.FSimGate(theta=theta, phi=phi), phase_exponent),
        cirq.google.PhasedFSimCharacterization(
            theta=theta, zeta=zeta, chi=chi, gamma=gamma, phi=phi
        ),
        characterization_index=5,
    )
    actual = cirq.unitary(corrected.as_circuit())

    assert cirq.equal_up_to_global_phase(actual, expected)
    assert corrected.moment_to_calibration == [None, 5, None]


def test_zeta_chi_gamma_calibration_for_moments():
    a, b = cirq.LineQubit.range(2)

    characterizations = [
        PhasedFSimCalibrationResult(
            parameters={(a, b): SQRT_ISWAP_PARAMETERS},
            gate=SQRT_ISWAP_GATE,
            options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
        )
    ]
    moment_allocations = [0]

    for circuit in [
        cirq.Circuit(cirq.FSimGate(theta=np.pi / 4, phi=0.0).on(a, b)),
        cirq.Circuit(cirq.FSimGate(theta=-np.pi / 4, phi=0.0).on(a, b)),
    ]:
        calibrated_circuit = workflow.make_zeta_chi_gamma_compensation_for_moments(
            workflow.CircuitWithCalibration(circuit, moment_allocations), characterizations
        )
        assert np.allclose(cirq.unitary(circuit), cirq.unitary(calibrated_circuit.circuit))
        assert calibrated_circuit.moment_to_calibration == [None, 0, None]


def test_zeta_chi_gamma_calibration_for_moments_invalid_argument_fails() -> None:
    a, b, c = cirq.LineQubit.range(3)

    with pytest.raises(ValueError):
        circuit_with_calibration = workflow.CircuitWithCalibration(cirq.Circuit(), [1])
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])

    with pytest.raises(ValueError):
        circuit_with_calibration = workflow.CircuitWithCalibration(
            cirq.Circuit(SQRT_ISWAP_GATE.on(a, b)), [None]
        )
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])

    with pytest.raises(ValueError):
        circuit_with_calibration = workflow.CircuitWithCalibration(
            cirq.Circuit(SQRT_ISWAP_GATE.on(a, b)), [0]
        )
        characterizations = [
            PhasedFSimCalibrationResult(
                parameters={},
                gate=SQRT_ISWAP_GATE,
                options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            )
        ]
        workflow.make_zeta_chi_gamma_compensation_for_moments(
            circuit_with_calibration, characterizations
        )

    with pytest.raises(workflow.IncompatibleMomentError):
        circuit_with_calibration = workflow.CircuitWithCalibration(
            cirq.Circuit(cirq.GlobalPhaseOperation(coefficient=1.0)), [None]
        )
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])

    with pytest.raises(workflow.IncompatibleMomentError):
        circuit_with_calibration = workflow.CircuitWithCalibration(
            cirq.Circuit(cirq.CZ.on(a, b)), [None]
        )
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])

    with pytest.raises(workflow.IncompatibleMomentError):
        circuit_with_calibration = workflow.CircuitWithCalibration(
            cirq.Circuit([SQRT_ISWAP_GATE.on(a, b), cirq.Z.on(c)]), [0]
        )
        characterizations = [
            PhasedFSimCalibrationResult(
                parameters={
                    (a, b): PhasedFSimCharacterization(
                        theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5
                    )
                },
                gate=SQRT_ISWAP_GATE,
                options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            )
        ]
        workflow.make_zeta_chi_gamma_compensation_for_moments(
            circuit_with_calibration, characterizations
        )


def test_run_zeta_chi_gamma_calibration_for_moments() -> None:
    parameters_ab = cirq.google.PhasedFSimCharacterization(zeta=0.5, chi=0.4, gamma=0.3)
    parameters_bc = cirq.google.PhasedFSimCharacterization(zeta=-0.5, chi=-0.4, gamma=-0.3)
    parameters_cd = cirq.google.PhasedFSimCharacterization(zeta=0.2, chi=0.3, gamma=0.4)

    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq.google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={
            (a, b): parameters_ab.merge_with(SQRT_ISWAP_PARAMETERS),
            (b, c): parameters_bc.merge_with(SQRT_ISWAP_PARAMETERS),
            (c, d): parameters_cd.merge_with(SQRT_ISWAP_PARAMETERS),
        }
    )

    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_GATE.on(a, b), SQRT_ISWAP_GATE.on(c, d)],
            [SQRT_ISWAP_GATE.on(b, c)],
        ]
    )

    options = cirq.google.FloquetPhasedFSimCalibrationOptions(
        characterize_theta=False,
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_phi=False,
    )

    calibrated_circuit, calibrations = workflow.run_zeta_chi_gamma_compensation_for_moments(
        circuit,
        engine_simulator,
        processor_id=None,
        gate_set=cirq.google.SQRT_ISWAP_GATESET,
        options=options,
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated_circuit.circuit),
        cirq.final_state_vector(circuit),
    )
    assert calibrations == [
        cirq.google.PhasedFSimCalibrationResult(
            gate=SQRT_ISWAP_GATE,
            parameters={(a, b): parameters_ab, (c, d): parameters_cd},
            options=options,
        ),
        cirq.google.PhasedFSimCalibrationResult(
            gate=SQRT_ISWAP_GATE, parameters={(b, c): parameters_bc}, options=options
        ),
    ]
    assert calibrated_circuit.moment_to_calibration == [None, None, 0, None, None, 1, None]


def test_run_zeta_chi_gamma_calibration_for_moments_no_chi() -> None:
    parameters_ab = cirq.google.PhasedFSimCharacterization(theta=np.pi / 4, zeta=0.5, gamma=0.3)
    parameters_bc = cirq.google.PhasedFSimCharacterization(theta=np.pi / 4, zeta=-0.5, gamma=-0.3)
    parameters_cd = cirq.google.PhasedFSimCharacterization(theta=np.pi / 4, zeta=0.2, gamma=0.4)

    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq.google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd},
        ideal_when_missing_parameter=True,
    )

    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_GATE.on(a, b), SQRT_ISWAP_GATE.on(c, d)],
            [SQRT_ISWAP_GATE.on(b, c)],
        ]
    )

    calibrated_circuit, *_ = workflow.run_zeta_chi_gamma_compensation_for_moments(
        circuit, engine_simulator, processor_id=None, gate_set=cirq.google.SQRT_ISWAP_GATESET
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated_circuit.circuit),
        cirq.final_state_vector(circuit),
    )
