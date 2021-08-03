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
import itertools
from typing import Optional
from unittest import mock

import numpy as np
import pytest

import cirq
import cirq_google
import cirq_google.calibration.workflow as workflow
import cirq_google.calibration.xeb_wrapper
from cirq.experiments import (
    random_rotations_between_grid_interaction_layers_circuit,
    XEBPhasedFSimCharacterizationOptions,
)
from cirq_google.calibration.engine_simulator import PhasedFSimEngineSimulator
from cirq_google.calibration.phased_fsim import (
    ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    PhaseCalibratedFSimGate,
    PhasedFSimCharacterization,
    PhasedFSimCalibrationResult,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION,
    LocalXEBPhasedFSimCalibrationRequest,
    LocalXEBPhasedFSimCalibrationOptions,
    XEBPhasedFSimCalibrationRequest,
    XEBPhasedFSimCalibrationOptions,
)

SQRT_ISWAP_INV_PARAMETERS = cirq_google.PhasedFSimCharacterization(
    theta=np.pi / 4, zeta=0.0, chi=0.0, gamma=0.0, phi=0.0
)
SYCAMORE_PARAMETERS = cirq_google.PhasedFSimCharacterization(
    theta=np.pi / 2, zeta=0.0, chi=0.0, gamma=0.0, phi=np.pi / 6
)
SQRT_ISWAP_GATE = cirq.FSimGate(7 * np.pi / 4, 0.0)
SQRT_ISWAP_INV_GATE = cirq.FSimGate(np.pi / 4, 0.0)


def _fsim_identity_converter(gate: cirq.Gate) -> Optional[PhaseCalibratedFSimGate]:
    if isinstance(gate, cirq.FSimGate):
        return PhaseCalibratedFSimGate(gate, 0.0)
    return None


def test_prepare_floquet_characterization_for_moment_none_for_measurements():
    a, b, c, d = cirq.LineQubit.range(4)
    moment = cirq.Moment(cirq.measure(a, b, c, d))
    assert (
        workflow.prepare_floquet_characterization_for_moment(
            moment, WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
        )
        is None
    )


@pytest.mark.parametrize(
    'options',
    [WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION],
)
def test_prepare_characterization_for_moment_none_for_measurements(options):
    a, b, c, d = cirq.LineQubit.range(4)
    moment = cirq.Moment(cirq.measure(a, b, c, d))
    assert workflow.prepare_characterization_for_moment(moment, options) is None


def test_prepare_floquet_characterization_for_moment_fails_for_non_gate_operation():
    moment = cirq.Moment(cirq.GlobalPhaseOperation(coefficient=1.0))
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_floquet_characterization_for_moment(
            moment, WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
        )


@pytest.mark.parametrize(
    'options',
    [WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION],
)
def test_prepare_characterization_for_moment_fails_for_non_gate_operation(options):
    moment = cirq.Moment(cirq.GlobalPhaseOperation(coefficient=1.0))
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_characterization_for_moment(moment, options)


def test_prepare_floquet_characterization_for_moment_fails_for_unsupported_gate():
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment(cirq.CZ(a, b))
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_floquet_characterization_for_moment(
            moment,
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            gates_translator=_fsim_identity_converter,
        )


@pytest.mark.parametrize(
    'options',
    [WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION],
)
def test_prepare_characterization_for_moment_fails_for_unsupported_gate(options):
    a, b = cirq.LineQubit.range(2)
    moment = cirq.Moment(cirq.CZ(a, b))
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_characterization_for_moment(
            moment,
            options,
            gates_translator=_fsim_identity_converter,
        )


def test_prepare_floquet_characterization_for_moment_fails_for_mixed_gates():
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


@pytest.mark.parametrize(
    'options',
    [WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION],
)
def test_prepare_characterization_for_moment_fails_for_mixed_gates(options):
    a, b, c, d = cirq.LineQubit.range(4)
    moment = cirq.Moment(
        [
            cirq.FSimGate(theta=np.pi / 4, phi=0.0).on(a, b),
            cirq.FSimGate(theta=np.pi / 8, phi=0.0).on(c, d),
        ]
    )
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_characterization_for_moment(
            moment,
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            gates_translator=_fsim_identity_converter,
        )


def test_prepare_floquet_characterization_for_moment_fails_for_mixed_moment():
    a, b, c = cirq.LineQubit.range(3)
    moment = cirq.Moment([cirq.FSimGate(theta=np.pi / 4, phi=0.0).on(a, b), cirq.Z.on(c)])
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_floquet_characterization_for_moment(
            moment, WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
        )


@pytest.mark.parametrize(
    'options',
    [WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION],
)
def test_prepare_characterization_for_moment_fails_for_mixed_moment(options):
    a, b, c = cirq.LineQubit.range(3)
    moment = cirq.Moment([cirq.FSimGate(theta=np.pi / 4, phi=0.0).on(a, b), cirq.Z.on(c)])
    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.prepare_characterization_for_moment(
            moment, WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
        )


def test_prepare_floquet_characterization_for_moments():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
            [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(b)],
        ]
    )
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(
        circuit, options=options
    )

    assert requests == [
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
    ]

    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, None]


def test_prepare_characterization_for_circuits_moments():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit_1 = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(b)],
        ]
    )
    circuit_2 = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
            [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(b)],
        ]
    )
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    circuits_with_calibration, requests = workflow.prepare_characterization_for_circuits_moments(
        [circuit_1, circuit_2], options=options
    )

    assert requests == [
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
    ]

    assert len(circuits_with_calibration) == 2
    assert circuits_with_calibration[0].circuit == circuit_1
    assert circuits_with_calibration[0].moment_to_calibration == [None, 0, None]
    assert circuits_with_calibration[1].circuit == circuit_2
    assert circuits_with_calibration[1].moment_to_calibration == [None, 1, None]


@pytest.mark.parametrize(
    'options_cls',
    [
        (
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            cirq_google.FloquetPhasedFSimCalibrationRequest,
        ),
        (ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION, cirq_google.XEBPhasedFSimCalibrationRequest),
    ],
)
def test_prepare_characterization_for_moments(options_cls):
    options, cls = options_cls
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
            [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(b)],
        ]
    )

    circuit_with_calibration, requests = workflow.prepare_characterization_for_moments(
        circuit, options=options
    )

    assert requests == [
        cls(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options),
    ]

    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, None]


@pytest.mark.parametrize(
    'options_cls',
    [
        (
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            cirq_google.FloquetPhasedFSimCalibrationRequest,
        ),
        (ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION, cirq_google.XEBPhasedFSimCalibrationRequest),
    ],
)
def test_prepare_characterization_for_moments_with_permit_mixed_moments(options_cls):
    options, cls = options_cls
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.Y(c)]),
            cirq.Moment([SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)]),
            cirq.Moment([cirq.X(a), SQRT_ISWAP_INV_GATE.on(b, c)], cirq.Y(d)),
            cirq.Moment([cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(b)]),
        ]
    )

    circuit_with_calibration, requests = workflow.prepare_characterization_for_moments(
        circuit, options=options, permit_mixed_moments=True
    )

    assert requests == [
        cls(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options),
    ]

    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, None]


def test_prepare_floquet_characterization_for_moments_merges_sub_sets():
    a, b, c, d, e = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
            [SQRT_ISWAP_INV_GATE.on(a, b)],
        ]
    )
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)])
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(
        circuit, options=options
    )

    assert requests == [
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 0, 1]


@pytest.mark.parametrize(
    'options_cls',
    [
        (
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            cirq_google.FloquetPhasedFSimCalibrationRequest,
        ),
        (ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION, cirq_google.XEBPhasedFSimCalibrationRequest),
    ],
)
def test_prepare_characterization_for_moments_merges_sub_sets(options_cls):
    options, cls = options_cls
    a, b, c, d, e = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
            [SQRT_ISWAP_INV_GATE.on(a, b)],
        ]
    )
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)])

    circuit_with_calibration, requests = workflow.prepare_characterization_for_moments(
        circuit, options=options
    )

    assert requests == [
        cls(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options),
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 0, 1]


def test_prepare_floquet_characterization_for_moments_merges_many_circuits():
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION
    a, b, c, d, e = cirq.LineQubit.range(5)

    circuit_1 = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
            [SQRT_ISWAP_INV_GATE.on(a, b)],
        ]
    )

    circuit_with_calibration_1, requests_1 = workflow.prepare_floquet_characterization_for_moments(
        circuit_1, options=options
    )

    assert requests_1 == [
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
    ]
    assert circuit_with_calibration_1.circuit == circuit_1
    assert circuit_with_calibration_1.moment_to_calibration == [None, 0, 1, 0]

    circuit_2 = cirq.Circuit([SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)])

    circuit_with_calibration_2, requests_2 = workflow.prepare_floquet_characterization_for_moments(
        circuit_2, options=options, initial=requests_1
    )

    assert requests_2 == [
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
    ]
    assert circuit_with_calibration_2.circuit == circuit_2
    assert circuit_with_calibration_2.moment_to_calibration == [1]


@pytest.mark.parametrize(
    'options_cls',
    [
        (
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            cirq_google.FloquetPhasedFSimCalibrationRequest,
        ),
        (ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION, cirq_google.XEBPhasedFSimCalibrationRequest),
    ],
)
def test_prepare_characterization_for_moments_merges_many_circuits(options_cls):
    options, cls = options_cls
    a, b, c, d, e = cirq.LineQubit.range(5)

    circuit_1 = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
            [SQRT_ISWAP_INV_GATE.on(a, b)],
        ]
    )

    circuit_with_calibration_1, requests_1 = workflow.prepare_characterization_for_moments(
        circuit_1, options=options
    )

    assert requests_1 == [
        cls(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options),
    ]
    assert circuit_with_calibration_1.circuit == circuit_1
    assert circuit_with_calibration_1.moment_to_calibration == [None, 0, 1, 0]

    circuit_2 = cirq.Circuit([SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)])

    circuit_with_calibration_2, requests_2 = workflow.prepare_characterization_for_moments(
        circuit_2, options=options, initial=requests_1
    )

    assert requests_2 == [
        cls(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options),
    ]
    assert circuit_with_calibration_2.circuit == circuit_2
    assert circuit_with_calibration_2.moment_to_calibration == [1]


def test_prepare_floquet_characterization_for_moments_does_not_merge_sub_sets_when_disabled():
    a, b, c, d, e = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
            [SQRT_ISWAP_INV_GATE.on(a, b)],
        ]
    )
    circuit += cirq.Circuit(
        [SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)],
        [SQRT_ISWAP_INV_GATE.on(b, c)],
    )
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(
        circuit, options=options, merge_subsets=False
    )

    assert requests == [
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b),), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 2, 3, 1]


@pytest.mark.parametrize(
    'options_cls',
    [
        (
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            cirq_google.FloquetPhasedFSimCalibrationRequest,
        ),
        (ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION, cirq_google.XEBPhasedFSimCalibrationRequest),
    ],
)
def test_prepare_characterization_for_moments_does_not_merge_sub_sets_when_disabled(options_cls):
    options, cls = options_cls
    a, b, c, d, e = cirq.LineQubit.range(5)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
            [SQRT_ISWAP_INV_GATE.on(a, b)],
        ]
    )
    circuit += cirq.Circuit(
        [SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)],
        [SQRT_ISWAP_INV_GATE.on(b, c)],
    )

    circuit_with_calibration, requests = workflow.prepare_characterization_for_moments(
        circuit, options=options, merge_subsets=False
    )

    assert requests == [
        cls(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((b, c),), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((a, b),), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options),
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 2, 3, 1]


def test_prepare_floquet_characterization_for_moments_merges_compatible_sets():
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    circuit = cirq.Circuit([cirq.X(a), cirq.Y(c)])
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(a, b)])
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)])
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(c, d)])
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(a, f), SQRT_ISWAP_INV_GATE.on(d, e)])
    options = WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION

    circuit_with_calibration, requests = workflow.prepare_floquet_characterization_for_moments(
        circuit, options=options
    )

    assert requests == [
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, f), (b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 0, 1]


@pytest.mark.parametrize(
    'options_cls',
    [
        (
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            cirq_google.FloquetPhasedFSimCalibrationRequest,
        ),
        (ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION, cirq_google.XEBPhasedFSimCalibrationRequest),
    ],
)
def test_prepare_characterization_for_moments_merges_compatible_sets(options_cls):
    options, cls = options_cls
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    circuit = cirq.Circuit([cirq.X(a), cirq.Y(c)])
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(a, b)])
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(b, c), SQRT_ISWAP_INV_GATE.on(d, e)])
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(c, d)])
    circuit += cirq.Moment([SQRT_ISWAP_INV_GATE.on(a, f), SQRT_ISWAP_INV_GATE.on(d, e)])

    circuit_with_calibration, requests = workflow.prepare_characterization_for_moments(
        circuit, options=options
    )

    assert requests == [
        cls(pairs=((a, b), (c, d)), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((a, f), (b, c), (d, e)), gate=SQRT_ISWAP_INV_GATE, options=options),
    ]
    assert circuit_with_calibration.circuit == circuit
    assert circuit_with_calibration.moment_to_calibration == [None, 0, 1, 0, 1]


def test_prepare_floquet_characterization_for_operations():
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
            [SQRT_ISWAP_INV_GATE.on(q00, q01), SQRT_ISWAP_INV_GATE.on(q10, q11)],
            [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(q01)],
        ]
    )

    requests_1 = workflow.prepare_floquet_characterization_for_operations(
        circuit_1, options=options
    )

    assert requests_1 == [
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q10, q11),), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q00, q01),), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
    ]

    # Prepare characterizations for a list of circuits.
    circuit_2 = cirq.Circuit(
        [
            [SQRT_ISWAP_INV_GATE.on(q00, q01), SQRT_ISWAP_INV_GATE.on(q10, q11)],
            [SQRT_ISWAP_INV_GATE.on(q00, q10), SQRT_ISWAP_INV_GATE.on(q01, q11)],
            [SQRT_ISWAP_INV_GATE.on(q10, q20), SQRT_ISWAP_INV_GATE.on(q11, q21)],
        ]
    )

    requests_2 = workflow.prepare_floquet_characterization_for_operations(
        [circuit_1, circuit_2], options=options
    )

    # The order of moments originates from HALF_GRID_STAGGERED_PATTERN.
    assert requests_2 == [
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q00, q10), (q11, q21)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q01, q11), (q10, q20)), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q10, q11),), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
        cirq_google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((q00, q01),), gate=SQRT_ISWAP_INV_GATE, options=options
        ),
    ]


@pytest.mark.parametrize(
    'options_cls',
    [
        (
            WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            cirq_google.FloquetPhasedFSimCalibrationRequest,
        ),
        (ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION, cirq_google.XEBPhasedFSimCalibrationRequest),
    ],
)
def test_prepare_characterization_for_operations(options_cls):
    options, cls = options_cls
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)

    # Prepare characterizations for a single circuit.
    circuit_1 = cirq.Circuit(
        [
            [cirq.X(q00), cirq.Y(q11)],
            [SQRT_ISWAP_INV_GATE.on(q00, q01), SQRT_ISWAP_INV_GATE.on(q10, q11)],
            [cirq.WaitGate(duration=cirq.Duration(micros=5.0)).on(q01)],
        ]
    )

    requests_1 = workflow.prepare_characterization_for_operations(circuit_1, options=options)

    assert requests_1 == [
        cls(pairs=((q10, q11),), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((q00, q01),), gate=SQRT_ISWAP_INV_GATE, options=options),
    ]

    # Prepare characterizations for a list of circuits.
    circuit_2 = cirq.Circuit(
        [
            [SQRT_ISWAP_INV_GATE.on(q00, q01), SQRT_ISWAP_INV_GATE.on(q10, q11)],
            [SQRT_ISWAP_INV_GATE.on(q00, q10), SQRT_ISWAP_INV_GATE.on(q01, q11)],
            [SQRT_ISWAP_INV_GATE.on(q10, q20), SQRT_ISWAP_INV_GATE.on(q11, q21)],
        ]
    )

    requests_2 = workflow.prepare_characterization_for_operations(
        [circuit_1, circuit_2], options=options
    )

    # The order of moments originates from HALF_GRID_STAGGERED_PATTERN.
    assert requests_2 == [
        cls(pairs=((q00, q10), (q11, q21)), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((q01, q11), (q10, q20)), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((q10, q11),), gate=SQRT_ISWAP_INV_GATE, options=options),
        cls(pairs=((q00, q01),), gate=SQRT_ISWAP_INV_GATE, options=options),
    ]


def test_prepare_floquet_characterization_for_operations_when_no_interactions():
    q00 = cirq.GridQubit(0, 0)
    q11 = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit([cirq.X(q00), cirq.X(q11)])

    assert workflow.prepare_floquet_characterization_for_operations(circuit) == []


@pytest.mark.parametrize(
    'options',
    [WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION],
)
def test_prepare_characterization_for_operations_when_no_interactions(options):
    q00 = cirq.GridQubit(0, 0)
    q11 = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit([cirq.X(q00), cirq.X(q11)])

    assert workflow.prepare_characterization_for_operations(circuit, options) == []


def test_prepare_floquet_characterization_for_operations_when_non_grid_fails():
    q00 = cirq.GridQubit(0, 0)
    q11 = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit(SQRT_ISWAP_INV_GATE.on(q00, q11))

    with pytest.raises(ValueError):
        workflow.prepare_floquet_characterization_for_operations(circuit)


@pytest.mark.parametrize(
    'options',
    [WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION],
)
def test_prepare_characterization_for_operations_when_non_grid_fails(options):
    q00 = cirq.GridQubit(0, 0)
    q11 = cirq.GridQubit(1, 1)
    circuit = cirq.Circuit(SQRT_ISWAP_INV_GATE.on(q00, q11))

    with pytest.raises(ValueError):
        workflow.prepare_characterization_for_operations(circuit, options)


def test_prepare_floquet_characterization_for_operations_when_multiple_gates_fails():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(
        [SQRT_ISWAP_INV_GATE.on(q00, q01), cirq.FSimGate(theta=0.0, phi=np.pi).on(q00, q01)]
    )

    with pytest.raises(ValueError):
        workflow.prepare_floquet_characterization_for_operations(
            circuit, gates_translator=_fsim_identity_converter
        )


@pytest.mark.parametrize(
    'options',
    [WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION, ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION],
)
def test_prepare_characterization_for_operations_when_multiple_gates_fails(options):
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    circuit = cirq.Circuit(
        [SQRT_ISWAP_INV_GATE.on(q00, q01), cirq.FSimGate(theta=0.0, phi=np.pi).on(q00, q01)]
    )

    with pytest.raises(ValueError):
        workflow.prepare_characterization_for_operations(
            circuit, gates_translator=_fsim_identity_converter, options=options
        )


def test_make_zeta_chi_gamma_compensation_for_operations():
    a, b, c, d = cirq.LineQubit.range(4)
    parameters_ab = cirq_google.PhasedFSimCharacterization(zeta=0.5, chi=0.4, gamma=0.3)
    parameters_bc = cirq_google.PhasedFSimCharacterization(zeta=-0.5, chi=-0.4, gamma=-0.3)
    parameters_cd = cirq_google.PhasedFSimCharacterization(zeta=0.2, chi=0.3, gamma=0.4)

    parameters_dict = {(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd}

    engine_simulator = cirq_google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={
            pair: parameters.merge_with(SQRT_ISWAP_INV_PARAMETERS)
            for pair, parameters in parameters_dict.items()
        }
    )

    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
        ]
    )

    options = cirq_google.FloquetPhasedFSimCalibrationOptions(
        characterize_theta=False,
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_phi=False,
    )

    characterizations = [
        PhasedFSimCalibrationResult(
            parameters={pair: parameters}, gate=SQRT_ISWAP_INV_GATE, options=options
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


def test_make_zeta_chi_gamma_compensation_for_operations_with_permit_mixed_moments_disabled():
    a, b, c, d = cirq.LineQubit.range(4)
    parameters_ab = cirq_google.PhasedFSimCharacterization(zeta=0.5, chi=0.4, gamma=0.3)
    parameters_bc = cirq_google.PhasedFSimCharacterization(zeta=-0.5, chi=-0.4, gamma=-0.3)
    parameters_cd = cirq_google.PhasedFSimCharacterization(zeta=0.2, chi=0.3, gamma=0.4)

    parameters_dict = {(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd}

    circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.Y(c)]),
            cirq.Moment([SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)]),
            cirq.Moment([cirq.X(a), SQRT_ISWAP_INV_GATE.on(b, c), cirq.Y(d)]),
        ]
    )

    options = cirq_google.FloquetPhasedFSimCalibrationOptions(
        characterize_theta=False,
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_phi=False,
    )

    characterizations = [
        PhasedFSimCalibrationResult(
            parameters={pair: parameters}, gate=SQRT_ISWAP_INV_GATE, options=options
        )
        for pair, parameters in parameters_dict.items()
    ]

    with pytest.raises(workflow.IncompatibleMomentError):
        workflow.make_zeta_chi_gamma_compensation_for_operations(
            circuit,
            characterizations,
        )


def test_make_zeta_chi_gamma_compensation_for_operations_with_permit_mixed_moments():
    a, b, c, d = cirq.LineQubit.range(4)
    parameters_ab = cirq_google.PhasedFSimCharacterization(zeta=0.5, chi=0.4, gamma=0.3)
    parameters_bc = cirq_google.PhasedFSimCharacterization(zeta=-0.5, chi=-0.4, gamma=-0.3)
    parameters_cd = cirq_google.PhasedFSimCharacterization(zeta=0.2, chi=0.3, gamma=0.4)

    parameters_dict = {(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd}

    engine_simulator = cirq_google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={
            pair: parameters.merge_with(SQRT_ISWAP_INV_PARAMETERS)
            for pair, parameters in parameters_dict.items()
        }
    )

    circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.Y(c)]),
            cirq.Moment([SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)]),
            cirq.Moment([cirq.X(a), SQRT_ISWAP_INV_GATE.on(b, c), cirq.Y(d)]),
        ]
    )

    options = cirq_google.FloquetPhasedFSimCalibrationOptions(
        characterize_theta=False,
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_phi=False,
    )

    characterizations = [
        PhasedFSimCalibrationResult(
            parameters={pair: parameters}, gate=SQRT_ISWAP_INV_GATE, options=options
        )
        for pair, parameters in parameters_dict.items()
    ]

    calibrated_circuit = workflow.make_zeta_chi_gamma_compensation_for_operations(
        circuit,
        characterizations,
        permit_mixed_moments=True,
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated_circuit),
        cirq.final_state_vector(circuit),
    )
    assert calibrated_circuit[5] == cirq.Moment(
        [cirq.X(a), SQRT_ISWAP_INV_GATE.on(b, c), cirq.Y(d)]
    )


def test_run_calibrations():
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

    job = cirq_google.engine.EngineJob('project_id', 'program_id', 'job_id', None)
    job._calibration_results = [result]

    engine = mock.MagicMock(spec=cirq_google.Engine)
    engine.run_calibration.return_value = job

    sampler = cirq_google.QuantumEngineSampler(
        engine=engine, processor_id='qproc', gate_set=cirq_google.FSIM_GATESET
    )

    progress_calls = []

    def progress(step: int, steps: int) -> None:
        progress_calls.append((step, steps))

    actual = workflow.run_calibrations([request], sampler, progress_func=progress)

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
            project_id='project_id',
            program_id='program_id',
            job_id='job_id',
        )
    ]

    assert actual == expected
    assert progress_calls == [(1, 1)]


def test_run_characterization_with_engine():
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

    job = cirq_google.engine.EngineJob('project_id', 'program_id', 'job_id', None)
    job._calibration_results = [result]

    engine = mock.MagicMock(spec=cirq_google.Engine)
    engine.run_calibration.return_value = job

    progress_calls = []

    def progress(step: int, steps: int) -> None:
        progress_calls.append((step, steps))

    actual = workflow.run_calibrations(
        [request], engine, 'qproc', cirq_google.FSIM_GATESET, progress_func=progress
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
            project_id='project_id',
            program_id='program_id',
            job_id='job_id',
        )
    ]

    assert actual == expected
    assert progress_calls == [(1, 1)]


def test_run_calibrations_empty():
    assert workflow.run_calibrations([], None, 'qproc', cirq_google.FSIM_GATESET) == []


def test_run_calibrations_fails_when_invalid_arguments():
    with pytest.raises(ValueError):
        assert workflow.run_calibrations(
            [], None, 'qproc', cirq_google.FSIM_GATESET, max_layers_per_request=0
        )

    request = FloquetPhasedFSimCalibrationRequest(
        gate=SQRT_ISWAP_INV_GATE,
        pairs=(),
        options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    )
    engine = mock.MagicMock(spec=cirq_google.Engine)

    with pytest.raises(ValueError):
        assert workflow.run_calibrations([request], engine, None, cirq_google.FSIM_GATESET)

    with pytest.raises(ValueError):
        assert workflow.run_calibrations([request], engine, 'qproc', None)

    with pytest.raises(ValueError):
        assert workflow.run_calibrations([request], 0, 'qproc', cirq_google.FSIM_GATESET)


def test_run_calibrations_with_simulator():
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = SQRT_ISWAP_INV_GATE

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
            gate=SQRT_ISWAP_INV_GATE,
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

    job = cirq_google.engine.EngineJob('project_id', 'program_id', 'job_id', None)
    job._calibration_results = [
        cirq_google.CalibrationResult(
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
    ]

    engine = mock.MagicMock(spec=cirq_google.Engine)
    engine.run_calibration.return_value = job

    circuit_with_calibration, requests = workflow.run_floquet_characterization_for_moments(
        circuit, engine, 'qproc', cirq_google.FSIM_GATESET, options=options
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
            project_id='project_id',
            program_id='program_id',
            job_id='job_id',
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
        cirq_google.PhasedFSimCharacterization(
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
        cirq_google.PhasedFSimCharacterization(
            theta=theta, zeta=zeta, chi=chi, gamma=gamma, phi=phi
        ),
        characterization_index=5,
    )
    actual = cirq.unitary(corrected.as_circuit())

    assert cirq.equal_up_to_global_phase(actual, expected)
    assert corrected.moment_to_calibration == [None, 5, None]


def test_make_zeta_chi_gamma_compensation_for_moments():
    a, b = cirq.LineQubit.range(2)

    moment_allocations = [0]

    for gate_to_calibrate, engine_gate in [
        (cirq.FSimGate(theta=np.pi / 4, phi=0.0), SQRT_ISWAP_INV_GATE),
        (cirq.FSimGate(theta=-np.pi / 4, phi=0.0), SQRT_ISWAP_INV_GATE),
        (cirq.ISwapPowGate(exponent=0.2), cirq.FSimGate(theta=0.1 * np.pi, phi=0.0)),
        (cirq.PhasedFSimGate(theta=0.1, phi=0.2), cirq.FSimGate(theta=0.1, phi=0.2)),
        (cirq.PhasedFSimGate(theta=0.1, phi=0.2, chi=0.3), cirq.FSimGate(theta=0.1, phi=0.2)),
        (cirq.PhasedISwapPowGate(exponent=0.2), cirq.FSimGate(theta=0.1 * np.pi, phi=0.0)),
        (
            cirq.PhasedISwapPowGate(exponent=0.2, phase_exponent=0.4),
            cirq.FSimGate(theta=0.1 * np.pi, phi=0.0),
        ),
        (cirq.CZ, cirq.FSimGate(theta=0.0, phi=np.pi)),
        (cirq.ops.CZPowGate(exponent=0.5), cirq.FSimGate(theta=0.0, phi=1.5 * np.pi)),
        (cirq_google.ops.SycamoreGate(), cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)),
    ]:
        circuit = cirq.Circuit(gate_to_calibrate.on(a, b))
        characterizations = [
            PhasedFSimCalibrationResult(
                # Assume that the engine gate is perfect.
                parameters={
                    (a, b): cirq_google.PhasedFSimCharacterization(
                        theta=engine_gate.theta, phi=engine_gate.phi, zeta=0.0, chi=0.0, gamma=0.0
                    )
                },
                gate=engine_gate,
                options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            )
        ]
        calibrated_circuit = workflow.make_zeta_chi_gamma_compensation_for_moments(
            workflow.CircuitWithCalibration(circuit, moment_allocations), characterizations
        )
        assert np.allclose(cirq.unitary(circuit), cirq.unitary(calibrated_circuit.circuit))
        assert calibrated_circuit.moment_to_calibration == [None, 0, None]


def test_make_zeta_chi_gamma_compensation_for_moments_wrong_engine_gate_error():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.FSimGate(theta=np.pi / 4, phi=0.2).on(a, b))
    characterizations = [
        PhasedFSimCalibrationResult(
            parameters={
                (a, b): cirq_google.PhasedFSimCharacterization(
                    theta=np.pi / 4, phi=0.2, zeta=0.0, chi=0.0, gamma=0.0
                )
            },
            gate=SQRT_ISWAP_INV_GATE,
            options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
        )
    ]
    with pytest.raises(ValueError, match="Engine gate .+ doesn't match characterized gate .+"):
        workflow.make_zeta_chi_gamma_compensation_for_moments(
            workflow.CircuitWithCalibration(circuit, [0]), characterizations
        )


def test_make_zeta_chi_gamma_compensation_for_moments_circuit():
    a, b = cirq.LineQubit.range(2)

    characterizations = [
        PhasedFSimCalibrationResult(
            parameters={(a, b): SQRT_ISWAP_INV_PARAMETERS},
            gate=SQRT_ISWAP_INV_GATE,
            options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
        )
    ]

    for circuit, expected_moment_to_calibration in [
        (cirq.Circuit(cirq.FSimGate(theta=np.pi / 4, phi=0.0).on(a, b)), [None, 0, None]),
        (
            cirq.Circuit([cirq.Z.on(a), cirq.FSimGate(theta=-np.pi / 4, phi=0.0).on(a, b)]),
            [None, None, 0, None],
        ),
    ]:
        calibrated_circuit = workflow.make_zeta_chi_gamma_compensation_for_moments(
            circuit, characterizations
        )
        assert np.allclose(cirq.unitary(circuit), cirq.unitary(calibrated_circuit.circuit))
        assert calibrated_circuit.moment_to_calibration == expected_moment_to_calibration


def test_zmake_zeta_chi_gamma_compensation_for_moments_invalid_argument_fails() -> None:
    a, b, c = cirq.LineQubit.range(3)

    with pytest.raises(ValueError):
        circuit_with_calibration = workflow.CircuitWithCalibration(cirq.Circuit(), [1])
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])

    with pytest.raises(ValueError):
        circuit_with_calibration = workflow.CircuitWithCalibration(
            cirq.Circuit(SQRT_ISWAP_INV_GATE.on(a, b)), [None]
        )
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])

    with pytest.raises(ValueError):
        workflow.make_zeta_chi_gamma_compensation_for_moments(
            cirq.Circuit(SQRT_ISWAP_INV_GATE.on(a, b)), []
        )

    with pytest.raises(ValueError):
        circuit_with_calibration = workflow.CircuitWithCalibration(
            cirq.Circuit(SQRT_ISWAP_INV_GATE.on(a, b)), [0]
        )
        characterizations = [
            PhasedFSimCalibrationResult(
                parameters={},
                gate=SQRT_ISWAP_INV_GATE,
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
            cirq.Circuit(cirq.CX.on(a, b)), [None]
        )
        workflow.make_zeta_chi_gamma_compensation_for_moments(circuit_with_calibration, [])

    with pytest.raises(workflow.IncompatibleMomentError):
        circuit_with_calibration = workflow.CircuitWithCalibration(
            cirq.Circuit([SQRT_ISWAP_INV_GATE.on(a, b), cirq.Z.on(c)]), [0]
        )
        characterizations = [
            PhasedFSimCalibrationResult(
                parameters={
                    (a, b): PhasedFSimCharacterization(
                        theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5
                    )
                },
                gate=SQRT_ISWAP_INV_GATE,
                options=WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            )
        ]
        workflow.make_zeta_chi_gamma_compensation_for_moments(
            circuit_with_calibration, characterizations
        )


def test_make_zeta_chi_gamma_compensation_for_moments_imperfect_gates():
    params_cz_ab = cirq_google.PhasedFSimCharacterization(
        zeta=0.02, chi=0.05, gamma=0.04, theta=0.0, phi=np.pi
    )
    params_cz_cd = cirq_google.PhasedFSimCharacterization(
        zeta=0.03, chi=0.08, gamma=0.03, theta=0.0, phi=np.pi
    )
    params_syc_ab = cirq_google.PhasedFSimCharacterization(
        zeta=0.01, chi=0.09, gamma=0.02, theta=np.pi / 2, phi=np.pi / 6
    )
    params_sqrt_iswap_ac = cirq_google.PhasedFSimCharacterization(
        zeta=0.05, chi=0.06, gamma=0.07, theta=np.pi / 4, phi=0.0
    )
    params_sqrt_iswap_bd = cirq_google.PhasedFSimCharacterization(
        zeta=0.01, chi=0.02, gamma=0.03, theta=np.pi / 4, phi=0.0
    )

    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq_google.PhasedFSimEngineSimulator.create_from_dictionary(
        parameters={
            (a, b): {
                cirq.FSimGate(theta=0, phi=np.pi): params_cz_ab,
                cirq_google.SYC: params_syc_ab,
            },
            (c, d): {cirq.FSimGate(theta=0, phi=np.pi): params_cz_cd},
            (a, c): {SQRT_ISWAP_INV_GATE: params_sqrt_iswap_ac},
            (b, d): {SQRT_ISWAP_INV_GATE: params_sqrt_iswap_bd},
        }
    )

    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.H(c)],
            [cirq.CZ.on(a, b), cirq.CZ.on(d, c)],
            [cirq_google.SYC.on(a, b)],
            [SQRT_ISWAP_GATE.on(a, c), SQRT_ISWAP_INV_GATE.on(b, d)],
        ]
    )

    options = cirq_google.FloquetPhasedFSimCalibrationOptions(
        characterize_theta=False,
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_phi=False,
    )

    characterizations = [
        cirq_google.PhasedFSimCalibrationResult(
            gate=cirq.FSimGate(theta=0, phi=np.pi),
            parameters={(a, b): params_cz_ab, (c, d): params_cz_cd},
            options=options,
        ),
        cirq_google.PhasedFSimCalibrationResult(
            gate=cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6),
            parameters={(a, b): params_syc_ab},
            options=options,
        ),
        cirq_google.PhasedFSimCalibrationResult(
            gate=cirq.FSimGate(theta=np.pi / 4, phi=0),
            parameters={(a, c): params_sqrt_iswap_ac, (b, d): params_sqrt_iswap_bd},
            options=options,
        ),
    ]

    circuit_with_calibration = workflow.make_zeta_chi_gamma_compensation_for_moments(
        circuit,
        characterizations,
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(circuit_with_calibration.circuit),
        cirq.final_state_vector(circuit),
    )


def test_run_zeta_chi_gamma_calibration_for_moments() -> None:
    parameters_ab = cirq_google.PhasedFSimCharacterization(zeta=0.5, chi=0.4, gamma=0.3)
    parameters_bc = cirq_google.PhasedFSimCharacterization(zeta=-0.5, chi=-0.4, gamma=-0.3)
    parameters_cd = cirq_google.PhasedFSimCharacterization(zeta=0.2, chi=0.3, gamma=0.4)

    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq_google.PhasedFSimEngineSimulator.create_from_dictionary(
        parameters={
            (a, b): {SQRT_ISWAP_INV_GATE: parameters_ab.merge_with(SQRT_ISWAP_INV_PARAMETERS)},
            (b, c): {cirq_google.ops.SYC: parameters_bc.merge_with(SYCAMORE_PARAMETERS)},
            (c, d): {SQRT_ISWAP_INV_GATE: parameters_cd.merge_with(SQRT_ISWAP_INV_PARAMETERS)},
        }
    )

    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [cirq_google.ops.SYC.on(b, c)],
        ]
    )

    options = cirq_google.FloquetPhasedFSimCalibrationOptions(
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
        gate_set=cirq_google.FSIM_GATESET,
        options=options,
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated_circuit.circuit),
        cirq.final_state_vector(circuit),
    )
    assert calibrations == [
        cirq_google.PhasedFSimCalibrationResult(
            gate=SQRT_ISWAP_INV_GATE,
            parameters={(a, b): parameters_ab, (c, d): parameters_cd},
            options=options,
        ),
        cirq_google.PhasedFSimCalibrationResult(
            gate=cirq_google.ops.SYC, parameters={(b, c): parameters_bc}, options=options
        ),
    ]
    assert calibrated_circuit.moment_to_calibration == [None, None, 0, None, None, 1, None]


def test_run_zeta_chi_gamma_calibration_for_moments_no_chi() -> None:
    parameters_ab = cirq_google.PhasedFSimCharacterization(theta=np.pi / 4, zeta=0.5, gamma=0.3)
    parameters_bc = cirq_google.PhasedFSimCharacterization(theta=np.pi / 4, zeta=-0.5, gamma=-0.3)
    parameters_cd = cirq_google.PhasedFSimCharacterization(theta=np.pi / 4, zeta=0.2, gamma=0.4)

    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq_google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd},
        ideal_when_missing_parameter=True,
    )

    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [SQRT_ISWAP_INV_GATE.on(a, b), SQRT_ISWAP_INV_GATE.on(c, d)],
            [SQRT_ISWAP_INV_GATE.on(b, c)],
        ]
    )

    calibrated_circuit, *_ = workflow.run_zeta_chi_gamma_compensation_for_moments(
        circuit, engine_simulator, processor_id=None, gate_set=cirq_google.SQRT_ISWAP_GATESET
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated_circuit.circuit),
        cirq.final_state_vector(circuit),
    )


_MOCK_ENGINE_SAMPLER = mock.MagicMock(
    spec=cirq_google.QuantumEngineSampler, _processor_ids=['my_fancy_processor'], _gate_set='test'
)


@pytest.mark.parametrize('sampler_engine', [cirq.Simulator, _MOCK_ENGINE_SAMPLER])
def test_run_local(sampler_engine, monkeypatch):
    called_times = 0

    def myfunc(
        calibration: LocalXEBPhasedFSimCalibrationRequest,
        sampler: cirq.Sampler,
    ):
        nonlocal called_times
        assert isinstance(calibration, LocalXEBPhasedFSimCalibrationRequest)
        assert sampler is not None
        called_times += 1
        return []

    # Note: you must patch specifically the function imported into `workflow`.
    monkeypatch.setattr('cirq_google.calibration.workflow.run_local_xeb_calibration', myfunc)

    qubit_indices = [
        (0, 5),
        (0, 6),
        (1, 6),
        (2, 6),
    ]
    qubits = [cirq.GridQubit(*idx) for idx in qubit_indices]

    circuits = [
        random_rotations_between_grid_interaction_layers_circuit(
            qubits,
            depth=depth,
            two_qubit_op_factory=lambda a, b, _: SQRT_ISWAP_INV_GATE.on(a, b),
            pattern=cirq.experiments.GRID_ALIGNED_PATTERN,
            seed=10,
        )
        for depth in [5, 10]
    ]

    options = LocalXEBPhasedFSimCalibrationOptions(
        fsim_options=XEBPhasedFSimCharacterizationOptions(
            characterize_zeta=True,
            characterize_gamma=True,
            characterize_chi=True,
            characterize_theta=False,
            characterize_phi=False,
            theta_default=np.pi / 4,
        ),
        n_processes=1,
    )

    characterization_requests = []
    for circuit in circuits:
        _, characterization_requests = workflow.prepare_characterization_for_moments(
            circuit, options=options, initial=characterization_requests
        )
    assert len(characterization_requests) == 2
    for cr in characterization_requests:
        assert isinstance(cr, LocalXEBPhasedFSimCalibrationRequest)

    workflow.run_calibrations(characterization_requests, sampler_engine)
    assert called_times == 2


def test_multiple_calibration_types_error():
    r1 = LocalXEBPhasedFSimCalibrationRequest(
        pairs=[], gate=None, options=LocalXEBPhasedFSimCalibrationOptions()
    )
    r2 = XEBPhasedFSimCalibrationRequest(
        pairs=[], gate=None, options=XEBPhasedFSimCalibrationOptions()
    )
    with pytest.raises(ValueError, match=r'must be of the same type\.'):
        workflow.run_calibrations([r1, r2], cirq.Simulator())
