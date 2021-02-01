from typing import Iterable, Tuple

import collections
import pytest

import numpy as np

from cirq.google.calibration.engine_simulator import (
    PhasedFSimEngineSimulator,
    SQRT_ISWAP_PARAMETERS,
)
from cirq.google.calibration import (
    FloquetPhasedFSimCalibrationOptions,
    IncompatibleMomentError,
    ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
)
import cirq


def test_floquet_get_calibrations() -> None:

    parameters_ab = cirq.google.PhasedFSimCharacterization(
        theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2
    )
    parameters_bc = cirq.google.PhasedFSimCharacterization(
        theta=0.8, zeta=-0.5, chi=-0.4, gamma=-0.3, phi=-0.2
    )
    parameters_cd_dict = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}
    parameters_cd = cirq.google.PhasedFSimCharacterization(**parameters_cd_dict)

    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd_dict}
    )

    requests = [_create_sqrt_iswap_request([(a, b), (c, d)]), _create_sqrt_iswap_request([(b, c)])]

    results = engine_simulator.get_calibrations(requests)

    assert results == [
        cirq.google.PhasedFSimCalibrationResult(
            gate=cirq.FSimGate(np.pi / 4, 0.0),
            parameters={(a, b): parameters_ab, (c, d): parameters_cd},
            options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
        ),
        cirq.google.PhasedFSimCalibrationResult(
            gate=cirq.FSimGate(np.pi / 4, 0.0),
            parameters={(b, c): parameters_bc},
            options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
        ),
    ]


# TODO: Test get_calibrations exceptions and options (missing angles, etc.)


def test_ideal_sqrt_iswap_simulates_correctly() -> None:
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)],
        ]
    )

    engine_simulator = PhasedFSimEngineSimulator.create_with_ideal_sqrt_iswap()

    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(circuit)

    assert cirq.allclose_up_to_global_phase(actual, expected)


def test_ideal_sqrt_iswap_simulates_correctly_invalid_circuit_fails() -> None:
    engine_simulator = PhasedFSimEngineSimulator.create_with_ideal_sqrt_iswap()

    with pytest.raises(IncompatibleMomentError):
        a, b = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([cirq.CZ.on(a, b)])
        engine_simulator.simulate(circuit)

    with pytest.raises(IncompatibleMomentError):
        circuit = cirq.Circuit(cirq.GlobalPhaseOperation(coefficient=1.0))
        engine_simulator.simulate(circuit)


def test_with_random_gaussian_sqrt_iswap_simulates_correctly() -> None:
    engine_simulator = PhasedFSimEngineSimulator.create_with_random_gaussian_sqrt_iswap(
        mean=SQRT_ISWAP_PARAMETERS
    )

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)],
        ]
    )

    calibrations = engine_simulator.get_calibrations(
        [_create_sqrt_iswap_request([(a, b), (c, d)]), _create_sqrt_iswap_request([(b, c)])]
    )
    parameters = collections.ChainMap(*(calibration.parameters for calibration in calibrations))

    expected_circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.X(c)],
            [
                cirq.PhasedFSimGate(**parameters[(a, b)].asdict()).on(a, b),
                cirq.PhasedFSimGate(**parameters[(c, d)].asdict()).on(c, d),
            ],
            [cirq.PhasedFSimGate(**parameters[(b, c)].asdict()).on(b, c)],
        ]
    )

    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)

    assert cirq.allclose_up_to_global_phase(actual, expected)


def test_from_dictionary_sqrt_iswap_simulates_correctly() -> None:
    parameters_ab = cirq.google.PhasedFSimCharacterization(
        theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2
    )
    parameters_bc = cirq.google.PhasedFSimCharacterization(
        theta=0.8, zeta=-0.5, chi=-0.4, gamma=-0.3, phi=-0.2
    )
    parameters_cd_dict = {'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5}

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)],
        ]
    )
    expected_circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.X(c)],
            [
                cirq.PhasedFSimGate(**parameters_ab.asdict()).on(a, b),
                cirq.PhasedFSimGate(**parameters_cd_dict).on(c, d),
            ],
            [cirq.PhasedFSimGate(**parameters_bc.asdict()).on(b, c)],
        ]
    )

    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={(a, b): parameters_ab, (b, c): parameters_bc, (c, d): parameters_cd_dict}
    )

    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)

    assert cirq.allclose_up_to_global_phase(actual, expected)


def test_from_dictionary_sqrt_iswap_ideal_when_missing_simulates_correctly() -> None:
    parameters_ab = cirq.google.PhasedFSimCharacterization(
        theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2
    )
    parameters_bc = cirq.google.PhasedFSimCharacterization(theta=0.8, zeta=-0.5, chi=-0.4)

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)],
        ]
    )
    expected_circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.X(c)],
            [
                cirq.PhasedFSimGate(**parameters_ab.asdict()).on(a, b),
                cirq.PhasedFSimGate(**SQRT_ISWAP_PARAMETERS.asdict()).on(c, d),
            ],
            [
                cirq.PhasedFSimGate(**parameters_bc.merge_with(SQRT_ISWAP_PARAMETERS).asdict()).on(
                    b, c
                )
            ],
        ]
    )

    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={(a, b): parameters_ab, (b, c): parameters_bc},
        ideal_when_missing_parameter=True,
        ideal_when_missing_gate=True,
    )

    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)

    assert cirq.allclose_up_to_global_phase(actual, expected)


def test_from_characterizations_sqrt_iswap_simulates_correctly() -> None:
    parameters_ab = cirq.google.PhasedFSimCharacterization(
        theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2
    )
    parameters_bc = cirq.google.PhasedFSimCharacterization(
        theta=0.8, zeta=-0.5, chi=-0.4, gamma=-0.3, phi=-0.2
    )
    parameters_cd = cirq.google.PhasedFSimCharacterization(
        theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5
    )

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.Y(c)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
            [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)],
        ]
    )
    expected_circuit = cirq.Circuit(
        [
            [cirq.X(a), cirq.X(c)],
            [
                cirq.PhasedFSimGate(**parameters_ab.asdict()).on(a, b),
                cirq.PhasedFSimGate(**parameters_cd.asdict()).on(c, d),
            ],
            [cirq.PhasedFSimGate(**parameters_bc.asdict()).on(b, c)],
        ]
    )

    engine_simulator = PhasedFSimEngineSimulator.create_from_characterizations_sqrt_iswap(
        characterizations=[
            cirq.google.PhasedFSimCalibrationResult(
                gate=cirq.FSimGate(np.pi / 4, 0.0),
                parameters={(a, b): parameters_ab, (c, d): parameters_cd},
                options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            ),
            cirq.google.PhasedFSimCalibrationResult(
                gate=cirq.FSimGate(np.pi / 4, 0.0),
                parameters={(b, c): parameters_bc},
                options=ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
            ),
        ]
    )

    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)

    assert cirq.allclose_up_to_global_phase(actual, expected)


def _create_sqrt_iswap_request(
    pairs: Iterable[Tuple[cirq.Qid, cirq.Qid]],
    options: FloquetPhasedFSimCalibrationOptions = ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
) -> cirq.google.FloquetPhasedFSimCalibrationRequest:
    return cirq.google.FloquetPhasedFSimCalibrationRequest(
        gate=cirq.FSimGate(np.pi / 4, 0.0), pairs=tuple(pairs), options=options
    )
