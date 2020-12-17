from typing import Iterable, Tuple, cast

from cirq.google.calibration.engine_simulator import (
    PhasedFSimEngineSimulator,
    SQRT_ISWAP_PARAMETERS
)
import cirq
import collections
import numpy as np


def test_floquet_get_calibrations() -> None:

    parameters_ab = cirq.google.PhasedFSimParameters(
        theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2
    )
    parameters_bc = cirq.google.PhasedFSimParameters(
        theta=0.8, zeta=-0.5, chi=-0.4, gamma=-0.3, phi=-0.2
    )
    parameters_cd_dict = {
        'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5
    }
    parameters_cd = cirq.google.PhasedFSimParameters(**parameters_cd_dict)

    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={
            (a, b): parameters_ab,
            (b, c): parameters_bc,
            (c, d): parameters_cd_dict
        }
    )

    requests = [
        _create_sqrt_iswap_request([(a, b), (c, d)]),
        _create_sqrt_iswap_request([(b, c)])
    ]

    results = engine_simulator.get_calibrations(requests)

    assert results == [
        cirq.google.PhasedFSimCalibrationResult(
            gate=cirq.FSimGate(np.pi / 4, 0.0),
            gate_set=cirq.google.SQRT_ISWAP_GATESET,
            parameters={
                (a, b): parameters_ab,
                (c, d): parameters_cd
            }
        ),
        cirq.google.PhasedFSimCalibrationResult(
            gate=cirq.FSimGate(np.pi / 4, 0.0),
            gate_set=cirq.google.SQRT_ISWAP_GATESET,
            parameters={
                (b, c): parameters_bc
            }
        )
    ]

# TODO: Test get_calibrations exceptions and options (missing angles, etc.)


def test_ideal_sqrt_iswap_simulates_correctly() -> None:
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([
        [cirq.X(a), cirq.Y(c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)]
    ])

    engine_simulator = PhasedFSimEngineSimulator.create_with_ideal_sqrt_iswap(cirq.Simulator())

    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(circuit)

    assert cirq.allclose_up_to_global_phase(actual, expected)


def test_with_random_gaussian_sqrt_iswap_simulates_correctly() -> None:
    engine_simulator = PhasedFSimEngineSimulator.create_with_random_gaussian_sqrt_iswap(
        mean=SQRT_ISWAP_PARAMETERS
    )

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([
        [cirq.X(a), cirq.Y(c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)]
    ])

    calibrations = engine_simulator.get_calibrations([
        _create_sqrt_iswap_request([(a, b), (c, d)]),
        _create_sqrt_iswap_request([(b, c)])
    ])
    parameters = collections.ChainMap(*(calibration.parameters for calibration in calibrations))

    expected_circuit = cirq.Circuit([
        [cirq.X(a), cirq.X(c)],
        [cirq.PhasedFSimGate(**parameters[(a, b)].asdict()).on(a, b),
         cirq.PhasedFSimGate(**parameters[(c, d)].asdict()).on(c, d)],
        [cirq.PhasedFSimGate(**parameters[(b, c)].asdict()).on(b, c)]
    ])

    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)

    assert cirq.allclose_up_to_global_phase(actual, expected)


def test_from_dictionary_sqrt_iswap_simulates_correctly() -> None:
    parameters_ab = cirq.google.PhasedFSimParameters(
        theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2
    )
    parameters_bc = cirq.google.PhasedFSimParameters(
        theta=0.8, zeta=-0.5, chi=-0.4, gamma=-0.3, phi=-0.2
    )
    parameters_cd_dict = {
        'theta': 0.1, 'zeta': 0.2, 'chi': 0.3, 'gamma': 0.4, 'phi': 0.5
    }

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([
        [cirq.X(a), cirq.Y(c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)]
    ])
    expected_circuit = cirq.Circuit([
        [cirq.X(a), cirq.X(c)],
        [cirq.PhasedFSimGate(**parameters_ab.asdict()).on(a, b),
         cirq.PhasedFSimGate(**parameters_cd_dict).on(c, d)],
        [cirq.PhasedFSimGate(**parameters_bc.asdict()).on(b, c)]
    ])

    engine_simulator = PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={
            (a, b): parameters_ab,
            (b, c): parameters_bc,
            (c, d): parameters_cd_dict
        }
    )

    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)

    assert cirq.allclose_up_to_global_phase(actual, expected)

# TODO: Test create_from_dictionary_sqrt_iswap ideal_when_missing_gate and
#  ideal_when_missing_parameter


def test_from_characterizations_sqrt_iswap_simulates_correctly() -> None:
    parameters_ab = cirq.google.PhasedFSimParameters(
        theta=0.6, zeta=0.5, chi=0.4, gamma=0.3, phi=0.2
    )
    parameters_bc = cirq.google.PhasedFSimParameters(
        theta=0.8, zeta=-0.5, chi=-0.4, gamma=-0.3, phi=-0.2
    )
    parameters_cd = cirq.google.PhasedFSimParameters(
        theta=0.1, zeta=0.2, chi=0.3, gamma=0.4, phi=0.5
    )

    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([
        [cirq.X(a), cirq.Y(c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)]
    ])
    expected_circuit = cirq.Circuit([
        [cirq.X(a), cirq.X(c)],
        [cirq.PhasedFSimGate(**parameters_ab.asdict()).on(a, b),
         cirq.PhasedFSimGate(**parameters_cd.asdict()).on(c, d)],
        [cirq.PhasedFSimGate(**parameters_bc.asdict()).on(b, c)]
    ])

    engine_simulator = PhasedFSimEngineSimulator.create_from_characterizations_sqrt_iswap(
        characterizations=[
            cirq.google.PhasedFSimCalibrationResult(
                gate=cirq.FSimGate(np.pi / 4, 0.0),
                gate_set=cirq.google.SQRT_ISWAP_GATESET,
                parameters={
                    (a, b): parameters_ab,
                    (c, d): parameters_cd
                }
            ),
            cirq.google.PhasedFSimCalibrationResult(
                gate=cirq.FSimGate(np.pi / 4, 0.0),
                gate_set=cirq.google.SQRT_ISWAP_GATESET,
                parameters={
                    (b, c): parameters_bc
                }
            )
        ]
    )

    actual = engine_simulator.final_state_vector(circuit)
    expected = cirq.final_state_vector(expected_circuit)

    assert cirq.allclose_up_to_global_phase(actual, expected)


def _create_sqrt_iswap_request(
        pairs: Iterable[Tuple[cirq.Qid, cirq.Qid]],
        options: cirq.google.FloquetPhasedFSimCalibrationOptions =
            cirq.google.FloquetPhasedFSimCalibrationOptions.all_options()
) -> cirq.google.FloquetPhasedFSimCalibrationRequest:
    return cirq.google.FloquetPhasedFSimCalibrationRequest(
        gate=cirq.FSimGate(np.pi / 4, 0.0),
        gate_set=cirq.google.SQRT_ISWAP_GATESET,
        pairs=tuple(pairs),
        options=options
    )