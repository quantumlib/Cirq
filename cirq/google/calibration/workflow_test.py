import cirq
import cirq.google.calibration.workflow as workflow
import itertools
import numpy as np
import pytest


SQRT_ISWAP_PARAMETERS = cirq.google.PhasedFSimCharacterization(
    theta=np.pi / 4,
    zeta=0.0,
    chi=0.0,
    gamma=0.0,
    phi=0.0
)
SQRT_ISWAP_GATE = cirq.FSimGate(np.pi / 4, 0.0)


def test_floquet_characterization_for_circuit() -> None:
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([
        [cirq.X(a), cirq.Y(c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)]
    ])
    options = cirq.google.FloquetPhasedFSimCalibrationOptions.all_except_for_chi_options()

    requests, mapping =  workflow.floquet_characterization_for_circuit(
        circuit, cirq.google.SQRT_ISWAP_GATESET, options=options)

    assert requests == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)),
            gate=SQRT_ISWAP_GATE,
            options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c),),
            gate=SQRT_ISWAP_GATE,
            options=options
        )
    ]
    assert mapping == [None, 0, 1]


def test_floquet_characterization_for_circuit_merges_sub_sets() -> None:
    a, b, c, d, e = cirq.LineQubit.range(5)
    circuit = cirq.Circuit([
        [cirq.X(a), cirq.Y(c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b)]
    ])
    circuit += cirq.Moment(
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c), cirq.FSimGate(np.pi / 4, 0.0).on(d, e)])
    options = cirq.google.FloquetPhasedFSimCalibrationOptions.all_except_for_chi_options()

    requests, mapping =  workflow.floquet_characterization_for_circuit(
        circuit, cirq.google.SQRT_ISWAP_GATESET, options=options)

    assert requests == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)),
            gate=SQRT_ISWAP_GATE,
            options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((b, c), (d, e)),
            gate=SQRT_ISWAP_GATE,
            options=options
        )
    ]
    assert mapping == [None, 0, 1, 0, 1]


def test_floquet_characterization_for_circuit_merges_compatible_sets() -> None:
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    circuit = cirq.Circuit([cirq.X(a), cirq.Y(c)])
    circuit += cirq.Moment([cirq.FSimGate(np.pi / 4, 0.0).on(a, b)])
    circuit += cirq.Moment([cirq.FSimGate(np.pi / 4, 0.0).on(b, c),
                            cirq.FSimGate(np.pi / 4, 0.0).on(d, e)])
    circuit += cirq.Moment([cirq.FSimGate(np.pi / 4, 0.0).on(c, d)])
    circuit += cirq.Moment([cirq.FSimGate(np.pi / 4, 0.0).on(a, f),
                            cirq.FSimGate(np.pi / 4, 0.0).on(d, e)])
    options = cirq.google.FloquetPhasedFSimCalibrationOptions.all_except_for_chi_options()

    requests, mapping =  workflow.floquet_characterization_for_circuit(
        circuit, cirq.google.SQRT_ISWAP_GATESET, options=options)

    assert requests == [
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, b), (c, d)),
            gate=SQRT_ISWAP_GATE,
            options=options
        ),
        cirq.google.calibration.FloquetPhasedFSimCalibrationRequest(
            pairs=((a, f), (b, c), (d, e)),
            gate=SQRT_ISWAP_GATE,
            options=options
        )
    ]
    assert mapping == [None, 0, 1, 0, 1]


@pytest.mark.parametrize(
    'theta,zeta,chi,gamma,phi',
    itertools.product(
        [0.1, 0.7],
        [-0.3, 0.1, 0.5],
        [-0.3, 0.2, 0.4],
        [-0.6, 0.1, 0.6],
        [0.2, 0.6]
    )
)
def test_create_corrected_fsim_gate(
        theta: float, zeta: float, chi: float, gamma: float, phi: float
) -> None:
    a, b = cirq.LineQubit.range(2)

    expected_gate = cirq.PhasedFSimGate(
        theta=theta,
        zeta=-zeta,
        chi=-chi,
        gamma=-gamma,
        phi=phi
    )
    expected = cirq.unitary(expected_gate)

    corrected_gate, corrected_mapping = workflow.create_corrected_fsim_gate(
        (a, b),
        cirq.FSimGate(theta=theta, phi=phi),
        cirq.google.PhasedFSimCharacterization(
            theta=theta,
            zeta=zeta,
            chi=chi,
            gamma=gamma,
            phi=phi
        ),
        5
    )
    actual = cirq.unitary(cirq.Circuit(corrected_gate))

    assert cirq.equal_up_to_global_phase(actual, expected)
    assert corrected_mapping == [None, 5, None]


def test_run_floquet_calibration() -> None:
    parameters_ab = cirq.google.PhasedFSimCharacterization(
        zeta=0.5, chi=0.4, gamma=0.3
    )
    parameters_bc = cirq.google.PhasedFSimCharacterization(
        zeta=-0.5, chi=-0.4, gamma=-0.3
    )
    parameters_cd = cirq.google.PhasedFSimCharacterization(
        zeta=0.2, chi=0.3, gamma=0.4
    )

    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq.google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={
            (a, b): parameters_ab.merge_with(SQRT_ISWAP_PARAMETERS),
            (b, c): parameters_bc.merge_with(SQRT_ISWAP_PARAMETERS),
            (c, d): parameters_cd.merge_with(SQRT_ISWAP_PARAMETERS)
        }
    )

    circuit = cirq.Circuit([
        [cirq.X(a), cirq.Y(c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)]
    ])

    (calibrated, calibrations, mapping, calibrated_parameters
     ) = workflow.run_floquet_phased_calibration_for_circuit(
        circuit,
        engine_simulator,
        processor_id=None,
        gate_set=cirq.google.SQRT_ISWAP_GATESET,
        options=cirq.google.FloquetPhasedFSimCalibrationOptions(
            characterize_theta=False,
            characterize_zeta=True,
            characterize_chi=True,
            characterize_gamma=True,
            characterize_phi=False
        )
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated), cirq.final_state_vector(circuit))
    assert calibrations == [
        cirq.google.PhasedFSimCalibrationResult(
            gate=cirq.FSimGate(np.pi / 4, 0.0),
            parameters={
                (a, b): parameters_ab,
                (c, d): parameters_cd
            }
        ),
        cirq.google.PhasedFSimCalibrationResult(
            gate=cirq.FSimGate(np.pi / 4, 0.0),
            parameters={
                (b, c): parameters_bc
            }
        )
    ]
    assert mapping == [None, None, 0, None, None, 1, None]
    assert calibrated_parameters == cirq.google.PhasedFSimCharacterization(
        zeta=0.0,
        chi=0.0,
        gamma=0.0
    )

# TODO: Check if calibration preserves moments.


def test_run_floquet_calibration_no_chi() -> None:
    parameters_ab = cirq.google.PhasedFSimCharacterization(
        theta=np.pi / 4, zeta=0.5, gamma=0.3
    )
    parameters_bc = cirq.google.PhasedFSimCharacterization(
        theta=np.pi / 4, zeta=-0.5, gamma=-0.3
    )
    parameters_cd = cirq.google.PhasedFSimCharacterization(
        theta=np.pi / 4, zeta=0.2, gamma=0.4
    )

    a, b, c, d = cirq.LineQubit.range(4)
    engine_simulator = cirq.google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        parameters={
            (a, b): parameters_ab,
            (b, c): parameters_bc,
            (c, d): parameters_cd
        },
        ideal_when_missing_parameter=True
    )

    circuit = cirq.Circuit([
        [cirq.X(a), cirq.Y(c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)]
    ])

    calibrated, *_ = workflow.run_floquet_phased_calibration_for_circuit(
        circuit,
        engine_simulator,
        processor_id=None,
        gate_set=cirq.google.SQRT_ISWAP_GATESET
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated), cirq.final_state_vector(circuit))
