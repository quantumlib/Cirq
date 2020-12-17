import cirq
import cirq.google.calibration.workflow as workflow
import itertools
import numpy as np
import pytest


SQRT_ISWAP_PARAMETERS = cirq.google.PhasedFSimParameters(
    theta=np.pi / 4,
    zeta=0.0,
    chi=0.0,
    gamma=0.0,
    phi=0.0
)


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
        cirq.google.PhasedFSimParameters(
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
    parameters_ab = cirq.google.PhasedFSimParameters(
        zeta=0.5, chi=0.4, gamma=0.3
    )
    parameters_bc = cirq.google.PhasedFSimParameters(
        zeta=-0.5, chi=-0.4, gamma=-0.3
    )
    parameters_cd = cirq.google.PhasedFSimParameters(
        zeta=0.2, chi=0.3, gamma=0.4
    )

    a, b, c, d = cirq.LineQubit.range(4)
    simulator = cirq.Simulator()
    engine_simulator = cirq.google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        simulator,
        parameters={
            (a, b): parameters_ab.other_when_none(SQRT_ISWAP_PARAMETERS),
            (b, c): parameters_bc.other_when_none(SQRT_ISWAP_PARAMETERS),
            (c, d): parameters_cd.other_when_none(SQRT_ISWAP_PARAMETERS)
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
        handler_name=None,
        gate_set=cirq.google.SQRT_ISWAP_GATESET,
        options=cirq.google.FloquetPhasedFSimCalibrationOptions(
            estimate_theta=False,
            estimate_zeta=True,
            estimate_chi=True,
            estimate_gamma=True,
            estimate_phi=False
        )
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated), cirq.final_state_vector(circuit))
    assert calibrations == [
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
    assert mapping == [None, None, 0, None, None, 1, None]
    assert calibrated_parameters == cirq.google.PhasedFSimParameters(
        zeta=0.0,
        chi=0.0,
        gamma=0.0
    )

# TODO: Check if calibration preserves moments.


def test_run_floquet_calibration_no_chi() -> None:
    parameters_ab = cirq.google.PhasedFSimParameters(
        theta=np.pi / 4, zeta=0.5, gamma=0.3
    )
    parameters_bc = cirq.google.PhasedFSimParameters(
        theta=np.pi / 4, zeta=-0.5, gamma=-0.3
    )
    parameters_cd = cirq.google.PhasedFSimParameters(
        theta=np.pi / 4, zeta=0.2, gamma=0.4
    )

    a, b, c, d = cirq.LineQubit.range(4)
    simulator = cirq.Simulator()
    engine_simulator = cirq.google.PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
        simulator,
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
        handler_name=None,
        gate_set=cirq.google.SQRT_ISWAP_GATESET
    )

    assert cirq.allclose_up_to_global_phase(
        engine_simulator.final_state_vector(calibrated), cirq.final_state_vector(circuit))
