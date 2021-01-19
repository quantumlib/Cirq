import cirq
import cirq.google.calibration.workflow as workflow
import numpy as np

SQRT_ISWAP_GATE = cirq.FSimGate(np.pi / 4, 0.0)


def test_floquet_characterization_for_circuit() -> None:
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([
        [cirq.X(a), cirq.Y(c)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(a, b), cirq.FSimGate(np.pi / 4, 0.0).on(c, d)],
        [cirq.FSimGate(np.pi / 4, 0.0).on(b, c)]
    ])
    options = cirq.google.FloquetPhasedFSimCalibrationOptions.all_except_for_chi_options()

    requests, mapping =  workflow.floquet_characterization_for_circuit(circuit, options=options)

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

    requests, mapping =  workflow.floquet_characterization_for_circuit(circuit, options=options)

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

    requests, mapping =  workflow.floquet_characterization_for_circuit(circuit, options=options)

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
