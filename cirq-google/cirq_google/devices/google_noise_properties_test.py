from typing import Dict, List, Tuple
import numpy as np
import cirq, cirq_google

# from cirq.testing import assert_equivalent_op_tree
from cirq.devices.noise_properties import (
    SYMMETRIC_TWO_QUBIT_GATES,
    SINGLE_QUBIT_GATES,
    TWO_QUBIT_GATES,
)
from cirq.devices.noise_utils import OpIdentifier

from cirq_google.devices.google_noise_properties import (
    GoogleNoiseProperties,
    NoiseModelFromGoogleNoiseProperties,
)


DEFAULT_GATE_NS: Dict[type, float] = {
    cirq.ZPowGate: 25.0,
    cirq.MeasurementGate: 4000.0,
    cirq.ResetChannel: 250.0,
    cirq.PhasedXZGate: 25.0,
    cirq.FSimGate: 32.0,
    cirq.ISwapPowGate: 32.0,
    cirq.CZPowGate: 32.0,
    # cirq.WaitGate is a special case.
}


# These properties are for testing purposes only - they are not representative
# of device behavior for any existing hardware.
def sample_noise_properties(
    system_qubits: List[cirq.Qid], qubit_pairs: List[Tuple[cirq.Qid, cirq.Qid]]
):
    return GoogleNoiseProperties(
        gate_times_ns=DEFAULT_GATE_NS,
        T1_ns={q: 1e5 for q in system_qubits},
        Tphi_ns={q: 2e5 for q in system_qubits},
        ro_fidelities={q: np.array([0.001, 0.01]) for q in system_qubits},
        gate_pauli_errors={
            **{OpIdentifier(g, q): 0.001 for g in SINGLE_QUBIT_GATES for q in system_qubits},
            **{OpIdentifier(g, q0, q1): 0.01 for g in TWO_QUBIT_GATES for q0, q1 in qubit_pairs},
        },
        fsim_errors={
            OpIdentifier(g, q0, q1): cirq.PhasedFSimGate(0.01, 0.03, 0.04, 0.05, 0.02)
            for g in SYMMETRIC_TWO_QUBIT_GATES
            for q0, q1 in qubit_pairs
        },
    )


def test_model_from_props():
    system_qubits = cirq.GridQubit.rect(2, 2)
    qubit_pairs = [
        (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
        (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)),
        (cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)),
        (cirq.GridQubit(0, 1), cirq.GridQubit(1, 1)),
        (cirq.GridQubit(1, 1), cirq.GridQubit(0, 1)),
        (cirq.GridQubit(1, 1), cirq.GridQubit(1, 0)),
        (cirq.GridQubit(1, 0), cirq.GridQubit(0, 0)),
        (cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)),
    ]
    props = sample_noise_properties(system_qubits, qubit_pairs)
    model = NoiseModelFromGoogleNoiseProperties(props)

    circuit = cirq.Circuit(
        cirq.H(system_qubits[0]),
        cirq.H(system_qubits[2]),
        cirq.CNOT(*system_qubits[0:2]),
        cirq.CNOT(*system_qubits[2:4]),
        cirq.measure(*system_qubits, key='m'),
    )
    print('circuit')
    print(circuit)
    syc_circuit = cirq_google.optimized_for_sycamore(circuit)
    print('syc_circuit')
    print(syc_circuit)
    noisy_circuit = syc_circuit.with_noise(model)
    print('noisy_circuit')
    print(noisy_circuit.moments)
    choi = cirq.kraus_to_choi(cirq.kraus(noisy_circuit.moments[4].operations[0]))
    print(choi)

    assert False


# TODO: all the other tests
