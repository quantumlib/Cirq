from typing import Sequence

import numpy as np
import pytest

import cirq


def make_bitstrings(samples: np.ndarray) -> Sequence[int]:
    assert samples.shape[1] == 2
    return [2 * b1 + b0 for b1, b0 in samples]


def sample_noisy_bitstrings(circuit: cirq.Circuit, depolarization: float,
                            n_samples: int) -> Sequence[int]:
    assert 0 <= depolarization <= 1
    n_incoherent = int(depolarization * n_samples)
    n_coherent = n_samples - n_incoherent
    incoherent_samples = np.random.randint(2, size=(n_incoherent, 2))
    if n_coherent > 0:
        sim = cirq.Simulator()
        r = sim.run(circuit, repetitions=n_coherent)
        coherent_samples = r.measurements['0,1']
        all_samples = np.concatenate((coherent_samples, incoherent_samples))
        return make_bitstrings(all_samples)
    return make_bitstrings(incoherent_samples)


@pytest.mark.parametrize('depolarization', (0, 0.25, 0.5, 0.75, 1.0))
def test_estimate_circuit_fidelity(depolarization):
    prng_state = np.random.get_state()
    np.random.seed(0)

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1),
                                    cirq.measure(q0, q1))
    bitstrings = sample_noisy_bitstrings(circuit, depolarization, 10000)
    f = cirq.estimate_circuit_fidelity(circuit, (q0, q1), bitstrings)
    assert np.isclose(f, 1 - depolarization, atol=0.026)

    np.random.set_state(prng_state)


def test_estimate_circuit_fidelity_invalid_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1),
                                    cirq.measure(q0, q1))
    bitstrings = sample_noisy_bitstrings(circuit, 0.9, 10)
    with pytest.raises(ValueError):
        cirq.estimate_circuit_fidelity(circuit, (q0, q1, q2), bitstrings)


def test_estimate_circuit_fidelity_invalid_bitstrings():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1),
                                    cirq.measure(q0, q1))
    bitstrings = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        cirq.estimate_circuit_fidelity(circuit, (q0, q1), bitstrings)
