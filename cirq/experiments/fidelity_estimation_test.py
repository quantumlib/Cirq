from typing import Sequence

import numpy as np
import pytest

import cirq

MEASUREMENT_KEY = 'm'


def sample_noisy_bitstrings(circuit: cirq.Circuit, depolarization: float,
                            n_samples: int) -> np.ndarray:
    assert 0 <= depolarization <= 1
    dim = np.product(circuit.qid_shape())
    n_incoherent = int(depolarization * n_samples)
    n_coherent = n_samples - n_incoherent
    incoherent_samples = np.random.randint(dim, size=n_incoherent)
    if n_coherent > 0:
        sim = cirq.Simulator()
        r = sim.run(circuit, repetitions=n_coherent)
        coherent_samples = r.data[MEASUREMENT_KEY].to_numpy()
        return np.concatenate((coherent_samples, incoherent_samples))
    return incoherent_samples


@pytest.mark.parametrize('depolarization', (0, 0.25, 0.5, 0.75, 1.0))
def test_compute_linear_xeb_fidelity(depolarization):
    prng_state = np.random.get_state()
    np.random.seed(0)

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1),
                                    cirq.measure(q0, q1, key=MEASUREMENT_KEY))
    bitstrings = sample_noisy_bitstrings(circuit, depolarization, 10000)
    f = cirq.compute_linear_xeb_fidelity(circuit, bitstrings, (q0, q1))
    assert np.isclose(f, 1 - depolarization, atol=0.026)

    np.random.set_state(prng_state)


def test_compute_linear_xeb_fidelity_invalid_qubits():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1),
                                    cirq.measure(q0, q1, key=MEASUREMENT_KEY))
    bitstrings = sample_noisy_bitstrings(circuit, 0.9, 10)
    with pytest.raises(ValueError):
        cirq.compute_linear_xeb_fidelity(circuit, bitstrings, (q0, q2))


def test_compute_linear_xeb_fidelity_invalid_bitstrings():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit.from_ops(cirq.H(q0), cirq.CNOT(q0, q1),
                                    cirq.measure(q0, q1, key=MEASUREMENT_KEY))
    bitstrings = [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        cirq.compute_linear_xeb_fidelity(circuit, bitstrings, (q0, q1))
