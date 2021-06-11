import cirq
import numpy as np
import pytest


def test_matrix_mixture_from_mixture():
    q0 = cirq.LineQubit(0)
    dp = cirq.depolarize(0.1)
    cc = cirq.MatrixMixture.from_mixture(dp, key='dp')
    assert cirq.measurement_key(cc) == 'dp'

    circuit = cirq.Circuit(cc.on(q0))
    sim = cirq.Simulator(seed=0)

    results = sim.simulate(circuit)
    assert 'dp' in results.measurements
    # The depolarizing channel is composed of four unitaries.
    assert results.measurements['dp'] in range(4)


def test_matrix_mixture_equality():
    dp_pt1 = cirq.depolarize(0.1)
    dp_pt2 = cirq.depolarize(0.2)
    cc_a1 = cirq.MatrixMixture.from_mixture(dp_pt1, key='a')
    cc_a2 = cirq.MatrixMixture.from_mixture(dp_pt2, key='a')
    cc_b1 = cirq.MatrixMixture.from_mixture(dp_pt1, key='b')

    # Even if their effect is the same, MatrixMixtures are not treated as equal
    # to other channels defined in Cirq.
    assert cc_a1 != dp_pt1
    assert cc_a1 != cc_a2
    assert cc_a1 != cc_b1
    assert cc_a2 != cc_b1

    mix = [
        (0.5, np.array([[1, 0], [0, 1]])),
        (0.5, np.array([[0, 1], [1, 0]])),
    ]
    half_flip = cirq.MatrixMixture(mix)
    mix_inv = list(reversed(mix))
    half_flip_inv = cirq.MatrixMixture(mix_inv)
    # Even though these have the same effect on the circuit, their measurement
    # behavior differs, so they are considered non-equal.
    assert half_flip != half_flip_inv


def test_matrix_mixture_remap_keys():
    dp = cirq.depolarize(0.1)
    cc = cirq.MatrixMixture.from_mixture(dp)
    assert cirq.with_measurement_key_mapping(cc, {'a': 'b'}) is NotImplemented

    cc_x = cirq.MatrixMixture.from_mixture(dp, key='x')
    assert cirq.with_measurement_key_mapping(cc_x, {'a': 'b'}) is cc_x

    cc_a = cirq.MatrixMixture.from_mixture(dp, key='a')
    cc_b = cirq.MatrixMixture.from_mixture(dp, key='b')
    assert cc_a != cc_b
    assert cirq.with_measurement_key_mapping(cc_a, {'a': 'b'}) == cc_b


def test_matrix_mixture_from_unitaries():
    q0 = cirq.LineQubit(0)
    mix = [
        (0.5, np.array([[1, 0], [0, 1]])),
        (0.5, np.array([[0, 1], [1, 0]])),
    ]
    half_flip = cirq.MatrixMixture(mix, key='flip')
    assert cirq.measurement_key(half_flip) == 'flip'

    circuit = cirq.Circuit(half_flip.on(q0), cirq.measure(q0, key='m'))
    sim = cirq.Simulator(seed=0)

    results = sim.simulate(circuit)
    assert 'flip' in results.measurements
    assert results.measurements['flip'] == results.measurements['m']


def test_mix_mismatch_fails():
    op2 = np.zeros((4, 4))
    op2[1][1] = 1
    mix = [
        (0.5, np.array([[1, 0], [0, 0]])),
        (0.5, op2),
    ]

    with pytest.raises(ValueError, match='Inconsistent unitary sizes'):
        _ = cirq.MatrixMixture(mixture=mix, key='m')


def test_nonqubit_mixture_fails():
    mix = [
        (0.5, np.array([[1, 0, 0], [0, 1, 0]])),
        (0.5, np.array([[0, 1, 0], [1, 0, 0]])),
    ]

    with pytest.raises(ValueError, match='Input mixture'):
        _ = cirq.MatrixMixture(mixture=mix, key='m')
