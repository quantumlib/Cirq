import cirq

import numpy as np


def test_produces_samples():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.H(a),
        cirq.CNOT(a, b),
        cirq.measure(a, key='a'),
        cirq.measure(b, key='b'),
    )

    result = cirq.StabilizerSampler().sample(c, repetitions=100)
    assert 5 < sum(result['a']) < 95
    assert np.all(result['a'] ^ result['b'] == 0)
