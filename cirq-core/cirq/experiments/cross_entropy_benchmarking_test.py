# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np
import pytest
import cirq

from cirq.experiments import (
    CrossEntropyResult,
    CrossEntropyResultDict,
    cross_entropy_benchmarking,
    build_entangling_layers,
)
from cirq.experiments.cross_entropy_benchmarking import CrossEntropyPair, SpecklePurityPair


@pytest.mark.usefixtures('closefigures')
def test_cross_entropy_benchmarking():
    # Check that the fidelities returned from a four-qubit XEB simulation are
    # close to 1 (deviations from 1 is expected due to finite number of
    # measurements).
    simulator = cirq.Simulator()
    qubits = cirq.GridQubit.square(2)

    # Sanity check single-qubit-gate causes error
    with pytest.raises(ValueError):
        build_entangling_layers(qubits, cirq.Z ** 0.91)

    # Build a sequence of CZ gates.
    interleaved_ops = build_entangling_layers(qubits, cirq.CZ ** 0.91)

    # Specify a set of single-qubit rotations. Pick prime numbers for the
    # exponent to avoid evolving the system into a basis state.
    single_qubit_rots = [
        [cirq.X ** 0.37],
        [cirq.Y ** 0.73, cirq.X ** 0.53],
        [cirq.Z ** 0.61, cirq.X ** 0.43],
        [cirq.Y ** 0.19],
    ]

    # Simulate XEB using the default single-qubit gate set without two-qubit
    # gates, XEB using the specified single-qubit gate set without two-qubit
    # gates, and XEB using the specified single-qubit gate set with two-qubit
    # gate. Check that the fidelities are close to 1.0 in all cases. Also,
    # check that a single XEB fidelity is returned if a single cycle number
    # is specified.
    results_0 = cross_entropy_benchmarking(
        simulator, qubits, num_circuits=3, repetitions=1000, cycles=range(4, 20, 5)
    )
    results_1 = cross_entropy_benchmarking(
        simulator,
        qubits,
        num_circuits=3,
        repetitions=1000,
        cycles=[4, 8, 12],
        scrambling_gates_per_cycle=single_qubit_rots,
    )
    results_2 = cross_entropy_benchmarking(
        simulator,
        qubits,
        benchmark_ops=interleaved_ops,
        num_circuits=3,
        repetitions=1000,
        cycles=[4, 8, 12],
        scrambling_gates_per_cycle=single_qubit_rots,
    )
    results_3 = cross_entropy_benchmarking(
        simulator,
        qubits,
        benchmark_ops=interleaved_ops,
        num_circuits=3,
        repetitions=1000,
        cycles=15,
        scrambling_gates_per_cycle=single_qubit_rots,
    )
    fidelities_0 = [datum.xeb_fidelity for datum in results_0.data]
    fidelities_1 = [datum.xeb_fidelity for datum in results_1.data]
    fidelities_2 = [datum.xeb_fidelity for datum in results_2.data]
    fidelities_3 = [datum.xeb_fidelity for datum in results_3.data]
    assert np.isclose(np.mean(fidelities_0), 1.0, atol=0.1)
    assert np.isclose(np.mean(fidelities_1), 1.0, atol=0.1)
    assert np.isclose(np.mean(fidelities_2), 1.0, atol=0.1)
    assert len(fidelities_3) == 1

    # Sanity test that plot runs.
    ax = plt.subplot()
    results_1.plot(ax)


def test_cross_entropy_result_depolarizing_models():
    prng = np.random.RandomState(59566)
    S = 0.8
    p = 0.99
    data = [
        CrossEntropyPair(num_cycle=d, xeb_fidelity=S * p ** d + prng.normal(scale=0.01))
        for d in range(10, 211, 20)
    ]
    purity_data = [
        SpecklePurityPair(num_cycle=d, purity=S * p ** (2 * d) + prng.normal(scale=0.01))
        for d in range(10, 211, 20)
    ]
    result = CrossEntropyResult(data=data, repetitions=1000, purity_data=purity_data)
    model = result.depolarizing_model()
    purity_model = result.purity_depolarizing_model()
    np.testing.assert_allclose(model.spam_depolarization, S, atol=1e-2)
    np.testing.assert_allclose(model.cycle_depolarization, p, atol=1e-2)
    np.testing.assert_allclose(purity_model.purity, p ** 2, atol=1e-2)


def test_cross_entropy_result_repr():
    result1 = CrossEntropyResult(
        data=[CrossEntropyPair(2, 0.9), CrossEntropyPair(5, 0.5)], repetitions=1000
    )
    result2 = CrossEntropyResult(
        data=[CrossEntropyPair(2, 0.9), CrossEntropyPair(5, 0.5)],
        repetitions=1000,
        purity_data=[SpecklePurityPair(2, 0.8), SpecklePurityPair(5, 0.3)],
    )
    cirq.testing.assert_equivalent_repr(result1)
    cirq.testing.assert_equivalent_repr(result2)


def test_cross_entropy_result_dict_repr():
    pair = tuple(cirq.LineQubit.range(2))
    result = CrossEntropyResult(
        data=[CrossEntropyPair(2, 0.9), CrossEntropyPair(5, 0.5)], repetitions=1000
    )
    result_dict = CrossEntropyResultDict(results={pair: result})
    cirq.testing.assert_equivalent_repr(result_dict)


def test_cross_entropy_result_purity_model_fails_with_no_data():
    data = [
        CrossEntropyPair(num_cycle=2, xeb_fidelity=0.9),
        CrossEntropyPair(num_cycle=4, xeb_fidelity=0.8),
    ]
    result = CrossEntropyResult(data=data, repetitions=1000)
    with pytest.raises(ValueError):
        _ = result.purity_depolarizing_model()
