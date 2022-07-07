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

import numpy as np
import pytest
import cirq

from cirq.experiments import CrossEntropyResult, CrossEntropyResultDict
from cirq.experiments.cross_entropy_benchmarking import CrossEntropyPair, SpecklePurityPair

_DEPRECATION_MESSAGE = 'Use cirq.experiments.xeb_fitting.XEBCharacterizationResult instead'
_DEPRECATION_RANDOM_CIRCUIT = 'Use cirq.experiments.random_quantum_circuit_generation instead'


def test_cross_entropy_result_depolarizing_models():
    with cirq.testing.assert_deprecated(_DEPRECATION_MESSAGE, deadline='v0.16'):
        prng = np.random.RandomState(59566)
        S = 0.8
        p = 0.99
        data = [
            CrossEntropyPair(num_cycle=d, xeb_fidelity=S * p**d + prng.normal(scale=0.01))
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
        np.testing.assert_allclose(purity_model.purity, p**2, atol=1e-2)


def test_cross_entropy_result_repr():
    with cirq.testing.assert_deprecated(_DEPRECATION_MESSAGE, deadline='v0.16'):
        result1 = CrossEntropyResult(
            data=[CrossEntropyPair(2, 0.9), CrossEntropyPair(5, 0.5)], repetitions=1000
        )
    with cirq.testing.assert_deprecated(_DEPRECATION_MESSAGE, deadline='v0.16'):
        result2 = CrossEntropyResult(
            data=[CrossEntropyPair(2, 0.9), CrossEntropyPair(5, 0.5)],
            repetitions=1000,
            purity_data=[SpecklePurityPair(2, 0.8), SpecklePurityPair(5, 0.3)],
        )
    with cirq.testing.assert_deprecated(_DEPRECATION_MESSAGE, deadline='v0.16', count=6):
        cirq.testing.assert_equivalent_repr(result1)
        cirq.testing.assert_equivalent_repr(result2)


def test_cross_entropy_result_dict_repr():
    pair = tuple(cirq.LineQubit.range(2))
    with cirq.testing.assert_deprecated(_DEPRECATION_MESSAGE, deadline='v0.16'):
        result = CrossEntropyResult(
            data=[CrossEntropyPair(2, 0.9), CrossEntropyPair(5, 0.5)], repetitions=1000
        )
    with cirq.testing.assert_deprecated(_DEPRECATION_MESSAGE, deadline='v0.16'):
        result_dict = CrossEntropyResultDict(results={pair: result})
        assert len(result_dict) == 1
    with cirq.testing.assert_deprecated(_DEPRECATION_MESSAGE, deadline='v0.16', count=6):
        cirq.testing.assert_equivalent_repr(result_dict)


def test_cross_entropy_result_purity_model_fails_with_no_data():
    with cirq.testing.assert_deprecated(_DEPRECATION_MESSAGE, deadline='v0.16'):
        data = [
            CrossEntropyPair(num_cycle=2, xeb_fidelity=0.9),
            CrossEntropyPair(num_cycle=4, xeb_fidelity=0.8),
        ]
        result = CrossEntropyResult(data=data, repetitions=1000)
        with pytest.raises(ValueError):
            _ = result.purity_depolarizing_model()
