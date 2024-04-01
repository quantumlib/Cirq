# Copyright 2024 The Cirq Developers
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


from unittest.mock import patch
import pytest

import numpy as np

import cirq
from cirq.transformers.gauge_compiling import GaugeTransformer, GaugeSelector


class GaugeTester:

    two_qubit_gate: cirq.Gate
    gauge_transformer: GaugeTransformer

    @pytest.mark.parametrize('generation_seed', [*range(5)])
    @pytest.mark.parametrize('transformation_seed', [*range(5)])
    def test_gauge_transformer(self, generation_seed, transformation_seed):
        c = cirq.testing.random_circuit(
            qubits=3,
            n_moments=3,
            op_density=1,
            gate_domain={self.two_qubit_gate: 2, cirq.X: 1, cirq.Y: 1, cirq.H: 1, cirq.Z: 1},
            random_state=generation_seed,
        )
        nc = self.gauge_transformer(c, prng=np.random.default_rng(transformation_seed))
        cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
            nc, c, qubit_map={q: q for q in c.all_qubits()}
        )

    @patch('cirq.transformers.gauge_compiling.gauge_compiling._select', autospec=True)
    @pytest.mark.parametrize('seed', [*range(5)])
    def test_all_gauges(self, mock_select, seed):
        assert isinstance(
            self.gauge_transformer.gauge_selector, GaugeSelector
        ), 'When using a custom selector, please override this method to properly test all gauges'
        c = cirq.Circuit(self.two_qubit_gate(cirq.LineQubit(0), cirq.LineQubit(1)))
        prng = np.random.default_rng(seed)
        for gauge in self.gauge_transformer.gauge_selector.gauges:
            mock_select.return_value = gauge
            assert self.gauge_transformer.gauge_selector(prng) == gauge
            nc = self.gauge_transformer(c, prng=prng)
            try:
                _ = cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
                    nc, c, qubit_map={q: q for q in c.all_qubits()}
                )
            except AssertionError as ex:
                raise AssertionError(f"{gauge=} didn't result in an equivalent circuit") from ex
