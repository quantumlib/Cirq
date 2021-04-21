# Copyright 2021 The Cirq Developers
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
from typing import List

import numpy as np

import cirq
from cirq.sim import act_on_args


def test_measurements():
    class DummyArgs(cirq.ActOnArgs):
        def _perform_measurement(self) -> List[int]:
            return [5, 3]

    args = DummyArgs(axes=[], prng=np.random.RandomState(), log_of_measurement_results={})
    args.measure("test", [1])
    assert args.log_of_measurement_results["test"] == [5]


def test_decompose():
    class DummyArgs(cirq.ActOnArgs):
        def _act_on_fallback_(self, action, allow_decompose):
            return True

    class Composite(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _decompose_(self, qubits):
            yield cirq.X(*qubits)

    args = DummyArgs(axes=[0], prng=np.random.RandomState(), log_of_measurement_results={})
    assert act_on_args.strat_act_on_from_apply_decompose(Composite(), args)
