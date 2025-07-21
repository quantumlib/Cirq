# Copyright 2025 The Cirq Developers
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


import pathlib

import cirq
from cirq.contrib.json import _class_resolver_dictionary
from cirq.testing.json import ModuleJsonTestSpec

TestSpec = ModuleJsonTestSpec(
    name="cirq.contrib",
    packages=[cirq.contrib],
    test_data_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[],
    should_not_be_serialized=["Unique", "CircuitDag"],
    resolver_cache=_class_resolver_dictionary(),
    deprecated={},
    # TODO: #7520 - create .json and .repr for these so they can be tested here
    tested_elsewhere=["QuantumVolumeResult", "SwapPermutationGate", "BayesianNetworkGate"],
)
