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

"""The cirq-pasqal test specification for JSON serialization tests.

The actual tests live in cirq.protocols.json_serialization_test.py.
See cirq-core/cirq/testing/json.py for a description of the framework.
"""

import pathlib

import cirq_pasqal
from cirq_pasqal.json_resolver_cache import _class_resolver_dictionary

from cirq.testing.json import ModuleJsonTestSpec

TestSpec = ModuleJsonTestSpec(
    name="cirq_pasqal",
    packages=[cirq_pasqal],
    test_data_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[],
    should_not_be_serialized=["PasqalNoiseModel", "PasqalSampler"],
    resolver_cache=_class_resolver_dictionary(),
    deprecated={},
)
