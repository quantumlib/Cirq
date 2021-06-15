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

import pathlib

import cirq_ionq
from cirq_ionq.json_resolver_cache import _class_resolver_dictionary

from cirq.testing.json import ModuleJsonTestSpec

TestSpec = ModuleJsonTestSpec(
    name="cirq_ionq",
    packages=[cirq_ionq],
    test_data_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[
        "SerializedProgram",
        "Calibration",
        "QPUResult",
        "IonQException",
        "IonQUnsuccessfulJobException",
        "IonQNotFoundException",
        "IonQAPIDevice",
        "Job",
        "SimulatorResult",
    ],
    should_not_be_serialized=[
        "Sampler",
        "Service",
        "Serializer",
    ],
    resolver_cache=_class_resolver_dictionary(),
    deprecated={},
)
