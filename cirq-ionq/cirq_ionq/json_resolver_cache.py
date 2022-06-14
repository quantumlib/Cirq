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

import functools
from typing import Dict

import cirq_ionq
from cirq.protocols.json_serialization import ObjectFactory


@functools.lru_cache()  # coverage: ignore
def _class_resolver_dictionary() -> Dict[str, ObjectFactory]:  # coverage: ignore
    return {
        "GPIGate": cirq_ionq.GPIGate,
        "GPI2Gate": cirq_ionq.GPI2Gate,
        "MSGate": cirq_ionq.MSGate,
        "IonQTargetGateset": cirq_ionq.IonQTargetGateset,
    }
