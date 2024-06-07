# Copyright 2024 Scaleway
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
from enum import Enum
from typing import List, Dict

from dataclasses import dataclass
from dataclasses_json import dataclass_json


class SerializationType(Enum):
    UNKOWN = 0
    QASM_V1 = 1
    QASM_V2 = 2
    QASM_V3 = 3
    QPY_V1 = 4
    JSON = 5


@dataclass_json
@dataclass
class CircuitPayload:
    serialization_type: SerializationType
    circuit_serialization: str


@dataclass_json
@dataclass
class RunPayload:
    circuits: List[CircuitPayload]
    options: Dict


@dataclass_json
@dataclass
class BackendPayload:
    name: str
    version: str
    options: Dict


@dataclass_json
@dataclass
class ClientPayload:
    user_agent: str


@dataclass_json
@dataclass
class JobPayload:
    client: ClientPayload
    backend: BackendPayload
    run: RunPayload
