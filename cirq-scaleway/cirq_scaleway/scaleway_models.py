from enum import Enum

from dataclasses import dataclass
from dataclasses_json import dataclass_json


class SerializationType(Enum):
    UNKOWN = 0
    QASM_V1 = 1
    QASM_V2 = 2
    QASM_V3 = 3
    JSON = 4


@dataclass_json
@dataclass
class CircuitPayload:
    serialization_type: SerializationType
    circuit_serialization: str


@dataclass_json
@dataclass
class RunPayload:
    circuit: CircuitPayload
    options: dict


@dataclass_json
@dataclass
class BackendPayload:
    name: str
    version: str
    options: dict


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
