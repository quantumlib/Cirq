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

"""Infrastructure for running quantum executables."""

import dataclasses
from typing import Any, Dict, Optional

import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
from cirq_google.workflow.quantum_executable import (
    ExecutableSpec,
)


@dataclasses.dataclass
class SharedRuntimeInfo:
    """Runtime information common to all `QuantumExecutable`s in an execution of a
    `QuantumExecutableGroup`.

    There is one `SharedRuntimeInfo` per `ExecutableGroupResult`.

    Args:
        run_id: A unique `str` identifier for this run.
    """

    run_id: str

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclasses.dataclass
class RuntimeInfo:
    """Runtime information relevant to a particular `QuantumExecutable`.

    There is one `RuntimeInfo` per `ExecutableResult`

    Args:
        execution_index: What order (in its `QuantumExecutableGroup`) this `QuantumExecutable` was
            executed.
    """

    execution_index: int

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclasses.dataclass
class ExecutableResult:
    """Results for a `QuantumExecutable`.

    Args:
        spec: The `ExecutableSpec` typifying the `QuantumExecutable`.
        runtime_info: A `RuntimeInfo` dataclass containing information gathered during execution
            of the `QuantumExecutable`.
        raw_data: The `cirq.Result` containing the data from the run.
    """

    spec: Optional[ExecutableSpec]
    runtime_info: RuntimeInfo
    raw_data: cirq.Result

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')
