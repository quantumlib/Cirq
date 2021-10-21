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

"""Runtime information dataclasses that accompany execution of executables."""

import dataclasses
from typing import Any, Dict, Optional, List

import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
from cirq_google.workflow._abstract_engine_processor_shim import AbstractEngineProcessorShim
from cirq_google.workflow.quantum_executable import (
    ExecutableSpec,
)


@dataclasses.dataclass
class SharedRuntimeInfo:
    """Runtime information common to all `cg.QuantumExecutable`s in an execution of a
    `cg.QuantumExecutableGroup`.

    There is one `cg.SharedRuntimeInfo` per `cg.ExecutableGroupResult`.

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
    """Runtime information relevant to a particular `cg.QuantumExecutable`.

    There is one `cg.RuntimeInfo` per `cg.ExecutableResult`

    Args:
        execution_index: What order (in its `cg.QuantumExecutableGroup`) this
            `cg.QuantumExecutable` was executed.
    """

    execution_index: int

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclasses.dataclass
class ExecutableResult:
    """Results for a `cg.QuantumExecutable`.

    Args:
        spec: The `cg.ExecutableSpec` typifying the `cg.QuantumExecutable`.
        runtime_info: A `cg.RuntimeInfo` dataclass containing information gathered during
            execution of the `cg.QuantumExecutable`.
        raw_data: The `cirq.Result` containing the data from the run.
    """

    spec: Optional[ExecutableSpec]
    runtime_info: RuntimeInfo
    raw_data: cirq.Result

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclasses.dataclass
class ExecutableGroupResult:
    """Results for a `cg.QuantumExecutableGroup`.

    Args:
        runtime_configuration: The `cg.QuantumRuntimeConfiguration` describing how the
            `cg.QuantumExecutableGroup` was requested to be executed.
        shared_runtime_info: A `cg.SharedRuntimeInfo` dataclass containing information gathered
            during execution of the `cg.QuantumExecutableGroup` which is relevant to all
            `executable_results`.
        executable_results: A list of `cg.ExecutableResult`. Each contains results and raw data
            for an individual `cg.QuantumExecutable`.
    """

    runtime_configuration: 'QuantumRuntimeConfiguration'
    shared_runtime_info: SharedRuntimeInfo
    executable_results: List[ExecutableResult]

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclasses.dataclass
class QuantumRuntimeConfiguration:
    """User-requested configuration of how to execute a given `cg.QuantumExecutableGroup`.

    Args:
        processor: The `cg.AbstractEngineProcessor` responsible for running circuits and providing
            device information.
        run_id: A unique `str` identifier for a run. If data already exists for the specified
            `run_id`, an exception will be raised. If not specified, we will generate a UUID4
            run identifier.
    """

    processor: AbstractEngineProcessorShim
    run_id: Optional[str] = None

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')
