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

import os
import uuid
from dataclasses import dataclass
from typing import List

import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
from cirq_google.workflow._abstract_engine_processor_shim import AbstractEngineProcessorShim
from cirq_google.workflow.quantum_executable import ExecutableSpec, QuantumExecutableGroup, \
    BitstringsMeasurement


@dataclass
class SharedRuntimeInfo:
    """Runtime information common to all `QuantumExecutable`s in an execution of a
    `QuantumExecutableGroup`.

    There is one `SharedRuntimeInfo` per `ExecutableGroupResult`.

    Args:
        run_id: A unique `str` identifier for this run.
    """
    run_id: str

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self):
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclass
class RuntimeInfo:
    """Runtime information relevant to a particular `QuantumExecutable`.

    There is one `RuntimeInfo` per `ExecutableResult`

    Args:
        execution_index: What order (in its `QuantumExecutableGroup`) this `QuantumExecutable` was
            executed.
    """
    execution_index: int

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self):
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclass
class ExecutableResult:
    """Results for a `QuantumExecutable`.

    Args:
        spec: The `ExecutableSpec` typifying the `QuantumExecutable`.
        runtime_info: A `RuntimeInfo` dataclass containing information gathered during execution
            of the `QuantumExecutable`.
        raw_data: The `cirq.Result` containing the data from the run.
    """
    spec: ExecutableSpec
    runtime_info: RuntimeInfo
    raw_data: cirq.Result

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self):
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclass
class ExecutableGroupResult:
    """Results for a `QuantumExecutableGroup`.

    Args:
        runtime_configuration: The `QuantumRuntimeConfiguration` describing how the
            `QuantumExecutableGroup` was requested to be executed.
        shared_runtime_info: A `SharedRuntimeInfo` dataclass containing information gathered
            during execution of the `QuantumExecutableGroup` which is relevant to all
            `executable_results`.
        executable_results: A list of `ExecutableResult`. Each contains results and raw data
            for an individual `QuantumExecutable`.
    """
    runtime_configuration: 'QuantumRuntimeConfiguration'
    shared_runtime_info: SharedRuntimeInfo
    executable_results: List[ExecutableResult]

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self):
        return _compat.dataclass_repr(self, namespace='cirq_google')


@dataclass
class QuantumRuntimeConfiguration:
    """User-requested configuration of how to execute a given `QuantumExecutableGroup`.

    Args:
        processor: The `AbstractEngineProcessor` responsible for running circuits and providing
            device information.
        run_id: A unique `str` identifier for a run. If data already exists for the specified
            `run_id`, an exception will be raised. If not specified, we will generate a UUID4
            run identifier.
    """
    processor: AbstractEngineProcessorShim
    run_id: str = None

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self):
        return _compat.dataclass_repr(self, namespace='cirq_google')


def execute(
        rt_config: QuantumRuntimeConfiguration,
        executable_group: QuantumExecutableGroup,
        base_data_dir: str = ".",
) -> ExecutableGroupResult:
    """Execute a `QuantumExecutableGroup` according to a `QuantumRuntimeConfiguration`.

    Args:
        rt_config: The `QuantumRuntimeConfiguration` specifying how to execute `executable_group`.
        executable_group: The `QuantumExecutableGroup` containing the executables to execute.
        base_data_dir: A filesystem path to write data. We write
            "{base_data_dir}/{run_id}/ExecutableGroupResult.json.gz"
            containing the `ExecutableGroupResult` as well as one file
            "{base_data_dir}/{run_id}/ExecutableResult.{i}.json.gz" per `ExecutableResult` as
            each executable result becomes available.

    Returns:
        The `ExecutableGroupResult` containing all data and metadata for an execution.
    """
    # run_id defaults logic.
    if rt_config.run_id is None:
        run_id = str(uuid.uuid4())
    else:
        run_id = rt_config.run_id

    # base_data_dir handling.
    if not base_data_dir:
        raise ValueError("Please provide a non-empty `base_data_dir`.")

    os.makedirs(f'{base_data_dir}/{run_id}', exist_ok=False)

    # Results object that we will fill in in the main loop.
    exegroup_result = ExecutableGroupResult(runtime_configuration=rt_config,
                                            shared_runtime_info=SharedRuntimeInfo(run_id=run_id),
                                            executable_results=[])
    cirq.to_json_gzip(exegroup_result, f'{base_data_dir}/{run_id}/ExecutableGroupResult.json.gz')

    # Loop over executables.
    sampler = rt_config.processor.get_sampler()
    print('# Executables:', len(executable_group), flush=True)
    for i, exe in enumerate(executable_group):
        runtime_info = RuntimeInfo(
            execution_index=i
        )

        if exe.params != tuple():
            raise NotImplementedError("Circuit params are not yet supported.")

        circuit = exe.circuit

        if not isinstance(exe.measurement, BitstringsMeasurement):
            raise NotImplementedError("Only `BitstringsMeasurement` are supported.")

        sampler_run_result = sampler.run(circuit, repetitions=exe.measurement.n_repetitions)

        exe_result = ExecutableResult(
            spec=exe.spec,
            runtime_info=runtime_info,
            raw_data=sampler_run_result,
        )
        cirq.to_json_gzip(exe_result, f'{base_data_dir}/{run_id}/ExecutableResult.{i}.json.gz')
        exegroup_result.executable_results.append(exe_result)
        print(i, end=' ', flush=True)

    return exegroup_result
