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

"""Runtime information dataclasses and execution of executables."""

import dataclasses
import os
import uuid
from typing import Any, Dict, Optional, List

import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict
from cirq_google.workflow._abstract_engine_processor_shim import AbstractEngineProcessorShim
from cirq_google.workflow.progress import _PrintLogger
from cirq_google.workflow.quantum_executable import (
    ExecutableSpec,
    QuantumExecutableGroup,
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
class ExecutableGroupResultFilesystemRecord:
    """Filename references to the constituent parts of a `cg.ExecutableGroupResult`.

    Args:
        runtime_configuration_path: A filename pointing to the `runtime_configuration` value.
        shared_runtime_info_path: A filename pointing to the `shared_runtime_info` value.
        executable_result_paths: A list of filenames pointing to the `executable_results` values.
        run_id: The unique `str` identifier from this run. This is used to locate the other
            values on disk.
    """

    runtime_configuration_path: str
    shared_runtime_info_path: str
    executable_result_paths: List[str]

    run_id: str

    def load(self, *, base_data_dir: str = ".") -> ExecutableGroupResult:
        """Using the filename references in this dataclass, load a `cg.ExecutableGroupResult`
        from its constituent parts.

        Args:
            base_data_dir: The base data directory. Files should be found at
                {base_data_dir}/{run_id}/{this class's paths}
        """
        data_dir = f"{base_data_dir}/{self.run_id}"
        return ExecutableGroupResult(
            runtime_configuration=cirq.read_json_gzip(
                f'{data_dir}/{self.runtime_configuration_path}'
            ),
            shared_runtime_info=cirq.read_json_gzip(f'{data_dir}/{self.shared_runtime_info_path}'),
            executable_results=[
                cirq.read_json_gzip(f'{data_dir}/{exe_path}')
                for exe_path in self.executable_result_paths
            ],
        )

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


def _safe_to_json(obj: Any, *, part_path: str, nominal_path: str, bak_path: str):
    """Safely update a json file.

    1. The new value is written to a "part" file
    2. The previous file atomically replaces the previous backup file, thereby becoming the
       current backup file.
    3. The part file is atomically renamed to the desired filename.
    """
    cirq.to_json_gzip(obj, part_path)
    if os.path.exists(nominal_path):
        os.replace(nominal_path, bak_path)
    os.replace(part_path, nominal_path)


def _update_updatable_files(
    egr_record: ExecutableGroupResultFilesystemRecord,
    shared_rt_info: SharedRuntimeInfo,
    data_dir: str,
):
    """Safely update ExecutableGroupResultFilesystemRecord.json.gz and SharedRuntimeInfo.json.gz
    during an execution run.
    """
    _safe_to_json(
        shared_rt_info,
        part_path=f'{data_dir}/SharedRuntimeInfo.json.gz.part',
        nominal_path=f'{data_dir}/SharedRuntimeInfo.json.gz',
        bak_path=f'{data_dir}/SharedRuntimeInfo.json.gz.bak',
    )
    _safe_to_json(
        egr_record,
        part_path=f'{data_dir}/ExecutableGroupResultFilesystemRecord.json.gz.part',
        nominal_path=f'{data_dir}/ExecutableGroupResultFilesystemRecord.json.gz',
        bak_path=f'{data_dir}/ExecutableGroupResultFilesystemRecord.json.gz.bak',
    )


def execute(
    rt_config: QuantumRuntimeConfiguration,
    executable_group: QuantumExecutableGroup,
    base_data_dir: str = ".",
) -> ExecutableGroupResult:
    """Execute a `cg.QuantumExecutableGroup` according to a `cg.QuantumRuntimeConfiguration`.

    The ExecutableGroupResult's constituent parts will be saved to disk as they become
    available. Within the "{base_data_dir}/{run_id}" directory we save:
        - The `cg.QuantumRuntimeConfiguration` at the start of the execution as a record
          of *how* the executable group was run.
        - A `cg.SharedRuntimeInfo` which is updated throughout the run.
        - An `cg.ExecutableResult` for each `cg.QuantumExecutable` as they become available.
        - A `cg.ExecutableGroupResultFilesystemRecord` which is updated throughout the run.

    Args:
        rt_config: The `cg.QuantumRuntimeConfiguration` specifying how to execute
            `executable_group`.
        executable_group: The `cg.QuantumExecutableGroup` containing the executables to execute.
        base_data_dir: Each data file will be written to the "{base_data_dir}/{run_id}/" directory,
            which must not already exist.

    Returns:
        The `cg.ExecutableGroupResult` containing all data and metadata for an execution.

    Raises:
        NotImplementedError: If an executable uses the `params` field or anything other than
            a BitstringsMeasurement measurement field.
        ValueError: If `base_data_dir` is not a valid directory.
    """
    # run_id defaults logic.
    if rt_config.run_id is None:
        run_id = str(uuid.uuid4())
    else:
        run_id = rt_config.run_id

    # base_data_dir handling.
    if not base_data_dir:
        # coverage: ignore
        raise ValueError("Please provide a non-empty `base_data_dir`.")

    # Set up data saving, save runtime configuration.
    data_dir = f'{base_data_dir}/{run_id}'
    os.makedirs(data_dir, exist_ok=False)
    egr_record = ExecutableGroupResultFilesystemRecord(
        runtime_configuration_path='QuantumRuntimeConfiguration.json.gz',
        shared_runtime_info_path='SharedRuntimeInfo.json.gz',
        executable_result_paths=[],
        run_id=run_id,
    )
    cirq.to_json_gzip(rt_config, f'{data_dir}/{egr_record.runtime_configuration_path}')

    # Set up to-be-updated objects.
    shared_rt_info = SharedRuntimeInfo(run_id=run_id)
    _update_updatable_files(egr_record, shared_rt_info, data_dir)
    executable_results = []

    sampler = rt_config.processor.get_sampler()
    logger = _PrintLogger(n_total=len(executable_group))
    logger.initialize()
    for i, exe in enumerate(executable_group):
        runtime_info = RuntimeInfo(execution_index=i)

        if exe.params != tuple():
            raise NotImplementedError("Circuit params are not yet supported.")

        circuit = exe.circuit

        if not hasattr(exe.measurement, 'n_repetitions'):
            raise NotImplementedError("Only `BitstringsMeasurement` are supported.")

        sampler_run_result = sampler.run(circuit, repetitions=exe.measurement.n_repetitions)

        exe_result = ExecutableResult(
            spec=exe.spec,
            runtime_info=runtime_info,
            raw_data=sampler_run_result,
        )
        # Do bookkeeping for finished ExecutableResult
        exe_result_path = f'ExecutableResult.{i}.json.gz'
        cirq.to_json_gzip(exe_result, f"{data_dir}/{exe_result_path}")
        executable_results.append(exe_result)
        egr_record.executable_result_paths.append(exe_result_path)

        _update_updatable_files(egr_record, shared_rt_info, data_dir)
        logger.consume_result(exe_result, shared_rt_info)
    logger.finalize()

    return ExecutableGroupResult(
        runtime_configuration=rt_config,
        shared_runtime_info=shared_rt_info,
        executable_results=executable_results,
    )
