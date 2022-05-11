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

"""Facilities for dealing with data input and output for the quantum runtime."""
import abc
import dataclasses
import os
from typing import Any, Dict, List, TYPE_CHECKING

import cirq
from cirq import _compat
from cirq.protocols import dataclass_json_dict

if TYPE_CHECKING:
    import cirq_google as cg


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

    @classmethod
    def from_json(
        cls, *, run_id: str, base_data_dir: str = "."
    ) -> 'ExecutableGroupResultFilesystemRecord':
        fn = f'{base_data_dir}/{run_id}/ExecutableGroupResultFilesystemRecord.json.gz'
        egr_record = cirq.read_json_gzip(fn)
        if not isinstance(egr_record, cls):
            raise ValueError(
                f"The file located at {fn} is not an `ExecutableGroupFilesystemRecord`."
            )
        if egr_record.run_id != run_id:
            raise ValueError(
                f"The loaded run_id {run_id} does not match the provided run_id {run_id}"
            )

        return egr_record

    def load(self, *, base_data_dir: str = ".") -> 'cg.ExecutableGroupResult':
        """Using the filename references in this dataclass, load a `cg.ExecutableGroupResult`
        from its constituent parts.

        Args:
            base_data_dir: The base data directory. Files should be found at
                {base_data_dir}/{run_id}/{this class's paths}
        """
        data_dir = f"{base_data_dir}/{self.run_id}"
        from cirq_google.workflow.quantum_runtime import ExecutableGroupResult

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

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return dataclass_json_dict(self)

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
    shared_rt_info: 'cg.SharedRuntimeInfo',
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


class _WorkflowSaver(abc.ABC):
    def initialize(
        self, rt_config: 'cg.QuantumRuntimeConfiguration', shared_rt_info: 'cg.SharedRuntimeInfo'
    ):
        """Initialize a data saving for a workflow execution.

        Args:
            rt_config: The immutable `cg.QuantumRuntimeConfiguation` for this run. This should
                be saved once, likely during initialization.
            shared_rt_info: The current `cg.SharedRuntimeInfo` for saving.
        """

    def consume_result(
        self, exe_result: 'cg.ExecutableResult', shared_rt_info: 'cg.SharedRuntimeInfo'
    ):
        """Consume an `cg.ExecutableResult` that has been completed.

        Args:
            exe_result: The completed `cg.ExecutableResult` to be saved.
            shared_rt_info: The current `cg.SharedRuntimeInfo` to be saved or updated.
        """

    def finalize(self, shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Called at the end of a workflow execution to finalize data saving.

        Args:
            shared_rt_info: The final `cg.SharedRuntimeInfo` to be saved or updated.
        """


class _FilesystemSaver(_WorkflowSaver):
    """Save data to the filesystem.

    The ExecutableGroupResult's constituent parts will be saved to disk as they become
    available. Within the "{base_data_dir}/{run_id}" directory we save:
        - The `cg.QuantumRuntimeConfiguration` at the start of the execution as a record
          of *how* the executable group was run.
        - A `cg.SharedRuntimeInfo` which is updated throughout the run.
        - An `cg.ExecutableResult` for each `cg.QuantumExecutable` as they become available.
        - A `cg.ExecutableGroupResultFilesystemRecord` which is updated throughout the run.

    Args:
        base_data_dir: Each data file will be written to the "{base_data_dir}/{run_id}/" directory,
            which must not already exist.
        run_id: Each data file will be written to the "{base_data_dir}/{run_id}/" directory,
            which must not already exist.
    """

    def __init__(self, base_data_dir, run_id):
        self.base_data_dir = base_data_dir
        self.run_id = run_id

        self._data_dir = f'{self.base_data_dir}/{self.run_id}'
        self._egr_record = None

    @property
    def data_dir(self) -> str:
        """The data directory, namely '{base_data_dir}/{run_id}"""
        return self._data_dir

    @property
    def egr_record(self) -> ExecutableGroupResultFilesystemRecord:
        """The `cg.ExecutablegroupResultFilesystemRecord` keeping track of all the paths for saved
        files."""
        return self._egr_record

    def initialize(
        self, rt_config: 'cg.QuantumRuntimeConfiguration', shared_rt_info: 'cg.SharedRuntimeInfo'
    ):
        """Initialize the filesystem for data saving

        Args:
            rt_config: The immutable `cg.QuantumRuntimeConfiguation` for this run. This is written
                once during this initialization.
            shared_rt_info: The initial `cg.SharedRuntimeInfo` to be saved to a file.
        """
        os.makedirs(self._data_dir, exist_ok=False)
        self._egr_record = ExecutableGroupResultFilesystemRecord(
            runtime_configuration_path='QuantumRuntimeConfiguration.json.gz',
            shared_runtime_info_path='SharedRuntimeInfo.json.gz',
            executable_result_paths=[],
            run_id=self.run_id,
        )
        cirq.to_json_gzip(
            rt_config, f'{self._data_dir}/{self._egr_record.runtime_configuration_path}'
        )
        _update_updatable_files(self._egr_record, shared_rt_info, self._data_dir)

    def consume_result(
        self, exe_result: 'cg.ExecutableResult', shared_rt_info: 'cg.SharedRuntimeInfo'
    ):
        """Save an `cg.ExecutableResult` that has been completed.

        Args:
            exe_result: The completed `cg.ExecutableResult` to be saved to
                'ExecutableResult.{i}.json.gz'
            shared_rt_info: The current `cg.SharedRuntimeInfo` to update SharedRuntimeInfo.json.gz.
        """
        i = exe_result.runtime_info.execution_index
        exe_result_path = f'ExecutableResult.{i}.json.gz'
        cirq.to_json_gzip(exe_result, f"{self._data_dir}/{exe_result_path}")
        self._egr_record.executable_result_paths.append(exe_result_path)

        _update_updatable_files(self._egr_record, shared_rt_info, self._data_dir)

    def finalize(self, shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Called at the end of a workflow execution to finalize data saving.

        Args:
            shared_rt_info: The final `cg.SharedRuntimeInfo` to be saved or updated.
        """
        _update_updatable_files(self.egr_record, shared_rt_info, self._data_dir)
