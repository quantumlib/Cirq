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


class _FilesystemSaver:
    def __init__(self, base_data_dir, run_id):
        self.base_data_dir = base_data_dir
        self.run_id = run_id

        self._data_dir = f'{self.base_data_dir}/{self.run_id}'
        self._egr_record = None

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def egr_record(self):
        return self._egr_record

    def initialize(self, rt_config: 'cg.QuantumRuntimeConfiguration',
                   shared_rt_info: 'cg.SharedRuntimeInfo'):
        os.makedirs(self._data_dir, exist_ok=False)
        self._egr_record = ExecutableGroupResultFilesystemRecord(
            runtime_configuration_path='QuantumRuntimeConfiguration.json.gz',
            shared_runtime_info_path='SharedRuntimeInfo.json.gz',
            executable_result_paths=[],
            run_id=self.run_id,
        )
        cirq.to_json_gzip(rt_config,
                          f'{self._data_dir}/{self._egr_record.runtime_configuration_path}')
        _update_updatable_files(self._egr_record, shared_rt_info, self._data_dir)

    def consume_one(self, exe_result: 'cg.ExecutableResult',
                    shared_rt_info: 'cg.SharedRuntimeInfo'):
        i = exe_result.runtime_info.execution_index
        exe_result_path = f'ExecutableResult.{i}.json.gz'
        cirq.to_json_gzip(exe_result, f"{self._data_dir}/{exe_result_path}")
        self._egr_record.executable_result_paths.append(exe_result_path)

        _update_updatable_files(self._egr_record, shared_rt_info, self._data_dir)
