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
import os

import numpy as np
import pytest

import cirq
import cirq_google as cg
from cirq_google.workflow.io import _FilesystemSaver


def cg_assert_equivalent_repr(value):
    """cirq.testing.assert_equivalent_repr with cirq_google.workflow imported."""
    return cirq.testing.assert_equivalent_repr(value, global_vals={'cirq_google': cg})


def test_egr_filesystem_record_repr():
    egr_fs_record = cg.ExecutableGroupResultFilesystemRecord(
        runtime_configuration_path='RuntimeConfiguration.json.gz',
        shared_runtime_info_path='SharedRuntimeInfo.jzon.gz',
        executable_result_paths=['ExecutableResult.1.json.gz', 'ExecutableResult.2.json.gz'],
        run_id='my-run-id',
    )
    cg_assert_equivalent_repr(egr_fs_record)


def test_egr_filesystem_record_from_json(tmpdir):
    run_id = 'my-run-id'
    egr_fs_record = cg.ExecutableGroupResultFilesystemRecord(
        runtime_configuration_path='RuntimeConfiguration.json.gz',
        shared_runtime_info_path='SharedRuntimeInfo.jzon.gz',
        executable_result_paths=['ExecutableResult.1.json.gz', 'ExecutableResult.2.json.gz'],
        run_id=run_id,
    )

    # Test 1: normal
    os.makedirs(f'{tmpdir}/{run_id}')
    cirq.to_json_gzip(
        egr_fs_record, f'{tmpdir}/{run_id}/ExecutableGroupResultFilesystemRecord.json.gz'
    )
    egr_fs_record2 = cg.ExecutableGroupResultFilesystemRecord.from_json(
        run_id=run_id, base_data_dir=tmpdir
    )
    assert egr_fs_record == egr_fs_record2

    # Test 2: bad object type
    cirq.to_json_gzip(
        cirq.Circuit(), f'{tmpdir}/{run_id}/ExecutableGroupResultFilesystemRecord.json.gz'
    )
    with pytest.raises(ValueError, match=r'.*not an `ExecutableGroupFilesystemRecord`.'):
        cg.ExecutableGroupResultFilesystemRecord.from_json(run_id=run_id, base_data_dir=tmpdir)

    # Test 3: Mismatched run id
    os.makedirs(f'{tmpdir}/questionable_run_id')
    cirq.to_json_gzip(
        egr_fs_record, f'{tmpdir}/questionable_run_id/ExecutableGroupResultFilesystemRecord.json.gz'
    )
    with pytest.raises(ValueError, match=r'.*does not match the provided run_id'):
        cg.ExecutableGroupResultFilesystemRecord.from_json(
            run_id='questionable_run_id', base_data_dir=tmpdir
        )


def test_filesystem_saver(tmpdir) -> None:
    run_id = 'asdf'
    fs_saver = _FilesystemSaver(base_data_dir=tmpdir, run_id=run_id)

    rt_config = cg.QuantumRuntimeConfiguration(
        processor_record=cg.SimulatedProcessorWithLocalDeviceRecord('rainbow'), run_id=run_id
    )
    shared_rt_info = cg.SharedRuntimeInfo(run_id=run_id)
    fs_saver.initialize(rt_config, shared_rt_info=shared_rt_info)

    # Test 1: assert fs_saver.initialize() has worked.
    rt_config2 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/QuantumRuntimeConfiguration.json.gz')
    shared_rt_info2 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/SharedRuntimeInfo.json.gz')
    assert rt_config == rt_config2
    assert shared_rt_info == shared_rt_info2

    # Test 2: assert `consume_result()` works.
    # you shouldn't actually mutate run_id in the shared runtime info, but we want to test
    # updating the shared rt info object:
    shared_rt_info.run_id = 'updated_run_id'
    exe_result = cg.ExecutableResult(
        spec=None,
        runtime_info=cg.RuntimeInfo(execution_index=0),
        raw_data=cirq.ResultDict(
            params=cirq.ParamResolver({}), measurements={'z': np.ones((100, 5))}
        ),
    )
    fs_saver.consume_result(exe_result=exe_result, shared_rt_info=shared_rt_info)

    shared_rt_info3 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/SharedRuntimeInfo.json.gz')
    exe_result3 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/ExecutableResult.0.json.gz')
    assert shared_rt_info == shared_rt_info3
    assert exe_result == exe_result3

    # Test 3: assert loading egr_record works.
    egr_record: cg.ExecutableGroupResultFilesystemRecord = cirq.read_json_gzip(
        f'{fs_saver.data_dir}/ExecutableGroupResultFilesystemRecord.json.gz'
    )
    assert egr_record == fs_saver.egr_record
    exegroup_result: cg.ExecutableGroupResult = egr_record.load(base_data_dir=tmpdir)
    assert exegroup_result.shared_runtime_info == shared_rt_info
    assert exegroup_result.runtime_configuration == rt_config
    assert exegroup_result.executable_results[0] == exe_result
