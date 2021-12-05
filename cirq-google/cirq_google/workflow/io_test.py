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

import cirq
import cirq_google as cg
import numpy as np
from cirq_google.workflow.io import _FilesystemSaver
from cirq_google.workflow.quantum_runtime_test import (
    _MockEngineProcessor,
    patch_cirq_default_resolvers,
)

assert patch_cirq_default_resolvers, 'Pytest fixture -- not unused import'


def cg_assert_equivalent_repr(value):
    """cirq.testing.assert_equivalent_repr with cirq_google.workflow imported."""
    return cirq.testing.assert_equivalent_repr(
        value,
        global_vals={
            'cirq_google': cg,
        },
    )


def test_egr_filesystem_record_repr():
    egr_fs_record = cg.ExecutableGroupResultFilesystemRecord(
        runtime_configuration_path='RuntimeConfiguration.json.gz',
        shared_runtime_info_path='SharedRuntimeInfo.jzon.gz',
        executable_result_paths=[
            'ExecutableResult.1.json.gz',
            'ExecutableResult.2.json.gz',
        ],
        run_id='my-run-id',
    )
    cg_assert_equivalent_repr(egr_fs_record)


def test_filesystem_saver(tmpdir, patch_cirq_default_resolvers):
    assert patch_cirq_default_resolvers
    run_id = 'asdf'
    fs_saver = _FilesystemSaver(base_data_dir=tmpdir, run_id=run_id)

    rt_config = cg.QuantumRuntimeConfiguration(processor=_MockEngineProcessor(), run_id=run_id)
    shared_rt_info = cg.SharedRuntimeInfo(run_id=run_id)
    fs_saver.initialize(rt_config, shared_rt_info=shared_rt_info)

    rt_config2 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/QuantumRuntimeConfiguration.json.gz')
    shared_rt_info2 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/SharedRuntimeInfo.json.gz')
    assert rt_config == rt_config2
    assert shared_rt_info == shared_rt_info2

    # you shouldn't actually mutate run_id in the shared runtime info, but we want to test
    # updating the shared rt info object:
    shared_rt_info.run_id = 'updated_run_id'
    exe_result = cg.ExecutableResult(
        spec=None,
        runtime_info=cg.RuntimeInfo(execution_index=0),
        raw_data=cirq.Result(params=cirq.ParamResolver({}), measurements={'z': np.ones((100, 5))}),
    )
    fs_saver.consume_result(exe_result=exe_result, shared_rt_info=shared_rt_info)

    shared_rt_info3 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/SharedRuntimeInfo.json.gz')
    exe_result3 = cirq.read_json_gzip(f'{tmpdir}/{run_id}/ExecutableResult.0.json.gz')
    assert shared_rt_info == shared_rt_info3
    assert exe_result == exe_result3

    egr_record: cg.ExecutableGroupResultFilesystemRecord = cirq.read_json_gzip(
        f'{fs_saver.data_dir}/ExecutableGroupResultFilesystemRecord.json.gz'
    )
    assert egr_record == fs_saver.egr_record
    exegroup_result: cg.ExecutableGroupResult = egr_record.load(base_data_dir=tmpdir)
    assert exegroup_result.shared_runtime_info == shared_rt_info
    assert exegroup_result.runtime_configuration == rt_config
    assert exegroup_result.executable_results[0] == exe_result
