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
import glob
import re
import uuid
from dataclasses import dataclass
from typing import List, cast, Any

import cirq
import cirq_google as cg
import numpy as np
import pytest
from cirq_google.workflow._abstract_engine_processor_shim import AbstractEngineProcessorShim
from cirq_google.workflow.quantum_executable_test import _get_quantum_executables, _get_example_spec


@dataclass
class _MockEngineProcessor(AbstractEngineProcessorShim):
    def get_device(self) -> cirq.Device:
        return cg.Sycamore23

    def get_sampler(self) -> cirq.Sampler:
        return cirq.ZerosSampler()

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google.testing'

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, attribute_names=[])


def cg_assert_equivalent_repr(value):
    """cirq.testing.assert_equivalent_repr with cirq_google.workflow imported."""
    return cirq.testing.assert_equivalent_repr(
        value,
        global_vals={
            'cirq_google': cg,
            '_MockEngineProcessor': _MockEngineProcessor,
        },
    )


def test_shared_runtime_info():
    shared_rtinfo = cg.SharedRuntimeInfo(run_id='my run')
    cg_assert_equivalent_repr(shared_rtinfo)


def test_runtime_info():
    rtinfo = cg.RuntimeInfo(execution_index=5)
    cg_assert_equivalent_repr(rtinfo)


def test_executable_result():
    rtinfo = cg.RuntimeInfo(execution_index=5)
    er = cg.ExecutableResult(
        spec=_get_example_spec(name='test-spec'),
        runtime_info=rtinfo,
        raw_data=cirq.Result(params=cirq.ParamResolver(), measurements={'z': np.ones((1_000, 4))}),
    )
    cg_assert_equivalent_repr(er)


def _testing_resolver(cirq_type: str):
    if cirq_type == 'cirq.google.testing._MockEngineProcessor':
        return _MockEngineProcessor


def _cg_read_json_gzip(fn):
    return cirq.read_json_gzip(fn, resolvers=[_testing_resolver] + cirq.DEFAULT_RESOLVERS)


@pytest.fixture
def patch_cirq_default_resolvers():
    backup = cirq.DEFAULT_RESOLVERS.copy()
    cirq.DEFAULT_RESOLVERS.insert(0, _testing_resolver)
    yield True
    cirq.DEFAULT_RESOLVERS = backup


def _assert_json_roundtrip(o, tmpdir):
    cirq.to_json_gzip(o, f'{tmpdir}/o.json')
    o2 = _cg_read_json_gzip(f'{tmpdir}/o.json')
    assert o == o2


def test_quantum_runtime_configuration():
    rt_config = cg.QuantumRuntimeConfiguration(
        processor=_MockEngineProcessor(),
        run_id='unit-test',
    )

    sampler = rt_config.processor.get_sampler()
    result = sampler.run(cirq.Circuit(cirq.measure(cirq.LineQubit(0), key='z')))
    assert isinstance(result, cirq.Result)

    assert isinstance(rt_config.processor.get_device(), cirq.Device)


def test_quantum_runtime_configuration_serialization(tmpdir):
    rt_config = cg.QuantumRuntimeConfiguration(
        processor=_MockEngineProcessor(),
        run_id='unit-test',
    )
    cg_assert_equivalent_repr(rt_config)
    _assert_json_roundtrip(rt_config, tmpdir)


def test_executable_group_result(tmpdir):
    egr = cg.ExecutableGroupResult(
        runtime_configuration=cg.QuantumRuntimeConfiguration(
            processor=_MockEngineProcessor(),
            run_id='unit-test',
        ),
        shared_runtime_info=cg.SharedRuntimeInfo(run_id='my run'),
        executable_results=[
            cg.ExecutableResult(
                spec=_get_example_spec(name=f'test-spec-{i}'),
                runtime_info=cg.RuntimeInfo(execution_index=i),
                raw_data=cirq.Result(
                    params=cirq.ParamResolver(), measurements={'z': np.ones((1_000, 4))}
                ),
            )
            for i in range(3)
        ],
    )
    cg_assert_equivalent_repr(egr)
    assert len(egr.executable_results) == 3
    _assert_json_roundtrip(egr, tmpdir)


def _load_result_by_hand(tmpdir: str, run_id: str) -> cg.ExecutableGroupResult:
    """Load `ExecutableGroupResult` "by hand" without using
    `ExecutableGroupResultFilesystemRecord`."""
    rt_config = cirq.read_json_gzip(f'{tmpdir}/{run_id}/QuantumRuntimeConfiguration.json.gz')
    shared_rt_info = cirq.read_json_gzip(f'{tmpdir}/{run_id}/SharedRuntimeInfo.json.gz')
    fns = glob.glob(f'{tmpdir}/{run_id}/ExecutableResult.*.json.gz')
    fns = sorted(
        fns,
        key=lambda s: int(cast(Any, re.search(r'ExecutableResult\.(\d+)\.json\.gz$', s)).group(1)),
    )
    assert len(fns) == 3
    exe_results: List[cg.ExecutableResult] = [cirq.read_json_gzip(fn) for fn in fns]
    return cg.ExecutableGroupResult(
        runtime_configuration=rt_config,
        shared_runtime_info=shared_rt_info,
        executable_results=exe_results,
    )


@pytest.mark.parametrize('run_id_in', ['unit_test_runid', None])
def test_execute(tmpdir, run_id_in, patch_cirq_default_resolvers):
    assert patch_cirq_default_resolvers
    rt_config = cg.QuantumRuntimeConfiguration(
        processor=_MockEngineProcessor(),
        run_id=run_id_in,
        qubit_placer=cg.NaiveQubitPlacer(),
    )
    executable_group = cg.QuantumExecutableGroup(_get_quantum_executables())
    returned_exegroup_result = cg.execute(
        rt_config=rt_config, executable_group=executable_group, base_data_dir=tmpdir
    )
    run_id = returned_exegroup_result.shared_runtime_info.run_id
    if run_id_in is not None:
        assert run_id_in == run_id
    else:
        assert isinstance(uuid.UUID(run_id), uuid.UUID)

    manual_exegroup_result = _load_result_by_hand(tmpdir, run_id)
    egr_record: cg.ExecutableGroupResultFilesystemRecord = cirq.read_json_gzip(
        f'{tmpdir}/{run_id}/ExecutableGroupResultFilesystemRecord.json.gz'
    )
    exegroup_result: cg.ExecutableGroupResult = egr_record.load(base_data_dir=tmpdir)

    # TODO(gh-4699): Don't null-out device once it's serializable.
    assert isinstance(returned_exegroup_result.shared_runtime_info.device, cg.SerializableDevice)
    returned_exegroup_result.shared_runtime_info.device = None

    assert returned_exegroup_result == exegroup_result
    assert manual_exegroup_result == exegroup_result
