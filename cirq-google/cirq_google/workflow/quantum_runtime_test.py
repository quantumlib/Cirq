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
import datetime
import glob
import re
import time
import uuid
from typing import List, cast, Any

import numpy as np
import pytest

import cirq
import cirq_google as cg
from cirq_google.workflow.quantum_executable_test import _get_quantum_executables, _get_example_spec
from cirq_google.workflow.quantum_runtime import _time_into_runtime_info


def cg_assert_equivalent_repr(value):
    """cirq.testing.assert_equivalent_repr with cirq_google.workflow imported."""
    return cirq.testing.assert_equivalent_repr(value, global_vals={'cirq_google': cg})


def test_shared_runtime_info():
    shared_rtinfo = cg.SharedRuntimeInfo(
        run_id='my run', run_start_time=datetime.datetime.now(tz=datetime.timezone.utc)
    )
    cg_assert_equivalent_repr(shared_rtinfo)


def test_runtime_info():
    rtinfo = cg.RuntimeInfo(execution_index=5)
    with _time_into_runtime_info(rtinfo, 'test'):
        pass
    cg_assert_equivalent_repr(rtinfo)


def test_executable_result():
    rtinfo = cg.RuntimeInfo(execution_index=5)
    er = cg.ExecutableResult(
        spec=_get_example_spec(name='test-spec'),
        runtime_info=rtinfo,
        raw_data=cirq.ResultDict(
            params=cirq.ParamResolver(), measurements={'z': np.ones((1_000, 4))}
        ),
    )
    cg_assert_equivalent_repr(er)


def _assert_json_roundtrip(o, tmpdir):
    cirq.to_json_gzip(o, f'{tmpdir}/o.json')
    o2 = cirq.read_json_gzip(f'{tmpdir}/o.json')
    assert o == o2


@pytest.fixture(params=['minimal', 'full'])
def rt_config(request):
    if request.param == 'minimal':
        return cg.QuantumRuntimeConfiguration(
            processor_record=cg.SimulatedProcessorWithLocalDeviceRecord('rainbow')
        )

    elif request.param == 'full':
        return cg.QuantumRuntimeConfiguration(
            processor_record=cg.SimulatedProcessorWithLocalDeviceRecord('rainbow'),
            run_id='unit-test',
            random_seed=52,
            qubit_placer=cg.RandomDevicePlacer(),
            target_gateset=cirq.CZTargetGateset(),
        )

    raise ValueError(f"Unknown flavor {request}")  # pragma: no cover


def test_quantum_runtime_configuration_sampler(rt_config):
    sampler = rt_config.processor_record.get_sampler()
    result = sampler.run(cirq.Circuit(cirq.measure(cirq.GridQubit(5, 3), key='z')))
    assert isinstance(result, cirq.Result)


def test_quantum_runtime_configuration_device(rt_config):
    assert isinstance(rt_config.processor_record.get_device(), cirq.Device)


def test_quantum_runtime_configuration_run_id(rt_config):
    if rt_config.run_id is not None:
        assert isinstance(rt_config.run_id, str)


def test_quantum_runtime_configuration_qubit_placer(rt_config):
    device = rt_config.processor_record.get_device()
    c, _ = rt_config.qubit_placer.place_circuit(
        cirq.Circuit(cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), key='z')),
        problem_topology=cirq.LineTopology(n_nodes=2),
        shared_rt_info=cg.SharedRuntimeInfo(run_id=rt_config.run_id, device=device),
        rs=np.random.RandomState(rt_config.random_seed),
    )
    if isinstance(rt_config.qubit_placer, cg.NaiveQubitPlacer):
        assert all(isinstance(q, cirq.LineQubit) for q in c.all_qubits())
    else:
        assert all(isinstance(q, cirq.GridQubit) for q in c.all_qubits())


def test_quantum_runtime_configuration_target_gateset(rt_config):
    c = cirq.Circuit(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)))
    if rt_config.target_gateset is not None:
        # CNOT = H CZ H
        c = cirq.optimize_for_target_gateset(c, gateset=rt_config.target_gateset)
        assert len(c) == 3


def test_quantum_runtime_configuration_serialization(tmpdir, rt_config):
    cg_assert_equivalent_repr(rt_config)
    _assert_json_roundtrip(rt_config, tmpdir)


def test_executable_group_result(tmpdir):
    egr = cg.ExecutableGroupResult(
        runtime_configuration=cg.QuantumRuntimeConfiguration(
            processor_record=cg.SimulatedProcessorWithLocalDeviceRecord('rainbow'),
            run_id='unit-test',
        ),
        shared_runtime_info=cg.SharedRuntimeInfo(run_id='my run'),
        executable_results=[
            cg.ExecutableResult(
                spec=_get_example_spec(name=f'test-spec-{i}'),
                runtime_info=cg.RuntimeInfo(execution_index=i),
                raw_data=cirq.ResultDict(
                    params=cirq.ParamResolver(), measurements={'z': np.ones((1_000, 4))}
                ),
            )
            for i in range(3)
        ],
    )
    cg_assert_equivalent_repr(egr)
    assert len(egr.executable_results) == 3
    _assert_json_roundtrip(egr, tmpdir)


def test_timing():
    rt = cg.RuntimeInfo(execution_index=0)
    with _time_into_runtime_info(rt, 'test_proc'):
        time.sleep(0.1)

    assert 'test_proc' in rt.timings_s
    assert rt.timings_s['test_proc'] > 0.05


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


def test_execute(tmpdir, rt_config) -> None:
    executable_group = cg.QuantumExecutableGroup(_get_quantum_executables())
    returned_exegroup_result = cg.execute(
        rt_config=rt_config, executable_group=executable_group, base_data_dir=tmpdir
    )
    run_id = returned_exegroup_result.shared_runtime_info.run_id
    if rt_config.run_id is not None:
        assert run_id == 'unit-test'
    else:
        assert isinstance(uuid.UUID(run_id), uuid.UUID)

    start_dt = returned_exegroup_result.shared_runtime_info.run_start_time
    end_dt = returned_exegroup_result.shared_runtime_info.run_end_time
    assert start_dt is not None
    assert end_dt is not None
    assert end_dt > start_dt
    assert end_dt <= datetime.datetime.now(tz=datetime.timezone.utc)

    manual_exegroup_result = _load_result_by_hand(tmpdir, run_id)
    egr_record: cg.ExecutableGroupResultFilesystemRecord = cirq.read_json_gzip(
        f'{tmpdir}/{run_id}/ExecutableGroupResultFilesystemRecord.json.gz'
    )
    exegroup_result: cg.ExecutableGroupResult = egr_record.load(base_data_dir=tmpdir)
    helper_loaded_result = cg.ExecutableGroupResultFilesystemRecord.from_json(
        run_id=run_id, base_data_dir=tmpdir
    ).load(base_data_dir=tmpdir)

    assert isinstance(returned_exegroup_result.shared_runtime_info.device, cirq.Device)

    assert returned_exegroup_result == exegroup_result
    assert manual_exegroup_result == exegroup_result
    assert helper_loaded_result == exegroup_result

    exe_result = returned_exegroup_result.executable_results[0]
    assert 'placement' in exe_result.runtime_info.timings_s
    assert 'run' in exe_result.runtime_info.timings_s
