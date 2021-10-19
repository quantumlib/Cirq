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
from dataclasses import dataclass

import cirq
import cirq_google as cg
import numpy as np
from cirq_google.workflow._abstract_engine_processor_shim import AbstractEngineProcessorShim
from cirq_google.workflow.quantum_executable_test import _get_example_spec


@dataclass
class _MockEngineProcessor(AbstractEngineProcessorShim):
    def get_device(self) -> cirq.Device:
        return cg.Sycamore23

    def get_sampler(self) -> cirq.Sampler:
        return cirq.ZerosSampler()

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, attribute_names=[], namespace='cirq.google.testing')


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


def _cg_read_json_gzip(fn):
    def _testing_resolver(cirq_type: str):
        if cirq_type == 'cirq.google.testing._MockEngineProcessor':
            return _MockEngineProcessor

    return cirq.read_json_gzip(fn, resolvers=[_testing_resolver] + cirq.DEFAULT_RESOLVERS)


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
