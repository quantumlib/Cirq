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
import cirq_google.workflow as cgw
from cirq_google.workflow._abstract_engine_processor_shim import AbstractEngineProcessorShim
from cirq_google.workflow.quantum_executable_test import _get_quantum_executables


class _MockEngineProcessor(AbstractEngineProcessorShim):

    def get_device(self) -> cirq.Device:
        return cg.Sycamore23

    def get_sampler(self) -> cirq.Sampler:
        return cirq.ZerosSampler()

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, attribute_names=[], namespace='cirq.google.testing')


def test_execute(tmpdir):
    rt_config = cgw.QuantumRuntimeConfiguration(
        processor=_MockEngineProcessor(),
        run_id='unittests',
    )
    executable_group = cgw.QuantumExecutableGroup(_get_quantum_executables())
    cgw.execute(rt_config=rt_config, executable_group=executable_group, base_data_dir=tmpdir)
