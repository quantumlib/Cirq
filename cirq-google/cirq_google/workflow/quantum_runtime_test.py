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
from cirq_google.workflow.quantum_executable_test import _get_example_spec


def cg_assert_equivalent_repr(value):
    """cirq.testing.assert_equivalent_repr with cirq_google.workflow imported."""
    return cirq.testing.assert_equivalent_repr(
        value,
        global_vals={
            'cirq_google': cg,
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
