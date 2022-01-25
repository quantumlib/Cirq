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
from cirq_google.workflow.progress import _PrintLogger


def test_print_logger(capsys):
    pl = _PrintLogger(n_total=10)
    shared_rt_info = cg.SharedRuntimeInfo(run_id='hi mom')
    pl.initialize()
    for i in range(10):
        exe_result = cg.ExecutableResult(
            spec=None,
            runtime_info=cg.RuntimeInfo(execution_index=i),
            raw_data=cirq.ResultDict(params=cirq.ParamResolver({}), measurements={}),
        )
        pl.consume_result(exe_result, shared_rt_info)
    pl.finalize()
    assert capsys.readouterr().out == (
        '\n\r1 / 10'
        '\r2 / 10'
        '\r3 / 10'
        '\r4 / 10'
        '\r5 / 10'
        '\r6 / 10'
        '\r7 / 10'
        '\r8 / 10'
        '\r9 / 10'
        '\r10 / 10\n'
    )
