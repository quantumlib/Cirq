# Copyright 2019 The Cirq Developers
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

from unittest import mock

import cirq
import cirq.google as cg


def test_run_circuit():
    engine = mock.Mock()
    sampler = cg.QuantumEngineSampler(engine=engine,
                                      processor_id='tmp',
                                      gate_set=cg.XMON)
    c = cirq.Circuit()
    params = [cirq.ParamResolver({'a': 1})]
    sampler.run_sweep(c, params, 5)
    engine.run_sweep.assert_called_with(gate_set=cg.XMON,
                                        params=params,
                                        processor_ids=['tmp'],
                                        program=c,
                                        repetitions=5)
