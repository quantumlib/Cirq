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
    circuit = cirq.Circuit()
    params = [cirq.ParamResolver({'a': 1})]
    sampler.run_sweep(circuit, params, 5)
    engine.run_sweep.assert_called_with(gate_set=cg.XMON,
                                        params=params,
                                        processor_ids=['tmp'],
                                        program=circuit,
                                        repetitions=5)


def test_run_engine_program():
    engine = mock.Mock()
    sampler = cg.QuantumEngineSampler(engine=engine,
                                      processor_id='tmp',
                                      gate_set=cg.XMON)
    program = mock.Mock(spec=cg.EngineProgram)
    params = [cirq.ParamResolver({'a': 1})]
    sampler.run_sweep(program, params, 5)
    program.run_sweep.assert_called_with(params=params,
                                         processor_ids=['tmp'],
                                         repetitions=5)
    engine.run_sweep.assert_not_called()


def test_engine_sampler_engine_property():
    engine = mock.Mock()
    sampler = cg.QuantumEngineSampler(engine=engine,
                                      processor_id='tmp',
                                      gate_set=cg.XMON)
    assert sampler.engine is engine
