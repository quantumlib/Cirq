# Copyright 2018 Google LLC
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

"""Tests for studies."""

from cirq.circuits import Circuit
from cirq.study import Study
from cirq.google.parameterized_value import ParameterizedValue
from cirq.google.resolver import ParamResolver
from cirq.google.xmon_gates import (ExpWGate, Exp11Gate, ExpZGate,
                                    XmonMeasurementGate)
from cirq.google.xmon_qubit import XmonQubit
from cirq.sim.google.xmon_simulator import Result, Simulator


def bit_flip_circuit(flip0, flip1):
    q1, q2 = XmonQubit(0, 0), XmonQubit(0, 1)
    g1, g2 = ExpWGate(half_turns=flip0)(q1), ExpWGate(half_turns=flip1)(q2)
    m1, m2 = XmonMeasurementGate('q1')(q1), XmonMeasurementGate('q2')(q2)
    circuit = Circuit()
    circuit.append([g1, g2, m1, m2])
    return circuit


def test_study_repetitions():
    sim = Simulator()
    circuit = bit_flip_circuit(1, 1)

    study = Study(executor=sim, program=circuit, repetitions=10)
    full_results = study.run_study()
    assert len(full_results) == 1
    repetition_results = full_results[0]
    assert len(repetition_results) == 10
    for result in repetition_results:
        assert isinstance(result, Result)
        assert result.measurements == {'q1': [True], 'q2': [True]}
        assert not result.param_dict


def test_study_parameters():
    sim = Simulator()
    circuit = bit_flip_circuit(ParameterizedValue('a'), ParameterizedValue('b'))

    resolvers = [ParamResolver({'a': b1, 'b': b2})
                 for b1 in range(2) for b2 in range(2)]

    study = Study(executor=sim, program=circuit, param_resolvers=resolvers,
                  repetitions=1)
    full_results = study.run_study()
    assert len(full_results) == 4
    for repetition_results in full_results:
        assert len(repetition_results) == 1
        result = repetition_results[0]
        expected = {'q1': [result.param_dict['a'] == 1],
                    'q2': [result.param_dict['b'] == 1]}
        assert expected == result.measurements


def test_study_param_and_reps():
    sim = Simulator()
    circuit = bit_flip_circuit(ParameterizedValue('a'), ParameterizedValue('b'))

    resolvers = [ParamResolver({'a': b1, 'b': b2})
                 for b1 in range(2) for b2 in range(2)]

    study = Study(executor=sim, program=circuit, param_resolvers=resolvers,
                  repetitions=3)
    full_results = study.run_study()
    assert len(full_results) == 4
    for repetition_results in full_results:
        assert len(repetition_results) == 3
        for result in repetition_results:
            expected = {'q1': [result.param_dict['a'] == 1],
                        'q2': [result.param_dict['b'] == 1]}
            assert expected == result.measurements

