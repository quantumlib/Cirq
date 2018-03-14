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

import itertools

import collections
import pytest

from cirq import study
from cirq.circuits import Circuit
from cirq.google import Simulator, XmonQubit, ExpWGate, XmonMeasurementGate
from cirq.study import ExecutorStudy
from cirq.study.parameterized_value import ParameterizedValue
from cirq.study.resolver import ParamResolver


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

    study = ExecutorStudy(executor=sim, program=circuit, repetitions=10)
    all_trials = study.run_study()
    assert len(all_trials) == 10
    for context, result in all_trials:
        assert context.param_dict == {}
        assert result.measurements == {'q1': [True], 'q2': [True]}
    # All repetition ids are used.
    assert set(range(10)) == {c.repetition_id for c,_ in all_trials}


def test_study_parameters():
    sim = Simulator()
    circuit = bit_flip_circuit(ParameterizedValue('a'), ParameterizedValue('b'))

    resolvers = [ParamResolver({'a': b1, 'b': b2})
                 for b1 in range(2) for b2 in range(2)]

    study = ExecutorStudy(executor=sim, program=circuit,
                          param_resolvers=resolvers,
                          repetitions=1)
    all_trials = study.run_study()
    assert len(all_trials) == 4
    for context, result in all_trials:
        expected = {'q1': [context.param_dict['a'] == 1],
                    'q2': [context.param_dict['b'] == 1]}
        assert expected == result.measurements
    # All parameters explored.
    assert (set(itertools.product([0, 1], [0, 1]))
            == {(c.param_dict['a'], c.param_dict['b']) for c, _ in all_trials})
    # And they always have a single repetition.
    assert 4 * [0] == [c.repetition_id for c,_ in all_trials]


def test_study_param_and_reps():
    sim = Simulator()
    circuit = bit_flip_circuit(ParameterizedValue('a'), ParameterizedValue('b'))

    resolvers = [ParamResolver({'a': b1, 'b': b2})
                 for b1 in range(2) for b2 in range(2)]

    study = ExecutorStudy(executor=sim, program=circuit,
                          param_resolvers=resolvers,
                          repetitions=3)
    all_trials = study.run_study()
    assert len(all_trials) == 3 * 4
    for context, result in all_trials:
        expected = {'q1': [context.param_dict['a'] == 1],
                    'q2': [context.param_dict['b'] == 1]}
        assert expected == result.measurements
    # All parameters explored.
    comb = list(itertools.product([0, 1], [0, 1]))
    # And we see the results exactly as many times as expected.
    expected = collections.Counter(comb * 3)
    assert expected == collections.Counter(
        (c.param_dict['a'], c.param_dict['b']) for c, _ in all_trials)



def test_study_executor_kwargs():
    sim = Simulator()
    circuit = bit_flip_circuit(1, 1)

    study = ExecutorStudy(executor=sim, program=circuit, repetitions=1,
                          initial_state=3)
    all_trials = study.run_study()
    assert len(all_trials ) == 1
    _, result = all_trials[0]
    assert result.measurements == {'q1': [False], 'q2': [False]}



class BadResult(study.TrialResult):
    pass


def test_bad_result():
    with pytest.raises(NotImplementedError):
        BadResult()


class BadContext(study.TrialContext):
    param_dict = {}


def test_context_missing_repetitions():
    with pytest.raises(NotImplementedError):
        BadContext()


class EvenWorseContext(study.TrialContext):
    pass


def test_context_missing_all_attributes():
    with pytest.raises(NotImplementedError):
        EvenWorseContext()
