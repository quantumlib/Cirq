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

import numpy as np
import pytest

from cirq import study
from cirq.circuits import Circuit
from cirq.google import Simulator, XmonQubit, ExpWGate, XmonMeasurementGate
from cirq.study import ExecutorStudy
from cirq.study.resolver import ParamResolver
from cirq.value import Symbol


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
    assert len(all_trials) == 1
    for result in all_trials:
        assert result.params.param_dict == {}
        assert result.repetitions == 10
        np.testing.assert_equal(result.measurements['q1'], [[True]] * 10)
        np.testing.assert_equal(result.measurements['q2'], [[True]] * 10)


def test_study_parameters():
    sim = Simulator()
    circuit = bit_flip_circuit(Symbol('a'), Symbol('b'))

    resolvers = [ParamResolver({'a': b1, 'b': b2})
                 for b1 in range(2) for b2 in range(2)]

    study = ExecutorStudy(executor=sim, program=circuit,
                          param_resolvers=resolvers,
                          repetitions=1)
    all_trials = study.run_study()
    assert len(all_trials) == 4
    for result in all_trials:
        assert result.repetitions == 1
        expect_a = result.params['a'] == 1
        expect_b = result.params['b'] == 1
        np.testing.assert_equal(result.measurements['q1'], [[expect_a]])
        np.testing.assert_equal(result.measurements['q2'], [[expect_b]])
    # All parameters explored.
    assert (set(itertools.product([0, 1], [0, 1]))
            == {(r.params['a'], r.params['b']) for r in all_trials})


def test_study_param_and_reps():
    sim = Simulator()
    circuit = bit_flip_circuit(Symbol('a'), Symbol('b'))

    resolvers = [ParamResolver({'a': b1, 'b': b2})
                 for b1 in range(2) for b2 in range(2)]

    study = ExecutorStudy(executor=sim, program=circuit,
                          param_resolvers=resolvers,
                          repetitions=3)
    all_trials = study.run_study()
    assert len(all_trials) == 4
    for result in all_trials:
        assert result.repetitions == 3
        expect_a = result.params['a'] == 1
        expect_b = result.params['b'] == 1
        np.testing.assert_equal(result.measurements['q1'], [[expect_a]] * 3)
        np.testing.assert_equal(result.measurements['q2'], [[expect_b]] * 3)
    # All parameters explored.
    # All parameters explored.
    assert (set(itertools.product([0, 1], [0, 1]))
            == {(r.params['a'], r.params['b']) for r in all_trials})



def test_study_executor_kwargs():
    sim = Simulator()
    circuit = bit_flip_circuit(1, 1)

    study = ExecutorStudy(executor=sim, program=circuit, repetitions=1,
                          initial_state=3)
    all_trials = study.run_study()
    assert len(all_trials ) == 1
    result = all_trials[0]
    np.testing.assert_equal(result.measurements['q1'], [[False]])
    np.testing.assert_equal(result.measurements['q2'], [[False]])


class BadResult(study.TrialResult):
    pass


def test_bad_result():
    with pytest.raises(NotImplementedError):
        BadResult()
