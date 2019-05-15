# Copyright 2018 The Cirq Developers
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

import numpy as np

import cirq

t1 = cirq.TrialResult(
            params=cirq.ParamResolver({'a': 2}),
            repetitions=2,
            measurements={'m': np.array([[1, 2]])})
t2 = cirq.TrialResult(
            params=cirq.ParamResolver({'b': 3}),
            repetitions=2,
            measurements={'m': np.array([[1, 2]])})
t3 = cirq.TrialResult(
            params=cirq.ParamResolver({'a': 2, 'b': 3}),
            repetitions=2,
            measurements={'m': np.array([[1, 2]])})


def test_empty():
    empty = cirq.SweepTrialResult()
    assert len(empty.trial_results) == 0
    assert len(empty.trials_where_params_match({})) == 0


def test_repr():
    v = cirq.SweepTrialResult([t1, t2])

    assert repr(v) == ("cirq.SweepTrialResult(trial_results=["
                       "cirq.TrialResult(params=cirq.ParamResolver({'a': 2}), "
                       "repetitions=2, measurements={'m': array([[1, 2]])}), "
                       "cirq.TrialResult(params=cirq.ParamResolver({'b': 3}), "
                       "repetitions=2, measurements={'m': array([[1, 2]])})])")


def test_str():
    result1 = cirq.TrialResult(
        params=cirq.ParamResolver({}),
        repetitions=5,
        measurements={
            'ab': np.array([[0, 1],
                            [0, 1],
                            [0, 1],
                            [1, 0],
                            [0, 1]]),
            'c': np.array([[0], [0], [1], [0], [1]])
        })
    result2 = cirq.TrialResult(
        params=cirq.ParamResolver({}),
        repetitions=5,
        measurements={
            'ab': np.array([[1, 1],
                            [0, 0],
                            [1, 1],
                            [0, 1],
                            [0, 0]]),
            'c': np.array([[1], [1], [0], [0], [0]])
        })
    v = cirq.SweepTrialResult([result1, result2])
    assert str(v) == '[{ab=00010, 11101\nc=00101}, {ab=10100, 10110\nc=11000}]'


def test_trial_result_equality():
    et = cirq.testing.EqualsTester()
    et.add_equality_group(cirq.SweepTrialResult(),
        cirq.SweepTrialResult([t1, t2, t3]).slice_where_params_match({'a': 3}),
        cirq.SweepTrialResult([t1, t2]).slice_where_params_match({'a': 2,
                                                                  'b': 3}))
    et.add_equality_group(cirq.SweepTrialResult([t1]))
    et.add_equality_group(cirq.SweepTrialResult([t2]))
    et.add_equality_group(cirq.SweepTrialResult([t3]),
        cirq.SweepTrialResult([t1, t2, t3]).slice_where_params_match({'a': 2,
                                                                      'b': 3}),
        cirq.SweepTrialResult([t3]).slice_where_params_match({}))
    et.add_equality_group(cirq.SweepTrialResult([t1, t2]),
        cirq.SweepTrialResult([t2, t1]))
    et.add_equality_group(cirq.SweepTrialResult([t2, t3]),
        cirq.SweepTrialResult([t3, t2]),
        cirq.SweepTrialResult([t1, t2, t3]).slice_where_params_match({'b': 3}))
    et.add_equality_group(cirq.SweepTrialResult([t1, t3]),
        cirq.SweepTrialResult([t3, t1]),
        cirq.SweepTrialResult([t1, t2, t3]).slice_where_params_match({'a': 2}))
    et.add_equality_group(cirq.SweepTrialResult([t1, t2, t3]),
        cirq.SweepTrialResult([t3, t1, t2]),
        cirq.SweepTrialResult([t2, t1, t3]))
    et.add_equality_group(cirq.SweepTrialResult([t1, t1, t2, t3]))

    et.add_equality_group(set([t1, t3]),
        set(cirq.SweepTrialResult(
                [t1, t2, t3]).trials_where_params_match({'a': 2})))


def test_text_diagram_jupyter():
    result1 = cirq.TrialResult(
        params=cirq.ParamResolver({}),
        repetitions=5,
        measurements={
            'ab': np.array([[0, 1],
                            [0, 1],
                            [0, 1],
                            [1, 0],
                            [0, 1]]),
            'c': np.array([[0], [0], [1], [0], [1]])
        })
    result2 = cirq.TrialResult(
        params=cirq.ParamResolver({}),
        repetitions=5,
        measurements={
            'ab': np.array([[1, 1],
                            [0, 0],
                            [1, 1],
                            [0, 1],
                            [0, 0]]),
            'c': np.array([[1], [1], [0], [0], [0]])
        })
    v = cirq.SweepTrialResult([result1, result2])

    # Test Jupyter console output from
    class FakePrinter:
        def __init__(self):
            self.text_pretty = ''
        def text(self, to_print):
            self.text_pretty += to_print
    p = FakePrinter()
    v._repr_pretty_(p, False)
    assert p.text_pretty == ('[{ab=00010, 11101\nc=00101}, '
                             '{ab=10100, 10110\nc=11000}]')

    # Test cycle handling
    p = FakePrinter()
    v._repr_pretty_(p, True)
    assert p.text_pretty == 'SweepTrialResult(...)'
