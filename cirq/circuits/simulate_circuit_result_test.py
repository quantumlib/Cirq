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

import itertools
import numpy as np

import cirq


def test_init():
    measurements = {'a': np.array([True])}
    final_state = np.array([0, 1])
    c = cirq.SimulateCircuitResult(measurements=measurements,
                                   final_state=final_state)
    assert c.measurements is measurements
    assert c.final_state is final_state


def test_str():
    empty = cirq.SimulateCircuitResult({}, np.array([1]))
    assert str(empty) == 'measurements: (none)\nfinal_state: [1]'

    multi = cirq.SimulateCircuitResult(
        {'a': np.array([True, True]), 'b': np.array([False])},
        np.array([0, 1, 0, 0]))
    assert str(multi) == 'measurements: a=11 b=0\nfinal_state: [0 1 0 0]'


def test_repr():
    multi = cirq.SimulateCircuitResult(
        {'a': np.array([True, True]), 'b': np.array([False])},
        np.array([0, 1, 0, 0]))
    assert repr(multi) == ("SimulateCircuitResult(measurements={"
                           "'a': array([ True,  True]), "
                           "'b': array([False])}, "
                           "final_state=array([0, 1, 0, 0]))")


def test_approx_eq():
    empty = cirq.SimulateCircuitResult({}, np.array([1]))

    assert empty.approx_eq(empty, atol=1e-2)
    assert empty.approx_eq(empty, atol=1e-6)
    assert empty.approx_eq(empty, atol=1e-2, ignore_global_phase=False)
    assert empty.approx_eq(empty, atol=1e-6, ignore_global_phase=False)

    empty_neg = cirq.SimulateCircuitResult({}, np.array([-1]))
    assert empty.approx_eq(empty_neg, atol=1e-2)
    assert empty.approx_eq(empty_neg, atol=1e-6)
    assert not empty.approx_eq(empty_neg, atol=1e-2, ignore_global_phase=False)
    assert not empty.approx_eq(empty_neg, atol=1e-6, ignore_global_phase=False)

    empty_near = cirq.SimulateCircuitResult({}, np.array([0.999]))
    assert empty.approx_eq(empty_near, atol=1e-2)
    assert not empty.approx_eq(empty_near, atol=1e-6)
    assert empty.approx_eq(empty_near, atol=1e-2, ignore_global_phase=False)
    assert not empty.approx_eq(empty_near,
                               atol=1e-6,
                               ignore_global_phase=False)

    just_on = cirq.SimulateCircuitResult({},
                                         np.array([0, 1]))
    near_on = cirq.SimulateCircuitResult({},
                                         np.array([0, 0.999]))
    just_off = cirq.SimulateCircuitResult({},
                                         np.array([1, 0]))
    collapse_on = cirq.SimulateCircuitResult({'a': np.array([True])},
                                             np.array([0, 1]))
    collapse_off = cirq.SimulateCircuitResult({'a': np.array([False])},
                                              np.array([1, 0]))
    other_collapse_on = cirq.SimulateCircuitResult({'b': np.array([True])},
                                                   np.array([0, 1]))
    multi_collapse_on = cirq.SimulateCircuitResult(
        {'a': np.array([True, True])},
        np.array([0, 1]))

    approx_eq_groups = [
        [just_on, near_on],
        [just_off],
        [collapse_on],
        [collapse_off],
        [other_collapse_on],
        [multi_collapse_on],
    ]
    for g1, g2 in itertools.product(approx_eq_groups, repeat=2):
        for e1, e2 in itertools.product(g1, g2):
            for g in [False, True]:
                assert (g1 is g2) == e1.approx_eq(e2,
                                                  atol=1e-2,
                                                  ignore_global_phase=g)
