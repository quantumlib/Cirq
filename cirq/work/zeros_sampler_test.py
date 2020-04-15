# Copyright 2020 The Cirq Developers
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
import sympy
from cirq import study

import cirq


def test_run_sweep():
    a = cirq.NamedQubit('a')
    c = cirq.Circuit([cirq.measure(a)])
    sampler = cirq.ZerosSampler()

    result = sampler.run_sweep(c, None, 2)

    assert len(result) == 1
    assert result[0].measurements.keys() == {'a'}
    assert np.all(result[0].measurements['a'] == [[False], [False]])


def test_sample():
    # Create a circuit whose measurements are always zeros, and check that
    # results of ZeroSampler on this circuit are identical to results of
    # actual simulation.
    qs = cirq.LineQubit.range(4)
    c = cirq.Circuit([cirq.CNOT(qs[0], qs[1]), cirq.X(qs[2]), cirq.X(qs[2])])
    c += cirq.Z(qs[3])**sympy.Symbol('p')
    c += [cirq.measure(q) for q in qs]
    # Z to even power is an identity.
    params = study.Points(sympy.Symbol('p'), [0, 2, 4, 6])

    result1 = cirq.ZerosSampler().sample(c, repetitions=10,
                                         params=params).sort_index(axis=1)
    result2 = cirq.Simulator().sample(c, repetitions=10,
                                      params=params).sort_index(axis=1)

    assert np.all(result1 == result2)
