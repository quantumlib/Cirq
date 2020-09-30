# Copyright 2020 The Cirq developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import cirq
from cirq.work.observable_measurement import measure_observables_df, _with_parameterized_layers


def test_with_parameterized_layers():
    qs = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        [
            cirq.H.on_each(*qs),
            cirq.CZ(qs[0], qs[1]),
            cirq.CZ(qs[1], qs[2]),
        ]
    )
    circuit2 = _with_parameterized_layers(circuit, qubits=qs, no_initialization=True)
    assert circuit != circuit2
    assert len(circuit2) == 3 + 3  # 3 original, then X, Y, measure layer

    circuit3 = _with_parameterized_layers(circuit, qubits=qs, no_initialization=False)
    assert circuit != circuit3
    assert circuit2 != circuit3
    assert len(circuit3) == 2 + 3 + 3


def test_Z():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit([cirq.X(q) ** 0.2])
    observable = 1 * cirq.Z(q)
    df = measure_observables_df(
        circuit,
        [observable],
        cirq.Simulator(seed=52),
        stopping_criteria='variance',
        stopping_criteria_val=1e-3 ** 2,
    )
    mean = df.loc[0]['mean']
    np.testing.assert_allclose(0.8, mean, atol=1e-2)


def test_X():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(
        [
            cirq.Y(q) ** 0.5,
            cirq.Z(q) ** 0.2,
        ]
    )
    observable = cirq.X(q)
    df = measure_observables_df(
        circuit,
        [observable],
        cirq.Simulator(seed=52),
        stopping_criteria='variance',
        stopping_criteria_val=1e-3 ** 2,
    )
    mean = df.loc[0]['mean']
    np.testing.assert_allclose(0.8, mean, atol=1e-2)


def test_Y():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(
        [
            cirq.X(q) ** -0.5,
            cirq.Z(q) ** 0.2,
        ]
    )
    observable = cirq.Y(q)
    df = measure_observables_df(
        circuit,
        [observable],
        cirq.Simulator(seed=52),
        stopping_criteria='variance',
        stopping_criteria_val=1e-3 ** 2,
    )
    mean = df.loc[0]['mean']
    np.testing.assert_allclose(0.8, mean, atol=1e-2)
