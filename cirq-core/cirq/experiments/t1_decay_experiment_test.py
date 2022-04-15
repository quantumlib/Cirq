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

import pytest

import numpy as np
import pandas as pd
import sympy

import cirq


def test_init_result():
    data = pd.DataFrame(
        columns=['delay_ns', 'false_count', 'true_count'],
        index=range(2),
        data=[[100.0, 0, 10], [1000.0, 10, 0]],
    )
    result = cirq.experiments.T1DecayResult(data)
    assert result.data is data


@pytest.mark.usefixtures('closefigures')
def test_plot_does_not_raise_error():
    class _TimeDependentDecay(cirq.NoiseModel):
        def noisy_moment(self, moment, system_qubits):
            duration = max(
                (
                    op.gate.duration
                    for op in moment.operations
                    if isinstance(op.gate, cirq.WaitGate)
                ),
                default=cirq.Duration(nanos=1),
            )
            yield cirq.amplitude_damp(1 - 0.99 ** duration.total_nanos()).on_each(system_qubits)
            yield moment

    results = cirq.experiments.t1_decay(
        sampler=cirq.DensityMatrixSimulator(noise=_TimeDependentDecay()),
        qubit=cirq.GridQubit(0, 0),
        num_points=3,
        repetitions=10,
        max_delay=cirq.Duration(nanos=500),
    )
    results.plot()


def test_result_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(
        lambda: cirq.experiments.T1DecayResult(
            data=pd.DataFrame(
                columns=['delay_ns', 'false_count', 'true_count'], index=[0], data=[[100.0, 2, 8]]
            )
        )
    )
    eq.add_equality_group(
        cirq.experiments.T1DecayResult(
            data=pd.DataFrame(
                columns=['delay_ns', 'false_count', 'true_count'],
                index=[0],
                data=[[100.0, 2, 50002]],
            )
        )
    )


def test_sudden_decay_results():
    class _SuddenDecay(cirq.NoiseModel):
        def noisy_moment(self, moment, system_qubits):
            duration = max(
                (
                    op.gate.duration
                    for op in moment.operations
                    if isinstance(op.gate, cirq.WaitGate)
                ),
                default=cirq.Duration(),
            )
            if duration > cirq.Duration(nanos=500):
                yield cirq.amplitude_damp(1).on_each(system_qubits)
            yield moment

    results = cirq.experiments.t1_decay(
        sampler=cirq.DensityMatrixSimulator(noise=_SuddenDecay()),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=10,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
    )
    assert results == cirq.experiments.T1DecayResult(
        data=pd.DataFrame(
            columns=['delay_ns', 'false_count', 'true_count'],
            index=range(4),
            data=[[100.0, 0, 10], [400.0, 0, 10], [700.0, 10, 0], [1000.0, 10, 0]],
        )
    )


def test_all_on_results():
    results = cirq.experiments.t1_decay(
        sampler=cirq.Simulator(),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=10,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
    )
    assert results == cirq.experiments.T1DecayResult(
        data=pd.DataFrame(
            columns=['delay_ns', 'false_count', 'true_count'],
            index=range(4),
            data=[[100.0, 0, 10], [400.0, 0, 10], [700.0, 0, 10], [1000.0, 0, 10]],
        )
    )


def test_all_off_results():
    results = cirq.experiments.t1_decay(
        sampler=cirq.DensityMatrixSimulator(noise=cirq.amplitude_damp(1)),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=10,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
    )
    assert results == cirq.experiments.T1DecayResult(
        data=pd.DataFrame(
            columns=['delay_ns', 'false_count', 'true_count'],
            index=range(4),
            data=[[100.0, 10, 0], [400.0, 10, 0], [700.0, 10, 0], [1000.0, 10, 0]],
        )
    )


@pytest.mark.usefixtures('closefigures')
def test_curve_fit_plot_works():
    good_fit = cirq.experiments.T1DecayResult(
        data=pd.DataFrame(
            columns=['delay_ns', 'false_count', 'true_count'],
            index=range(4),
            data=[[100.0, 6, 4], [400.0, 10, 0], [700.0, 10, 0], [1000.0, 10, 0]],
        )
    )

    good_fit.plot(include_fit=True)


@pytest.mark.usefixtures('closefigures')
def test_curve_fit_plot_warning():
    bad_fit = cirq.experiments.T1DecayResult(
        data=pd.DataFrame(
            columns=['delay_ns', 'false_count', 'true_count'],
            index=range(4),
            data=[[100.0, 10, 0], [400.0, 10, 0], [700.0, 10, 0], [1000.0, 10, 0]],
        )
    )

    with pytest.warns(RuntimeWarning, match='Optimal parameters could not be found for curve fit'):
        bad_fit.plot(include_fit=True)


@pytest.mark.parametrize('t1', [200, 500, 700])
def test_noise_model_continous(t1):
    class GradualDecay(cirq.NoiseModel):
        def __init__(self, t1: float):
            self.t1 = t1

        def noisy_moment(self, moment, system_qubits):
            duration = max(
                (
                    op.gate.duration
                    for op in moment.operations
                    if isinstance(op.gate, cirq.WaitGate)
                ),
                default=cirq.Duration(nanos=0),
            )
            if duration > cirq.Duration(nanos=0):
                # Found a wait gate in this moment.
                return cirq.amplitude_damp(1 - np.exp(-duration.total_nanos() / self.t1)).on_each(
                    system_qubits
                )
            return moment

    results = cirq.experiments.t1_decay(
        sampler=cirq.DensityMatrixSimulator(noise=GradualDecay(t1)),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=10,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
    )

    assert np.isclose(results.constant, t1, 50)


@pytest.mark.parametrize('gamma', [0.01, 0.05, 0.1])
def test_noise_model_discrete(gamma):
    results = cirq.experiments.t1_decay(
        sampler=cirq.DensityMatrixSimulator(
            noise=cirq.NoiseModel.from_noise_model_like(cirq.amplitude_damp(gamma))
        ),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=100,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
    )

    data = results.data
    probs = data['true_count'] / (data['true_count'] + data['false_count'])

    # Check that there is no decay in probability over time
    np.testing.assert_allclose(probs, np.mean(probs), atol=0.2)


def test_bad_args():
    with pytest.raises(ValueError, match='repetitions <= 0'):
        _ = cirq.experiments.t1_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=0,
            max_delay=cirq.Duration(micros=1),
        )

    with pytest.raises(ValueError, match='max_delay < min_delay'):
        _ = cirq.experiments.t1_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=10,
            min_delay=cirq.Duration(micros=1),
            max_delay=cirq.Duration(micros=0),
        )

    with pytest.raises(ValueError, match='min_delay < 0'):
        _ = cirq.experiments.t1_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=10,
            max_delay=cirq.Duration(micros=1),
            min_delay=cirq.Duration(micros=-1),
        )

    with pytest.raises(ValueError, match='sympy expressions'):
        _ = cirq.experiments.t1_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=10,
            max_delay=cirq.Duration(micros=sympy.Symbol('t')),
            min_delay=cirq.Duration(micros=sympy.Symbol('t')),
        )


def test_str():
    result = cirq.experiments.T1DecayResult(
        data=pd.DataFrame(
            columns=['delay_ns', 'false_count', 'true_count'],
            index=range(2),
            data=[[100.0, 0, 10], [1000.0, 10, 0]],
        )
    )

    assert str(result) == (
        'T1DecayResult with data:\n'
        '   delay_ns  false_count  true_count\n'
        '0     100.0            0          10\n'
        '1    1000.0           10           0'
    )
    cirq.testing.assert_equivalent_repr(result)

    class FakePrinter:
        def __init__(self):
            self.text_pretty = ''

        def text(self, to_print):
            self.text_pretty += to_print

    p = FakePrinter()
    result._repr_pretty_(p, False)
    assert p.text_pretty == str(result)

    p = FakePrinter()
    result._repr_pretty_(p, True)
    assert p.text_pretty == 'T1DecayResult(...)'
