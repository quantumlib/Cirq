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

import pytest

import pandas as pd
import sympy

import cirq
import cirq.experiments.t2_decay_experiment as t2


def test_init_t2_decay_result():
    x_data = pd.DataFrame(
        columns=['delay_ns', 0, 1],
        index=range(2),
        data=[
            [100.0, 0, 10],
            [1000.0, 10, 0],
        ],
    )
    y_data = pd.DataFrame(
        columns=['delay_ns', 0, 1],
        index=range(2),
        data=[
            [100.0, 5, 5],
            [1000.0, 5, 5],
        ],
    )
    result = cirq.experiments.T2DecayResult(x_data, y_data)
    assert result

    bad_data = pd.DataFrame(
        columns=['delay_ms', 0, 1],
        index=range(2),
        data=[
            [100.0, 0, 10],
            [1000.0, 10, 0],
        ],
    )
    with pytest.raises(ValueError):
        cirq.experiments.T2DecayResult(bad_data, y_data)
    with pytest.raises(ValueError):
        cirq.experiments.T2DecayResult(x_data, bad_data)


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

    results = cirq.experiments.t2_decay(
        sampler=cirq.DensityMatrixSimulator(noise=_TimeDependentDecay()),
        qubit=cirq.GridQubit(0, 0),
        num_points=3,
        repetitions=10,
        max_delay=cirq.Duration(nanos=500),
    )
    results.plot_expectations()
    results.plot_bloch_vector()


def test_result_eq():
    example_data = pd.DataFrame(
        columns=['delay_ns', 0, 1],
        index=range(5),
        data=[
            [200.0, 0, 100],
            [400.0, 20, 80],
            [600.0, 40, 60],
            [800.0, 60, 40],
            [1000.0, 80, 20],
        ],
    )
    other_data = pd.DataFrame(
        columns=['delay_ns', 0, 1],
        index=range(5),
        data=[
            [200.0, 0, 100],
            [400.0, 19, 81],
            [600.0, 39, 61],
            [800.0, 59, 41],
            [1000.0, 79, 21],
        ],
    )
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.experiments.T2DecayResult(example_data, example_data))

    eq.add_equality_group(cirq.experiments.T2DecayResult(other_data, example_data))
    eq.add_equality_group(cirq.experiments.T2DecayResult(example_data, other_data))


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

    results = cirq.experiments.t2_decay(
        sampler=cirq.DensityMatrixSimulator(noise=_SuddenDecay()),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=500,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
    )

    assert (results.expectation_pauli_y['value'][0:2] == -1).all()
    assert (results.expectation_pauli_y['value'][2:] < 0.20).all()

    # X Should be approximately zero
    assert (abs(results.expectation_pauli_x['value']) < 0.20).all()


@pytest.mark.parametrize('experiment_type', [t2.ExperimentType.HAHN_ECHO, t2.ExperimentType.CPMG])
def test_spin_echo_cancels_out_constant_rate_phase(experiment_type):
    class _TimeDependentPhase(cirq.NoiseModel):
        def noisy_moment(self, moment, system_qubits):
            duration = max(
                (
                    op.gate.duration
                    for op in moment.operations
                    if isinstance(op.gate, cirq.WaitGate)
                ),
                default=cirq.Duration(nanos=1),
            )
            phase = duration.total_nanos() / 100.0
            yield (cirq.Y ** phase).on_each(system_qubits)
            yield moment

    pulses = [1] if experiment_type == t2.ExperimentType.CPMG else None
    results = cirq.experiments.t2_decay(
        sampler=cirq.DensityMatrixSimulator(noise=_TimeDependentPhase()),
        qubit=cirq.GridQubit(0, 0),
        num_points=10,
        repetitions=100,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
        num_pulses=pulses,
        experiment_type=experiment_type,
    )

    assert (results.expectation_pauli_y['value'] < -0.8).all()


@pytest.mark.parametrize(
    'experiment_type',
    [t2.ExperimentType.RAMSEY, t2.ExperimentType.HAHN_ECHO, t2.ExperimentType.CPMG],
)
def test_all_on_results(experiment_type):
    pulses = [1] if experiment_type == t2.ExperimentType.CPMG else None
    results = t2.t2_decay(
        sampler=cirq.Simulator(),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=500,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
        num_pulses=pulses,
        experiment_type=experiment_type,
    )

    assert (results.expectation_pauli_y['value'] == -1.0).all()

    # X Should be approximately zero
    assert (abs(results.expectation_pauli_x['value']) < 0.20).all()


@pytest.mark.parametrize(
    'experiment_type',
    [t2.ExperimentType.RAMSEY, t2.ExperimentType.HAHN_ECHO, t2.ExperimentType.CPMG],
)
def test_all_off_results(experiment_type):
    pulses = [1] if experiment_type == t2.ExperimentType.CPMG else None
    results = t2.t2_decay(
        sampler=cirq.DensityMatrixSimulator(noise=cirq.amplitude_damp(1)),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=10,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
        num_pulses=pulses,
        experiment_type=experiment_type,
    )
    assert results == cirq.experiments.T2DecayResult(
        x_basis_data=pd.DataFrame(
            columns=['delay_ns', 0, 1],
            index=range(4),
            data=[
                [100.0, 10, 0],
                [400.0, 10, 0],
                [700.0, 10, 0],
                [1000.0, 10, 0],
            ],
        ),
        y_basis_data=pd.DataFrame(
            columns=['delay_ns', 0, 1],
            index=range(4),
            data=[
                [100.0, 10, 0],
                [400.0, 10, 0],
                [700.0, 10, 0],
                [1000.0, 10, 0],
            ],
        ),
    )


@pytest.mark.parametrize(
    'experiment_type',
    [t2.ExperimentType.RAMSEY, t2.ExperimentType.HAHN_ECHO, t2.ExperimentType.CPMG],
)
def test_custom_delay_sweep(experiment_type):
    pulses = [1] if experiment_type == t2.ExperimentType.CPMG else None
    results = t2.t2_decay(
        sampler=cirq.DensityMatrixSimulator(noise=cirq.amplitude_damp(1)),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=10,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
        experiment_type=experiment_type,
        num_pulses=pulses,
        delay_sweep=cirq.Points('delay_ns', [1.0, 10.0, 100.0, 1000.0, 10000.0]),
    )
    assert results == cirq.experiments.T2DecayResult(
        x_basis_data=pd.DataFrame(
            columns=['delay_ns', 0, 1],
            index=range(5),
            data=[
                [1.0, 10, 0],
                [10.0, 10, 0],
                [100.0, 10, 0],
                [1000.0, 10, 0],
                [10000.0, 10, 0],
            ],
        ),
        y_basis_data=pd.DataFrame(
            columns=['delay_ns', 0, 1],
            index=range(5),
            data=[
                [1.0, 10, 0],
                [10.0, 10, 0],
                [100.0, 10, 0],
                [1000.0, 10, 0],
                [10000.0, 10, 0],
            ],
        ),
    )


def test_multiple_pulses():
    results = t2.t2_decay(
        sampler=cirq.DensityMatrixSimulator(noise=cirq.amplitude_damp(1)),
        qubit=cirq.GridQubit(0, 0),
        num_points=4,
        repetitions=10,
        min_delay=cirq.Duration(nanos=100),
        max_delay=cirq.Duration(micros=1),
        experiment_type=t2.ExperimentType.CPMG,
        num_pulses=[1, 2, 3, 4],
        delay_sweep=cirq.Points('delay_ns', [1.0, 10.0, 100.0, 1000.0, 10000.0]),
    )
    data = [
        [1.0, 1, 10, 0],
        [1.0, 2, 10, 0],
        [1.0, 3, 10, 0],
        [1.0, 4, 10, 0],
        [10.0, 1, 10, 0],
        [10.0, 2, 10, 0],
        [10.0, 3, 10, 0],
        [10.0, 4, 10, 0],
        [100.0, 1, 10, 0],
        [100.0, 2, 10, 0],
        [100.0, 3, 10, 0],
        [100.0, 4, 10, 0],
        [1000.0, 1, 10, 0],
        [1000.0, 2, 10, 0],
        [1000.0, 3, 10, 0],
        [1000.0, 4, 10, 0],
        [10000.0, 1, 10, 0],
        [10000.0, 2, 10, 0],
        [10000.0, 3, 10, 0],
        [10000.0, 4, 10, 0],
    ]
    assert results == cirq.experiments.T2DecayResult(
        x_basis_data=pd.DataFrame(
            columns=['delay_ns', 'num_pulses', 0, 1],
            index=range(20),
            data=data,
        ),
        y_basis_data=pd.DataFrame(
            columns=['delay_ns', 'num_pulses', 0, 1],
            index=range(20),
            data=data,
        ),
    )
    expected = pd.DataFrame(
        columns=['delay_ns', 'num_pulses', 'value'],
        index=range(20),
        data=[
            [1.0, 1, -1.0],
            [1.0, 2, -1.0],
            [1.0, 3, -1.0],
            [1.0, 4, -1.0],
            [10.0, 1, -1.0],
            [10.0, 2, -1.0],
            [10.0, 3, -1.0],
            [10.0, 4, -1.0],
            [100.0, 1, -1.0],
            [100.0, 2, -1.0],
            [100.0, 3, -1.0],
            [100.0, 4, -1.0],
            [1000.0, 1, -1.0],
            [1000.0, 2, -1.0],
            [1000.0, 3, -1.0],
            [1000.0, 4, -1.0],
            [10000.0, 1, -1.0],
            [10000.0, 2, -1.0],
            [10000.0, 3, -1.0],
            [10000.0, 4, -1.0],
        ],
    )
    assert results.expectation_pauli_x.equals(expected)


def test_bad_args():
    with pytest.raises(ValueError, match='repetitions <= 0'):
        _ = cirq.experiments.t2_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=0,
            max_delay=cirq.Duration(micros=1),
        )

    with pytest.raises(ValueError, match='max_delay < min_delay'):
        _ = cirq.experiments.t2_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=10,
            min_delay=cirq.Duration(micros=1),
            max_delay=cirq.Duration(micros=0),
        )

    with pytest.raises(ValueError, match='min_delay < 0'):
        _ = cirq.experiments.t2_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=10,
            max_delay=cirq.Duration(micros=1),
            min_delay=cirq.Duration(micros=-1),
        )

    with pytest.raises(ValueError, match='CPMG'):
        _ = cirq.experiments.t2_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=100,
            num_pulses=[1, 2, 3, 4],
            max_delay=cirq.Duration(micros=1),
            experiment_type=t2.ExperimentType.RAMSEY,
        )

    with pytest.raises(ValueError, match='num_pulses'):
        _ = cirq.experiments.t2_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=100,
            max_delay=cirq.Duration(micros=1),
            experiment_type=t2.ExperimentType.CPMG,
        )

    with pytest.raises(ValueError, match='delay_ns'):
        _ = cirq.experiments.t2_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=10,
            max_delay=cirq.Duration(micros=10),
            min_delay=cirq.Duration(micros=1),
            delay_sweep=cirq.Linspace(sympy.Symbol('t'), start=10, stop=2000, length=10),
        )
    sweep1 = cirq.Linspace(sympy.Symbol('delay_ns'), start=10, stop=100, length=10)
    sweep2 = cirq.Linspace(sympy.Symbol('t'), start=20, stop=200, length=10)
    product = cirq.Product(sweep1, sweep2)
    with pytest.raises(ValueError, match='delay_ns'):
        _ = cirq.experiments.t2_decay(
            sampler=cirq.Simulator(),
            qubit=cirq.GridQubit(0, 0),
            num_points=4,
            repetitions=10,
            max_delay=cirq.Duration(micros=10),
            min_delay=cirq.Duration(micros=1),
            delay_sweep=product,
        )


def test_cpmg_circuit():
    """Tests sub-component to make sure CPMG circuit is generated correctly."""
    q = cirq.GridQubit(1, 1)
    t = sympy.Symbol('t')
    circuit = t2._cpmg_circuit(q, t, 2)
    expected = cirq.Circuit(
        cirq.Y(q) ** 0.5,
        cirq.wait(q, nanos=t),
        cirq.X(q),
        cirq.wait(q, nanos=2 * t * sympy.Symbol('pulse_0')),
        cirq.X(q) ** sympy.Symbol('pulse_0'),
        cirq.wait(q, nanos=2 * t * sympy.Symbol('pulse_1')),
        cirq.X(q) ** sympy.Symbol('pulse_1'),
        cirq.wait(q, nanos=t),
    )
    assert circuit == expected


def test_cpmg_sweep():
    sweep = t2._cpmg_sweep([1, 3, 5])
    expected = cirq.Zip(
        cirq.Points('pulse_0', [1, 1, 1]),
        cirq.Points('pulse_1', [0, 1, 1]),
        cirq.Points('pulse_2', [0, 1, 1]),
        cirq.Points('pulse_3', [0, 0, 1]),
        cirq.Points('pulse_4', [0, 0, 1]),
    )
    assert sweep == expected


def test_str():
    x_data = pd.DataFrame(
        columns=['delay_ns', 0, 1],
        index=range(2),
        data=[
            [100.0, 0, 10],
            [1000.0, 10, 0],
        ],
    )
    y_data = pd.DataFrame(
        columns=['delay_ns', 0, 1],
        index=range(2),
        data=[
            [100.0, 5, 5],
            [1000.0, 5, 5],
        ],
    )
    result = cirq.experiments.T2DecayResult(x_data, y_data)

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
    assert p.text_pretty == 'T2DecayResult(...)'
