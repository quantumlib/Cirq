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

from typing import Any, TYPE_CHECKING

import pandas as pd
import sympy
from matplotlib import pyplot as plt

from cirq import circuits, devices, ops, study, work, value
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


def t1_decay(sampler: work.Sampler,
             *,
             qubit: devices.GridQubit,
             num_points: int,
             max_delay: 'cirq.DURATION_LIKE',
             min_delay: 'cirq.DURATION_LIKE' = None,
             repetitions: int = 1000) -> 'cirq.experiments.T1DecayResult':
    """Runs a t1 decay experiment.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        num_points: The number of evenly spaced delays to test.
        max_delay: The largest delay to test.
        min_delay: The smallest delay to test. Defaults to no delay.
        repetitions: The number of repetitions of the circuit for each delay.

    Returns:
        A T1DecayResult object that stores and can plot the data.
    """
    var = sympy.Symbol('delay_ns')

    sweep = study.Linspace(var,
                           start=value.Duration(min_delay).total_nanos(),
                           stop=value.Duration(max_delay).total_nanos(),
                           length=num_points)

    circuit = circuits.Circuit(
        ops.X(qubit),
        ops.WaitGate(value.Duration(nanos=var)).on(qubit),
        ops.measure(qubit, key='output'),
    )

    results = sampler.sample(circuit, params=sweep, repetitions=repetitions)

    # Count by combined (delay_ns, output) pairs.
    pair_counts = results.data.groupby(['delay_ns', 'output']).size()

    # Filter false counts and true counts into separate data frames.
    df = pair_counts.to_frame('count').reset_index('output')
    output = df['output']
    false_counts = df[output == 0]['count'].to_frame('false_count')
    true_counts = df[output == 1]['count'].to_frame('true_count')

    # Merge into a common table with delay_ns, false_count, true_count cols.
    merged = false_counts.join(true_counts, how='outer').reset_index()
    filled = merged.fillna(0).astype({
        'false_count': 'int64',
        'true_count': 'int64'
    })
    return T1DecayResult(filled)


class T1DecayResult:
    """Results from a Rabi oscillation experiment."""

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: A data frame with three columns:
                delay_ns, false_count, true_count.
        """
        assert list(data.columns) == ['delay_ns', 'false_count', 'true_count']
        self._data = data

    @property
    def data(self) -> pd.DataFrame:
        """A data frame with delay_ns, false_count, true_count columns."""
        return self._data

    def plot(self, show: bool = False, **plot_kwargs: Any) -> None:
        """Plots excited state probability vs the Rabi angle (angle of rotation
        around the x-axis).

        Args:
            show: If set to True, `matplotlib.pyplot.show()` is called.
            **plot_kwargs: Arguments to be passed to matplotlib.pyplot.plot.
        """
        fig = plt.figure()
        ax = plt.gca()
        ax.set_ylim([0, 1])

        xs = self._data['delay_ns']
        ts = self._data['true_count']
        fs = self._data['false_count']

        plt.plot(xs, ts / (fs + ts), 'ro-', figure=fig, **plot_kwargs)
        plt.xlabel(
            r"Delay between initialization and measurement (nanoseconds)",
            figure=fig)
        plt.ylabel('Excited State Probability', figure=fig)
        plt.title('T1 Decay Experiment Data')
        if show:
            plt.show()

    def __str__(self):
        return f'T1DecayResult with data:\n{self.data}'

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.data.equals(other.data)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return f'cirq.experiments.T1DecayResult(data={proper_repr(self.data)})'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('T1DecayResult(...)')
        else:
            p.text(str(self))
