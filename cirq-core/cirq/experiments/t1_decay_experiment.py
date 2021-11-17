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

from typing import Any, Optional, TYPE_CHECKING

import warnings
import pandas as pd
import sympy
from matplotlib import pyplot as plt
import numpy as np


from cirq import circuits, ops, study, value, _import
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq

# We initialize optimize lazily, otherwise it slows global import speed.
optimize = _import.LazyLoader("optimize", globals(), "scipy.optimize")


def t1_decay(
    sampler: 'cirq.Sampler',
    *,
    qubit: 'cirq.Qid',
    num_points: int,
    max_delay: 'cirq.DURATION_LIKE',
    min_delay: 'cirq.DURATION_LIKE' = None,
    repetitions: int = 1000,
) -> 'cirq.experiments.T1DecayResult':
    """Runs a t1 decay experiment.

    Initializes a qubit into the |1⟩ state, waits for a variable amount of time,
    and measures the qubit. Plots how often the |1⟩ state is observed for each
    amount of waiting.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        num_points: The number of evenly spaced delays to test.
        max_delay: The largest delay to test.
        min_delay: The smallest delay to test. Defaults to no delay.
        repetitions: The number of repetitions of the circuit for each delay.

    Returns:
        A T1DecayResult object that stores and can plot the data.

    Raises:
        ValueError: If the supplied parameters are not valid: negative repetitions,
            max delay less than min, or min delay less than 0.
    """
    min_delay_dur = value.Duration(min_delay)
    max_delay_dur = value.Duration(max_delay)

    if repetitions <= 0:
        raise ValueError('repetitions <= 0')
    if max_delay_dur < min_delay_dur:
        raise ValueError('max_delay < min_delay')
    if min_delay_dur < 0:
        raise ValueError('min_delay < 0')
    var = sympy.Symbol('delay_ns')

    sweep = study.Linspace(
        var,
        start=min_delay_dur.total_nanos(),
        stop=max_delay_dur.total_nanos(),
        length=num_points,
    )

    circuit = circuits.Circuit(
        ops.X(qubit),
        ops.wait(qubit, nanos=var),
        ops.measure(qubit, key='output'),
    )

    results = sampler.sample(circuit, params=sweep, repetitions=repetitions)

    # Cross tabulate into a delay_ns, false_count, true_count table.
    tab = pd.crosstab(results.delay_ns, results.output)
    tab.rename_axis(None, axis="columns", inplace=True)
    tab = tab.rename(columns={0: 'false_count', 1: 'true_count'}).reset_index()
    for col_index, name in [(1, 'false_count'), (2, 'true_count')]:
        if name not in tab:
            tab.insert(col_index, name, [0] * tab.shape[0])

    return T1DecayResult(tab)


class T1DecayResult:
    """Results from a Rabi oscillation experiment."""

    def __init__(self, data: pd.DataFrame):
        """Inits T1DecayResult.

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

    @property
    def constant(self) -> float:
        """The t1 decay constant."""

        def exp_decay(x, t1):
            return np.exp(-x / t1)

        xs = self._data['delay_ns']
        ts = self._data['true_count']
        fs = self._data['false_count']
        probs = ts / (fs + ts)

        # Find the point closest to probability of 1/e
        guess_index = np.argmin(np.abs(probs - 1.0 / np.e))
        t1_guess = xs[guess_index]

        # Fit to exponential decay to find the t1 constant
        try:
            popt, _ = optimize.curve_fit(exp_decay, xs, probs, p0=[t1_guess])
            t1 = popt[0]
            return t1
        except RuntimeError:
            warnings.warn("Optimal parameters could not be found for curve fit", RuntimeWarning)
            return np.nan

    def plot(
        self, ax: Optional[plt.Axes] = None, include_fit: bool = False, **plot_kwargs: Any
    ) -> plt.Axes:
        """Plots the excited state probability vs the amount of delay.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            include_fit: boolean to include exponential decay fit on graph
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        assert ax is not None
        ax.set_ylim(ymin=0, ymax=1)

        xs = self._data['delay_ns']
        ts = self._data['true_count']
        fs = self._data['false_count']

        ax.plot(xs, ts / (fs + ts), 'ro-', **plot_kwargs)

        if include_fit and not np.isnan(self.constant):
            ax.plot(xs, np.exp(-xs / self.constant), label='curve fit')
            plt.legend()

        ax.set_xlabel(r"Delay between initialization and measurement (nanoseconds)")
        ax.set_ylabel('Excited State Probability')
        ax.set_title('T1 Decay Experiment Data')
        if show_plot:
            fig.show()
        return ax

    def __str__(self) -> str:
        return f'T1DecayResult with data:\n{self.data}'

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.data.equals(other.data)

    def __ne__(self, other) -> bool:
        return not self == other

    def __repr__(self) -> str:
        return f'cirq.experiments.T1DecayResult(data={proper_repr(self.data)})'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('T1DecayResult(...)')
        else:
            p.text(str(self))
