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
import enum

from typing import Any, Optional, TYPE_CHECKING

import pandas as pd
import sympy
from matplotlib import pyplot as plt

from cirq import circuits, devices, ops, study, value, work
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


class ExperimentType(enum.Enum):
    RAMSEY = 1  # Often denoted as t2*
    HAHN_ECHO = 2  # Spin echo or t2
    CPMG = 3  # Carr-Purcell-Meiboom-Gill sequence


_T2_COLUMNS = ['delay_ns', 0, 1]


def t2_decay(
        sampler: work.Sampler,
        *,
        qubit: devices.GridQubit,
        experiment_type: 'ExperimentType' = ExperimentType.RAMSEY,
        num_points: int,
        max_delay: 'cirq.DURATION_LIKE',
        min_delay: 'cirq.DURATION_LIKE' = None,
        repetitions: int = 1000,
        delay_sweep: Optional[study.Sweep] = None,
) -> 'cirq.experiments.T2DecayResult':
    """Runs a t2 transverse relaxation experiment.

    Initializes a qubit into a superposition state, evolves the system using
    rules determined by the experiment type and by the delay parameters,
    then rotates back for measurement.  This will measure the phase decoherence
    decay.  This experiment has three types of T2 metrics, each which measure
    a different slice of the noise spectrum.

    For the Ramsey experiment type (often denoted T2*), the state will be
    prepared with a square root Y gate (`cirq.Y ** 0.5`) and then waits for
    a variable amount of time.  After this time, it will do basic state
    tomography to measure the expectation of the Pauli-X and Pauli-Y operators
    by performing either a `cirq.Y ** -0.5` or `cirq.X ** -0.5`.  The square of
    these two measurements is summed to determine the length of the Bloch
    vector. This experiment measures the phase decoherence of the system under
    free evolution.

    For the Hahn echo experiment (often denoted T2 or spin echo), the state
    will also be prepared with a square root Y gate (`cirq.Y ** 0.5`).
    However, during the mid-point of the delay time being measured, a pi-pulse
    (`cirq.X`) gate will be applied to cancel out inhomogeneous dephasing.
    The same method of measuring the final state as Ramsey experiment is applied
    after the second half of the delay period.

    CPMG, or the Carr-Purcell-Meiboom-Gill sequence, is currently not
    implemented.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubit: The qubit under test.
        experiment_type: The type of T2 test to run.
        num_points: The number of evenly spaced delays to test.
        max_delay: The largest delay to test.
        min_delay: The smallest delay to test. Defaults to no delay.
        repetitions: The number of repetitions of the circuit
             for each delay and for each tomography result.
        delay_sweep: Optional range of time delays to sweep across.  This should
             be a SingleSweep using the 'delay_ns' with values in integer number
             of nanoseconds.  If specified, this will override the max_delay and
             min_delay parameters.  If not specified, the experiment will sweep
             from min_delay to max_delay with linear steps.
    Returns:
        A T2DecayResult object that stores and can plot the data.
    """
    min_delay_dur = value.Duration(min_delay)
    max_delay_dur = value.Duration(max_delay)

    # Input validation
    if repetitions <= 0:
        raise ValueError('repetitions <= 0')
    if max_delay_dur < min_delay_dur:
        raise ValueError('max_delay < min_delay')
    if min_delay_dur < 0:
        raise ValueError('min_delay < 0')

    # Initialize values used in sweeps
    delay_var = sympy.Symbol('delay_ns')
    inv_x_var = sympy.Symbol('inv_x')
    inv_y_var = sympy.Symbol('inv_y')

    if not delay_sweep:
        delay_sweep = study.Linspace(delay_var,
                                     start=min_delay_dur.total_nanos(),
                                     stop=max_delay_dur.total_nanos(),
                                     length=num_points)
    if delay_sweep.keys != ['delay_ns']:
        raise ValueError('delay_sweep must be a SingleSweep '
                         'with delay_ns parameter')

    if experiment_type == ExperimentType.RAMSEY:
        # Ramsey T2* experiment
        # Use sqrt(Y) to flip to the equator.
        # Evolve the state for a given amount of delay time
        # Then measure the state in both X and Y bases.

        circuit = circuits.Circuit(
            ops.Y(qubit)**0.5,
            ops.WaitGate(value.Duration(nanos=delay_var))(qubit),
            ops.X(qubit)**inv_x_var,
            ops.Y(qubit)**inv_y_var,
            ops.measure(qubit, key='output'),
        )
        tomography_sweep = study.Zip(
            study.Points('inv_x', [0.0, -0.5]),
            study.Points('inv_y', [-0.5, 0.0]),
        )
        sweep = study.Product(delay_sweep, tomography_sweep)
    elif experiment_type == ExperimentType.HAHN_ECHO:
        # Hahn / Spin Echo T2 experiment
        # Use sqrt(Y) to flip to the equator.
        # Evolve the state for half the given amount of delay time
        # Flip the state using an X gate
        # Evolve the state for half the given amount of delay time
        # Then measure the state in both X and Y bases.

        circuit = circuits.Circuit(
            ops.Y(qubit)**0.5,
            ops.WaitGate(value.Duration(nanos=0.5 * delay_var))(qubit),
            ops.X(qubit),
            ops.WaitGate(value.Duration(nanos=0.5 * delay_var))(qubit),
            ops.X(qubit)**inv_x_var,
            ops.Y(qubit)**inv_y_var,
            ops.measure(qubit, key='output'),
        )
        tomography_sweep = study.Zip(
            study.Points('inv_x', [0.0, 0.5]),
            study.Points('inv_y', [-0.5, 0.0]),
        )
        sweep = study.Product(delay_sweep, tomography_sweep)
    else:
        raise ValueError(f'Experiment type {experiment_type} not supported')

    # Tabulate measurements into a histogram
    results = sampler.sample(circuit, params=sweep, repetitions=repetitions)

    y_basis_measurements = results[abs(results.inv_y) > 0]
    x_basis_measurements = results[abs(results.inv_x) > 0]
    x_basis_tabulation = pd.crosstab(x_basis_measurements.delay_ns,
                                     x_basis_measurements.output).reset_index()
    y_basis_tabulation = pd.crosstab(y_basis_measurements.delay_ns,
                                     y_basis_measurements.output).reset_index()

    # If all measurements are 1 or 0, fill in the missing column with all zeros.
    for tab in [x_basis_tabulation, y_basis_tabulation]:
        for col_index, name in [(1, 0), (2, 1)]:
            if name not in tab:
                tab.insert(col_index, name, [0] * tab.shape[0])

    # Return the results in a container object
    return T2DecayResult(x_basis_tabulation, y_basis_tabulation)


class T2DecayResult:
    """Results from a T2 decay experiment.

     This object is a container for the measurement results in each basis
     for each amount of delay.  These can be used to calculate Pauli
     expectation values, length of the Bloch vector, and various fittings of
     the data to calculate estimated T2 phase decay times.
     """

    def __init__(self, x_basis_data: pd.DataFrame, y_basis_data: pd.DataFrame):
        """
        Args:
            data: A data frame with three columns:
                delay_ns, false_count, true_count.
        """
        x_cols = list(x_basis_data.columns)
        y_cols = list(y_basis_data.columns)
        if any(col not in x_cols for col in _T2_COLUMNS):
            raise ValueError(f'x_basis_data must have columns {_T2_COLUMNS} '
                             f'but had {list(x_basis_data.columns)}')
        if any(col not in y_cols for col in _T2_COLUMNS):
            raise ValueError(f'y_basis_data must have columns {_T2_COLUMNS} '
                             f'but had {list(y_basis_data.columns)}')
        self._x_basis_data = x_basis_data
        self._y_basis_data = y_basis_data
        self._expectation_pauli_x = self._expectation(x_basis_data)
        self._expectation_pauli_y = self._expectation(y_basis_data)

    def _expectation(self, data) -> pd.DataFrame:
        """Calculates the expected value of the Pauli operator.

        Assuming that the data is measured in the Pauli basis of the operator,
        then the expectation of the Pauli operator would be +1 if the
        measurement is all ones and -1 if the measurement is all zeros.

        Returns:
            Data frame with two columns 'delay_ns' and 'value'
        """
        xs = data['delay_ns']
        ones = data[1]
        zeros = data[0]
        pauli_expectation = (2 * (ones / (ones + zeros))) - 1.0
        return pd.DataFrame({'delay_ns': xs, 'value': pauli_expectation})

    @property
    def expectation_pauli_x(self) -> pd.DataFrame:
        """A data frame with delay_ns, value columns.

        This value contains the expectation of the Pauli X operator as
        estimated by measurement outcomes.
        """
        return self._expectation_pauli_x

    @property
    def expectation_pauli_y(self) -> pd.DataFrame:
        """A data frame with delay_ns, value columns.

        This value contains the expectation of the Pauli X operator as
        estimated by measurement outcomes.
        """
        return self._expectation_pauli_y

    def plot_expectations(self,
                          ax: Optional[plt.Axes] = None,
                          **plot_kwargs: Any) -> plt.Axes:
        """Plots the expectation values of Pauli operators versus delay time.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes containing the plot.
        """
        show_plot = not ax
        if show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        assert ax is not None
        ax.set_ylim(ymin=-2, ymax=2)

        # Plot different expectation values in different colors.
        ax.plot(self._expectation_pauli_x['delay_ns'],
                self._expectation_pauli_x['value'],
                'bo-',
                label='<X>',
                **plot_kwargs)
        ax.plot(self._expectation_pauli_y['delay_ns'],
                self._expectation_pauli_y['value'],
                'go-',
                label='<Y>',
                **plot_kwargs)

        ax.set_xlabel(
            r"Delay between initialization and measurement (nanoseconds)")
        ax.set_ylabel('Pauli Operator Expectation')
        ax.set_title('T2 Decay Pauli Expectations')
        ax.legend()
        if show_plot:
            fig.show()
        return ax

    def plot_bloch_vector(self,
                          ax: Optional[plt.Axes] = None,
                          **plot_kwargs: Any) -> plt.Axes:
        """Plots the estimated length of the Bloch vector versus time.

        This plot estimates the Bloch Vector by squaring the Pauli expectation
        value of X and adding it to the square of the Pauli expectation value of
        Y.  This essentially projects the state into the XY plane.

        Note that Z expectation is not considered, since T1 related amplitude
        damping will generally push this value towards |0>
        (expectation <Z> = -1) which will significantly distort the T2 numbers.

        Args:
            ax: the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            **plot_kwargs: Arguments to be passed to 'plt.Axes.plot'.

        Returns:
            The plt.Axes containing the plot.
         """
        show_plot = not ax
        if show_plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        assert ax is not None
        ax.set_ylim(ymin=0, ymax=1)

        # Estimate length of Bloch vector (projected to xy plane)
        # by squaring <X> and <Y> expectation values
        bloch_vector = (self._expectation_pauli_x**2 +
                        self._expectation_pauli_y**2)

        ax.plot(self._expectation_pauli_x['delay_ns'], bloch_vector, 'r+-',
                **plot_kwargs)
        ax.set_xlabel(
            r"Delay between initialization and measurement (nanoseconds)")
        ax.set_ylabel('Bloch Vector X-Y Projection Squared')
        ax.set_title('T2 Decay Experiment Data')
        if show_plot:
            fig.show()
        return ax

    def __str__(self):
        return (f'T2DecayResult with data:\n'
                f'<X>\n{self._x_basis_data}\n<Y>\n{self._y_basis_data}')

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (self._expectation_pauli_x.equals(other._expectation_pauli_x) and
                self._expectation_pauli_y.equals(other._expectation_pauli_y))

    def __repr__(self):
        return (f'cirq.experiments.T2DecayResult('
                f'x_basis_data={proper_repr(self._x_basis_data)}, '
                f'y_basis_data={proper_repr(self._y_basis_data)})')

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('T2DecayResult(...)')
        else:
            p.text(str(self))
