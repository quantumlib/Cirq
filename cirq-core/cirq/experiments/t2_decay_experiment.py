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

from typing import Any, List, Optional, TYPE_CHECKING, Union

import pandas as pd
import sympy
from matplotlib import pyplot as plt

from cirq import circuits, ops, study, value
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


class ExperimentType(enum.Enum):
    RAMSEY = 1  # Often denoted as t2*
    HAHN_ECHO = 2  # Spin echo or t2
    CPMG = 3  # Carr-Purcell-Meiboom-Gill sequence


_T2_COLUMNS = ['delay_ns', 0, 1]


# TODO(#3388) Add documentation for Raises.
# pylint: disable=missing-raises-doc
def t2_decay(
    sampler: 'cirq.Sampler',
    *,
    qubit: 'cirq.Qid',
    experiment_type: 'ExperimentType' = ExperimentType.RAMSEY,
    num_points: int,
    max_delay: 'cirq.DURATION_LIKE',
    min_delay: 'cirq.DURATION_LIKE' = None,
    repetitions: int = 1000,
    delay_sweep: Optional[study.Sweep] = None,
    num_pulses: List[int] = None,
) -> Union['cirq.experiments.T2DecayResult', List['cirq.experiments.T2DecayResult']]:
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
    by performing either a `cirq.Y ** -0.5` or `cirq.X ** 0.5`.  The square of
    these two measurements is summed to determine the length of the Bloch
    vector. This experiment measures the phase decoherence of the system under
    free evolution.

    For the Hahn echo experiment (often denoted T2 or spin echo), the state
    will also be prepared with a square root Y gate (`cirq.Y ** 0.5`).
    However, during the mid-point of the delay time being measured, a pi-pulse
    (`cirq.X`) gate will be applied to cancel out inhomogeneous dephasing.
    The same method of measuring the final state as Ramsey experiment is applied
    after the second half of the delay period.  See the animation on the wiki
    page https://en.wikipedia.org/wiki/Spin_echo for a visual illustration
    of this experiment.

    CPMG, or the Carr-Purcell-Meiboom-Gill sequence, involves using a sqrt(Y)
    followed by a sequence of pi pulses (X gates) in a specific timing pattern:

        π/2, t, π, 2t, π, ... 2t, π, t

    The first pulse, a sqrt(Y) gate, will put the qubit's state on the Bloch
    equator.  After a delay, successive X gates will refocus dehomogenous
    phase effects by causing them to precess in opposite directions and
    averaging their effects across the entire pulse train.

    This pulse pattern has two variables that can be adjusted.  The first,
    denoted as 't' in the above sequence, is delay, which can be specified
    with `delay_min` and `delay_max` or by using a `delay_sweep`, similar to
    the other experiments.  The second variable is the number of pi pulses
    (X gates).  This can be specified as a list of integers using the
    `num_pulses` parameter.  If multiple different pulses are specified,
    the data will be presented in a data frame with two
    indices (delay_ns and num_pulses).

    See the following reference for more information about CPMG pulse trains:
    Meiboom, S., and D. Gill, “Modified spin-echo method for measuring nuclear
    relaxation times”, Rev. Sci. Inst., 29, 688–691 (1958).
    https://doi.org/10.1063/1.1716296

    Note that interpreting T2 data is fairly tricky and subtle, as it can
    include other effects that need to be accounted for.  For instance,
    amplitude damping (T1) will present as T2 noise and needs to be
    appropriately compensated for to find a true measure of T2.  Due to this
    subtlety and lack of standard way to interpret the data, the fitting
    of the data to an exponential curve and the extrapolation of an actual
    T2 time value is left as an exercise to the reader.

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
        num_pulses: For CPMG, a list of the number of pulses to use.
             If multiple pulses are specified, each will be swept on.
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
    if num_pulses and experiment_type != ExperimentType.CPMG:
        raise ValueError('num_pulses is only valid for CPMG experiments.')

    # Initialize values used in sweeps
    delay_var = sympy.Symbol('delay_ns')
    inv_x_var = sympy.Symbol('inv_x')
    inv_y_var = sympy.Symbol('inv_y')
    max_pulses = max(num_pulses) if num_pulses else 0

    if not delay_sweep:
        delay_sweep = study.Linspace(
            delay_var,
            start=min_delay_dur.total_nanos(),
            stop=max_delay_dur.total_nanos(),
            length=num_points,
        )
    if delay_sweep.keys != ['delay_ns']:
        raise ValueError('delay_sweep must be a SingleSweep with delay_ns parameter')

    if experiment_type == ExperimentType.RAMSEY:
        # Ramsey T2* experiment
        # Use sqrt(Y) to flip to the equator.
        # Evolve the state for a given amount of delay time
        # Then measure the state in both X and Y bases.

        circuit = circuits.Circuit(
            ops.Y(qubit) ** 0.5,
            ops.wait(qubit, nanos=delay_var),
        )
    else:
        if experiment_type == ExperimentType.HAHN_ECHO:
            # Hahn / Spin Echo T2 experiment
            # Use sqrt(Y) to flip to the equator.
            # Evolve the state for the given amount of delay time
            # Flip the state using an X gate
            # Evolve the state for the given amount of delay time
            # Then measure the state in both X and Y bases.
            num_pulses = [0]
            # This is equivalent to a CPMG experiment with zero pulses
            # and will follow the same code path.

        # Carr-Purcell-Meiboom-Gill sequence.
        # Performs the following sequence
        # π/2 - wait(t) - π - wait(2t) - ... - π - wait(t)
        # There will be N π pulses (X gates)
        # where N sweeps over the values of num_pulses
        #
        if not num_pulses:
            raise ValueError('At least one value must be given for num_pulses in a CPMG experiment')
        circuit = _cpmg_circuit(qubit, delay_var, max_pulses)

    # Add simple state tomography
    circuit.append(ops.X(qubit) ** inv_x_var)
    circuit.append(ops.Y(qubit) ** inv_y_var)
    circuit.append(ops.measure(qubit, key='output'))
    tomography_sweep = study.Zip(
        study.Points('inv_x', [0.0, 0.5]),
        study.Points('inv_y', [-0.5, 0.0]),
    )

    if num_pulses and max_pulses > 0:
        pulse_sweep = _cpmg_sweep(num_pulses)
        sweep = study.Product(delay_sweep, pulse_sweep, tomography_sweep)
    else:
        sweep = study.Product(delay_sweep, tomography_sweep)

    # Tabulate measurements into a histogram
    results = sampler.sample(circuit, params=sweep, repetitions=repetitions)

    y_basis_measurements = results[abs(results.inv_y) > 0].copy()
    x_basis_measurements = results[abs(results.inv_x) > 0].copy()

    if num_pulses and len(num_pulses) > 1:
        cols = tuple(f'pulse_{t}' for t in range(max_pulses))
        x_basis_measurements['num_pulses'] = x_basis_measurements.loc[:, cols].sum(axis=1)
        y_basis_measurements['num_pulses'] = y_basis_measurements.loc[:, cols].sum(axis=1)

    x_basis_tabulation = _create_tabulation(x_basis_measurements)
    y_basis_tabulation = _create_tabulation(y_basis_measurements)

    # Return the results in a container object
    return T2DecayResult(x_basis_tabulation, y_basis_tabulation)


# pylint: enable=missing-raises-doc
def _create_tabulation(measurements: pd.DataFrame) -> pd.DataFrame:
    """Returns a sum of 0 and 1 results per index from a list of measurements."""
    if 'num_pulses' in measurements.columns:
        cols = [measurements.delay_ns, measurements.num_pulses]
    else:
        cols = [measurements.delay_ns]
    tabulation = pd.crosstab(cols, measurements.output).reset_index()
    # If all measurements are 1 or 0, fill in the missing column with all zeros.
    for col_index, name in [(1, 0), (2, 1)]:
        if name not in tabulation:
            tabulation.insert(col_index, name, [0] * tabulation.shape[0])
    return tabulation


def _cpmg_circuit(qubit: 'cirq.Qid', delay_var: sympy.Symbol, max_pulses: int) -> 'cirq.Circuit':
    """Creates a CPMG circuit for a given qubit.

    The circuit will look like:

      sqrt(Y) - wait(delay_var) - X - wait(2*delay_var) - ... - wait(delay_var)

    with max_pulses number of X gates.

    The X gates are paramterizd by 'pulse_N' symbols so that pulses can be
    turned on and off.  This is done to combine circuits with different pulses
    into the same paramterized circuit.
    """
    circuit = circuits.Circuit(ops.Y(qubit) ** 0.5, ops.wait(qubit, nanos=delay_var), ops.X(qubit))
    for n in range(max_pulses):
        pulse_n_on = sympy.Symbol(f'pulse_{n}')
        circuit.append(ops.wait(qubit, nanos=2 * delay_var * pulse_n_on))
        circuit.append(ops.X(qubit) ** pulse_n_on)
    circuit.append(ops.wait(qubit, nanos=delay_var))
    return circuit


def _cpmg_sweep(num_pulses: List[int]):
    """Returns a sweep for a circuit created by _cpmg_circuit.

    The circuit in _cpmg_circuit parameterizes the pulses, so this function
    fills in the parameters for each pulse.  For instance, if we want 3 pulses,
    pulse_0, pulse_1, and pulse_2 should be 1 and the rest of the pulses should
    be 0.
    """
    pulse_points = []
    for n in range(max(num_pulses)):
        pulse_points.append(study.Points(f'pulse_{n}', [1 if p > n else 0 for p in num_pulses]))
    return study.Zip(*pulse_points)


class T2DecayResult:
    """Results from a T2 decay experiment.

    This object is a container for the measurement results in each basis
    for each amount of delay.  These can be used to calculate Pauli
    expectation values, length of the Bloch vector, and various fittings of
    the data to calculate estimated T2 phase decay times.
    """

    # TODO(#3388) Add documentation for Raises.
    # pylint: disable=missing-raises-doc
    def __init__(self, x_basis_data: pd.DataFrame, y_basis_data: pd.DataFrame):
        """Inits T2DecayResult.

        Args:
            data: A data frame with three columns:
                delay_ns, false_count, true_count.
        """
        x_cols = list(x_basis_data.columns)
        y_cols = list(y_basis_data.columns)
        if any(col not in x_cols for col in _T2_COLUMNS):
            raise ValueError(
                f'x_basis_data must have columns {_T2_COLUMNS} '
                f'but had {list(x_basis_data.columns)}'
            )
        if any(col not in y_cols for col in _T2_COLUMNS):
            raise ValueError(
                f'y_basis_data must have columns {_T2_COLUMNS} '
                f'but had {list(y_basis_data.columns)}'
            )
        self._x_basis_data = x_basis_data
        self._y_basis_data = y_basis_data
        self._expectation_pauli_x = self._expectation(x_basis_data)
        self._expectation_pauli_y = self._expectation(y_basis_data)

    # pylint: enable=missing-raises-doc
    def _expectation(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates the expected value of the Pauli operator.

        Assuming that the data is measured in the Pauli basis of the operator,
        then the expectation of the Pauli operator would be +1 if the
        measurement is all ones and -1 if the measurement is all zeros.

        Returns:
            Data frame with columns 'delay_ns', 'num_pulses' and 'value'
            The num_pulses column will only exist if multiple pulses
            were requestd in the T2 experiment.
        """
        delay = data['delay_ns']
        ones = data[1]
        zeros = data[0]
        pauli_expectation = (2 * (ones / (ones + zeros))) - 1.0
        if 'num_pulses' in data.columns:
            return pd.DataFrame(
                {'delay_ns': delay, 'num_pulses': data['num_pulses'], 'value': pauli_expectation}
            )
        return pd.DataFrame({'delay_ns': delay, 'value': pauli_expectation})

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

    def plot_expectations(self, ax: Optional[plt.Axes] = None, **plot_kwargs: Any) -> plt.Axes:
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
        ax.plot(
            self._expectation_pauli_x['delay_ns'],
            self._expectation_pauli_x['value'],
            'bo-',
            label='<X>',
            **plot_kwargs,
        )
        ax.plot(
            self._expectation_pauli_y['delay_ns'],
            self._expectation_pauli_y['value'],
            'go-',
            label='<Y>',
            **plot_kwargs,
        )

        ax.set_xlabel(r"Delay between initialization and measurement (nanoseconds)")
        ax.set_ylabel('Pauli Operator Expectation')
        ax.set_title('T2 Decay Pauli Expectations')
        ax.legend()
        if show_plot:
            fig.show()
        return ax

    def plot_bloch_vector(self, ax: Optional[plt.Axes] = None, **plot_kwargs: Any) -> plt.Axes:
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
        bloch_vector = self._expectation_pauli_x ** 2 + self._expectation_pauli_y ** 2

        ax.plot(self._expectation_pauli_x['delay_ns'], bloch_vector, 'r+-', **plot_kwargs)
        ax.set_xlabel(r"Delay between initialization and measurement (nanoseconds)")
        ax.set_ylabel('Bloch Vector X-Y Projection Squared')
        ax.set_title('T2 Decay Experiment Data')
        if show_plot:
            fig.show()
        return ax

    def __str__(self):
        return f'T2DecayResult with data:\n<X>\n{self._x_basis_data}\n<Y>\n{self._y_basis_data}'

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._expectation_pauli_x.equals(
            other._expectation_pauli_x
        ) and self._expectation_pauli_y.equals(other._expectation_pauli_y)

    def __repr__(self):
        return (
            f'cirq.experiments.T2DecayResult('
            f'x_basis_data={proper_repr(self._x_basis_data)}, '
            f'y_basis_data={proper_repr(self._y_basis_data)})'
        )

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('T2DecayResult(...)')
        else:
            p.text(str(self))
