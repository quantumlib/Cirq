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
"""Single qubit readout experiments using parallel or isolated statistics."""
import dataclasses
import time
from typing import cast, Any, Dict, Iterable, List, Optional, TYPE_CHECKING

import sympy
import numpy as np
import matplotlib.pyplot as plt
import cirq.vis.heatmap as cirq_heatmap
import cirq.vis.histogram as cirq_histogram
from cirq.devices import grid_qubit
from cirq import circuits, ops, study

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass
class SingleQubitReadoutCalibrationResult:
    """Result of estimating single qubit readout error.

    Attributes:
        zero_state_errors: A dictionary from qubit to probability of measuring
            a 1 when the qubit is initialized to |0⟩.
        one_state_errors: A dictionary from qubit to probability of measuring
            a 0 when the qubit is initialized to |1⟩.
        repetitions: The number of repetitions that were used to estimate the
            probabilities.
        timestamp: The time the data was taken, in seconds since the epoch.
    """

    zero_state_errors: Dict['cirq.Qid', float]
    one_state_errors: Dict['cirq.Qid', float]
    repetitions: int
    timestamp: float

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'zero_state_errors': list(self.zero_state_errors.items()),
            'one_state_errors': list(self.one_state_errors.items()),
            'repetitions': self.repetitions,
            'timestamp': self.timestamp,
        }

    def plot_heatmap(
        self,
        axs: Optional[tuple[plt.Axes, plt.Axes]] = None,
        annotation_format: str = '0.1%',
        **plot_kwargs: Any,
    ) -> tuple[plt.Axes, plt.Axes]:
        """Plot a heatmap of the readout errors. If qubits are not cirq.GridQubits, throws an error.

        Args:
            axs: a tuple of the plt.Axes to plot on. If not given, a new figure is created,
                plotted on, and shown.
            annotation_format: The format string for the numbers in the heatmap.
            **plot_kwargs: Arguments to be passed to 'cirq.Heatmap.plot()'.
        Returns:
            The two plt.Axes containing the plot.

        Raises:
            ValueError: axs does not contain two plt.Axes
            TypeError: qubits are not cirq.GridQubits
        """

        if axs is None:
            _, axs_v = plt.subplots(1, 2, dpi=200, facecolor='white', figsize=(12, 4))
            axs_v = cast(np.ndarray, axs_v)
            axs = cast(tuple[plt.Axes, plt.Axes], (axs_v[0], axs_v[1]))
        else:
            if (
                not isinstance(axs, (tuple, list, np.ndarray))
                or len(axs) != 2
                or type(axs[0]) != plt.Axes
                or type(axs[1]) != plt.Axes
            ):  # pragma: no cover
                raise ValueError('axs should be a length-2 tuple of plt.Axes')  # pragma: no cover
        for ax, title, data in zip(
            axs,
            ['$|0\\rangle$ errors', '$|1\\rangle$ errors'],
            [self.zero_state_errors, self.one_state_errors],
        ):
            data_with_grid_qubit_keys = {}
            for qubit in data:
                if type(qubit) != grid_qubit.GridQubit:
                    raise TypeError(f'{qubit} must be of type cirq.GridQubit')  # pragma: no cover
                data_with_grid_qubit_keys[qubit] = data[qubit]  # just for typecheck
            _, _ = cirq_heatmap.Heatmap(data_with_grid_qubit_keys).plot(
                ax, annotation_format=annotation_format, title=title, **plot_kwargs
            )
        return axs[0], axs[1]

    def plot_integrated_histogram(
        self,
        ax: Optional[plt.Axes] = None,
        cdf_on_x: bool = False,
        axis_label: str = 'Readout error rate',
        semilog: bool = True,
        median_line: bool = True,
        median_label: Optional[str] = 'median',
        mean_line: bool = False,
        mean_label: Optional[str] = 'mean',
        show_zero: bool = False,
        title: Optional[str] = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot the readout errors using cirq.integrated_histogram().

        Args:
            ax: The axis to plot on. If None, we generate one.
            cdf_on_x: If True, flip the axes compared the above example.
            axis_label: Label for x axis (y-axis if cdf_on_x is True).
            semilog: If True, force the x-axis to be logarithmic.
            median_line: If True, draw a vertical line on the median value.
            median_label: If drawing median line, optional label for it.
            mean_line: If True, draw a vertical line on the mean value.
            mean_label: If drawing mean line, optional label for it.
            title: Title of the plot. If None, we assign "N={len(data)}".
            show_zero: If True, moves the step plot up by one unit by prepending 0
                to the data.
            **kwargs: Kwargs to forward to `ax.step()`. Some examples are
                color: Color of the line.
                linestyle: Linestyle to use for the plot.
                lw: linewidth for integrated histogram.
                ms: marker size for a histogram trace.
        Returns:
            The axis that was plotted on.
        """

        ax = cirq_histogram.integrated_histogram(
            data=self.zero_state_errors,
            ax=ax,
            cdf_on_x=cdf_on_x,
            semilog=semilog,
            median_line=median_line,
            median_label=median_label,
            mean_line=mean_line,
            mean_label=mean_label,
            show_zero=show_zero,
            color='C0',
            label='$|0\\rangle$ errors',
            **kwargs,
        )
        ax = cirq_histogram.integrated_histogram(
            data=self.one_state_errors,
            ax=ax,
            cdf_on_x=cdf_on_x,
            axis_label=axis_label,
            semilog=semilog,
            median_line=median_line,
            median_label=median_label,
            mean_line=mean_line,
            mean_label=mean_label,
            show_zero=show_zero,
            title=title,
            color='C1',
            label='$|1\\rangle$ errors',
            **kwargs,
        )
        ax.legend(loc='best')
        ax.set_ylabel('Percentile')
        return ax

    @classmethod
    def _from_json_dict_(
        cls, zero_state_errors, one_state_errors, repetitions, timestamp, **kwargs
    ):
        return cls(
            zero_state_errors=dict(zero_state_errors),
            one_state_errors=dict(one_state_errors),
            repetitions=repetitions,
            timestamp=timestamp,
        )

    def __repr__(self) -> str:
        return (
            'cirq.experiments.SingleQubitReadoutCalibrationResult('
            f'zero_state_errors={self.zero_state_errors!r}, '
            f'one_state_errors={self.one_state_errors!r}, '
            f'repetitions={self.repetitions!r}, '
            f'timestamp={self.timestamp!r})'
        )


def estimate_single_qubit_readout_errors(
    sampler: 'cirq.Sampler', *, qubits: Iterable['cirq.Qid'], repetitions: int = 1000
) -> SingleQubitReadoutCalibrationResult:
    """Estimate single-qubit readout error.

    For each qubit, prepare the |0⟩ state and measure. Calculate how often a 1
    is measured. Also, prepare the |1⟩ state and calculate how often a 0 is
    measured. The state preparations and measurements are done in parallel,
    i.e., for the first experiment, we actually prepare every qubit in the |0⟩
    state and measure them simultaneously.

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: The qubits being tested.
        repetitions: The number of measurement repetitions to perform.

    Returns:
        A SingleQubitReadoutCalibrationResult storing the readout error
        probabilities as well as the number of repetitions used to estimate
        the probabilities. Also stores a timestamp indicating the time when
        data was finished being collected from the sampler.
    """
    num_qubits = len(list(qubits))
    return estimate_parallel_single_qubit_readout_errors(
        sampler=sampler,
        qubits=qubits,
        repetitions=repetitions,
        trials=2,
        bit_strings=np.array([[0] * num_qubits, [1] * num_qubits]),
    )


def estimate_parallel_single_qubit_readout_errors(
    sampler: 'cirq.Sampler',
    *,
    qubits: Iterable['cirq.Qid'],
    trials: int = 20,
    repetitions: int = 1000,
    trials_per_batch: Optional[int] = None,
    bit_strings: Optional[np.ndarray] = None,
) -> SingleQubitReadoutCalibrationResult:
    """Estimate single qubit readout error using parallel operations.

    For each trial, prepare and then measure a random computational basis
    bitstring on qubits using gates in parallel.
    Returns a SingleQubitReadoutCalibrationResult which can be used to
    compute readout errors for each qubit.

    Args:
        sampler: The `cirq.Sampler` used to run the circuits.
        qubits: The qubits being tested.
        repetitions: The number of measurement repetitions to perform for
            each trial.
        trials: The number of bitstrings to prepare.
        trials_per_batch:  If provided, split the experiment into batches
            with this number of trials in each batch.
        bit_strings: Optional numpy array of shape (trials, qubits) where the
            first dimension is the number of the trial and the second
            dimension is the qubit (ordered by the qubit order from
            the qubits parameter).  Each value should be a 0 or 1 which
            specifies which state the qubit should be prepared into during
            that trial.  If not provided, the function will generate random
            bit strings for you.

    Returns:
        A SingleQubitReadoutCalibrationResult storing the readout error
        probabilities as well as the number of repetitions used to estimate
        the probabilities. Also stores a timestamp indicating the time when
        data was finished being collected from the sampler.  Note that,
        if there did not exist a trial where a given qubit was set to |0〉,
        the zero-state error will be set to `nan` (not a number).  Likewise
        for qubits with no |1〉trial and one-state error.

    Raises:
        ValueError: If the number of trials, repetitions, or trials_per batch is
            negative, or if bit_strings is not a numpy array or of the wrong
            shape.
    """
    qubits = list(qubits)

    if trials <= 0:
        raise ValueError("Must provide non-zero trials for readout calibration.")
    if repetitions <= 0:
        raise ValueError("Must provide non-zero repetition for readout calibration.")
    if bit_strings is None:
        bit_strings = np.random.randint(0, 2, size=(trials, len(qubits)))
    else:
        if not hasattr(bit_strings, 'shape') or bit_strings.shape != (trials, len(qubits)):
            raise ValueError(
                'bit_strings must be numpy array '
                f'of shape (trials, qubits) ({trials}, {len(qubits)}) '
                f"but was {bit_strings.shape if hasattr(bit_strings, 'shape') else None}"
            )
        if not np.all((bit_strings == 0) | (bit_strings == 1)):
            raise ValueError('bit_strings values must be all 0 or 1')
    if trials_per_batch is None:
        trials_per_batch = trials
    if trials_per_batch <= 0:
        raise ValueError("Must provide non-zero trials_per_batch for readout calibration.")

    all_sweeps: List[study.Sweepable] = []
    num_batches = (trials + trials_per_batch - 1) // trials_per_batch

    # Initialize circuits
    flip_symbols = sympy.symbols(f'flip_0:{len(qubits)}')
    flip_circuit = circuits.Circuit(
        [ops.X(q) ** s for q, s in zip(qubits, flip_symbols)],
        [ops.measure_each(*qubits, key_func=repr)],
    )
    all_circuits = [flip_circuit] * num_batches
    # Initialize sweeps
    for batch in range(num_batches):
        single_sweeps = []
        for qubit_idx in range(len(qubits)):
            trial_range = range(
                batch * trials_per_batch, min((batch + 1) * trials_per_batch, trials)
            )
            single_sweeps.append(
                study.Points(
                    key=f'flip_{qubit_idx}',
                    points=[bit_strings[bit][qubit_idx] for bit in trial_range],
                )
            )
        total_sweeps = study.Zip(*single_sweeps)
        all_sweeps.append(total_sweeps)

    # Execute circuits
    results = sampler.run_batch(all_circuits, all_sweeps, repetitions=repetitions)
    timestamp = time.time()

    # Analyze results
    zero_state_trials = np.zeros((1, len(qubits)))
    one_state_trials = np.zeros((1, len(qubits)))
    zero_state_totals = np.zeros((1, len(qubits)))
    one_state_totals = np.zeros((1, len(qubits)))
    trial_idx = 0
    for batch_result in results:
        for trial_result in batch_result:
            all_measurements = trial_result.data[[repr(x) for x in qubits]].to_numpy()
            sample_counts = np.einsum('ij->j', all_measurements)

            zero_state_trials += sample_counts * (1 - bit_strings[trial_idx])
            zero_state_totals += repetitions * (1 - bit_strings[trial_idx])
            one_state_trials += (repetitions - sample_counts) * bit_strings[trial_idx]
            one_state_totals += repetitions * bit_strings[trial_idx]

            trial_idx += 1

    zero_state_errors = {
        q: (
            zero_state_trials[0][qubit_idx] / zero_state_totals[0][qubit_idx]
            if zero_state_totals[0][qubit_idx] > 0
            else np.nan
        )
        for qubit_idx, q in enumerate(qubits)
    }
    one_state_errors = {
        q: (
            one_state_trials[0][qubit_idx] / one_state_totals[0][qubit_idx]
            if one_state_totals[0][qubit_idx] > 0
            else np.nan
        )
        for qubit_idx, q in enumerate(qubits)
    }

    return SingleQubitReadoutCalibrationResult(
        zero_state_errors=zero_state_errors,
        one_state_errors=one_state_errors,
        repetitions=repetitions,
        timestamp=timestamp,
    )
