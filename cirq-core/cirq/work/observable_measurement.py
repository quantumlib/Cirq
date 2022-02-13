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

import abc
import dataclasses
import itertools
import os
import tempfile
import warnings
from typing import (
    Optional,
    Union,
    Iterable,
    Dict,
    List,
    Tuple,
    TYPE_CHECKING,
    Set,
    Sequence,
    Any,
)

import numpy as np
import pandas as pd
import sympy
from cirq import circuits, study, ops, value, protocols
from cirq._doc import document
from cirq.work.observable_grouping import group_settings_greedy, GROUPER_T
from cirq.work.observable_measurement_data import (
    BitstringAccumulator,
    ObservableMeasuredResult,
    flatten_grouped_results,
)
from cirq.work.observable_settings import (
    InitObsSetting,
    observables_to_settings,
    _MeasurementSpec,
)

if TYPE_CHECKING:
    import cirq
    from cirq.value.product_state import _NamedOneQubitState

MAX_REPETITIONS_PER_JOB = 3_000_000
document(
    MAX_REPETITIONS_PER_JOB,
    """The maximum repetitions allowed in a single batch job.

    This depends on the Sampler executing your batch job. It is set to be
    tens of minutes assuming ~kilosamples per second.
    """,
)


def _with_parameterized_layers(
    circuit: 'cirq.AbstractCircuit',
    qubits: Sequence['cirq.Qid'],
    needs_init_layer: bool,
) -> 'cirq.Circuit':
    """Return a copy of the input circuit with parameterized single-qubit rotations.

    These rotations flank the circuit: the initial two layers of X and Y gates
    are given parameter names "{qubit}-Xi" and "{qubit}-Yi" and are used
    to set up the initial state. If `needs_init_layer` is False,
    these two layers of gates are omitted.

    The final two layers of X and Y gates are given parameter names
    "{qubit}-Xf" and "{qubit}-Yf" and are use to change the frame of the
    qubit before measurement, effectively measuring in bases other than Z.
    """
    x_beg_mom = circuits.Moment([ops.X(q) ** sympy.Symbol(f'{q}-Xi') for q in qubits])
    y_beg_mom = circuits.Moment([ops.Y(q) ** sympy.Symbol(f'{q}-Yi') for q in qubits])
    x_end_mom = circuits.Moment([ops.X(q) ** sympy.Symbol(f'{q}-Xf') for q in qubits])
    y_end_mom = circuits.Moment([ops.Y(q) ** sympy.Symbol(f'{q}-Yf') for q in qubits])
    meas_mom = circuits.Moment([ops.measure(*qubits, key='z')])
    if needs_init_layer:
        total_circuit = circuits.Circuit([x_beg_mom, y_beg_mom])
        total_circuit += circuit.unfreeze()
    else:
        total_circuit = circuit.unfreeze()
    total_circuit.append([x_end_mom, y_end_mom, meas_mom])
    return total_circuit


class StoppingCriteria(abc.ABC):
    """An abstract object that queries a BitstringAccumulator to figure out
    whether that `meas_spec` is complete."""

    @abc.abstractmethod
    def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
        """Return the number of additional repetitions to take.

        StoppingCriteria should be respectful and have some notion of a
        maximum number of repetitions per chunk.
        """


@dataclasses.dataclass(frozen=True)
class VarianceStoppingCriteria(StoppingCriteria):
    """Stop sampling when average variance per term drops below a variance bound."""

    variance_bound: float
    repetitions_per_chunk: int = 10_000

    def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
        if len(accumulator.bitstrings) == 0:
            return self.repetitions_per_chunk

        cov = accumulator.covariance()
        n_terms = cov.shape[0]
        sum_variance = np.sum(cov)
        var_of_the_e = sum_variance / len(accumulator.bitstrings)
        vpt = var_of_the_e / n_terms

        if vpt <= self.variance_bound:
            # Done
            return 0
        return self.repetitions_per_chunk

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)


@dataclasses.dataclass(frozen=True)
class RepetitionsStoppingCriteria(StoppingCriteria):
    """Stop sampling when the number of repetitions has been reached."""

    total_repetitions: int
    repetitions_per_chunk: int = 10_000

    def more_repetitions(self, accumulator: BitstringAccumulator) -> int:
        done = accumulator.n_repetitions
        todo = self.total_repetitions - done
        if todo <= 0:
            return 0

        to_do_next = min(self.repetitions_per_chunk, todo)
        return to_do_next

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)


_OBS_TO_PARAM_VAL: Dict[Tuple['cirq.Pauli', bool], Tuple[float, float]] = {
    (ops.X, False): (0, -1 / 2),
    (ops.X, True): (0, +1 / 2),
    (ops.Y, False): (1 / 2, 0),
    (ops.Y, True): (-1 / 2, 0),
    (ops.Z, False): (0, 0),
    (ops.Z, True): (1, 0),
}
"""Mapping from single-qubit Pauli observable to the X- and Y-rotation parameter values. The
second element in the key is whether to measure in the positive or negative (flipped) basis
for readout symmetrization."""

_STATE_TO_PARAM_VAL: Dict['_NamedOneQubitState', Tuple[float, float]] = {
    value.KET_PLUS: (0, +1 / 2),
    value.KET_MINUS: (0, -1 / 2),
    value.KET_IMAG: (-1 / 2, 0),
    value.KET_MINUS_IMAG: (+1 / 2, 0),
    value.KET_ZERO: (0, 0),
    value.KET_ONE: (1, 0),
}
"""Mapping from an initial _NamedOneQubitState to the X- and Y-rotation parameter values."""


def _get_params_for_setting(
    setting: InitObsSetting,
    flips: Iterable[bool],
    qubits: Sequence['cirq.Qid'],
    needs_init_layer: bool,
) -> Dict[str, float]:
    """Return the parameter dictionary for the given setting.

    This must be used in conjunction with a circuit generated by
    `_with_parameterized_layers`. `flips` (used for readout symmetrization)
    should be of the same length as `qubits` and will modify the parameters
    to also include a bit flip (`X`). Code responsible for running the
    circuit should make sure to flip bits back prior to analysis.

    Like `_with_parameterized_layers`, we omit params for initialization gates
    if we know that `setting.init_state` is the all-zeros state and
    `needs_init_layer` is False.
    """
    setting = _pad_setting(setting, qubits)
    params = {}
    for qubit, flip in itertools.zip_longest(qubits, flips):
        if qubit is None or flip is None:
            raise ValueError("`qubits` and `flips` must be equal length")
        # When getting the one-qubit state / observable for this qubit,
        # you may be wondering what if there's no observable specified
        # for that qubit. We mandate that by the time you get to this stage,
        # each _max_setting has
        # weight(in_state) == weight(out_operator) == len(qubits)
        # See _pad_setting
        pauli = setting.observable[qubit]
        xf_param, yf_param = _OBS_TO_PARAM_VAL[pauli, flip]
        params[f'{qubit}-Xf'] = xf_param
        params[f'{qubit}-Yf'] = yf_param

        if needs_init_layer:
            state = setting.init_state[qubit]
            xi_param, yi_param = _STATE_TO_PARAM_VAL[state]
            params[f'{qubit}-Xi'] = xi_param
            params[f'{qubit}-Yi'] = yi_param

    return params


def _pad_setting(
    max_setting: InitObsSetting,
    qubits: Sequence['cirq.Qid'],
    pad_init_state_with=value.KET_ZERO,
    pad_obs_with: 'cirq.Gate' = ops.Z,
) -> InitObsSetting:
    """Pad `max_setting`'s `init_state` and `observable` with `pad_xx_with` operations
    (defaults:  |0> and Z) so each max_setting has the same qubits. We need this
    to be the case so we can fill in all the parameters, see `_get_params_for_setting`.
    """
    obs = max_setting.observable
    assert obs.coefficient == 1, "Only the max_setting should be padded."
    for qubit in qubits:
        if not qubit in obs:
            obs *= pad_obs_with(qubit)

    init_state = max_setting.init_state
    init_state_original_qubits = init_state.qubits
    for qubit in qubits:
        if not qubit in init_state_original_qubits:
            init_state *= pad_init_state_with(qubit)

    return InitObsSetting(init_state=init_state, observable=obs)


def _aggregate_n_repetitions(next_chunk_repetitions: Set[int]) -> int:
    """A stopping criteria can request a different number of more_repetitions for each
    measurement spec. For batching efficiency, we take the max and issue a warning in this case."""
    if len(next_chunk_repetitions) == 1:
        return list(next_chunk_repetitions)[0]

    reps = max(next_chunk_repetitions)
    warnings.warn(
        f"The stopping criteria specified a various numbers of "
        f"repetitions to perform next. To be able to submit as a single "
        f"sweep, the largest value will be used: {reps}."
    )
    return reps


def _check_meas_specs_still_todo(
    meas_specs: List[_MeasurementSpec],
    accumulators: Dict[_MeasurementSpec, BitstringAccumulator],
    stopping_criteria: StoppingCriteria,
) -> Tuple[List[_MeasurementSpec], int]:
    """Filter `meas_specs` in case some are done.

    In the sampling loop in `measure_grouped_settings`, we submit
    each `meas_spec` in chunks. This function contains the logic for
    removing `meas_spec`s from the loop if they are done.
    """
    still_todo = []
    repetitions_set: Set[int] = set()
    for meas_spec in meas_specs:
        accumulator = accumulators[meas_spec]
        more_repetitions = stopping_criteria.more_repetitions(accumulator)

        if more_repetitions < 0:
            raise ValueError(
                "Stopping criteria's `more_repetitions` should return 0 or a positive number."
            )
        if more_repetitions == 0:
            continue

        repetitions_set.add(more_repetitions)
        still_todo.append(meas_spec)

    if len(still_todo) == 0:
        return still_todo, 0

    repetitions = _aggregate_n_repetitions(repetitions_set)
    total_repetitions = len(still_todo) * repetitions
    if total_repetitions > MAX_REPETITIONS_PER_JOB:
        old_repetitions = repetitions
        repetitions = MAX_REPETITIONS_PER_JOB // len(still_todo)

        if repetitions < 10:
            raise ValueError(
                "You have requested too many parameter settings to batch your job effectively. "
                "Consider fewer sweeps or manually splitting sweeps into multiple jobs."
            )

        warnings.warn(
            f"The number of requested sweep parameters is high. To avoid a batched job with more "
            f"than {MAX_REPETITIONS_PER_JOB} shots, the number of shots per call to run_sweep "
            f"(per parameter value) will be throttled from {old_repetitions} to {repetitions}."
        )

    return still_todo, repetitions


@dataclasses.dataclass(frozen=True)
class _FlippyMeasSpec:
    """Internally, each MeasurementSpec class is split into two
    _FlippyMeasSpecs to support readout symmetrization.

    Bitstring results are combined, so this should be opaque to the user.
    """

    meas_spec: _MeasurementSpec
    flips: np.ndarray
    qubits: Sequence['cirq.Qid']

    def param_tuples(self, *, needs_init_layer=True):
        yield from _get_params_for_setting(
            self.meas_spec.max_setting,
            flips=self.flips,
            qubits=self.qubits,
            needs_init_layer=needs_init_layer,
        ).items()
        yield from self.meas_spec.circuit_params.items()


def _subdivide_meas_specs(
    meas_specs: Iterable[_MeasurementSpec],
    repetitions: int,
    qubits: Sequence['cirq.Qid'],
    readout_symmetrization: bool,
) -> Tuple[List[_FlippyMeasSpec], int]:
    """Split measurement specs into sub-jobs for readout symmetrization

    In readout symmetrization, we first run the "normal" circuit followed
    by running the circuit with flipped measurement.
    One _MeasurementSpec is split into two _FlippyMeasSpecs. These are run
    separately but accumulated according to their shared _MeasurementSpec.
    """
    n_qubits = len(qubits)
    flippy_mspecs = []
    for meas_spec in meas_specs:
        all_normal = np.zeros(n_qubits, dtype=bool)
        flippy_mspecs.append(
            _FlippyMeasSpec(
                meas_spec=meas_spec,
                flips=all_normal,
                qubits=qubits,
            )
        )

        if readout_symmetrization:
            all_flipped = np.ones(n_qubits, dtype=bool)
            flippy_mspecs.append(
                _FlippyMeasSpec(
                    meas_spec=meas_spec,
                    flips=all_flipped,
                    qubits=qubits,
                )
            )

    if readout_symmetrization:
        repetitions //= 2

    return flippy_mspecs, repetitions


def _to_sweep(param_tuples):
    """Turn param tuples into a sweep."""
    to_sweep = [dict(pt) for pt in param_tuples]
    to_sweep = study.to_sweep(to_sweep)
    return to_sweep


def _parse_checkpoint_options(
    checkpoint: bool, checkpoint_fn: Optional[str], checkpoint_other_fn: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Parse the checkpoint-oriented options in `measure_grouped_settings`.

    This function contains the validation and defaults logic. Please see
    `measure_grouped_settings` for documentation on these args.

    Returns:
        checkpoint_fn, checkpoint_other_fn: Parsed or default filenames for primary and previous
            checkpoint files.

    Raises:
        ValueError: If a `checkpoint_fn` was specified, but `checkpoint` was False, if the
            `checkpoint_fn` is not of the form filename.json, or if `checkout_fn` and
            `checkpoint_other_fn` are the same filename.
    """
    if not checkpoint:
        if checkpoint_fn is not None or checkpoint_other_fn is not None:
            raise ValueError(
                "Checkpoint filenames were provided but `checkpoint` was set to False."
            )
        return None, None

    if checkpoint_fn is None:
        checkpoint_dir = tempfile.mkdtemp()
        chk_basename = 'observables'
        checkpoint_fn = f'{checkpoint_dir}/{chk_basename}.json'

    if checkpoint_other_fn is None:
        checkpoint_dir = os.path.dirname(checkpoint_fn)
        chk_basename = os.path.basename(checkpoint_fn)
        chk_basename, dot, ext = chk_basename.rpartition('.')
        if chk_basename == '' or dot != '.' or ext == '':
            raise ValueError(
                f"You specified `checkpoint_fn={checkpoint_fn!r}` which does not follow the "
                f"pattern of 'filename.extension'. Please follow this pattern or fully specify "
                f"`checkpoint_other_fn`."
            )

        if ext != 'json':
            raise ValueError(
                "Please use a `.json` filename or fully "
                "specify checkpoint_fn and checkpoint_other_fn"
            )
        if checkpoint_dir == '':
            checkpoint_other_fn = f'{chk_basename}.prev.json'
        else:
            checkpoint_other_fn = f'{checkpoint_dir}/{chk_basename}.prev.json'

    if checkpoint_fn == checkpoint_other_fn:
        raise ValueError(
            f"`checkpoint_fn` and `checkpoint_other_fn` were set to the same "
            f"filename: {checkpoint_fn}. Please use two different filenames."
        )

    return checkpoint_fn, checkpoint_other_fn


@dataclasses.dataclass(frozen=True)
class CheckpointFileOptions:
    """Options to configure "checkpointing" to save intermediate results.

    Args:
        checkpoint: If set to True, save cumulative raw results at the end
            of each iteration of the sampling loop. Load in these results
            with `cirq.read_json`.
        checkpoint_fn: The filename for the checkpoint file. If `checkpoint`
            is set to True and this is not specified, a file in a temporary
            directory will be used.
        checkpoint_other_fn: The filename for another checkpoint file, which
            contains the previous checkpoint. This lets us avoid losing data if
            a failure occurs during checkpoint writing. If `checkpoint`
            is set to True and this is not specified, a file in a temporary
            directory will be used. If `checkpoint` is set to True and
            `checkpoint_fn` is specified but this argument is *not* specified,
            "{checkpoint_fn}.prev.json" will be used.
    """

    checkpoint: bool = False
    checkpoint_fn: Optional[str] = None
    checkpoint_other_fn: Optional[str] = None

    def __post_init__(self):
        fn, other_fn = _parse_checkpoint_options(
            self.checkpoint, self.checkpoint_fn, self.checkpoint_other_fn
        )
        object.__setattr__(self, 'checkpoint_fn', fn)
        object.__setattr__(self, 'checkpoint_other_fn', other_fn)

    def maybe_to_json(self, obj: Any):
        """Call `cirq.to_json with `value` according to the configuration options in this class.

        If `checkpoint=False`, nothing will happen. Otherwise, we will use `checkpoint_fn` and
        `checkpoint_other_fn` as the destination JSON file as described in the class docstring.
        """
        if not self.checkpoint:
            return
        assert self.checkpoint_fn is not None, 'mypy'
        assert self.checkpoint_other_fn is not None, 'mypy'
        if os.path.exists(self.checkpoint_fn):
            os.replace(self.checkpoint_fn, self.checkpoint_other_fn)
        protocols.to_json(obj, self.checkpoint_fn)


def _needs_init_layer(grouped_settings: Dict[InitObsSetting, List[InitObsSetting]]) -> bool:
    """Helper function to go through init_states and determine if any of them need an
    initialization layer of single-qubit gates."""
    for max_setting in grouped_settings.keys():
        if any(st is not value.KET_ZERO for _, st in max_setting.init_state):
            return True
    return False


def measure_grouped_settings(
    circuit: 'cirq.AbstractCircuit',
    grouped_settings: Dict[InitObsSetting, List[InitObsSetting]],
    sampler: 'cirq.Sampler',
    stopping_criteria: StoppingCriteria,
    *,
    readout_symmetrization: bool = False,
    circuit_sweep: 'cirq.Sweepable' = None,
    readout_calibrations: Optional[BitstringAccumulator] = None,
    checkpoint: CheckpointFileOptions = CheckpointFileOptions(),
) -> List[BitstringAccumulator]:
    """Measure a suite of grouped InitObsSetting settings.

    This is a low-level API for accessing the observable measurement
    framework. See also `measure_observables` and `measure_observables_df`.

    Args:
        circuit: The circuit. This can contain parameters, in which case
            you should also specify `circuit_sweep`.
        grouped_settings: A series of setting groups expressed as a dictionary.
            The key is the max-weight setting used for preparing single-qubit
            basis-change rotations. The value is a list of settings
            compatible with the maximal setting you desire to measure.
            Automated routing algorithms like `group_settings_greedy` can
            be used to construct this input.
        sampler: A sampler.
        stopping_criteria: A StoppingCriteria object that can report
            whether enough samples have been sampled.
        readout_symmetrization: If set to True, each `meas_spec` will be
            split into two runs: one normal and one where a bit flip is
            incorporated prior to measurement. In the latter case, the
            measured bit will be flipped back classically and accumulated
            together. This causes readout error to appear symmetric,
            p(0|0) = p(1|1).
        circuit_sweep: Additional parameter sweeps for parameters contained
            in `circuit`. The total sweep is the product of the circuit sweep
            with parameter settings for the single-qubit basis-change rotations.
        readout_calibrations: The result of `calibrate_readout_error`.
        checkpoint: Options to set up optional checkpointing of intermediate
            data for each iteration of the sampling loop. See the documentation
            for `CheckpointFileOptions` for more. Load in these results with
            `cirq.read_json`.

    Raises:
        ValueError: If readout calibration is specified, but `readout_symmetrization
            is not True.
    """
    if readout_calibrations is not None and not readout_symmetrization:
        raise ValueError("Readout calibration only works if `readout_symmetrization` is enabled.")

    qubits = sorted({q for ms in grouped_settings.keys() for q in ms.init_state.qubits})
    qubit_to_index = {q: i for i, q in enumerate(qubits)}

    needs_init_layer = _needs_init_layer(grouped_settings)
    measurement_param_circuit = _with_parameterized_layers(circuit, qubits, needs_init_layer)

    # meas_spec provides a key for accumulators.
    # meas_specs_todo is a mutable list. We will pop things from it as various
    # specs are measured to the satisfaction of the stopping criteria
    accumulators = {}
    meas_specs_todo = []
    for max_setting, param_resolver in itertools.product(
        grouped_settings.keys(), study.to_resolvers(circuit_sweep)
    ):
        circuit_params = param_resolver.param_dict
        meas_spec = _MeasurementSpec(max_setting=max_setting, circuit_params=circuit_params)
        accumulator = BitstringAccumulator(
            meas_spec=meas_spec,
            simul_settings=grouped_settings[max_setting],
            qubit_to_index=qubit_to_index,
            readout_calibration=readout_calibrations,
        )
        accumulators[meas_spec] = accumulator
        meas_specs_todo += [meas_spec]

    while True:
        meas_specs_todo, repetitions = _check_meas_specs_still_todo(
            meas_specs=meas_specs_todo,
            accumulators=accumulators,
            stopping_criteria=stopping_criteria,
        )
        if len(meas_specs_todo) == 0:
            break

        flippy_meas_specs, repetitions = _subdivide_meas_specs(
            meas_specs=meas_specs_todo,
            repetitions=repetitions,
            qubits=qubits,
            readout_symmetrization=readout_symmetrization,
        )

        resolved_params = [
            flippy_ms.param_tuples(needs_init_layer=needs_init_layer)
            for flippy_ms in flippy_meas_specs
        ]
        resolved_params = _to_sweep(resolved_params)

        results = sampler.run_sweep(
            program=measurement_param_circuit, params=resolved_params, repetitions=repetitions
        )

        assert len(results) == len(
            flippy_meas_specs
        ), 'Not as many results received as sweeps requested!'

        for flippy_ms, result in zip(flippy_meas_specs, results):
            accumulator = accumulators[flippy_ms.meas_spec]
            bitstrings = np.logical_xor(flippy_ms.flips, result.measurements['z'])
            accumulator.consume_results(bitstrings.astype(np.uint8, casting='safe'))

        checkpoint.maybe_to_json(list(accumulators.values()))

    return list(accumulators.values())


_GROUPING_FUNCS: Dict[str, GROUPER_T] = {
    'greedy': group_settings_greedy,
}


def _parse_grouper(grouper: Union[str, GROUPER_T] = group_settings_greedy) -> GROUPER_T:
    """Logic for turning a named grouper into one of the build-in groupers in support of the
    high-level `measure_observables` API."""
    if isinstance(grouper, str):
        try:
            grouper = _GROUPING_FUNCS[grouper.lower()]
        except KeyError:
            raise ValueError(f"Unknown grouping function {grouper}")
    return grouper


def _get_all_qubits(
    circuit: 'cirq.AbstractCircuit',
    observables: Iterable['cirq.PauliString'],
) -> List['cirq.Qid']:
    """Helper function for `measure_observables` to get all qubits from a circuit and a
    collection of observables."""
    qubit_set = set()
    for obs in observables:
        qubit_set |= set(obs.qubits)
    qubit_set |= circuit.all_qubits()
    return sorted(qubit_set)


def measure_observables(
    circuit: 'cirq.AbstractCircuit',
    observables: Iterable['cirq.PauliString'],
    sampler: Union['cirq.Simulator', 'cirq.Sampler'],
    stopping_criteria: StoppingCriteria,
    *,
    readout_symmetrization: bool = False,
    circuit_sweep: Optional['cirq.Sweepable'] = None,
    grouper: Union[str, GROUPER_T] = group_settings_greedy,
    readout_calibrations: Optional[BitstringAccumulator] = None,
    checkpoint: CheckpointFileOptions = CheckpointFileOptions(),
) -> List[ObservableMeasuredResult]:
    """Measure a collection of PauliString observables for a state prepared by a Circuit.

    If you need more control over the process, please see `measure_grouped_settings` for a
    lower-level API. If you would like your results returned as a pandas DataFrame,
    please see `measure_observables_df`.

    Args:
        circuit: The circuit used to prepare the state to measure. This can contain parameters,
            in which case you should also specify `circuit_sweep`.
        observables: A collection of PauliString observables to measure. These will be grouped
            into simultaneously-measurable groups, see `grouper` argument.
        sampler: The sampler.
        stopping_criteria: A StoppingCriteria object to indicate how precisely to sample
            measurements for estimating observables.
        readout_symmetrization: If set to True, each run will be split into two: one normal and
            one where a bit flip is incorporated prior to measurement. In the latter case, the
            measured bit will be flipped back classically and accumulated together. This causes
            readout error to appear symmetric, p(0|0) = p(1|1).
        circuit_sweep: Additional parameter sweeps for parameters contained in `circuit`. The
            total sweep is the product of the circuit sweep with parameter settings for the
            single-qubit basis-change rotations.
        grouper: Either "greedy" or a function that groups lists of `InitObsSetting`. See the
            documentation for the `grouped_settings` argument of `measure_grouped_settings` for
            full details.
        readout_calibrations: The result of `calibrate_readout_error`.
        checkpoint: Options to set up optional checkpointing of intermediate data for each
            iteration of the sampling loop. See the documentation for `CheckpointFileOptions` for
            more. Load in these results with `cirq.read_json`.

    Returns:
        A list of ObservableMeasuredResult; one for each input PauliString.
    """
    qubits = _get_all_qubits(circuit, observables)
    settings = list(observables_to_settings(observables, qubits))
    actual_grouper = _parse_grouper(grouper)
    grouped_settings = actual_grouper(settings)

    accumulators = measure_grouped_settings(
        circuit=circuit,
        grouped_settings=grouped_settings,
        sampler=sampler,
        stopping_criteria=stopping_criteria,
        circuit_sweep=circuit_sweep,
        readout_symmetrization=readout_symmetrization,
        readout_calibrations=readout_calibrations,
        checkpoint=checkpoint,
    )
    return flatten_grouped_results(accumulators)


def measure_observables_df(
    circuit: 'cirq.AbstractCircuit',
    observables: Iterable['cirq.PauliString'],
    sampler: Union['cirq.Simulator', 'cirq.Sampler'],
    stopping_criteria: StoppingCriteria,
    *,
    readout_symmetrization: bool = False,
    circuit_sweep: Optional['cirq.Sweepable'] = None,
    grouper: Union[str, GROUPER_T] = group_settings_greedy,
    readout_calibrations: Optional[BitstringAccumulator] = None,
    checkpoint: CheckpointFileOptions = CheckpointFileOptions(),
):
    """Measure observables and return resulting data as a Pandas dataframe.

    Please see `measure_observables` for argument documentation.
    """
    results = measure_observables(
        circuit=circuit,
        observables=observables,
        sampler=sampler,
        stopping_criteria=stopping_criteria,
        readout_symmetrization=readout_symmetrization,
        circuit_sweep=circuit_sweep,
        grouper=grouper,
        readout_calibrations=readout_calibrations,
        checkpoint=checkpoint,
    )
    df = pd.DataFrame(res.as_dict() for res in results)
    return df
