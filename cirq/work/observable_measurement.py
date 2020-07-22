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

import abc
import dataclasses
import itertools
import os
import tempfile
import warnings
from typing import Optional, Union, Iterable, Dict, List, Tuple, \
    TYPE_CHECKING, Set, Sequence

import numpy as np
import pandas as pd
import sympy

from cirq import circuits, study, ops, value, protocols
from cirq.work.observable_grouping import group_settings_greedy
from cirq.work.observable_measurement_data import BitstringAccumulator
from cirq.work.observable_settings import InitObsSetting, observables_to_settings, zeros_state, \
    MeasurementSpec

if TYPE_CHECKING:
    import cirq


def _with_parameterized_layers(circuit: 'cirq.Circuit',
                               qubits: Sequence['cirq.Qid'],
                               no_initialization: bool,
                               ) -> 'cirq.Circuit':
    """Return a copy of the input circuit with a parameterized single-qubit
    rotations.

    These rotations flank the circuit: the initial two layers of X and Y gates
    are given parameter names "{qubit}-Xi" and "{qubit}-Yi" and are used
    to set up the initial state. If `no_initialization` is set to True,
    these two layers of gates are omitted.

    The final two layers of X and Y gates are given parameter names
    "{qubit}-Xf" and "{qubit}-Yf" and are use to change the frame of the
    qubit before measurement, effectively measuring in bases other than Z.
    """
    x_beg_mom = ops.Moment([
        ops.X(q) ** sympy.Symbol(f'{q}-Xi')
        for q in qubits
    ])
    y_beg_mom = ops.Moment([
        ops.Y(q) ** sympy.Symbol(f'{q}-Yi')
        for q in qubits
    ])
    x_end_mom = ops.Moment([
        ops.X(q) ** sympy.Symbol(f'{q}-Xf')
        for q in qubits
    ])
    y_end_mom = ops.Moment([
        ops.Y(q) ** sympy.Symbol(f'{q}-Yf')
        for q in qubits
    ])
    meas_mom = ops.Moment([ops.measure(*qubits, key='z')])
    if no_initialization:
        total_circuit = circuit.copy()
    else:
        total_circuit = circuits.Circuit([x_beg_mom, y_beg_mom])
        total_circuit += circuit.copy()
    total_circuit.append([x_end_mom, y_end_mom, meas_mom])
    return total_circuit


class StoppingCriteria(abc.ABC):
    """An abstract object that queries a BitstringAccumulator to figure out
    whether that `meas_spec` is complete."""

    @abc.abstractmethod
    def more_repetitions(self, accumulator) -> int:
        """Return the number of additional repetitions to take.

        StoppingCriteria should be respectful and have some notion of a
        maximum number of repetitions per chunk.
        """


@dataclasses.dataclass(frozen=True)
class VarianceStoppingCriteria(StoppingCriteria):
    """Stop sampling when average variance per term drops
    below a variance bound."""
    variance_bound: float
    repetitions_per_chunk: int = 10_000

    def more_repetitions(self, accumulator: BitstringAccumulator):
        if len(accumulator.bitstrings) == 0:
            return self.repetitions_per_chunk

        cov = accumulator.covariance()
        o = np.ones(cov.shape[0])
        sum_variance = o @ cov @ o.T
        var_of_the_e = sum_variance / len(accumulator.bitstrings)
        vpt = var_of_the_e / len(o)

        done = vpt <= self.variance_bound
        if done:
            return 0
        return self.repetitions_per_chunk


@dataclasses.dataclass(frozen=True)
class RepetitionsStoppingCriteria(StoppingCriteria):
    """Stop sampling when the number of repetitions has been reached."""
    total_repetitions: int
    repetitions_per_chunk: int = 10_000

    def more_repetitions(self, accumulator: BitstringAccumulator):
        done = accumulator.n_repetitions
        todo = self.total_repetitions - done
        if todo <= 0:
            return 0

        to_do_next = min(self.repetitions_per_chunk, todo)
        return to_do_next


_PAULI_TO_PARAM_VAL = {
    (ops.X, False): (0, -1 / 2),
    (ops.X, True): (0, +1 / 2),
    (ops.Y, False): (1 / 2, 0),
    (ops.Y, True): (-1 / 2, 0),
    (ops.Z, False): (0, 0),
    (ops.Z, True): (1, 0),
}

_STATE_TO_PARAM_VAL = {
    value.KET_PLUS: (0, +1 / 2),
    value.KET_MINUS: (0, -1 / 2),
    value.KET_IMAG: (-1 / 2, 0),
    value.KET_MINUS_IMAG: (+1 / 2, 0),
    value.KET_ZERO: (0, 0),
    value.KET_ONE: (1, 0),
}


def _get_params_for_setting(setting: InitObsSetting,
                            flips: Iterable[bool],
                            qubits: Sequence['cirq.Qid'],
                            no_initialization: bool,
                            ) -> 'cirq.ParamDictType':
    """Return the parameter dictionary for the given setting.

    This must be used in conjunction with a circuit generated by
    `_with_parameterized_layers`. `flips` (used for readout symmetrization)
    should be of the same length as `qubits` and will modify the parameters
    to also include a bit flip (`X`). Code responsible for running the
    circuit should make sure to flip bits back prior to analysis.
    """
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
        xf_param, yf_param = _PAULI_TO_PARAM_VAL[pauli, flip]
        params[f'{qubit}-Xf'] = xf_param
        params[f'{qubit}-Yf'] = yf_param

        if not no_initialization:
            state = setting.init_state[qubit]
            xi_param, yi_param = _STATE_TO_PARAM_VAL[state]
            params[f'{qubit}-Xi'] = xi_param
            params[f'{qubit}-Yi'] = yi_param

    return params


def _aggregate_n_repetitions(next_chunk_repetitions: Set[int]):
    """In theory, each stopping criteria can request a different number
    of repetitions for the next chunk. For batching efficiency, we take the
    max and issue a warning in this case."""
    if len(next_chunk_repetitions) == 1:
        return list(next_chunk_repetitions)[0]

    reps = max(next_chunk_repetitions)
    warnings.warn("Your stopping criteria recommended a various numbers of "
                  "repetitions to perform next. So we can submit as a single "
                  "sweep, we will be taking the maximum of {}".format(reps))
    return reps


def _check_meas_specs_still_todo(
        meas_specs: List[MeasurementSpec],
        accumulators: Dict[MeasurementSpec, BitstringAccumulator],
        stopping_criteria: StoppingCriteria):
    """Filter `meas_specs` in case some are done.

    In the sampling loop in `measure_grouped_settings`, we submit
    each `meas_spec` in chunks. This function contains the logic for
    removing `meas_spec`s from the loop if they are done.
    """
    still_todo = []
    repetitions = set()
    for meas_spec in meas_specs:
        accumulator = accumulators[meas_spec]
        more_repetitions = stopping_criteria.more_repetitions(accumulator)

        if more_repetitions < 0:
            raise ValueError("Stopping criteria's `more_repetitions` "
                             "should return 0 or a positive number")
        if more_repetitions == 0:
            continue

        repetitions.add(more_repetitions)
        still_todo.append(meas_spec)

    if len(still_todo) == 0:
        repetitions = 0
        return still_todo, repetitions

    repetitions = _aggregate_n_repetitions(repetitions)
    total_repetitions = len(still_todo) * repetitions
    if total_repetitions > 3_000_000:
        old_repetitions = repetitions
        repetitions = 3_000_000 // len(still_todo)

        if repetitions < 10:
            raise ValueError("Too many parameter settings. Split it up.")

        warnings.warn(
            "You've requested a lot of parameters. We're throttling the "
            "number of shots per call to run_sweep (per parameter value) "
            "from {} to {}".format(old_repetitions, repetitions))

    return still_todo, repetitions


def _pad_setting(max_setting: InitObsSetting,
                 qubits: List['cirq.Qid'],
                 pad_init_state_with=value.KET_ZERO,
                 pad_obs_with=ops.Z) -> InitObsSetting:
    """Pad max_setting's init_state and observable with `pad_xx_with` ops
    (default |0>, Z) so each max_setting has the same qubits. We need this
    to be the case so we can fill in all the parameters.
    """
    obs = max_setting.observable
    assert obs.coefficient == 1, "Only max_setting's should be padded."
    new_obs = obs.copy()
    for qubit in qubits:
        if not qubit in new_obs:
            new_obs *= pad_obs_with(qubit)

    init_state = max_setting.init_state
    init_state_original_qubits = init_state.qubits
    for qubit in qubits:
        if not qubit in init_state_original_qubits:
            init_state *= pad_init_state_with(qubit)

    return InitObsSetting(init_state=init_state,
                          observable=new_obs)


def _trick_into_sweep(param_tuples):
    """Turn param tuples into a sweep.

    TODO: this may no longer be necessary.
    """
    trick_into_sweep = [dict(pt) for pt in param_tuples]
    trick_into_sweep = study.to_sweep(trick_into_sweep)
    return trick_into_sweep


@dataclasses.dataclass(frozen=True)
class _FlippyMeasSpec:
    """Internally, each MeasurementSpec class is split into two
    _FlippyMeasSpecs to support readout symmetrization.

    Bitstring results are combined, so this should be opaque to the user.
    """
    meas_spec: MeasurementSpec
    flips: np.ndarray
    qubits: Sequence['cirq.Qid']

    def param_tuples(self, no_initialization=False):
        yield from _get_params_for_setting(self.meas_spec.max_setting, flips=self.flips,
                                           qubits=self.qubits,
                                           no_initialization=no_initialization).items()
        yield from self.meas_spec.circuit_params.items()


def _subdivide_meas_specs(meas_specs: Iterable[MeasurementSpec],
                          repetitions: int,
                          qubits: Sequence['cirq.Qid'],
                          readout_symmetrization: bool) \
        -> Tuple[List[_FlippyMeasSpec], int]:
    """Split measurement specs into sub-jobs for readout symmetrization

    In readout symmetrization, we first run the "normal" circuit followed
    by running the circuit with flipped measurement.
    One MeasurementSpec is split into two _FlippyMeasSpecs. These are run
    separately but accumulated according to their shared MeasurementSpec.
    """
    n_qubits = len(qubits)
    flippy_mspecs = []
    for meas_spec in meas_specs:
        all_normal = np.zeros(n_qubits, dtype=bool)
        flippy_mspecs.append(_FlippyMeasSpec(
            meas_spec=meas_spec,
            flips=all_normal,
            qubits=qubits,
        ))

        if readout_symmetrization:
            all_flipped = np.ones(n_qubits, dtype=bool)
            flippy_mspecs.append(_FlippyMeasSpec(
                meas_spec=meas_spec,
                flips=all_flipped,
                qubits=qubits,
            ))

    if readout_symmetrization:
        repetitions //= 2

    return flippy_mspecs, repetitions


def _get_qubits(max_settings: Iterable[InitObsSetting]):
    """Helper function to find all the qubits in a suite of settings used
    in `measure_grouped_settings`.

    Qubits are returned in sorted order.
    """
    qubits = set()
    for max_setting in max_settings:
        max_in_st = max_setting.init_state
        qubits |= set(max_in_st.qubits)
    return sorted(qubits)


def _is_all_zeros_init(init_state: value.ProductState):
    """Helper function to determine whether we need the initial layer of
    rotations used in `measure_grouped_settings`.
    """
    for q, st in init_state:
        if st != value.KET_ZERO:
            return False

    return True


def _parse_checkpoint_options(
        checkpoint: bool,
        checkpoint_fn: Optional[str],
        checkpoint_other_fn: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """The user can specify these three arguments. This function
    contains the validation and defaults logic.
    """
    if not checkpoint:
        if checkpoint_fn is not None or checkpoint_other_fn is not None:
            raise ValueError("Checkpoint filenames were provided "
                             "but `checkpoint` was set to False.")
        return None, None

    if checkpoint_fn is None:
        checkpoint_dir = tempfile.mkdtemp()
        chk_basename = 'observables'
        checkpoint_fn = f'{checkpoint_dir}/{chk_basename}.json'

    if checkpoint_other_fn is None:
        checkpoint_dir = os.path.dirname(checkpoint_fn)
        chk_basename = os.path.basename(checkpoint_fn)
        chk_basename, _, ext = chk_basename.rpartition('.')
        if ext != 'json':
            raise ValueError("Please use a `.json` filename or fully "
                             "specify checkpoint_fn and checkpoint_other_fn")
        checkpoint_other_fn = f'{checkpoint_dir}/{chk_basename}.prev.json'

    print(f"We will save checkpoint files to {checkpoint_fn}")
    return checkpoint_fn, checkpoint_other_fn


def measure_grouped_settings(
        circuit: circuits.Circuit,
        grouped_settings: Dict[InitObsSetting, List[InitObsSetting]],
        sampler: 'cirq.Sampler',
        stopping_criteria: StoppingCriteria,
        *,
        readout_symmetrization: bool = False,
        circuit_sweep: 'cirq.Sweepable' = None,
        readout_calibrations: Optional[BitstringAccumulator] = None,
        checkpoint: bool = False,
        checkpoint_fn: Optional[str] = None,
        checkpoint_other_fn: Optional[str] = None,

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
        checkpoint: If set to True, save cumulative raw results at the end
            of each iteration of the sampling loop.
        checkpoint_fn: The filename for the checkpoint file. If `checkpoint`
            is set to True and this is not specified, a file in a temporary
            directory will be used.
        checkpoint_other_fn: The filename for another checkpoint file, which
            contains the previous checkpoint. If `checkpoint`
            is set to True and this is not specified, a file in a temporary
            directory will be used. If `checkpoint` is set to True and
            `checkpoint_fn` is specified but this argument is *not* specified,
            "{checkpoint_fn}.prev.json" will be used.
    """

    checkpoint_fn, checkpoint_other_fn = _parse_checkpoint_options(
        checkpoint=checkpoint, checkpoint_fn=checkpoint_fn,
        checkpoint_other_fn=checkpoint_other_fn)
    qubits = _get_qubits(grouped_settings.keys())
    qubit_to_index = {q: i for i, q in enumerate(qubits)}

    no_initialization = True
    for max_setting in grouped_settings.keys():
        if not _is_all_zeros_init(max_setting.init_state):
            no_initialization = False
            break

    measurement_param_circuit = _with_parameterized_layers(circuit, qubits, no_initialization)
    grouped_settings = {_pad_setting(max_setting, qubits): settings
                        for max_setting, settings in grouped_settings.items()}

    if circuit_sweep is None:
        circuit_sweep = study.UnitSweep

    # meas_spec provides a key for accumulators.
    # meas_specs_todo is a mutable list. We will pop things from it as various
    # specs are measured to the satisfaction of the stopping criteria
    accumulators = {}
    meas_specs_todo = []
    for max_setting, circuit_params in itertools.product(grouped_settings.keys(),
                                                         circuit_sweep.param_tuples()):
        # The type annotation for Param is just `Iterable`.
        # We make sure that it's truly a tuple.
        circuit_params = dict(circuit_params)

        meas_spec = MeasurementSpec(
            max_setting=max_setting,
            circuit_params=circuit_params)
        accumulator = BitstringAccumulator(
            meas_spec=meas_spec,
            simul_settings=grouped_settings[max_setting],
            qubit_to_index=qubit_to_index,
            readout_calibration=readout_calibrations)
        accumulators[meas_spec] = accumulator
        meas_specs_todo += [meas_spec]

    while True:
        meas_specs_todo, repetitions = _check_meas_specs_still_todo(
            meas_specs=meas_specs_todo,
            accumulators=accumulators,
            stopping_criteria=stopping_criteria)
        if len(meas_specs_todo) == 0:
            break

        flippy_meas_specs, repetitions = _subdivide_meas_specs(
            meas_specs=meas_specs_todo,
            repetitions=repetitions,
            qubits=qubits,
            readout_symmetrization=readout_symmetrization)

        resolved_params = [flippy_ms.param_tuples(no_initialization=no_initialization)
                           for flippy_ms in flippy_meas_specs]
        resolved_params = _trick_into_sweep(resolved_params)
        total_samples = repetitions * len(resolved_params)
        print("Total samples", total_samples)

        results = sampler.run_sweep(
            program=measurement_param_circuit,
            params=resolved_params,
            repetitions=repetitions)

        assert len(results) == len(flippy_meas_specs)
        for flippy_ms, result in zip(flippy_meas_specs, results):
            accumulator = accumulators[flippy_ms.meas_spec]
            bitstrings = np.logical_xor(flippy_ms.flips, result.measurements['z'])
            accumulator.consume_results(bitstrings)

        if checkpoint:
            if os.path.exists(checkpoint_fn):
                os.rename(checkpoint_fn, checkpoint_other_fn)
            protocols.to_json(list(accumulators.values()), checkpoint_fn)

    return list(accumulators.values())


_GROUPING_FUNCS = {
    'greedy': group_settings_greedy,
}

_STOPPING_CRITS = {
    'repetitions': RepetitionsStoppingCriteria,
    'variance': VarianceStoppingCriteria,
}


def _parse_stopping_criteria(stopping_criteria: Union[str, StoppingCriteria],
                             stopping_criteria_val: Optional[float] = None) -> StoppingCriteria:
    """Logic for turning a named stopping_criteria and value to one of the
    built-in stopping criteria in support of an easy high-level API,
    see `measure_observables`.
    """
    if isinstance(stopping_criteria, str):
        stopping_criteria_cls = _STOPPING_CRITS[stopping_criteria]
        stopping_criteria = stopping_criteria_cls(stopping_criteria_val)
    return stopping_criteria


def measure_observables(
        circuit: circuits.Circuit,
        observables: Iterable[ops.PauliString],
        sampler: Union['cirq.Simulator', 'cirq.Sampler'],
        stopping_criteria: Union[str, StoppingCriteria],
        stopping_criteria_val: Optional[float] = None,
        *,
        readout_symmetrization=True,
        circuit_sweep: 'cirq.Sweepable' = None,
        grouper=group_settings_greedy,
        readout_calibrations: Optional[BitstringAccumulator] = None,
        checkpoint: bool = False,
        checkpoint_fn: Optional[str] = None,
        checkpoint_other_fn: Optional[str] = None,
):
    """Measure a suite of PauliSum observables.

    If you need more control over the process, please see
    `measure_grouped_settings` for a lower-level API.
    If you would like your results returned as a pandas DataFrame,
    please see `measure_observables_df`.

    Args:
        circuit: The circuit. This can contain parameters, in which case
            you should also specify `circuit_sweep`.
        observables: A collection of PauliString observables to measure.
            These will be grouped into simultaneously-measurable groups,
            see `grouper` argument.
        sampler: A sampler.
        stopping_criteria: Either a StoppingCriteria object or one of
            'variance', 'repetitions'. In the latter case, you must
            also specify `stopping_criteria_val`.
        stopping_criteria_val: The value used for named stopping criteria.
            If you specified 'repetitions', this is the number of repetitions.
            If you specified 'variance', this is the variance.
        readout_symmetrization: If set to True, each run will be
            split into two: one normal and one where a bit flip is
            incorporated prior to measurement. In the latter case, the
            measured bit will be flipped back classically and accumulated
            together. This causes readout error to appear symmetric,
            p(0|0) = p(1|1).
        circuit_sweep: Additional parameter sweeps for parameters contained
            in `circuit`. The total sweep is the product of the circuit sweep
            with parameter settings for the single-qubit basis-change rotations.
        grouper: Either "greedy" or a function that groups lists of
            `InitObsSetting`. See the documentation for the `grouped_settings`
            argument of `measure_grouped_settings` for full details.
        readout_calibrations: The result of `calibrate_readout_error`.
        checkpoint: If set to True, save cumulative raw results at the end
            of each iteration of the sampling loop.
        checkpoint_fn: The filename for the checkpoint file. If `checkpoint`
            is set to True and this is not specified, a file in a temporary
            directory will be used.
        checkpoint_other_fn: The filename for another checkpoint file, which
            contains the previous checkpoint. If `checkpoint`
            is set to True and this is not specified, a file in a temporary
            directory will be used. If `checkpoint` is set to True and
            `checkpoint_fn` is specified but this argument is *not* specified,
            "{checkpoint_fn}.prev.json" will be used.
    """
    qubits = set()
    for obs in observables:
        qubits |= set(obs.qubits)
    qubits |= circuit.all_qubits()
    qubits = sorted(qubits)
    settings = list(observables_to_settings(observables, qubits))

    if isinstance(grouper, str):
        try:
            grouper = _GROUPING_FUNCS[grouper.lower()]
        except KeyError:
            raise ValueError("Unknown grouping function {}".format(grouper))
    grouped_settings = grouper(settings)

    stopping_criteria = _parse_stopping_criteria(
        stopping_criteria, stopping_criteria_val)

    return measure_grouped_settings(
        circuit=circuit,
        grouped_settings=grouped_settings,
        sampler=sampler,
        stopping_criteria=stopping_criteria,
        circuit_sweep=circuit_sweep,
        readout_symmetrization=readout_symmetrization,
        readout_calibrations=readout_calibrations,
        checkpoint=checkpoint,
        checkpoint_fn=checkpoint_fn,
        checkpoint_other_fn=checkpoint_other_fn,
    )


def measure_observables_df(
        circuit: circuits.Circuit,
        observables: Iterable[ops.PauliString],
        sampler: Union['cirq.Simulator', 'cirq.Sampler'],
        stopping_criteria: Union[str, StoppingCriteria],
        stopping_criteria_val: Optional[float] = None,
        params: 'cirq.Sweepable' = None,
        grouper=group_settings_greedy,
        symmetrize_readout=True,
        readout_calibrations: Optional[BitstringAccumulator] = None,
):
    """Measure observables and return resulting data as a dataframe."""
    accumulators = measure_observables(
        circuit=circuit, observables=observables, sampler=sampler,
        stopping_criteria=stopping_criteria, stopping_criteria_val=stopping_criteria_val,
        circuit_sweep=params, grouper=grouper, readout_symmetrization=symmetrize_readout,
        readout_calibrations=readout_calibrations, checkpoint=True,
    )

    df = pd.DataFrame(list(itertools.chain.from_iterable(
        acc.records for acc in accumulators)))
    return df


def calibrate_readout_error(
        qubits: Iterable[ops.Qid],
        sampler: Union['cirq.Simulator', 'cirq.Sampler'],
        stopping_criteria: Union[str, StoppingCriteria],
        stopping_criteria_val: Optional[float] = None,
):
    stopping_criteria = _parse_stopping_criteria(
        stopping_criteria, stopping_criteria_val)

    # We know there won't be any fancy sweeps or observables so we can
    # get away with more repetitions per job
    stopping_criteria = dataclasses.replace(stopping_criteria,
                                            repetitions_per_chunk=100_000)

    # Simultaneous readout characterization:
    # We can measure all qubits simultaneously (i.e. _max_setting is ZZZ..ZZ
    # for all qubits). We will extract individual qubit quantities, so there
    # are `n_qubits` TomographySettings, each responsible for one <Z>.
    #
    # Readout symmetrization means we just need to measure the "identity"
    # circuit. In reality, this corresponds to measuring I for half the time
    # and X for the other half.
    init_state = zeros_state(qubits)
    max_setting = InitObsSetting(
        init_state=init_state,
        observable=ops.PauliString({q: ops.Z for q in qubits})
    )
    grouped_settings = {max_setting: [
        InitObsSetting(init_state=init_state,
                       observable=ops.PauliString({q: ops.Z}))
        for q in qubits
    ]}

    result = measure_grouped_settings(
        circuit=circuits.Circuit(),
        grouped_settings=grouped_settings,
        sampler=sampler,
        stopping_criteria=stopping_criteria,
        circuit_sweep=study.UnitSweep,
        readout_symmetrization=True,
    )
    result = list(result)
    assert len(result) == 1
    result = result[0]
    return result
