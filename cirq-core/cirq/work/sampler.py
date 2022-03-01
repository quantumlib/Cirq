# Copyright 2018 The Cirq Developers
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
"""Abstract base class for things sampling quantum circuits."""

import abc
import collections
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import pandas as pd

from cirq import ops, protocols, study
from cirq.work.observable_measurement import (
    measure_observables,
    RepetitionsStoppingCriteria,
    CheckpointFileOptions,
)
from cirq.work.observable_settings import _hashable_param

if TYPE_CHECKING:
    import cirq


class Sampler(metaclass=abc.ABCMeta):
    """Something capable of sampling quantum circuits. Simulator or hardware."""

    def run(
        self,
        program: 'cirq.AbstractCircuit',
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        repetitions: int = 1,
    ) -> 'cirq.Result':
        """Samples from the given Circuit.

        Args:
            program: The circuit to sample from.
            param_resolver: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result for a run.
        """
        return self.run_sweep(program, param_resolver, repetitions)[0]

    async def run_async(
        self,
        program: 'cirq.AbstractCircuit',
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        repetitions: int = 1,
    ) -> 'cirq.Result':
        """Asynchronously samples from the given Circuit.

        Args:
            program: The circuit to sample from.
            param_resolver: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result for a run.
        """
        results = await self.run_sweep_async(program, param_resolver, repetitions)
        return results[0]

    def sample(
        self,
        program: 'cirq.AbstractCircuit',
        *,
        repetitions: int = 1,
        params: 'cirq.Sweepable' = None,
    ) -> 'pd.DataFrame':
        """Samples the given Circuit, producing a pandas data frame.

        Args:
            program: The circuit to sample from.
            repetitions: The number of times to sample the program, for each
                parameter mapping.
            params: Maps symbols to one or more values. This argument can be
                a dictionary, a list of dictionaries, a `cirq.Sweep`, a list of
                `cirq.Sweep`, etc. The program will be sampled `repetition`
                times for each mapping. Defaults to a single empty mapping.

        Returns:
            A `pandas.DataFrame` with a row for each sample, and a column for
            each measurement key as well as a column for each symbolic
            parameter.  Measurement results are stored as a big endian integer
            representation with one bit for each measured qubit in the key.
            See `cirq.big_endian_int_to_bits` and similar functions for how
            to convert this integer into bits.
            There is an also index column containing the repetition number,
            for each parameter assignment.

        Raises:
            ValueError: If a supplied sweep is invalid.

        Examples:
            >>> a, b, c = cirq.LineQubit.range(3)
            >>> sampler = cirq.Simulator()
            >>> circuit = cirq.Circuit(cirq.X(a),
            ...                        cirq.measure(a, key='out'))
            >>> print(sampler.sample(circuit, repetitions=4))
               out
            0    1
            1    1
            2    1
            3    1

            >>> circuit = cirq.Circuit(cirq.X(a),
            ...                        cirq.CNOT(a, b),
            ...                        cirq.measure(a, b, c, key='out'))
            >>> print(sampler.sample(circuit, repetitions=4))
               out
            0    6
            1    6
            2    6
            3    6

            >>> circuit = cirq.Circuit(cirq.X(a)**sympy.Symbol('t'),
            ...                        cirq.measure(a, key='out'))
            >>> print(sampler.sample(
            ...     circuit,
            ...     repetitions=3,
            ...     params=[{'t': 0}, {'t': 1}]))
               t  out
            0  0    0
            1  0    0
            2  0    0
            0  1    1
            1  1    1
            2  1    1
        """

        sweeps_list = study.to_sweeps(params)
        keys = sorted(sweeps_list[0].keys) if sweeps_list else []
        for sweep in sweeps_list:
            if sweep and set(sweep.keys) != set(keys):
                raise ValueError(
                    'Inconsistent sweep parameters. '
                    f'One sweep had {repr(keys)} '
                    f'while another had {repr(sorted(sweep.keys))}.'
                )

        results = []
        for sweep in sweeps_list:
            sweep_results = self.run_sweep(program, params=sweep, repetitions=repetitions)
            for resolver, result in zip(sweep, sweep_results):
                param_values_once = [resolver.value_of(key) for key in keys]
                param_table = pd.DataFrame(data=[param_values_once] * repetitions, columns=keys)
                results.append(pd.concat([param_table, result.data], axis=1))

        return pd.concat(results)

    @abc.abstractmethod
    def run_sweep(
        self,
        program: 'cirq.AbstractCircuit',
        params: 'cirq.Sweepable',
        repetitions: int = 1,
    ) -> Sequence['cirq.Result']:
        """Samples from the given Circuit.

        This allows for sweeping over different parameter values,
        unlike the `run` method.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result list for this run; one for each possible parameter resolver.
        """

    async def run_sweep_async(
        self,
        program: 'cirq.AbstractCircuit',
        params: 'cirq.Sweepable',
        repetitions: int = 1,
    ) -> Sequence['cirq.Result']:
        """Asynchronously samples from the given Circuit.

        By default, this method invokes `run_sweep` synchronously and simply
        exposes its result is an awaitable. Child classes that are capable of
        true asynchronous sampling should override it to use other strategies.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result list for this run; one for each possible parameter resolver.
        """
        return self.run_sweep(program, params=params, repetitions=repetitions)

    def run_batch(
        self,
        programs: Sequence['cirq.AbstractCircuit'],
        params_list: Optional[List['cirq.Sweepable']] = None,
        repetitions: Union[int, List[int]] = 1,
    ) -> Sequence[Sequence['cirq.Result']]:
        """Runs the supplied circuits.

        Each circuit provided in `programs` will pair with the optional
        associated parameter sweep provided in the `params_list`, and be run
        with the associated repetitions provided in `repetitions` (if
        `repetitions` is an integer, then all runs will have that number of
        repetitions). If `params_list` is specified, then the number of
        circuits is required to match the number of sweeps. Similarly, when
        `repetitions` is a list, the number of circuits is required to match
        the length of this list.

        By default, this method simply invokes `run_sweep` sequentially for
        each (circuit, parameter sweep, repetitions) tuple. Child classes that
        are capable of sampling batches more efficiently should override it to
        use other strategies. Note that child classes may have certain
        requirements that must be met in order for a speedup to be possible,
        such as a constant number of repetitions being used for all circuits.
        Refer to the documentation of the child class for any such requirements.

        Args:
            programs: The circuits to execute as a batch.
            params_list: Parameter sweeps to use with the circuits. The number
                of sweeps should match the number of circuits and will be
                paired in order with the circuits.
            repetitions: Number of circuit repetitions to run. Can be specified
                as a single value to use for all runs, or as a list of values,
                one for each circuit.

        Returns:
            A list of lists of TrialResults. The outer list corresponds to
            the circuits, while each inner list contains the TrialResults
            for the corresponding circuit, in the order imposed by the
            associated parameter sweep.

        Raises:
            ValueError: If length of `programs` is not equal to the length
                of `params_list` or the length of `repetitions`.
        """
        if params_list is None:
            params_list = [None] * len(programs)
        if len(programs) != len(params_list):
            raise ValueError(
                'len(programs) and len(params_list) must match. '
                f'Got {len(programs)} and {len(params_list)}.'
            )
        if isinstance(repetitions, int):
            repetitions = [repetitions] * len(programs)
        if len(programs) != len(repetitions):
            raise ValueError(
                'len(programs) and len(repetitions) must match. '
                f'Got {len(programs)} and {len(repetitions)}.'
            )
        return [
            self.run_sweep(circuit, params=params, repetitions=repetitions)
            for circuit, params, repetitions in zip(programs, params_list, repetitions)
        ]

    def sample_expectation_values(
        self,
        program: 'cirq.AbstractCircuit',
        observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
        *,
        num_samples: int,
        params: 'cirq.Sweepable' = None,
        permit_terminal_measurements: bool = False,
    ) -> Sequence[Sequence[float]]:
        """Calculates estimated expectation values from samples of a circuit.

        Please see also `cirq.work.measure_observables` for more control over how to measure
        a suite of observables.

        This method can be run on any device or simulator that supports circuit sampling. Compare
        with `simulate_expectation_values` in simulator.py, which is limited to simulators
        but provides exact results.

        Args:
            program: The circuit which prepares a state from which we sample expectation values.
            observables: A list of observables for which to calculate expectation values.
            num_samples: The number of samples to take. Increasing this value increases the
                statistical accuracy of the estimate.
            params: Parameters to run with the program.
            permit_terminal_measurements: If the provided circuit ends in a measurement, this
                method will generate an error unless this is set to True. This is meant to
                prevent measurements from ruining expectation value calculations.

        Returns:
            A list of expectation-value lists. The outer index determines the sweep, and the inner
            index determines the observable. For instance, results[1][3] would select the fourth
            observable measured in the second sweep.

        Raises:
            ValueError: If the number of samples was not positive, if empty observables were
                supplied, or if the provided circuit has terminal measurements and
                `permit_terminal_measurements` is true.
        """
        if num_samples <= 0:
            raise ValueError(
                f'Expectation values require at least one sample. Received: {num_samples}.'
            )
        if not observables:
            raise ValueError('At least one observable must be provided.')
        if not permit_terminal_measurements and program.are_any_measurements_terminal():
            raise ValueError(
                'Provided circuit has terminal measurements, which may '
                'skew expectation values. If this is intentional, set '
                'permit_terminal_measurements=True.'
            )

        # Wrap input into a list of pauli sum
        pauli_sums: List['cirq.PauliSum'] = (
            [ops.PauliSum.wrap(o) for o in observables]
            if isinstance(observables, List)
            else [ops.PauliSum.wrap(observables)]
        )
        del observables

        # Flatten Pauli Sum into one big list of Pauli String
        # Keep track of which Pauli Sum each one was from.
        flat_pstrings: List['cirq.PauliString'] = []
        pstring_to_psum_i: Dict['cirq.PauliString', int] = {}
        for psum_i, pauli_sum in enumerate(pauli_sums):
            for pstring in pauli_sum:
                flat_pstrings.append(pstring)
                pstring_to_psum_i[pstring] = psum_i

        # Flatten Circuit Sweep into one big list of Params.
        # Keep track of their indices so we can map back.
        flat_params: List[Dict[str, float]] = [pr.param_dict for pr in study.to_resolvers(params)]
        circuit_param_to_sweep_i: Dict[FrozenSet[Tuple[str, float]], int] = {
            _hashable_param(param.items()): i for i, param in enumerate(flat_params)
        }

        obs_meas_results = measure_observables(
            circuit=program,
            observables=flat_pstrings,
            sampler=self,
            stopping_criteria=RepetitionsStoppingCriteria(total_repetitions=num_samples),
            readout_symmetrization=False,
            circuit_sweep=params,
            checkpoint=CheckpointFileOptions(checkpoint=False),
        )

        # Results are ordered by how they're grouped. Since we want the (circuit_sweep, pauli_sum)
        # nesting structure, we place the measured values according to the back-mappings we set up
        # above. We also do the sum operation to aggregate multiple PauliString measured values
        # for a given PauliSum.
        nested_results: List[List[float]] = [[0] * len(pauli_sums) for _ in range(len(flat_params))]
        for res in obs_meas_results:
            param_i = circuit_param_to_sweep_i[_hashable_param(res.circuit_params.items())]
            psum_i = pstring_to_psum_i[res.setting.observable]
            nested_results[param_i][psum_i] += res.mean

        return nested_results

    @staticmethod
    def _get_measurement_shapes(
        circuit: 'cirq.AbstractCircuit',
    ) -> Dict[str, Tuple[int, Tuple[int, ...]]]:
        """Gets the shapes of measurements in the given circuit.

        Returns:
            A mapping from measurement key name to a tuple of (num_instances, qid_shape),
            where num_instances is the number of times that key appears in the circuit and
            qid_shape is the shape of measured qubits for the key, as determined by the
            `cirq.qid_shape` protocol.

        Raises:
            ValueError: if the qid_shape of different instances of the same measurement
            key disagree.
        """
        qid_shapes: Dict[str, Tuple[int, ...]] = {}
        num_instances: Dict[str, int] = collections.Counter()
        for op in circuit.all_operations():
            key = protocols.measurement_key_name(op, default=None)
            if key is not None:
                qid_shape = protocols.qid_shape(op)
                prev_qid_shape = qid_shapes.setdefault(key, qid_shape)
                if qid_shape != prev_qid_shape:
                    raise ValueError(
                        "Different qid shapes for repeated measurement: "
                        f"key={key!r}, prev_qid_shape={prev_qid_shape}, qid_shape={qid_shape}"
                    )
                num_instances[key] += 1
        return {k: (num_instances[k], qid_shape) for k, qid_shape in qid_shapes.items()}
