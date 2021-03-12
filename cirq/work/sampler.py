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

from typing import List, Optional, TYPE_CHECKING, Union
import abc

import pandas as pd

from cirq import ops, study
from cirq.work import group_settings_greedy, observables_to_settings
from cirq.work.observable_measurement import (
    _get_params_for_setting,
    _with_parameterized_layers,
)

if TYPE_CHECKING:
    import cirq


class Sampler(metaclass=abc.ABCMeta):
    """Something capable of sampling quantum circuits. Simulator or hardware."""

    def run(
        self,
        program: 'cirq.Circuit',
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        repetitions: int = 1,
    ) -> 'cirq.Result':
        """Samples from the given Circuit.

        By default, the `run_async` method invokes this method on another
        thread. So this method is supposed to be thread safe.

        Args:
            program: The circuit to sample from.
            param_resolver: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result for a run.
        """
        return self.run_sweep(program, study.ParamResolver(param_resolver), repetitions)[0]

    def sample(
        self,
        program: 'cirq.Circuit',
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

    def sample_expectation_values(
        self,
        program: 'cirq.Circuit',
        observables: Union['cirq.PauliSumLike', List['cirq.PauliSumLike']],
        *,
        num_samples: int,
        params: 'cirq.Sweepable' = None,
        permit_terminal_measurements: bool = False,
    ) -> List[List[float]]:
        """Calculates estimated expectation values from samples of a circuit.

        This method can be run on any device or simulator that supports circuit
        sampling. Compare with `simulate_expectation_values` in simulator.py,
        which is limited to simulators but provides exact results.

        Args:
            program: The circuit to simulate.
            observables: A list of observables for which to calculate
                expectation values.
            num_samples: The number of samples to take. Increasing this value
                increases the accuracy of the estimate.
            params: Parameters to run with the program.
            permit_terminal_measurements: If the provided circuit ends in a
                measurement, this method will generate an error unless this
                is set to True. This is meant to prevent measurements from
                ruining expectation value calculations.

        Returns:
            A list of expectation-value lists. The outer index determines the
            sweep, and the inner index determines the observable. For instance,
            results[1][3] would select the fourth observable measured in the
            second sweep.
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
        qubits = ops.QubitOrder.DEFAULT.order_for(program.all_qubits())
        qmap = {q: i for i, q in enumerate(qubits)}
        num_qubits = len(qubits)
        psums = (
            [ops.PauliSum.wrap(o) for o in observables]
            if isinstance(observables, List)
            else [ops.PauliSum.wrap(observables)]
        )

        obs_pstrings: List[ops.PauliString] = []
        for psum in psums:
            pstring_aggregate = {}
            for pstring in psum:
                pstring_aggregate.update(pstring.items())
            obs_pstrings.append(ops.PauliString(pstring_aggregate))

        # List of observable settings with the same indexing as 'observables'.
        obs_settings = list(observables_to_settings(obs_pstrings, qubits))
        # Map of setting-groups to the settings they cover.
        sampling_groups = group_settings_greedy(obs_settings)
        sampling_params = {
            max_setting: _get_params_for_setting(
                setting=max_setting,
                flips=[False] * num_qubits,
                qubits=qubits,
                needs_init_layer=False,
            )
            for max_setting in sampling_groups
        }

        # Parameterized circuit for observable measurement.
        mod_program = _with_parameterized_layers(program, qubits, needs_init_layer=False)

        input_params = list(params) if params else {}
        num_input_param_values = len(list(study.to_resolvers(input_params)))
        # Pairing of input sweeps with each required observable rotation.
        sweeps = study.to_sweeps(input_params)
        mod_sweep = study.ListSweep(sampling_params.values())
        all_sweeps = [study.Product(sweep, mod_sweep) for sweep in sweeps]

        # Results sampled from the modified circuit. Parameterization ensures
        # that all 'z' results map directly to observables.
        samples = self.sample(mod_program, repetitions=num_samples, params=all_sweeps)
        results: List[List[float]] = [[0] * len(psums) for _ in range(num_input_param_values)]

        for max_setting, grouped_settings in sampling_groups.items():
            ev_params = sampling_params[max_setting]
            # A set of num_samples integer-packed results for this setting.
            series = [True] * len(samples)
            for k, v in ev_params.items():
                series = series & (samples[k] == v)

            for sweep_idx, pr in enumerate(study.to_resolvers(input_params)):
                subseries = series
                for k in pr:
                    subseries = subseries & (samples[k] == pr.value_of(k))
                int_results = samples.loc[subseries, 'z']
                for setting in grouped_settings:
                    obs_idx = obs_settings.index(setting)
                    psum = psums[obs_idx]
                    ev = 0
                    for pstr in psum:
                        sum_val = 0
                        for int_result in int_results:
                            val = 1
                            for q in pstr.qubits:
                                bit_idx = num_qubits - qmap[q] - 1
                                if (int_result >> bit_idx) & 1:
                                    val *= -1
                            sum_val += val
                        ev += sum_val * pstr.coefficient
                    results[sweep_idx][obs_idx] = ev / num_samples

        return results

    @abc.abstractmethod
    def run_sweep(
        self,
        program: 'cirq.Circuit',
        params: 'cirq.Sweepable',
        repetitions: int = 1,
    ) -> List['cirq.Result']:
        """Samples from the given Circuit.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result list for this run; one for each possible parameter
            resolver.
        """

    async def run_async(self, program: 'cirq.Circuit', *, repetitions: int) -> 'cirq.Result':
        """Asynchronously samples from the given Circuit.

        By default, this method invokes `run` synchronously and simply exposes
        its result is an awaitable. Child classes that are capable of true
        asynchronous sampling should override it to use other strategies.

        Args:
            program: The circuit to sample from.
            repetitions: The number of times to sample.

        Returns:
            An awaitable Result.
        """
        return self.run(program, repetitions=repetitions)

    async def run_sweep_async(
        self,
        program: 'cirq.Circuit',
        params: 'cirq.Sweepable',
        repetitions: int = 1,
    ) -> List['cirq.Result']:
        """Asynchronously sweeps and samples from the given Circuit.

        By default, this method invokes `run_sweep` synchronously and simply
        exposes its result is an awaitable. Child classes that are capable of
        true asynchronous sampling should override it to use other strategies.

        Args:
            program: The circuit to sample from.
            params: One or more mappings from parameter keys to parameter values
                to use. For each parameter assignment, `repetitions` samples
                will be taken.
            repetitions: The number of times to sample.

        Returns:
            An awaitable Result.
        """
        return self.run_sweep(program, params=params, repetitions=repetitions)

    def run_batch(
        self,
        programs: List['cirq.Circuit'],
        params_list: Optional[List['cirq.Sweepable']] = None,
        repetitions: Union[int, List[int]] = 1,
    ) -> List[List['cirq.Result']]:
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
