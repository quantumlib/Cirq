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

from typing import List, TYPE_CHECKING
import abc

import pandas as pd

from cirq import study

if TYPE_CHECKING:
    import cirq


class Sampler(metaclass=abc.ABCMeta):
    """Something capable of sampling quantum circuits. Simulator or hardware."""

    def run(
            self,
            program: 'cirq.Circuit',
            param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
            repetitions: int = 1,
    ) -> 'cirq.TrialResult':
        """Samples from the given Circuit.

        By default, the `run_async` method invokes this method on another
        thread. So this method is supposed to be thread safe.

        Args:
            program: The circuit to sample from.
            param_resolver: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            TrialResult for a run.
        """
        return self.run_sweep(program, study.ParamResolver(param_resolver),
                              repetitions)[0]

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
                    f'while another had {repr(sorted(sweep.keys))}.')

        results = []
        for sweep in sweeps_list:
            sweep_results = self.run_sweep(program,
                                           params=sweep,
                                           repetitions=repetitions)
            for resolver, result in zip(sweep, sweep_results):
                param_values_once = [resolver.value_of(key) for key in keys]
                param_table = pd.DataFrame(data=[param_values_once] *
                                           repetitions,
                                           columns=keys)
                results.append(pd.concat([param_table, result.data], axis=1))

        return pd.concat(results)

    @abc.abstractmethod
    def run_sweep(
            self,
            program: 'cirq.Circuit',
            params: 'cirq.Sweepable',
            repetitions: int = 1,
    ) -> List['cirq.TrialResult']:
        """Samples from the given Circuit.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """

    async def run_async(self, program: 'cirq.Circuit', *,
                        repetitions: int) -> 'cirq.TrialResult':
        """Asynchronously samples from the given Circuit.

        By default, this method invokes `run` synchronously and simply exposes
        its result is an awaitable. Child classes that are capable of true
        asynchronous sampling should override it to use other strategies.

        Args:
            program: The circuit to sample from.
            repetitions: The number of times to sample.

        Returns:
            An awaitable TrialResult.
        """
        return self.run(program, repetitions=repetitions)

    async def run_sweep_async(
            self,
            program: 'cirq.Circuit',
            params: 'cirq.Sweepable',
            repetitions: int = 1,
    ) -> List['cirq.TrialResult']:
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
            An awaitable TrialResult.
        """
        return self.run_sweep(program, params=params, repetitions=repetitions)

    def run_batch(
            self,
            programs: List['cirq.Circuit'],
            params_list: List['cirq.Sweepable'] = None,
            repetitions: int = 1,
    ) -> List['cirq.TrialResult']:
        """Runs the supplied circuits.

        Each circuit provided in `programs` will pair with the associated
        parameter sweep provided in the `params_list`. The number of circuits
        is required to match the number of sweeps.

        By default, this method simply invokes `run_sweep` sequentially for
        each (circuit, parameter sweep) pair. Child classes that are capable of
        sampling batches more efficiently should override it to use other
        strategies.

        Args:
            programs: The circuits to execute as a batch.
            params_list: Parameter sweeps to use with the circuits. The number
                of sweeps should match the number of circuits and will be
                paired in order with the circuits.
            repetitions: Number of circuit repetitions to run. Each sweep value
                of each circuit in the batch will run with the same repetitions.

        Returns:
            A list of TrialResults. All TrialResults for the first circuit are
            listed first, then the TrialResults for the second, etc.
            The TrialResults for a circuit are listed in the order imposed by
            the associated parameter sweep.
        """
        if not params_list or len(programs) != len(params_list):
            raise ValueError('Number of circuits and sweeps must match')
        return [
            trial_result for circuit, params in zip(programs, params_list)
            for trial_result in self.run_sweep(
                circuit, params=params, repetitions=repetitions)
        ]
