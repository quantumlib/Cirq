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
