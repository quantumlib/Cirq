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

from typing import List, TYPE_CHECKING, Dict, Any

import numpy as np
import pandas as pd

from cirq import study, value, ops

if TYPE_CHECKING:
    import cirq


class Sampler(metaclass=value.ABCMetaImplementAnyOneOf):
    """Something capable of sampling quantum circuits. Simulator or hardware.

    Child classes must implement at least one of the following methods:

        - run
        - run_sweep
        - sample_dict
        - sample_dicts

    The other methods will be derived from the method that was implemented.
    """

    def _run_from_run_sweep(
            self,
            program: 'cirq.Circuit',
            param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
            repetitions: int = 1,
    ) -> 'cirq.TrialResult':
        return self.run_sweep(program, study.ParamResolver(param_resolver),
                              repetitions)[0]

    def _run_from_sample_dicts(
            self,
            program: 'cirq.Circuit',
            param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
            repetitions: int = 1,
    ) -> 'cirq.TrialResult':
        param_resolver = study.ParamResolver(param_resolver)
        dicts = self.sample_dicts(program,
                                  repetitions=repetitions,
                                  params=param_resolver)
        shapes = {
            op.gate.key: tuple(q.dimension for q in op.qubits)
            for op in program.all_operations()
            if isinstance(op.gate, ops.MeasurementGate)
        }
        return study.TrialResult(
            params=param_resolver,
            measurements={
                key: np.array([
                    value.big_endian_int_to_digits(d[key], base=shapes[key])
                    for d in dicts
                ]) for key in program.all_measurement_keys()
            })

    def _sample_dict_from_sample_dicts(
            self,
            program: 'cirq.Circuit',
            *,
            params: 'cirq.ParamResolverOrSimilarType' = None,
    ) -> Dict[str, Any]:
        return self.sample_dicts(program, params=params, repetitions=1)[0]

    def _sample_dicts_from_run(
            self,
            program: 'cirq.Circuit',
            *,
            repetitions: int,
            params: 'cirq.ParamResolverOrSimilarType' = None,
    ) -> List[Dict[str, int]]:
        return self.run(program, param_resolver=params,
                        repetitions=repetitions).as_dicts()

    def _sample_dicts_from_sample_dict(
            self,
            program: 'cirq.Circuit',
            *,
            repetitions: int,
            params: 'cirq.ParamResolverOrSimilarType' = None,
    ) -> List[Dict[str, int]]:
        return [
            self.sample_dict(program, params=params) for _ in range(repetitions)
        ]

    def _run_sweep_from_run(
            self,
            program: 'cirq.Circuit',
            params: 'cirq.Sweepable',
            repetitions: int = 1,
    ) -> List['cirq.TrialResult']:
        return [
            self.run(program, param_resolver=param, repetitions=repetitions)
            for param in study.to_sweep(params)
        ]

    @value.alternative(requires='run_sweep', implementation=_run_from_run_sweep)
    @value.alternative(requires='sample_dicts',
                       implementation=_run_from_sample_dicts)
    def run(
            self,
            program: 'cirq.Circuit',
            param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
            repetitions: int = 1,
    ) -> 'cirq.TrialResult':
        """Samples from the given Circuit.

        Args:
            program: The circuit to sample from.
            param_resolver: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            TrialResult for a run.
        """

    @value.alternative(requires='sample_dicts',
                       implementation=_sample_dict_from_sample_dicts)
    def sample_dict(
            self,
            program: 'cirq.Circuit',
            *,
            params: 'cirq.ParamResolverOrSimilarType' = None,
    ) -> Dict[str, Any]:
        """Samples the circuit and returns the result as a dictionary.

        The dictionary maps measurement keys to measurement results. Measurement
        results are integers. Multi-qubit measurements are fused into an integer
        in a big endian fashion. For example, cirq.measure(x, y, z) will produce
        the binary integer 0b_xyz where x, y, and z are the bits resulting from
        measuring qubits X, Y, and Z respectively.

        Args:
            program: The circuit to sample measurement results from.
            params: Parameters for parameterized gates in the circuit. Defaults
                to no parameters.

        Returns:
            A dictionary mapping measurement keys to measurement results for one
            sample of the circuit.
        """

    @value.alternative(requires='run', implementation=_sample_dicts_from_run)
    @value.alternative(requires='sample_dict',
                       implementation=_sample_dicts_from_sample_dict)
    def sample_dicts(
            self,
            program: 'cirq.Circuit',
            *,
            repetitions: int,
            params: 'cirq.ParamResolverOrSimilarType' = None,
    ) -> List[Dict[str, int]]:
        """Repeatedly samples the circuit, returning results as a list of dicts.

        Each dictionary maps measurement keys to measurement results.
        Measurement results are integers. Multi-qubit measurements are fused
        into an integer in a big endian fashion. For example,
        cirq.measure(x, y, z) will produce the binary integer 0b_xyz where x, y,
        and z are the bits resulting from measuring qubits X, Y, and Z
        respectively.

        Args:
            program: The circuit to sample measurement results from.
            params: Parameters for parameterized gates in the circuit. Defaults
                to no parameters.
            repetitions: How many times to sample from the circuit. The number
                of entries in the returned list.

        Returns:
            A list of dictionaries. The length of the list is the number of
            repetitions. Each dictionary is the results from one run of the
            circuit, and maps measurement keys to the measurement results from
            that run.
        """

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
            each measurement result as well as a column for each symbolic
            parameter. There is an also index column containing the repetition
            number, for each parameter assignment.

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

    @value.alternative(requires='run', implementation=_run_sweep_from_run)
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
