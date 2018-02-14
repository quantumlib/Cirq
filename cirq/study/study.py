# Copyright 2018 Google LLC
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

"""Defines studies and related classes."""

import abc

from typing import Dict, Iterable, Union

from cirq.circuits import Circuit
from cirq.google import ParamResolver  # TODO(dabacon): Move to study.
from cirq.schedules import Schedule


class Study(object):
    """A study is a collection of repeated trials run by an executor.

    Examples of executors include local and remote simulator and quantum
    computer execution APIs.
    """

    def __init__(self,
        executor: 'Executor',
        program: Union[Circuit, Schedule],
        param_resolvers: Iterable[ParamResolver] = None,
        repetitions: int = 0,
        **executor_kwags: Dict):
        self.executor = executor
        self.program = program
        self.param_resolvers = param_resolvers
        self.repetitions = repetitions
        self.executor_kwags = executor_kwags

    def run_study(self) -> Iterable[Iterable['Result']]:
        """Runs the study for all parameters and repetitions.

        Returns:
            An iterable of iterables of Results. The outer iterable is for
            each resolved parameter. The inner iterable is for each
            reptition.
        """
        trial_results = []
        for param_resolver in self.param_resolvers or [None]:
            repetitions = []
            for rep in range(self.repetitions):
                trial_result = self.executor.run(program=self.program,
                                                 param_resolver=param_resolver,
                                                 **self.executor_kwags)
                repetitions.append(trial_result)
            trial_results.append(repetitions)
        return trial_results


class Executor(metaclass=abc.ABCMeta):
    """Encapsulates running a Circuit or Scheduler for fixed parameters."""

    @abc.abstractmethod
    def run(self,
        program: Union[Circuit, Schedule],
        param_resolver: ParamResolver) -> 'Result':
        """Run the program using the parameters described in the ParamResolver.

        Args:
            program: Either the Circuit or Schedule to run. Some executors
                only support one of these types.
            param_resolver: Resolves parameters in the program.

        Returns:
            Result or subclass for the result of the execution.
        """

        # TODO(dabacon): Async run.


class ResultMeta(type):
    """Metaclass that asserts measurements and param_dict attributes exist."""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        if not hasattr(obj, 'measurements'):
            raise NotImplementedError(
                'Results must have a measurements attribute.')
        if not hasattr(obj, 'param_dict'):
            raise NotImplementedError(
                'Results must have a param_dict attribute.')
        return obj


class Result(metaclass=ResultMeta):
    """Results are required to have measurements and param_dict attributes.

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results. If a key is reused, the measurement values are returned
            in the order they appear in the Circuit being simulated.
        param_dict: A dictionary produce by the ParamResolver mapping parameter
            keys to actual parameter values that produced this result.
    """
