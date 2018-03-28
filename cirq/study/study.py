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

"""Defines studies and related classes.

Studies can be used to run repeated trials against different program
paramters and number of repetitions.

Example use:
    sim = Simulator()
    circuit = my_circuit()

    study = ExecutorStudy(executor=sim, program=circuit, repetitions=10)
    all_trials = study.run_study()
    # all_trials is a list of 10 (context, result) where context contains
    # the reptition id, and result contains the program's results.
"""

from typing import Any, Iterable, Union

from cirq import abc
from cirq.circuits import Circuit
from cirq.schedules import Schedule
from cirq.study import ParamResolver


class StudyInterface(metaclass=abc.ABCMeta):
    """An interface for running repeated trials over different parameters."""

    @abc.abstractmethod
    def run_study(self) -> Iterable['TrialResult']:
        """Runs the study.

        Returns:
            An iterable of results, each representing a single run of the
            executor, a trial. A trial may itself involve multiple repetitions
            for the same set of circuit parameters.
        """


class ExecutorStudy(StudyInterface):
    """A study is a collection of repeated trials run by an executor.

    Here an Executor runs a single quantum program within a context of
    each individual trials.
    """

    def __init__(
            self,
            executor: 'Executor',
            program: Union[Circuit, Schedule],
            param_resolvers: Iterable[ParamResolver] = None,
            repetitions: int = 0,
            **executor_kwags: Any) -> None:
        self.executor = executor
        self.program = program
        self.param_resolvers = param_resolvers or [ParamResolver({})]
        self.repetitions = repetitions
        self.executor_kwags = executor_kwags

    def run_study(self) -> Iterable['TrialResult']:
        """Runs the study for all parameters and repetitions.

        Returns:
            An iterable of trial results for each parameter setting.
        """
        trial_results = []
        for param_resolver in self.param_resolvers:
            result = self.executor.run(  # type: ignore
                program=self.program,
                param_resolver=param_resolver,
                repetitions=self.repetitions,
                **self.executor_kwags)
            trial_results.append(result)
        return trial_results


class Executor(metaclass=abc.ABCMeta):
    """Encapsulates running a Circuit or Scheduler for fixed parameters.

    Executors are used to run a single quantum program for a fixed
    context of parameter values.
    """

    @abc.abstractmethod
    def run(
            self,
            program: Union[Circuit, Schedule],
            param_resolver: ParamResolver,
            repetitions: int
    ) -> 'TrialResult':
        """Run the program using the parameters described in the ParamResolver.

        Args:
            program: Either the Circuit or Schedule to run. Some executors
                only support one of these types.
            param_resolver: Resolves parameters in the program.

        Returns:
            Results of the given run of the circuit.
        """

        # TODO(dabacon): Async run.


class TrialResultMeta(type):
    """Metaclass that asserts measurements attributes exist."""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        if not hasattr(obj, 'params'):
            raise NotImplementedError(
                'TrialResult subclasses must have a params attribute.')
        if not hasattr(obj, 'repetitions'):
            raise NotImplementedError(
                'TrialResult subclasses must have a repetitions attribute.')
        if not hasattr(obj, 'measurements'):
            raise NotImplementedError(
                'TrialResult subclasses must have a measurements attribute.')
        return obj


class TrialResult(metaclass=TrialResultMeta):
    """The results of a single execution (trial).

    Attributes:
        params: A ParamResolver of settings used for this result.
        repetitions: Number of repetitions included in this result.
        measurements: A dictionary from measurement gate key to measurement
            results. The value for each key is a 2-D array of booleans, with
            the first index running over the repetitions, and the second index
            running over the occurrences of measurement gates with this key in
            the circuit that was simulated.
    """
