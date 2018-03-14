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

import abc
from typing import Dict, Iterable, Tuple, Union

from cirq.circuits import Circuit
from cirq.schedules import Schedule
from cirq.study import ParamResolver


class StudyInterface(metaclass=abc.ABCMeta):
    """An interface for running repeated trials over different parameters."""

    @abc.abstractmethod
    def run_study(self) -> Iterable[Tuple['TrialContext', 'TrailResult']]:
        """Runs the study.

        Returns:
            An iterable of tuples. Each element represents a single run of the
            executor, a trial. The tuple is of the form (context, result)
            where context is a TrialContext, describing the settings for this
            trial, and result is a TrailResult, describing the results of the
            trial.
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
            **executor_kwags: Dict) -> None:
        self.executor = executor
        self.program = program
        self.param_resolvers = param_resolvers or [ParamResolver({})]
        self.repetitions = repetitions
        self.executor_kwags = executor_kwags

    def run_study(self) -> Iterable[Tuple['TrialContext', 'TrailResult']]:
        """Runs the study for all parameters and repetitions.

        Returns:
            An iterable of tuples. Each element represents a single run of the
            executor, a trial. The tuple is of the form (context, result)
            where context is a TrialContext, describing the settings for this
            trial, and result is a TrailResult, describing the results of the
            trial.
        """
        trial_results = []
        for param_resolver in self.param_resolvers:
            for rep_id in range(self.repetitions):
                (context, result) = self.executor.run(
                    program=self.program,
                    param_resolver=param_resolver,
                    **self.executor_kwags)
                context.repetition_id = rep_id
                trial_results.append((context, result))
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
            param_resolver: ParamResolver
    ) -> Tuple['TrialContext', 'TrialResult']:
        """Run the program using the parameters described in the ParamResolver.

        Args:
            program: Either the Circuit or Schedule to run. Some executors
                only support one of these types.
            param_resolver: Resolves parameters in the program.

        Returns:
            (context, result): A tuple of TrailContext and TrialResult objects
                describing the context that generated the result, and the
                result, respectively.
        """

        # TODO(dabacon): Async run.


class TrialContextMeta(type):
    """Metaclass that asserts param_dict and repetition_id attributes exist."""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        if not hasattr(obj, 'param_dict'):
            raise NotImplementedError(
                'TrialContext subclasses must have a param_dict attribute.')
        if not hasattr(obj, 'repetition_id'):
            raise NotImplementedError(
                'TrialContext subclasses must have a repetition_id attribute.')
        return obj


class TrialContext(metaclass=TrialContextMeta):
    """The context of for a single execution (trial).

    Attributes:
        param_dict: A dictionary produce by a ParamResolver mapping parameter
            keys to actual parameter values that produced this result.
        repetition_id: An integer labeling the repetition (repetition for a
            fixed param_dict).
    """


class TrialResultMeta(type):
    """Metaclass that asserts measurements attributes exist."""

    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        if not hasattr(obj, 'measurements'):
            raise NotImplementedError(
                'TrialResult subclasses must have a measurements attribute.')
        return obj


class TrialResult(metaclass=TrialResultMeta):
    """The results of a single execution (trial).

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results. If a key is reused, the measurement values are returned
            in the order they appear in the Circuit being simulated.
    """
