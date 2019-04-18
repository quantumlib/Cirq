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
from typing import (List, Union)

from cirq import circuits, schedules, study


class Sampler(metaclass=abc.ABCMeta):
    """Something capable of sampling quantum circuits. Simulator or hardware."""

    def run(
            self,
            program: Union[circuits.Circuit, schedules.Schedule],
            param_resolver: 'study.ParamResolverOrSimilarType' = None,
            repetitions: int = 1,
    ) -> study.TrialResult:
        """Samples from the given Circuit or Schedule.

        Args:
            program: The circuit or schedule to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.

        Returns:
            TrialResult for a run.
        """
        return self.run_sweep(program, study.ParamResolver(param_resolver),
                              repetitions)[0]

    @abc.abstractmethod
    def run_sweep(
            self,
            program: Union[circuits.Circuit, schedules.Schedule],
            params: study.Sweepable,
            repetitions: int = 1,
    ) -> List[study.TrialResult]:
        """Samples from the given Circuit or Schedule.

        In contrast to run, this allows for sweeping over different parameter
        values.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """
