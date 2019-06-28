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

from typing import Awaitable, List, Union
import abc
import asyncio
import threading

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

        By default, the `run_async` method invokes this method on another
        thread. So this method is supposed to be thread safe.

        Args:
            program: The circuit or schedule to sample from.
            param_resolver: Parameters to run with the program.
            repetitions: The number of times to sample.

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
            program: The circuit or schedule to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """

    async def run_async(self,
                        program: Union[circuits.Circuit, schedules.Schedule], *,
                        repetitions: int) -> Awaitable[study.TrialResult]:
        """Asynchronously samples from the given Circuit or Schedule.

        By default, this method calls `run` on another thread and yields the
        result via the asyncio event loop. However, child classes are free to
        override it to use other strategies.

        Args:
            program: The circuit or schedule to sample from.
            repetitions: The number of times to sample.

        Returns:
            An awaitable TrialResult.
        """
        loop = asyncio.get_event_loop()
        done = loop.create_future()  # type: asyncio.Future

        def run():
            try:
                result = self.run(program, repetitions=repetitions)
            except Exception as exc:
                loop.call_soon_threadsafe(done.set_exception, exc)
            else:
                loop.call_soon_threadsafe(done.set_result, result)

        t = threading.Thread(target=run)
        t.start()
        return await done
