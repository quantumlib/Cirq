# Copyright 2022 The Cirq Developers
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

from typing import Optional, Sequence, TYPE_CHECKING, Union, cast

import cirq
import duet

if TYPE_CHECKING:
    import cirq_google as cg


class ProcessorSampler(cirq.Sampler):
    """A wrapper around AbstractProcessor to implement the cirq.Sampler interface."""

    def __init__(self, *, processor: 'cg.engine.AbstractProcessor'):
        """Inits ProcessorSampler.

        Args:
            processor: AbstractProcessor instance to use.
        """
        self._processor = processor

    async def run_sweep_async(
        self, program: 'cirq.AbstractCircuit', params: cirq.Sweepable, repetitions: int = 1
    ) -> Sequence['cg.EngineResult']:
        job = await self._processor.run_sweep_async(
            program=program, params=params, repetitions=repetitions
        )
        return await job.results_async()

    run_sweep = duet.sync(run_sweep_async)

    async def run_batch_async(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        params_list: Optional[Sequence[cirq.Sweepable]] = None,
        repetitions: Union[int, Sequence[int]] = 1,
    ) -> Sequence[Sequence['cg.EngineResult']]:
        """Runs the supplied circuits.

        In order to gain a speedup from using this method instead of other run
        methods, the following conditions must be satisfied:
            1. All circuits must measure the same set of qubits.
            2. The number of circuit repetitions must be the same for all
               circuits. That is, the `repetitions` argument must be an integer,
               or else a list with identical values.
        """
        params_list, repetitions = self._normalize_batch_args(programs, params_list, repetitions)
        if len(set(repetitions)) == 1:
            # All repetitions are the same so batching can be done efficiently
            job = await self._processor.run_batch_async(
                programs=programs, params_list=params_list, repetitions=repetitions[0]
            )
            return await job.batched_results_async()
        # Varying number of repetitions so no speedup
        return cast(
            Sequence[Sequence['cg.EngineResult']],
            await super().run_batch_async(programs, params_list, repetitions),
        )

    run_batch = duet.sync(run_batch_async)

    @property
    def processor(self) -> 'cg.engine.AbstractProcessor':
        return self._processor
