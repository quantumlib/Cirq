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

    def __init__(
        self,
        *,
        processor: 'cg.engine.AbstractProcessor',
        run_name: str = "",
        device_config_name: str = "",
    ):
        """Inits ProcessorSampler.

        Either both `run_name` and `device_config_name` must be set, or neither of
        them must be set. If none of them are set, a default internal device configuration
        will be used.

        Args:
            processor: AbstractProcessor instance to use.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.

        Raises:
            ValueError: If  only one of `run_name` and `device_config_name` are specified.
        """
        if bool(run_name) ^ bool(device_config_name):
            raise ValueError('Cannot specify only one of `run_name` and `device_config_name`')

        self._processor = processor
        self._run_name = run_name
        self._device_config_name = device_config_name

    async def run_sweep_async(
        self, program: 'cirq.AbstractCircuit', params: cirq.Sweepable, repetitions: int = 1
    ) -> Sequence['cg.EngineResult']:
        job = await self._processor.run_sweep_async(
            program=program,
            params=params,
            repetitions=repetitions,
            run_name=self._run_name,
            device_config_name=self._device_config_name,
        )
        return await job.results_async()

    run_sweep = duet.sync(run_sweep_async)

    async def run_batch_async(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        params_list: Optional[Sequence[cirq.Sweepable]] = None,
        repetitions: Union[int, Sequence[int]] = 1,
    ) -> Sequence[Sequence['cg.EngineResult']]:
        return cast(
            Sequence[Sequence['cg.EngineResult']],
            await super().run_batch_async(programs, params_list, repetitions),
        )

    run_batch = duet.sync(run_batch_async)

    @property
    def processor(self) -> 'cg.engine.AbstractProcessor':
        return self._processor
