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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast, TYPE_CHECKING

import duet

import cirq

if TYPE_CHECKING:
    import cirq_google as cg


class ProcessorSampler(cirq.Sampler):
    """A wrapper around AbstractProcessor to implement the cirq.Sampler interface."""

    def __init__(
        self,
        *,
        processor: cg.engine.AbstractProcessor,
        run_name: str = "",
        snapshot_id: str = "",
        device_config_name: str = "",
        max_concurrent_jobs: int = 100,
        jobs_per_batch: int = 1,
    ):
        """Inits ProcessorSampler.

        Either both (`run_name` or `snapshot_id`) and `device_config_name` must be set, or neither
        of them must be set. If none of them are set, a default internal device configuration
        will be used.

        Args:
            processor: AbstractProcessor instance to use.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor.
            snapshot_id: A unique identifier for an immutable snapshot reference.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
            max_concurrent_jobs: The maximum number of jobs to be sent
                concurrently to the Engine. This client-side throttle can be
                used to proactively reduce load to the backends and avoid quota
                violations when pipelining circuit executions.
            jobs_per_batch:  If set to greater than 1, this will batch multiple
                circuits within the same API call when calling run_batch() or
                run_batch_async() up to a maximum of `jobs_per_batch`.

        Raises:
            ValueError: If  only one of `run_name` and `device_config_name` are specified.
        """
        if (bool(run_name) or bool(snapshot_id)) ^ bool(device_config_name):
            raise ValueError('Cannot specify only one of `run_name` and `device_config_name`')

        self._processor = processor
        self._run_name = run_name
        self._snapshot_id = snapshot_id
        self._device_config_name = device_config_name
        self._concurrent_job_limiter = duet.Limiter(max_concurrent_jobs)
        self._jobs_per_batch = jobs_per_batch

    async def run_sweep_async(
        self,
        program: (
            cirq.AbstractCircuit
            | Sequence[cirq.AbstractCircuit]
            | Mapping[str, cirq.AbstractCircuit]
        ),
        params: cirq.Sweepable | Sequence[cirq.Sweepable],
        repetitions: int = 1,
    ) -> Sequence[cg.EngineResult]:
        async with self._concurrent_job_limiter:
            job = await self._processor.run_sweep_async(
                program=program,
                params=params,
                repetitions=repetitions,
                run_name=self._run_name,
                snapshot_id=self._snapshot_id,
                device_config_name=self._device_config_name,
            )

            return await job.results_async()

    run_sweep = duet.sync(run_sweep_async)

    async def run_batch_async(
        self,
        programs: Sequence[cirq.AbstractCircuit] | Mapping[str, cirq.AbstractCircuit],
        params_list: Sequence[cirq.Sweepable] | None = None,
        repetitions: int | Sequence[int] = 1,
    ) -> Sequence[Sequence[cg.EngineResult]]:
        if self._jobs_per_batch > 1:
            # Treat programs as a sequence for iteration, but keep keys if it's a mapping
            prog_keys = list(programs.keys()) if isinstance(programs, Mapping) else []
            prog_values = (
                list(programs.values()) if isinstance(programs, Mapping) else list(programs)
            )

            params_list, repetitions = self._normalize_batch_args(
                prog_values, params_list, repetitions
            )
            # Batch programs that have the same number of repetitions.
            program_batches = []
            params_list_batches = []
            repetition_batches = []

            i = 0
            while i < len(prog_values):
                batch_reps = repetitions[i]
                batch_programs = {prog_keys[i]: prog_values[i]} if prog_keys else [prog_values[i]]
                batch_params = [params_list[i]]
                i += 1
                while (
                    i < len(prog_values)
                    and len(batch_programs) < self._jobs_per_batch
                    and repetitions[i] == batch_reps
                ):
                    if isinstance(batch_programs, dict):
                        batch_programs[prog_keys[i]] = prog_values[i]
                    else:
                        batch_programs.append(prog_values[i])
                    batch_params.append(params_list[i])
                    i += 1
                program_batches.append(batch_programs)
                params_list_batches.append(batch_params)
                repetition_batches.append(batch_reps)

            return await duet.pstarmap_async(
                self.run_sweep_async, zip(program_batches, params_list_batches, repetition_batches)
            )

        prog_values = list(programs.values()) if isinstance(programs, Mapping) else list(programs)
        return cast(
            Sequence[Sequence['cg.EngineResult']],
            await super().run_batch_async(prog_values, params_list, repetitions),
        )

    run_batch = duet.sync(run_batch_async)

    @property
    def processor(self) -> cg.engine.AbstractProcessor:
        return self._processor

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id  # pragma: no cover

    @property
    def device_config_name(self) -> str:
        return self._device_config_name

    @property
    def max_concurrent_jobs(self) -> int:
        assert self._concurrent_job_limiter.capacity is not None
        return self._concurrent_job_limiter.capacity
