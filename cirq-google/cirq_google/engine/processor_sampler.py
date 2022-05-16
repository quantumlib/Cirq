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

from typing import List, Optional, Sequence, TYPE_CHECKING, Union, cast

import cirq

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

    def run_sweep(
        self, program: 'cirq.AbstractCircuit', params: cirq.Sweepable, repetitions: int = 1
    ) -> Sequence['cg.EngineResult']:
        job = self._processor.run_sweep(program=program, params=params, repetitions=repetitions)
        return job.results()

    def run_batch(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        params_list: Optional[List[cirq.Sweepable]] = None,
        repetitions: Union[int, List[int]] = 1,
    ) -> Sequence[Sequence['cg.EngineResult']]:
        """Runs the supplied circuits.

        In order to gain a speedup from using this method instead of other run
        methods, the following conditions must be satisfied:
            1. All circuits must measure the same set of qubits.
            2. The number of circuit repetitions must be the same for all
               circuits. That is, the `repetitions` argument must be an integer,
               or else a list with identical values.
        """
        if isinstance(repetitions, List) and len(programs) != len(repetitions):
            raise ValueError(
                'len(programs) and len(repetitions) must match. '
                f'Got {len(programs)} and {len(repetitions)}.'
            )
        if isinstance(repetitions, int) or len(set(repetitions)) == 1:
            # All repetitions are the same so batching can be done efficiently
            if isinstance(repetitions, List):
                repetitions = repetitions[0]
            job = self._processor.run_batch(
                programs=programs, params_list=params_list, repetitions=repetitions
            )
            return job.batched_results()
        # Varying number of repetitions so no speedup
        return cast(
            Sequence[Sequence['cg.EngineResult']],
            super().run_batch(programs, params_list, repetitions),
        )

    @property
    def processor(self) -> 'cg.engine.AbstractProcessor':
        return self._processor
