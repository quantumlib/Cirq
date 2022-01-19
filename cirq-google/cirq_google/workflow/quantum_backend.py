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
import dataclasses
from typing import Optional, Iterable

import cirq
import cirq_google as cg


class EngineBackend(cg.engine.EngineProcessor):
    """An EngineProcessor typified by its processor_id.

    This is serializable and relies on the GOOGLE_CLOUD_PROJECT environment
    variable to set up the actual connection.
    """

    def __init__(self, processor_id: str):
        engine = cg.get_engine()
        super().__init__(
            project_id=engine.project_id,
            processor_id=processor_id,
            context=engine.context,
        )

    def get_sampler(self, gate_set: Optional['cg.Serializer'] = None) -> 'cg.QuantumEngineSampler':
        assert gate_set is None
        return super().get_sampler(cg.SQRT_ISWAP_GATESET)

    def get_device(self, gate_sets: Iterable['cg.Serializer'] = ()) -> 'cirq.Device':

        assert gate_sets is ()
        return super().get_device(cg.SQRT_ISWAP_GATESET)

    def __repr__(self):
        return f'cirq_google.EngineBackend({self.processor_id!r})'

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, ['processor_id'])


class SimulatedBackend(cg.engine.SimulatedLocalProcessor):
    """Simulated Engine Processor

    Args:
        processor_id: The processor id we are emulating
        noise_strength: To do noisy simulation, set this to a positive float. The default
            of `0` will result in a noiseless state-vector simulation. If `float('inf')`
            is provided the simulator will be `cirq.ZerosSampler`. Otherwise, use
            a depolarizing model with this probability of noise.
    """

    def __init__(self, processor_id: str, noise_strength: float = 0):

        super().__init__(
            processor_id=processor_id,
            sampler=self._init_sampler(noise_strength),
            device=self._init_device(processor_id),
        )
        self.noise_strength = noise_strength

    def _init_device(self, processor_id: str) -> cirq.Device:
        return cg.get_engine_device(processor_id)

    def _init_sampler(self, noise_strength: float) -> cirq.Sampler:
        if noise_strength == 0:
            return cirq.Simulator()
        if noise_strength == float('inf'):
            return cirq.ZerosSampler()

        return cirq.DensityMatrixSimulator(noise=cirq.depolarize(p=noise_strength))

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        return cirq.obj_to_dict_helper(self, ['processor_id', 'noise_strength'])

    def descriptive_name(self):
        if self.noise_strength == 0:
            suffix = 'simulator'
        elif self.noise_strength == float('inf'):
            suffix = 'zeros'
        else:
            suffix = f'p={self.noise_strength:.3e}'
        return f'{self.processor_id}-{suffix}'

    def __eq__(self, other):
        if not isinstance(other, SimulatedBackend):
            return False

        return (self.processor_id, self.noise_strength) == (
            other.processor_id,
            other.noise_strength,
        )

    def __repr__(self):
        return f'cirq_google.{self.__class__.__name__}({self.processor_id!r}, noise_strength={self.noise_strength!r})'


_DEVICES_BY_ID = {
    'rainbow': cg.Sycamore23,
    'weber': cg.Sycamore,
}


class SimulatedBackendWithLocalDevice(SimulatedBackend):
    def _init_device(self, processor_id: str) -> cirq.Device:
        return _DEVICES_BY_ID[processor_id]
