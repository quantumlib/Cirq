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
import abc
import dataclasses

import cirq
import cirq_google as cg
from cirq._compat import dataclass_repr


class ProcessorRecord(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_processor(self) -> 'cg.engine.AbstractProcessor':
        pass

    def get_sampler(self) -> 'cirq.Sampler':
        return self.get_processor().get_sampler()

    def get_device(self) -> 'cirq.Device':
        return self.get_processor().get_device()


@dataclasses.dataclass(frozen=True)
class EngineProcessorRecord(ProcessorRecord):
    """An EngineProcessor typified by its processor_id.

    This is serializable and relies on the GOOGLE_CLOUD_PROJECT environment
    variable to set up the actual connection.
    """

    processor_id: str

    def get_processor(self) -> cg.EngineProcessor:
        engine = cg.get_engine()
        return cg.EngineProcessor(
            project_id=engine.project_id,
            processor_id=self.processor_id,
            context=engine.context,
        )

    def get_sampler(self) -> 'cg.QuantumEngineSampler':
        return self.get_processor().get_sampler(cg.SQRT_ISWAP_GATESET)

    def get_device(self) -> 'cirq.Device':
        # Issues mocking out the gateset, so ignore coverage
        # coverage: ignore
        return self.get_processor().get_device([cg.SQRT_ISWAP_GATESET])

    def __repr__(self):
        return dataclass_repr(self, namespace='cirq_google')

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclasses.dataclass(frozen=True)
class SimulatedProcessorRecord(ProcessorRecord):
    """Simulated Engine Processor

    Args:
        processor_id: The processor id we are emulating
        noise_strength: To do noisy simulation, set this to a positive float. The default
            of `0` will result in a noiseless state-vector simulation. If `float('inf')`
            is provided the simulator will be `cirq.ZerosSampler`. Otherwise, use
            a depolarizing model with this probability of noise.
    """

    processor_id: str
    noise_strength: float = 0

    def get_processor(self) -> 'cg.engine.AbstractProcessor':
        return cg.engine.SimulatedLocalProcessor(
            processor_id=self.processor_id,
            sampler=self.get_sampler(),
            device=self.get_device(),
        )

    def get_device(self) -> 'cirq.Device':
        return cg.get_engine_device(self.processor_id)

    def get_sampler(
        self,
    ) -> 'cirq.Sampler':
        if self.noise_strength == 0:
            return cirq.Simulator()
        if self.noise_strength == float('inf'):
            return cirq.ZerosSampler()

        return cirq.DensityMatrixSimulator(
            noise=cirq.ConstantQubitNoiseModel(cirq.depolarize(p=self.noise_strength))
        )

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)

    def descriptive_name(self):
        if self.noise_strength == 0:
            suffix = 'simulator'
        elif self.noise_strength == float('inf'):
            suffix = 'zeros'
        else:
            suffix = f'p={self.noise_strength:.3e}'
        return f'{self.processor_id}-{suffix}'

    def __repr__(self):
        return dataclass_repr(self, namespace='cirq_google')


_DEVICES_BY_ID = {
    'rainbow': cg.Sycamore23,
    'weber': cg.Sycamore,
}


class SimulatedProcessorWithLocalDeviceRecord(SimulatedProcessorRecord):
    def get_device(self) -> 'cirq.Device':
        return _DEVICES_BY_ID[self.processor_id]
