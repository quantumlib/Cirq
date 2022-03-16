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
from typing import Any, Dict

import cirq
import cirq_google as cg
from cirq._compat import dataclass_repr
from cirq_google.engine.virtual_engine_factory import (
    MOST_RECENT_TEMPLATES,
    _create_device_spec_from_template,
)


class ProcessorRecord(metaclass=abc.ABCMeta):
    """A serializable record that maps to a particular `cg.engine.AbstractProcessor`."""

    @abc.abstractmethod
    def get_processor(self) -> 'cg.engine.AbstractProcessor':
        """Using this classes' attributes, return a unique `cg.engine.AbstractProcessor`

        This is the primary method that descendants must implement.
        """

    def get_sampler(self) -> 'cirq.Sampler':
        """Return a `cirq.Sampler` for the processor specified by this class.

        The default implementation delegates to `self.get_processor()`.
        """
        return self.get_processor().get_sampler()

    def get_device(self) -> 'cirq.Device':
        """Return a `cirq.Device` for the processor specified by this class.

        The default implementation delegates to `self.get_processor()`.
        """
        return self.get_processor().get_device()


@dataclasses.dataclass(frozen=True)
class EngineProcessorRecord(ProcessorRecord):
    """A serializable record of processor_id to map to a `cg.EngineProcessor`.

    This class presumes the GOOGLE_CLOUD_PROJECT environment
    variable is set to establish a connection to the cloud service.

    Args:
        processor_id: The processor id.
    """

    processor_id: str

    def get_processor(self) -> 'cg.EngineProcessor':
        """Return a `cg.EngineProcessor` for the specified processor_id."""
        engine = cg.get_engine()
        return engine.get_processor(self.processor_id)

    def __repr__(self):
        return dataclass_repr(self, namespace='cirq_google')

    def __str__(self) -> str:
        return self.processor_id

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self)


@dataclasses.dataclass(frozen=True)
class SimulatedProcessorRecord(ProcessorRecord):
    """A serializable record mapping a processor_id and optional noise spec to a simulator-backed
    mock of `cg.AbstractProcessor`.

    Args:
        processor_id: The processor id we are emulating
        noise_strength: To do noisy simulation, set this to a positive float. The default
            of `0` will result in a noiseless state-vector simulation. If `float('inf')`
            is provided the simulator will be `cirq.ZerosSampler`. Otherwise, use
            a depolarizing model with this probability of noise.
    """

    processor_id: str
    noise_strength: float = 0

    def get_processor(self) -> 'cg.engine.SimulatedLocalProcessor':
        """Return a `cg.SimulatedLocalProcessor` for the specified processor_id."""
        return cg.engine.SimulatedLocalProcessor(
            processor_id=self.processor_id,
            sampler=self._get_input_sampler(),
            device=self._get_input_device(),
        )

    def _get_input_device(self) -> 'cirq.Device':
        """Return a `cg.SerializableDevice` for the specified processor_id.

        This method presumes the GOOGLE_CLOUD_PROJECT environment
        variable is set to establish a connection to the cloud service.

        This function is used to initialize this class's `processor`.
        """
        return cg.get_engine_device(self.processor_id)

    def _get_input_sampler(
        self,
    ) -> 'cirq.Sampler':
        """Return a local `cirq.Sampler` based on the `noise_strength` attribute.

        If `self.noise_strength` is `0` return a noiseless state-vector simulator.
        If it's set to `float('inf')` the simulator will be `cirq.ZerosSampler`.
        Otherwise, we return a density matrix simulator with a depolarizing model with
        `noise_strength` probability of noise.

        This function is used to initialize this class's `processor`.
        """
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

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.dataclass_json_dict(self)

    def __str__(self) -> str:
        """A pretty string name combining processor_id and noise_strength into a unique name."""
        if self.noise_strength == 0:
            suffix = 'simulator'
        elif self.noise_strength == float('inf'):
            suffix = 'zeros'
        else:
            suffix = f'depol({self.noise_strength:.3e})'
        return f'{self.processor_id}-{suffix}'

    def __repr__(self) -> str:
        return dataclass_repr(self, namespace='cirq_google')


class SimulatedProcessorWithLocalDeviceRecord(SimulatedProcessorRecord):
    """A serializable record mapping a processor_id and optional noise spec to a
    completely local cg.AbstractProcessor

    Args:
        processor_id: The processor id we are emulating
        noise_strength: To do noisy simulation, set this to a positive float. The default
            of `0` will result in a noiseless state-vector simulation. If `float('inf')`
            is provided the simulator will be `cirq.ZerosSampler`. Otherwise, use
            a depolarizing model with this probability of noise.
    """

    def _get_input_device(self) -> 'cirq.Device':
        """Return a `cg.SerializableDevice` for the specified processor_id.

        Only 'rainbow' and 'weber' are recognized processor_ids and the device information
        may not be up-to-date, as it is completely local.
        """

        gate_sets = [cg.FSIM_GATESET]
        device_spec = _create_device_spec_from_template(MOST_RECENT_TEMPLATES[self.processor_id])
        device = cg.SerializableDevice.from_proto(device_spec, gate_sets)
        return device
