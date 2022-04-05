# Copyright 2021 The Cirq Developers
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

"""Classes for representing device noise.

NoiseProperties is an abstract class for capturing metrics of a device that can
be translated into noise models. NoiseModelFromNoiseProperties consumes those
noise models to produce a single noise model which replicates device noise.
"""

import abc
from typing import Iterable, Sequence, TYPE_CHECKING, List

from cirq import _compat, _import, ops, protocols, devices
from cirq.devices.noise_utils import (
    PHYSICAL_GATE_TAG,
)

circuits = _import.LazyLoader("circuits", globals(), "cirq.circuits.circuit")

if TYPE_CHECKING:
    import cirq


class NoiseProperties(abc.ABC):
    """Noise-defining properties for a quantum device."""

    @abc.abstractmethod
    def build_noise_models(self) -> List['cirq.NoiseModel']:
        """Construct all NoiseModels associated with this NoiseProperties."""


class NoiseModelFromNoiseProperties(devices.NoiseModel):
    def __init__(self, noise_properties: NoiseProperties) -> None:
        """Creates a Noise Model from a NoiseProperties object that can be used with a Simulator.

        Args:
            noise_properties: the NoiseProperties object to be converted to a Noise Model.

        Raises:
            ValueError: if no NoiseProperties object is specified.
        """
        self._noise_properties = noise_properties
        self.noise_models = self._noise_properties.build_noise_models()

    def is_virtual(self, op: 'cirq.Operation') -> bool:
        """Returns True if an operation is virtual.

        Device-specific subclasses should implement this method to mark any
        operations which their device handles outside the quantum hardware.

        Args:
            op: an operation to check for virtual indicators.

        Returns:
            True if `op` is virtual.
        """
        return False

    @_compat.deprecated(deadline='v0.16', fix='Use is_virtual instead.')
    def virtual_predicate(self, op: 'cirq.Operation') -> bool:
        return self.is_virtual(op)

    def noisy_moments(
        self, moments: Iterable['cirq.Moment'], system_qubits: Sequence['cirq.Qid']
    ) -> Sequence['cirq.OP_TREE']:
        # Split multi-qubit measurements into single-qubit measurements.
        # These will be recombined after noise is applied.
        split_measure_moments = []
        multi_measurements = {}
        for moment in moments:
            split_measure_ops = []
            for op in moment:
                if not protocols.is_measurement(op):
                    split_measure_ops.append(op)
                    continue
                m_key = protocols.measurement_key_obj(op)
                multi_measurements[m_key] = op
                for q in op.qubits:
                    split_measure_ops.append(ops.measure(q, key=m_key))
            split_measure_moments.append(circuits.Moment(split_measure_ops))

        # Append PHYSICAL_GATE_TAG to non-virtual ops in the input circuit,
        # using `self.virtual_predicate` to determine virtuality.
        new_moments = []
        for moment in split_measure_moments:
            virtual_ops = {op for op in moment if self.is_virtual(op)}
            physical_ops = [
                op.with_tags(PHYSICAL_GATE_TAG) for op in moment if op not in virtual_ops
            ]
            # Both physical and virtual operations remain in the circuit, but
            # only ops with PHYSICAL_GATE_TAG will receive noise.
            if virtual_ops:
                # Only subclasses will trigger this case.
                new_moments.append(circuits.Moment(virtual_ops))  # coverage: ignore
            if physical_ops:
                new_moments.append(circuits.Moment(physical_ops))

        split_measure_circuit = circuits.Circuit(new_moments)

        # Add noise from each noise model. The PHYSICAL_GATE_TAGs added
        # previously allow noise models to distinguish physical gates from
        # those added by other noise models.
        noisy_circuit = split_measure_circuit.copy()
        for model in self.noise_models:
            noisy_circuit = noisy_circuit.with_noise(model)

        # Recombine measurements.
        final_moments = []
        for moment in noisy_circuit:
            combined_measure_ops = []
            restore_keys = set()
            for op in moment:
                if not protocols.is_measurement(op):
                    combined_measure_ops.append(op)
                    continue
                restore_keys.add(protocols.measurement_key_obj(op))
            for key in restore_keys:
                combined_measure_ops.append(multi_measurements[key])
            final_moments.append(circuits.Moment(combined_measure_ops))
        return final_moments
