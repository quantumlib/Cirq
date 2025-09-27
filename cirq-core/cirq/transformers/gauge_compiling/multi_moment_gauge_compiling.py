# Copyright 2025 The Cirq Developers
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

"""Defines the abstraction for multi-moment gauge compiling as a cirq transformer."""

import abc

import numpy as np

from cirq import circuits, ops
from cirq.transformers import transformer_api


@transformer_api.transformer
class MultiMomentGaugeTransformer(abc.ABC):
    """A gauge transformer that wraps target blocks of moments with single-qubit gates.

    In detail, a "gauging moment" of single-qubit gates is inserted before a target block of
    moments. These gates are then commuted through the block, resulting in a corresponding
    moment of gates after it.

        q₀: ... ───LG0───╭───────────╮────RG0───...
                         │           │
        q₁: ... ───LG1───┤  moments  ├────RG1───...
                         │   to be   │
        q₂: ... ───LG2───┤ gauged on ├────RG2───...
                         │           │
        q₃: ... ───LG3───╰───────────╯────RG3───...
    """

    def __init__(
        self,
        target: ops.Gate | ops.Gateset | ops.GateFamily,
        supported_gates: ops.Gateset = ops.Gateset(),
    ) -> None:
        """Constructs a MultiMomentGaugeTransformer.

        Args:
            target: Specifies the two-qubit gates, gate families, or gate sets that will
              be targeted during gauge compiling. The gauge moment must contain at least
              one of the target gates.
            supported_gates: Determines what other gates, in addition to the target gates,
              are permitted within the gauge moments. If a moment contains a gate not found
              in either target or supported_gates, it won't be gauged.
        """
        self.target = ops.GateFamily(target) if isinstance(target, ops.Gate) else target
        self.supported_gates = (
            ops.GateFamily(supported_gates)
            if isinstance(supported_gates, ops.Gate)
            else supported_gates
        )

    @abc.abstractmethod
    def gauge_on_moments(self, moments_to_gauge: list[circuits.Moment]) -> list[circuits.Moment]:
        """Gauges a block of moments.

        Args:
            moments_to_gauge: A list of moments to be gauged.

        Returns:
            A list of moments after gauging.
        """

    @abc.abstractmethod
    def sample_left_moment(
        self, active_qubits: frozenset[ops.Qid], rng: np.random.Generator
    ) -> circuits.Moment:
        """Samples a random single-qubit moment to be inserted before the target block.

        Args:
            active_qubits: The qubits on which the sampled gates should be applied.
            rng: A pseudorandom number generator.

        Returns:
            The sampled moment.
        """

    def is_target_moment(
        self, moment: circuits.Moment, context: transformer_api.TransformerContext | None = None
    ) -> bool:
        """Checks if a moment is a target for gauging.

        A moment is a target moment if it contains at least one target op and
        all its operations are supported by this transformer.
        """
        has_target_gates: bool = False
        for op in moment:
            if (
                context
                and isinstance(op, ops.TaggedOperation)
                and set(op.tags).intersection(context.tags_to_ignore)
            ):  # skip the moment if the op is tagged with a tag in tags_to_ignore
                return False
            if op.gate:
                if op in self.target:
                    has_target_gates = True
                elif op not in self.supported_gates:
                    return False
        return has_target_gates

    def __call__(
        self,
        circuit: circuits.AbstractCircuit,
        *,
        context: transformer_api.TransformerContext | None = None,
    ) -> circuits.AbstractCircuit:
        if context is None:
            context = transformer_api.TransformerContext(deep=False)
        if context.deep:
            raise ValueError('GaugeTransformer cannot be used with deep=True')
        output_moments: list[circuits.Moment] = []
        moments_to_gauge: list[circuits.Moment] = []
        for moment in circuit:
            if self.is_target_moment(moment, context):
                moments_to_gauge.append(moment)
            else:
                if moments_to_gauge:
                    output_moments.extend(self.gauge_on_moments(moments_to_gauge))
                    moments_to_gauge.clear()
                output_moments.append(moment)
        if moments_to_gauge:
            output_moments.extend(self.gauge_on_moments(moments_to_gauge))

        return circuits.Circuit.from_moments(*output_moments)
