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

import attrs
import numpy as np

from cirq import circuits, ops
from cirq.transformers import transformer_api


@transformer_api.transformer
@attrs.frozen
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

    Attributes:
        target: The target gate, gate family or gateset, must exist in each of the moment in
          the "moments to be gauged".
        supported_gates: The gates that are supported in the "moments to be gauged".
    """

    target: ops.GateFamily | ops.Gateset
    supported_gates: ops.GateFamily | ops.Gateset

    @abc.abstractmethod
    def sample_left_moment(
        self, active_qubits: frozenset[ops.Qid], prng: np.random.Generator
    ) -> circuits.Moment:
        """Samples a random single-qubit moment to be inserted before the target block.

        Args:
            active_qubits: The qubits on which the sampled gates should be applied.
            prng: A pseudorandom number generator.

        Returns:
            The sampled moment.
        """

    @abc.abstractmethod
    def gauge_on_moments(
        self, moments_to_gauge: list[circuits.Moment], prng: np.random.Generator
    ) -> list[circuits.Moment]:
        """Gauges a block of moments.

        Args:
            moments_to_gauge: A list of moments to be gauged.
            prng: A pseudorandom number generator.

        Returns:
            A list of moments after gauging.
        """

    def is_target_moment(
        self, moment: circuits.Moment, context: transformer_api.TransformerContext | None = None
    ) -> bool:
        """Checks if a moment is a target for gauging.

        A moment is a target moment if it contains at least one target op and
        all its operations are supported by this transformer.
        """
        # skip the moment if the moment is tagged to be ignored
        if context and set(moment.tags).intersection(context.tags_to_ignore):
            return False

        has_target_gates: bool = False
        for op in moment:
            if (
                context
                and isinstance(op, ops.TaggedOperation)
                and set(op.tags).intersection(context.tags_to_ignore)
            ):  # skip the moment if the op is tagged to be ignored
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
        rng_or_seed: np.random.Generator | int | None = None,
    ) -> circuits.AbstractCircuit:
        """Apply the transformer to the given circuit.

        Args:
            circuit: The circuit to transform.
            context: `cirq.TransformerContext` storing common configurable options for transformers.
            prng: A pseudorandom number generator.

        Returns:
            The transformed circuit.

        Raises:
            ValueError: if the TransformerContext has deep=True.

        """
        if context is None:
            context = transformer_api.TransformerContext(deep=False)
        if context.deep:
            raise ValueError('GaugeTransformer cannot be used with deep=True')
        rng = (
            rng_or_seed
            if isinstance(rng_or_seed, np.random.Generator)
            else np.random.default_rng(rng_or_seed)
        )

        output_moments: list[circuits.Moment] = []
        moments_to_gauge: list[circuits.Moment] = []
        for moment in circuit:
            if self.is_target_moment(moment, context):
                moments_to_gauge.append(moment)
            else:
                if moments_to_gauge:
                    output_moments.extend(self.gauge_on_moments(moments_to_gauge, rng))
                    moments_to_gauge.clear()
                output_moments.append(moment)
        if moments_to_gauge:
            output_moments.extend(self.gauge_on_moments(moments_to_gauge, rng))

        return circuits.Circuit.from_moments(*output_moments)
