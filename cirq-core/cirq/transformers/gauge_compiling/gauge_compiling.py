# Copyright 2024 The Cirq Developers
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

"""Creates the abstraction for gauge compiling as a cirq transformer."""

from typing import Callable, Tuple, Optional, Sequence, Union, List
import abc
import itertools
import functools

from dataclasses import dataclass
from attrs import frozen, field
import numpy as np

from cirq.transformers import transformer_api
from cirq import ops, circuits


class Gauge(abc.ABC):
    """A gauge replaces a two qubit gate with an equivalent subcircuit.
    0: pre_q0───────two_qubit_gate───────post_q0
                        |
    1: pre_q1───────two_qubit_gate───────post_q1

    The Gauge class in general represents a family of closely related gauges
    (e.g. random z-rotations); Use `sample` method to get a specific gauge.
    """

    def weight(self) -> float:
        """Returns the relative frequency for selecting this gauge."""
        return 1.0

    @abc.abstractmethod
    def sample(self, gate: ops.Gate, prng: np.random.Generator) -> "ConstantGauge":
        """Returns a ConstantGauge sampled from a family of gauges.

        Args:
            gate: The two qubit gate to replace.
            prng: A numpy random number generator.

        Returns:
            A ConstantGauge.
        """


@frozen
class ConstantGauge(Gauge):
    """A gauge that replaces a two qubit gate with a constant gauge."""

    two_qubit_gate: ops.Gate
    pre_q0: Tuple[ops.Gate, ...] = field(
        default=(), converter=lambda g: (g,) if isinstance(g, ops.Gate) else tuple(g)
    )
    pre_q1: Tuple[ops.Gate, ...] = field(
        default=(), converter=lambda g: (g,) if isinstance(g, ops.Gate) else tuple(g)
    )
    post_q0: Tuple[ops.Gate, ...] = field(
        default=(), converter=lambda g: (g,) if isinstance(g, ops.Gate) else tuple(g)
    )
    post_q1: Tuple[ops.Gate, ...] = field(
        default=(), converter=lambda g: (g,) if isinstance(g, ops.Gate) else tuple(g)
    )
    swap_qubits: bool = False

    def sample(self, gate: ops.Gate, prng: np.random.Generator) -> "ConstantGauge":
        return self

    @property
    def pre(self) -> Tuple[Tuple[ops.Gate, ...], Tuple[ops.Gate, ...]]:
        """A tuple (ops to apply to q0, ops to apply to q1)."""
        return self.pre_q0, self.pre_q1

    @property
    def post(self) -> Tuple[Tuple[ops.Gate, ...], Tuple[ops.Gate, ...]]:
        """A tuple (ops to apply to q0, ops to apply to q1)."""
        return self.post_q0, self.post_q1

    def on(self, q0: ops.Qid, q1: ops.Qid) -> ops.Operation:
        """Returns the operation that replaces the two qubit gate."""
        if self.swap_qubits:
            return self.two_qubit_gate(q1, q0)
        return self.two_qubit_gate(q0, q1)


@frozen
class SameGateGauge(Gauge):
    """Same as ConstantGauge but the new two-qubit gate equals the old gate."""

    pre_q0: Tuple[ops.Gate, ...] = field(
        default=(), converter=lambda g: (g,) if isinstance(g, ops.Gate) else tuple(g)
    )
    pre_q1: Tuple[ops.Gate, ...] = field(
        default=(), converter=lambda g: (g,) if isinstance(g, ops.Gate) else tuple(g)
    )
    post_q0: Tuple[ops.Gate, ...] = field(
        default=(), converter=lambda g: (g,) if isinstance(g, ops.Gate) else tuple(g)
    )
    post_q1: Tuple[ops.Gate, ...] = field(
        default=(), converter=lambda g: (g,) if isinstance(g, ops.Gate) else tuple(g)
    )
    swap_qubits: bool = False

    def sample(self, gate: ops.Gate, prng: np.random.Generator) -> ConstantGauge:
        return ConstantGauge(
            two_qubit_gate=gate,
            pre_q0=self.pre_q0,
            pre_q1=self.pre_q1,
            post_q0=self.post_q0,
            post_q1=self.post_q1,
            swap_qubits=self.swap_qubits,
        )


def _select(choices: Sequence[Gauge], probabilites: np.ndarray, prng: np.random.Generator) -> Gauge:
    return choices[prng.choice(len(choices), p=probabilites)]


@dataclass(frozen=True)
class GaugeSelector:
    """Samples a gauge from a list of gauges."""

    gauges: Sequence[Gauge]

    @functools.cached_property
    def _weights(self) -> np.ndarray:
        weights = np.array([g.weight() for g in self.gauges])
        return weights / np.sum(weights)

    def __call__(self, prng: np.random.Generator) -> Gauge:
        """Randomly selects a gauge with probability proportional to its weight."""
        return _select(self.gauges, self._weights, prng)


@transformer_api.transformer
class GaugeTransformer:
    def __init__(
        self,
        # target can be either a specific gate, gatefamily or gateset
        # which allows matching parametric gates.
        target: Union[ops.Gate, ops.Gateset, ops.GateFamily],
        gauge_selector: Callable[[np.random.Generator], Gauge],
    ) -> None:
        """Constructs a GaugeTransformer.

        Args:
            target: Target two-qubit gate, a gate-family or a gate-set of two-qubit gates.
            gauge_selector: A callable that takes a numpy random number generator
                as an argument and returns a Gauge.
        """
        self.target = ops.GateFamily(target) if isinstance(target, ops.Gate) else target
        self.gauge_selector = gauge_selector

    def __call__(
        self,
        circuit: circuits.AbstractCircuit,
        *,
        context: Optional[transformer_api.TransformerContext] = None,
        prng: Optional[np.random.Generator] = None,
    ) -> circuits.AbstractCircuit:
        rng = np.random.default_rng() if prng is None else prng
        if context is None:
            context = transformer_api.TransformerContext(deep=False)
        if context.deep:
            raise ValueError('GaugeTransformer cannot be used with deep=True')
        new_moments = []
        left: List[List[ops.Operation]] = []
        right: List[List[ops.Operation]] = []
        for moment in circuit:
            left.clear()
            right.clear()
            center: List[ops.Operation] = []
            for op in moment:
                if isinstance(op, ops.TaggedOperation) and set(op.tags).intersection(
                    context.tags_to_ignore
                ):
                    center.append(op)
                    continue
                if op.gate is not None and len(op.qubits) == 2 and op in self.target:
                    gauge = self.gauge_selector(rng).sample(op.gate, rng)
                    q0, q1 = op.qubits
                    left.extend([g(q) for g in gs] for q, gs in zip(op.qubits, gauge.pre))
                    center.append(gauge.on(q0, q1))
                    right.extend([g(q) for g in gs] for q, gs in zip(op.qubits, gauge.post))
                else:
                    center.append(op)
            if left:
                new_moments.extend(_build_moments(left))
            new_moments.append(center)
            if right:
                new_moments.extend(_build_moments(right))
        return circuits.Circuit.from_moments(*new_moments)


def _build_moments(operation_by_qubits: List[List[ops.Operation]]) -> List[List[ops.Operation]]:
    """Builds moments from a list of operations grouped by qubits.

    Returns a list of moments from a list whose ith element is a list of operations applied
    to qubit i.
    """
    moments = []
    for moment in itertools.zip_longest(*operation_by_qubits):
        moments.append([op for op in moment if op is not None])
    return moments
