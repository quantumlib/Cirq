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

import abc
import functools
import itertools
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sympy
from attrs import field, frozen

from cirq import circuits, ops
from cirq.protocols import unitary_protocol
from cirq.protocols.has_unitary_protocol import has_unitary
from cirq.study import sweepable
from cirq.study.sweeps import Points, Zip
from cirq.transformers import transformer_api
from cirq.transformers.analytical_decompositions import single_qubit_decompositions


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


@frozen
class TwoQubitGateSymbolizer:
    """Parameterizes two qubit gates with symbols.

    Attributes:
        symbolizer: A callable that takes a two-qubit gate and a sequence of symbols,
            and returns a tuple containing the parameterized gate and a dictionary
            mapping symbol names to their values.
        n_symbols: The number of symbols to use for parameterization.
    """

    symbolizer_fn: Callable[[ops.Gate, Sequence[sympy.Symbol]], Tuple[ops.Gate, Dict[str, Real]]]
    n_symbols: int

    def __call__(
        self, two_qubit_gate: ops.Gate, symbols: Sequence[sympy.Symbol]
    ) -> Tuple[ops.Gate, Dict[str, Real]]:
        """Symbolizes a two qubit gate to a parameterized gate.

        Args:
            two_qubit_gate: The 2 qubit gate to be symbolized.
            symbols: A sequence of sympy symbols to use for parameterization.

        Returns:
            A tuple containing the parameterized gate and a dictionary mapping
            symbol names to their values.

        Raises:
            ValueError: If the provided symbols do not match the expected number.
        """
        if len(symbols) != self.n_symbols:
            raise ValueError(
                f"Expect {self.n_symbols} symbols, but got {len(symbols)} symbols in {symbols}"
            )
        return self.symbolizer_fn(two_qubit_gate, symbols)


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
        two_qubit_gate_symbolizer: Optional[TwoQubitGateSymbolizer] = None,
    ) -> None:
        """Constructs a GaugeTransformer.

        Args:
            target: Target two-qubit gate, a gate-family or a gate-set of two-qubit gates.
            gauge_selector: A callable that takes a numpy random number generator
                as an argument and returns a Gauge.
            two_qubit_gate_symbolizer: A symbolizer to symbolize 2 qubit gates.
        """
        self.target = ops.GateFamily(target) if isinstance(target, ops.Gate) else target
        self.gauge_selector = gauge_selector
        self.two_qubit_gate_symbolizer = two_qubit_gate_symbolizer

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

    def as_sweep(
        self,
        circuit: circuits.AbstractCircuit,
        *,
        N: int,
        context: Optional[transformer_api.TransformerContext] = None,
        prng: Optional[np.random.Generator] = None,
    ) -> Tuple[circuits.AbstractCircuit, sweepable.Sweepable]:
        """Generates a parameterized circuit with *N* sets of sweepable parameters.

        Args:
            circuit: The input circuit to be processed by gauge compiling.
            N: The number of parameter sets to generate.
            context: A `cirq.TransformerContext` storing common configurable options for
                the transformers.
            prng: A pseudo-random number generator to select a gauge within a gauge cluster.
        """

        rng = np.random.default_rng() if prng is None else prng
        if context is None:
            context = transformer_api.TransformerContext(deep=False)
        if context.deep:
            raise ValueError('GaugeTransformer cannot be used with deep=True')
        new_moments: List[List[ops.Operation]] = []  # Store parameterized circuits.
        phxz_sid = itertools.count()
        two_qubit_gate_sid = itertools.count()
        # Map from "((pre|post),$qid,$moment_id)" to gate parameters.
        # E.g., {(post,q1,2): {"x_exponent": "x1", "z_exponent": "z1", "axis_phase": "a1"}}
        phxz_symbols_by_locs: Dict[Tuple[str, ops.Qid, int], Dict[str, sympy.Symbol]] = {}
        # Map from "($q0,$q1,$moment_id)" to gate parameters.
        # E.g., {(q0,q1,0): ["s0"]}.
        two_qubit_gate_symbols_by_locs: Dict[Tuple[ops.Qid, ops.Qid, int], List[sympy.Symbol]] = {}

        def single_qubit_next_symbol() -> Dict[str, sympy.Symbol]:
            sid = next(phxz_sid)
            return _parameterize_to_phxz(sid)

        def two_qubit_gate_next_symbol_list(n: int) -> List[sympy.Symbol]:
            """Returns symbols for 2 qubit gate parameterization."""
            sid = next(two_qubit_gate_sid)
            symbols: List[sympy.Symbol] = [sympy.Symbol(f"s{sid}_{sub}") for sub in range(n)]
            return symbols

        # Build parameterized circuit.
        for moment_id, moment in enumerate(circuit):
            center_moment: List[ops.Operation] = []
            left_moment: List[ops.Operation] = []
            right_moment: List[ops.Operation] = []
            for op in moment:
                if isinstance(op, ops.TaggedOperation) and set(op.tags).intersection(
                    context.tags_to_ignore
                ):
                    center_moment.append(op)
                    continue
                if op.gate is not None and op in self.target:
                    random_gauge = self.gauge_selector(rng).sample(op.gate, rng)
                    # Build symbols for 2-qubit-gates if the transformer might transform it,
                    # otherwise, keep it as it is.
                    if self.two_qubit_gate_symbolizer is not None:
                        symbols: list[sympy.Symbol] = two_qubit_gate_next_symbol_list(
                            self.two_qubit_gate_symbolizer.n_symbols
                        )
                        two_qubit_gate_symbols_by_locs[(op.qubits[0], op.qubits[1], moment_id)] = (
                            symbols
                        )
                        parameterized_2_qubit_gate, _ = self.two_qubit_gate_symbolizer(
                            random_gauge.two_qubit_gate, symbols
                        )
                        center_moment.append(parameterized_2_qubit_gate.on(*op.qubits))
                    else:
                        center_moment.append(op)
                    # Build symbols for the gauge, for a 2-qubit gauge, symbols will be built for
                    # pre/post q0/q1 and the new 2-qubit gate if the 2-qubit gate is updated in
                    # the gauge compiling.
                    for prefix, q in itertools.product(["pre", "post"], op.qubits):
                        xza_by_symbols = single_qubit_next_symbol()  # xza in phased xz gate.
                        phxz_symbols_by_locs[(prefix, q, moment_id)] = xza_by_symbols
                        new_op = ops.PhasedXZGate(**xza_by_symbols).on(q)
                        if prefix == "pre":
                            left_moment.append(new_op)
                        else:
                            right_moment.append(new_op)
                else:
                    center_moment.append(op)
            new_moments.extend(
                [moment for moment in [left_moment, center_moment, right_moment] if moment]
            )

        # Initialize the map from symbol names to their N values.
        values_by_params: Dict[str, List[float]] = {
            **{
                str(symbol): []
                for symbols_by_names in phxz_symbols_by_locs.values()
                for symbol in symbols_by_names.values()
            },
            **{
                str(symbol): []
                for symbols in two_qubit_gate_symbols_by_locs.values()
                for symbol in symbols
            },
        }

        # Assign values for parameters via randomly chosen GaugeSelector.
        for _ in range(N):
            for moment_id, moment in enumerate(circuit):
                for op in moment:
                    if isinstance(op, ops.TaggedOperation) and set(op.tags).intersection(
                        context.tags_to_ignore
                    ):
                        continue
                    if op.gate is not None and len(op.qubits) == 2 and op in self.target:
                        gauge = self.gauge_selector(rng).sample(op.gate, rng)

                        # Get the params for 2 qubit gates.
                        if self.two_qubit_gate_symbolizer is not None:
                            _, vals_by_symbols = self.two_qubit_gate_symbolizer(
                                gauge.two_qubit_gate,
                                [
                                    *two_qubit_gate_symbols_by_locs[
                                        (op.qubits[0], op.qubits[1], moment_id)
                                    ]
                                ],
                            )
                            for symbol_str, val in vals_by_symbols.items():
                                values_by_params[symbol_str].append(float(val))

                        # Get the params of pre/post q0/q1 gates.
                        for pre_or_post, idx in itertools.product(["pre", "post"], [0, 1]):
                            gates = getattr(gauge, f"{pre_or_post}_q{idx}")
                            phxz_params = _gate_sequence_to_phxz_params(
                                gates,
                                phxz_symbols_by_locs[(pre_or_post, op.qubits[idx], moment_id)],
                            )
                            for key, value in phxz_params.items():
                                values_by_params[key].append(float(value))

        sweeps: List[Points] = [
            Points(key=key, points=values) for key, values in values_by_params.items()
        ]

        return circuits.Circuit.from_moments(*new_moments), Zip(*sweeps)


def _build_moments(operation_by_qubits: List[List[ops.Operation]]) -> List[List[ops.Operation]]:
    """Builds moments from a list of operations grouped by qubits.

    Returns a list of moments from a list whose ith element is a list of operations applied
    to qubit i.
    """
    moments = []
    for moment in itertools.zip_longest(*operation_by_qubits):
        moments.append([op for op in moment if op is not None])
    return moments


def _parameterize_to_phxz(symbol_id: int) -> Dict[str, sympy.Symbol]:
    """Returns symbolized parameters for the gate."""

    # Parameterize single qubit gate to parameterized PhasedXZGate.
    phased_xz_params = {
        "x_exponent": sympy.Symbol(f"x{symbol_id}"),
        "z_exponent": sympy.Symbol(f"z{symbol_id}"),
        "axis_phase_exponent": sympy.Symbol(f"a{symbol_id}"),
    }
    return phased_xz_params


def _gate_sequence_to_phxz_params(
    gates: Tuple[ops.Gate, ...], xza_by_symbols: Dict[str, sympy.Symbol]
) -> Dict[str, float]:
    identity_gate_in_phxz = {
        str(xza_by_symbols["x_exponent"]): 0.0,
        str(xza_by_symbols["z_exponent"]): 0.0,
        str(xza_by_symbols["axis_phase_exponent"]): 0.0,
    }
    if not gates:
        return identity_gate_in_phxz
    for gate in gates:
        if not has_unitary(gate) or gate.num_qubits() != 1:
            raise ValueError(
                "Invalid gate sequence to be converted to PhasedXZGate."
                f"Found incompatiable gate {gate} in sequence."
            )
    phxz = (
        single_qubit_decompositions.single_qubit_matrix_to_phxz(
            functools.reduce(
                np.matmul, [unitary_protocol.unitary(gate) for gate in reversed(gates)]
            )
        )
        or ops.I
    )
    if phxz is ops.I:  # Identity gate
        return identity_gate_in_phxz
    # Check the gate type, needs to be a PhasedXZ gate.
    if not isinstance(phxz, ops.PhasedXZGate):
        raise ValueError("Failed to convert the gate sequence to a PhasedXZ gate.")
    if phxz is not None:
        return {
            str(xza_by_symbols["x_exponent"]): phxz.x_exponent,
            str(xza_by_symbols["z_exponent"]): phxz.z_exponent,
            str(xza_by_symbols["axis_phase_exponent"]): phxz.axis_phase_exponent,
        }
