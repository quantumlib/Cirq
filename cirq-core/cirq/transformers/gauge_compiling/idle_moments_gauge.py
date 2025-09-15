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
from __future__ import annotations

from typing import Iterator, Sequence, TYPE_CHECKING

import attrs
import numpy as np

import cirq.circuits as circuits
import cirq.ops as ops
import cirq.protocols as protocols
import cirq.transformers.transformer_api as transformer_api

if TYPE_CHECKING:
    import cirq

_PAULIS: tuple[cirq.Gate, ...] = (ops.I, ops.X, ops.Y, ops.Z)  # type: ignore[has-type]
_CLIFFORDS = tuple(ops.SingleQubitCliffordGate.all_single_qubit_cliffords)
_INV_CLIFFORDS = tuple(c**-1 for c in ops.SingleQubitCliffordGate.all_single_qubit_cliffords)

_NAME_TO_GATES = {'pauli': _PAULIS, 'clifford': _CLIFFORDS, 'inv_clifford': _INV_CLIFFORDS}


def _gauges_arg_converter(gauges: str | Sequence[cirq.Gate] = 'clifford') -> tuple[cirq.Gate, ...]:
    if isinstance(gauges, str):
        return _NAME_TO_GATES[gauges]
    return tuple(gauges)


def _repr_fn(gauges: tuple[cirq.Gate, ...]) -> str:
    if gauges is _PAULIS:
        return '"pauli"'
    if gauges is _CLIFFORDS:
        return '"clifford"'
    if gauges is _INV_CLIFFORDS:
        return '"inv_clifford"'
    return repr(gauges)


def _get_structure(
    active: list[tuple[int, bool]],
    min_length: int,
    n: int,
    gauge_beginning: bool,
    gauge_ending: bool,
) -> Iterator[tuple[int, int]]:
    assert active
    if gauge_beginning:
        stop, is_mergable = active[0]
        if min_length <= stop:
            if is_mergable:
                yield (0, stop)
            else:
                yield (0, stop - 1)

    for i in range(len(active) - 1):
        left_pos, left_is_mergable = active[i]
        right_pos, right_is_mergable = active[i + 1]
        if right_pos - left_pos - 1 >= min_length:
            yield (left_pos + 1 - left_is_mergable, right_pos - 1 + right_is_mergable)

    if gauge_ending:
        stop, is_mergable = active[-1]
        if min_length <= n - stop - 1:
            if is_mergable:
                yield (stop, n - 1)
            else:
                yield (stop + 1, n - 1)


def _merge(g1: cirq.Gate, g2: cirq.Gate, q: cirq.Qid, tags: Sequence) -> cirq.Operation:
    u1 = protocols.unitary(g1)
    u2 = protocols.unitary(g2)
    return ops.PhasedXZGate.from_matrix(u2 @ u1)(q).with_tags(*tags)


@transformer_api.transformer
@attrs.frozen
class IdleMomentsGauge:
    r"""A transformer that inserts identity-preserving "gauge" gates around idle qubit moments.

    This transformer identifies sequences of consecutive idle moments on a single qubit
    that meet a `min_length` threshold. For each such sequence, it inserts a randomly
    selected gate `G` from `gauges` at the start of the idle period and its inverse `G^-1`
    from `gauges_inverse` at the end. This ensures the logical circuit behavior remains
    unchanged ($G \cdot G^{-1} = I$).

    The primary goal is to introduce specific structure into idle periods, which is
    useful for experiments.

    Attributes:
        min_length: Minimum number of consecutive idle moments for a gauge to be applied (>= 1).

        gauges: A sequence of `cirq.Gate` objects to randomly select from.
            Can be a custom tuple or a string alias:
            - `"pauli"`: Uses single-qubit Pauli gates (I, X, Y, Z).
            - `"clifford"`: Uses all 24 single-qubit Clifford gates.

        gauges_inverse: An optional sequence of `cirq.Gate` objects representing
            the inverses of gates in `gauges`. The `k`-th gate in `gauges_inverse`
            must be the inverse of the `k`-th gate in `gauges`. If not provided,
            it's automatically computed:
            - `"pauli"` defaults to `"pauli"`.
            - `"clifford"` defaults to `_INV_CLIFFORDS` (inverses of Clifford gates).
            - Custom gate sequences have their inverses computed.
            This positional correspondence is enforced by an internal assertion to
            ensure $G \cdot G^{-1} = I$.

        gauge_beginning: If `True`, applies a gauge to idle moments at the circuit's start,
            before any other qubit operation. Defaults to `False`.

        gauge_ending: If `True`, applies a gauge to idle moments at the circuit's end,
            after the last qubit operation. Defaults to `False`.
    """

    min_length: int = attrs.field(
        validator=(attrs.validators.instance_of(int), attrs.validators.ge(1))
    )
    gauges: tuple[cirq.Gate, ...] = attrs.field(
        default='clifford', converter=_gauges_arg_converter, repr=_repr_fn
    )
    gauges_inverse: tuple[cirq.Gate, ...] = attrs.field(
        converter=_gauges_arg_converter, repr=_repr_fn
    )
    gauge_beginning: bool = False
    gauge_ending: bool = False

    @gauges_inverse.default
    def _gauges_inverse_default(self) -> tuple[cirq.Gate, ...]:
        if self.gauges is _PAULIS:
            return _PAULIS
        if self.gauges is _CLIFFORDS:
            return _INV_CLIFFORDS
        if self.gauges is _INV_CLIFFORDS:
            return _CLIFFORDS
        return tuple(g**-1 for g in self.gauges)

    def __attrs_post_init__(self):
        if self.gauges is _PAULIS:
            assert self.gauges_inverse is _PAULIS
        elif self.gauges is _CLIFFORDS:
            assert self.gauges_inverse is _INV_CLIFFORDS
        elif self.gauges is _INV_CLIFFORDS:
            assert self.gauges_inverse is _CLIFFORDS
        else:
            identity = np.eye(2)
            assert np.all(
                np.all(protocols.unitary(g * g_adj), identity)
                for g, g_adj in zip(self.gauges, self.gauges_inverse, strict=True)
            )

    def __call__(
        self,
        circuit: cirq.AbstractCircuit,
        *,
        context: transformer_api.TransformerContext | None = None,
        rng_or_seed: np.random.Generator | int | None = None,
    ):
        """Apply the IdleMomentGauge transformer.

        Args:
            circuit: The circuit to process.
            context: The TransformerContext.
            rng_or_seed: The source of randomness.

        Returns:
            A transformed circuit.

        Raises:
            ValueError: if the TransformerContext has deep=True.
        """
        rng = (
            rng_or_seed
            if isinstance(rng_or_seed, np.random.Generator)
            else np.random.default_rng(rng_or_seed)
        )
        context = (
            context
            if isinstance(context, transformer_api.TransformerContext)
            else transformer_api.TransformerContext(deep=False)
        )
        if context.deep:
            raise ValueError("IdleMomentsGauge doesn't support deep TransformerContext")

        tags_to_ignore = frozenset(context.tags_to_ignore)
        all_qubits = circuit.all_qubits()

        active_moments: dict[cirq.Qid, list[tuple[int, bool]]] = {q: [] for q in all_qubits}
        for m_id, moment in enumerate(circuit):
            if tags_to_ignore & frozenset(moment.tags):
                for q in all_qubits:
                    active_moments[q].append((m_id, False))
            else:
                for op in moment:
                    if len(op.qubits) == 1:
                        is_mergable = True
                        if tags_to_ignore & frozenset(op.tags):
                            is_mergable = False
                    else:
                        is_mergable = False
                    for q in op.qubits:
                        active_moments[q].append((m_id, is_mergable))

        single_qubit_moments = [{q: op for op in m if len(op.qubits) == 1} for m in circuit]
        non_single_qubit_moments = [[op for op in m if len(op.qubits) != 1] for m in circuit]

        for q, active in active_moments.items():
            for s, e in _get_structure(
                active, self.min_length, len(circuit), self.gauge_beginning, self.gauge_ending
            ):
                gate_index = rng.choice(len(self.gauges))
                gate = self.gauges[gate_index]
                gate_inv = self.gauges_inverse[gate_index]

                if existing_op := single_qubit_moments[s].get(q, None):
                    existing_gate = existing_op.gate
                    assert existing_gate is not None
                    single_qubit_moments[s][q] = _merge(existing_gate, gate, q, existing_op.tags)
                else:
                    single_qubit_moments[s][q] = gate(q)

                if existing_op := single_qubit_moments[e].get(q, None):
                    existing_gate = existing_op.gate
                    assert existing_gate is not None
                    single_qubit_moments[e][q] = _merge(
                        gate_inv, existing_gate, q, existing_op.tags
                    )
                else:
                    single_qubit_moments[e][q] = gate_inv(q)

        return circuits.Circuit.from_moments(
            *(
                [op for op in sq.values()] + nsq
                for sq, nsq in zip(single_qubit_moments, non_single_qubit_moments, strict=True)
            )
        )
