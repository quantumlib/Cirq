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

from typing import TYPE_CHECKING, Sequence
import attrs
import numpy as np

import cirq.transformers.transformer_api as transformer_api
import cirq.ops as ops
import cirq.protocols as protocols

if TYPE_CHECKING:
    import cirq

_PAULIS = (ops.I, ops.X, ops.Y, ops.Z)
_CLIFFORDS = tuple(ops.SingleQubitCliffordGate.all_single_qubit_cliffords)
_INV_CLIFFORDS = tuple(c**-1 for c in ops.SingleQubitCliffordGate.all_single_qubit_cliffords)

_NAME_TO_GATES = {
    'pauli': _PAULIS,
    'clifford': _CLIFFORDS,
    'inv_clifford': _INV_CLIFFORDS,
}


def _gauges_arg_converter(gauges: str|Sequence[cirq.Gate] = 'clifford') -> tuple[cirq.Gate, ...]:
    if isinstance(gauges, str):
        return _NAME_TO_GATES[gauges]
    return tuple(gauges)



def _repr_fn(gauges: tuple[cirq.Gate, ...]) -> str:
    if gauges is _PAULIS or gauges == _PAULIS: return '"pauli"'
    if gauges is _CLIFFORDS: return '"clifford"'
    if gauges is _INV_CLIFFORDS: return '"inv_clifford"'
    return str(gauges)

def _get_structure(active: list[tuple[int, bool]], min_length: int, n: int, gauge_beginning: bool, gauge_ending: bool) -> list[tuple[int, int]]:
    assert active
    structure = []
    if gauge_beginning:
        stop, is_mergable = active[0]
        if min_length <= stop:
            if is_mergable:
                structure.append((0, stop))
            else:
                structure.append((0, stop - 1))
    
    for i in range(len(active) - 1):
        left_pos, left_is_mergable = active[i]
        right_pos, right_is_mergable = active[i + 1]

        structure.append((left_pos+1-left_is_mergable, right_pos-1+right_is_mergable))

    if gauge_ending:
        stop, is_mergable = active[-1]
        if min_length <= n - stop - 1:
            if is_mergable:
                structure.append((stop, n-1))
            else:
                structure.append((stop+1, n-1))
    
    return structure


def _merge(g1: cirq.Gate, g2: cirq.Gate) -> cirq.Gate:
    u1 = protocols.unitary(g1)
    u2 = protocols.unitary(g2)
    return ops.PhasedXZGate.from_matrix(u2 @ u1)

@transformer_api.transformer
@attrs.frozen
class IdleMomentsGauge:
    """A Gauge that encloses idle moments with a gate selected from the provided gauge and its adjoint.
    
    Attributes:
        min_length: The transformer will gauge the idle sequence only if its length
            is >= min_length.
        gauges: A sequence of gates to sample from. This parameter can also be one
            of "pauli", "clifford", and "inv_clifford". 
        gauges_inverse: The inverse of the given gauges. This is an optional argument
            that defaults to computing the inverse of the given `gauges`.
        gauge_beginning: Whether to apply the gauge to idle moments at the beginning
            before any operation is applied to the qubit.
        gauge_ending: Whether to apply the gauge to idle moments at the end after the
            last operation is applied to the qubit.
    """
    min_length: int = attrs.field(validator=(attrs.validators.instance_of(int), attrs.validators.ge(1)))
    gauges: tuple[cirq.Gate, ...] = attrs.field(default='clifford', converter=_gauges_arg_converter, repr=_repr_fn)
    gauges_inverse: tuple[cirq.Gate, ...] = attrs.field(converter=_gauges_arg_converter, repr=_repr_fn)
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
                np.all(protocols.unitary(g * g_adj), identity) for g, g_adj in zip(self.gauges, self.gauges_inverse, strict=True)
            )



    def __call__(self, circuit: cirq.AbstractCircuit, *, context: transformer_api.TransformerContext | None = None, rng_or_seed: np.random.Generator | int | None = None):
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
        rng = rng_or_seed if isinstance(rng_or_seed, np.random.Generator) else np.random.default_rng(rng_or_seed)
        context = context if isinstance(context, transformer_api.TransformerContext) else transformer_api.TransformerContext(deep=False)
        if context.deep:
            raise ValueError("IdleMomentsGauge doesn't support deep TransformerContext")

        tags_to_ignore = frozenset(context.tags_to_ignore)
        all_qubits = circuit.all_qubits()

        active_moments = {q:[] for q in all_qubits}
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

        single_qubit_moments = [
            {q:op.gate for op in m if len(op.qubits) == 1} for m in circuit
        ]
        non_single_qubit_moments = [
            [op for op in m if len(op.qubits) != 1] for m in circuit
        ]

        for q, active in active_moments.items():
            for s, e in _get_structure(active, self.min_length, len(circuit), self.gauge_beginning, self.gauge_ending):
                gate_index = rng.choice(len(self.gauges))
                gate = self.gauges[gate_index]
                gate_inv = self.gauges[gate_index]


                if existing_gate := single_qubit_moments[s].get(q, None):
                    single_qubit_moments[s][q] = _merge(existing_gate, gate)
                else:
                    single_qubit_moments[s][q] = gate
                
                if existing_gate := single_qubit_moments[e].get(q, None):
                    single_qubit_moments[e][q] = _merge(gate_inv, existing_gate)
                else:
                    single_qubit_moments[e][q] = gate_inv

        final_moments = [
            [g(q) for q, g in sq.items()] + nsq for sq, nsq in zip(single_qubit_moments, non_single_qubit_moments, strict=True)
        ]

        return cirq.Circuit.from_moments(*final_moments)


if __name__ == '__main__':
    tr = IdleMomentsGauge(2, gauges='pauli', gauge_beginning=True)
    print(tr)

    import cirq
    # c = cirq.Circuit.from_moments(cirq.X(cirq.q(0)), [], [], cirq.X(cirq.q(0)))
    # c = cirq.Circuit.from_moments([], [], cirq.X(cirq.q(0)))
    c = cirq.Circuit.from_moments([], [], cirq.X(cirq.q(0)).with_tags('ignore'))
    print(c)
    print(tr(c, context=cirq.TransformerContext(tags_to_ignore=("ignore",))))
