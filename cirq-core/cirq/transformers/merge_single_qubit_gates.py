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

"""Transformer passes to combine adjacent single-qubit rotations."""

import re
from typing import Dict, Hashable, List, Optional, Tuple, TYPE_CHECKING

import sympy

from cirq import circuits, ops, protocols
from cirq.study.sweeps import Points, Sweep, Zip
from cirq.study.result import TMeasurementKey
from cirq.transformers import align, merge_k_qubit_gates, transformer_api, transformer_primitives
from cirq.transformers.tag_transformers import index_tags, remove_tags
from cirq.transformers.analytical_decompositions import single_qubit_decompositions

if TYPE_CHECKING:
    import cirq


@transformer_api.transformer
def merge_single_qubit_gates_to_phased_x_and_z(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    atol: float = 1e-8,
) -> 'cirq.Circuit':
    """Replaces runs of single qubit rotations with `cirq.PhasedXPowGate` and `cirq.ZPowGate`.

    Specifically, any run of non-parameterized single-qubit unitaries will be replaced by an
    optional PhasedX operation followed by an optional Z operation.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """

    def rewriter(op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        u = protocols.unitary(op)
        if protocols.num_qubits(op) == 0:
            return ops.GlobalPhaseGate(u[0, 0]).on()
        return [
            g(op.qubits[0])
            for g in single_qubit_decompositions.single_qubit_matrix_to_phased_x_z(u, atol)
        ]

    return merge_k_qubit_gates.merge_k_qubit_unitaries(
        circuit, k=1, context=context, rewriter=rewriter
    )


@transformer_api.transformer
def merge_single_qubit_gates_to_phxz(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    merge_tag: Optional[str] = None,  # Change to func
    preserve_tags: bool = False,
    atol: float = 1e-8,
) -> 'cirq.Circuit':
    """Replaces runs of single qubit rotations with a single optional `cirq.PhasedXZGate`.

    Specifically, any run of non-parameterized single-qubit unitaries will be replaced by an
    optional PhasedXZ.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        merge_tag: If provided, tag merged PhXZ gate with it.
        preserve_tags: If true, the union of tags from all merged operations will be appended to the merge op.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """

    def rewriter(circuit_op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':

        u = protocols.unitary(circuit_op)
        if protocols.num_qubits(circuit_op) == 0:
            return ops.GlobalPhaseGate(u[0, 0]).on()

        gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(u, atol) or ops.I
        tags: set[Hashable] = {merge_tag} if merge_tag else set()
        if preserve_tags:
            for op in circuit_op.circuit.all_operations():
                tags.update(op.tags)
        phxz_op = gate.on(circuit_op.qubits[0])
        return phxz_op.with_tags(*tags)

    return merge_k_qubit_gates.merge_k_qubit_unitaries(
        circuit, k=1, context=context, rewriter=rewriter
    )


@transformer_api.transformer
def merge_single_qubit_moments_to_phxz(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    atol: float = 1e-8,
) -> 'cirq.Circuit':
    """Merges adjacent moments with only 1-qubit rotations to a single moment with PhasedXZ gates.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """
    tags_to_ignore = set(context.tags_to_ignore) if context else set()

    def can_merge_moment(m: 'cirq.Moment'):
        return all(
            protocols.num_qubits(op) == 1
            and protocols.has_unitary(op)
            and tags_to_ignore.isdisjoint(op.tags)
            for op in m
        )

    def merge_func(m1: 'cirq.Moment', m2: 'cirq.Moment') -> Optional['cirq.Moment']:
        if not (can_merge_moment(m1) and can_merge_moment(m2)):
            return None
        ret_ops = []
        for q in m1.qubits | m2.qubits:
            op1, op2 = m1.operation_at(q), m2.operation_at(q)
            if op1 and op2:
                mat = protocols.unitary(op2) @ protocols.unitary(op1)
                gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(mat, atol)
                if gate:
                    ret_ops.append(gate(q))
            else:
                op = op1 or op2
                assert op is not None
                if isinstance(op.gate, ops.PhasedXZGate):
                    ret_ops.append(op)
                else:
                    gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(
                        protocols.unitary(op), atol
                    )
                    if gate:
                        ret_ops.append(gate(q))
        return circuits.Moment(ret_ops)

    return transformer_primitives.merge_moments(
        circuit,
        merge_func,
        deep=context.deep if context else False,
        tags_to_ignore=tuple(tags_to_ignore),
    ).unfreeze(copy=False)


def _values_of_sweep(sweep: Sweep, key: TMeasurementKey):
    p = sympy.Symbol(key) if isinstance(key, str) else key
    return [resolver.value_of(p) for resolver in sweep]


def _parameterize_merged_circuits(
    merged_circuits: List['cirq.Circuit'],
    merge_tags: frozenset[Hashable],
    new_symbols: frozenset[str],
    remaining_symbols: frozenset[str],
    sweep: Sweep,
) -> Sweep:
    """Parameterizes the merged circuits and returns a new sweep."""
    values_by_params: Dict[str, List[float]] = {
        **{s: [] for s in new_symbols},  # New symbols introduced during merging
        **{
            s: _values_of_sweep(sweep, s) for s in remaining_symbols
        },  # Existing symbols in ops that were not merged, e.g., symbols in 2-qubit gates
    }

    for merged_circuit in merged_circuits:
        for op in merged_circuit.all_operations():
            the_merge_tag = merge_tags.intersection(op.tags)
            if not the_merge_tag:
                continue
            sid = the_merge_tag.pop().split("_")[-1]
            x, z, a = 0.0, 0.0, 0.0  # Identity gate's parameters
            if isinstance(op.gate, ops.PhasedXZGate):
                x, z, a = op.gate.x_exponent, op.gate.z_exponent, op.gate.axis_phase_exponent
            elif op.gate is not ops.I:
                raise RuntimeError(
                    f"Expected the merged gate to be a PhasedXZGate or IdentityGate,"
                    f" but got {op.gate}."
                )
            values_by_params[f"x{sid}"].append(x)
            values_by_params[f"z{sid}"].append(z)
            values_by_params[f"a{sid}"].append(a)

    return Zip(*[Points(key=key, points=values) for key, values in values_by_params.items()])


def merge_single_qubit_gates_to_phxz_symbolized(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    merge_tag: Optional[str] = None,
    preserve_tags: bool = False,
    sweep: Sweep,
    atol: float = 1e-8,
) -> Tuple['cirq.Circuit', Sweep]:
    """Merge consecutive single qubit gates as PhasedXZ Gates. Symbolize if any of the consecutive
      gates is symbolized.

    Example:
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> c = cirq.Circuit(\
                    cirq.X(q0),\
                    cirq.CZ(q0,q1)**sympy.Symbol("cz_exp"),\
                    cirq.Y(q0)**sympy.Symbol("y_exp"),\
                    cirq.X(q0))
        >>> print(c)
        0: ───X───@──────────Y^y_exp───X───
                  │
        1: ───────@^cz_exp─────────────────
        >>> new_circuit, new_sweep = cirq.merge_single_qubit_gates_to_phxz_symbolized(\
                c, sweep=cirq.Zip(cirq.Points(key="cz_exp", points=[0, 1]),\
                                  cirq.Points(key="y_exp",  points=[0, 1])))
        >>> print(new_circuit)
        0: ───PhXZ(a=-1,x=1,z=0)───@──────────PhXZ(a=a0,x=x0,z=z0)───
                                   │
        1: ────────────────────────@^cz_exp──────────────────────────
        >>> assert new_sweep[0] == cirq.ParamResolver({'a0': -1, 'x0': 1, 'z0': 0, 'cz_exp': 0})
        >>> assert new_sweep[1] == cirq.ParamResolver({'a0': -0.5, 'x0': 0, 'z0': -1, 'cz_exp': 1})

    Args:
        circuit: Input circuit to transform. It will not be modified.
        sweep: Sweep of the symbols in the input circuit, updated Sweep will be returned
            based on the transformation.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """
    deep = context.deep if context else False

    # Tag symbolized single-qubit op.
    symbolized_single_tag = "_symbolized_single"

    circuit_tagged = transformer_primitives.map_operations(
        circuit,
        lambda op, _: (
            op.with_tags(symbolized_single_tag)
            if protocols.is_parameterized(op) and len(op.qubits) == 1
            else op
        ),
        deep=deep,
    )

    # Step 0, isolate single qubit symbolized symbols and resolve the circuit on them.
    single_qubit_gate_symbols: frozenset[sympy.Symbol] = frozenset(
        set().union(
            *[
                protocols.parameter_symbols(op) if symbolized_single_tag in op.tags else set()
                for op in circuit_tagged.all_operations()
            ]
        )
    )
    # If all single qubit gates are not parameterized, call the nonparamerized version of
    # the transformer.
    if not single_qubit_gate_symbols:
        return (
            merge_single_qubit_gates_to_phxz(
                circuit,
                context=context,
                merge_tag=merge_tag,
                preserve_tags=preserve_tags,
                atol=atol,
            ),
            sweep,
        )
    # Remaining symbols, e.g., 2 qubit gates' symbols. Sweep of those symbols keeps unchanged.
    remaining_symbols: frozenset[sympy.Symbol] = frozenset(
        protocols.parameter_symbols(circuit) - single_qubit_gate_symbols
    )
    sweep_of_single: Sweep = Zip(
        *[Points(key=k, points=_values_of_sweep(sweep, k)) for k in single_qubit_gate_symbols]
    )
    # Get all resolved circuits from all sets of resolvers in the sweep.
    resolved_circuits = [
        protocols.resolve_parameters(circuit_tagged, resolver) for resolver in sweep_of_single
    ]

    # Step 1, merge single qubit gates of resolved circuits and tag merged gates.
    merged_circuits: List['cirq.Circuit'] = []
    merge_tags: set[Hashable] = set()
    phxz_symbols: set[sympy.Symbols] = set()
    _merge_tag = merge_tag or "tmp-phxz"
    for cid, resolved_circuit in enumerate(resolved_circuits):
        merged_circuit = index_tags(
            merge_single_qubit_gates_to_phxz(
                resolved_circuit,
                context=context,
                merge_tag=_merge_tag,
                preserve_tags=True,
                atol=atol,
            ),
            target_tags={_merge_tag},
            skip_op_fn=lambda op: not symbolized_single_tag in set(op.tags),
        )

        merge_tags_at_cid = {
            tag
            for op in merged_circuit.all_operations()
            for tag in op.tags
            if isinstance(tag, str) and re.fullmatch(f"{_merge_tag}_\\d+", tag)
        }

        if cid == 0:
            merge_tags = merge_tags_at_cid
            phxz_symbols = set().union(
                *[
                    set([tag.replace(f"{_merge_tag}_", s) for s in ["x", "z", "a"]])
                    for tag in merge_tags
                ]
            )
        elif merge_tags != merge_tags_at_cid:
            raise RuntimeError(
                "Different resolvers in sweep resulted in different merged structures."
            )
        merged_circuits.append(merged_circuit)

    # Step 2, get the new symbolzied circuit by mapping merged operations.
    def _map_func(op: 'cirq.Operation', _):
        """Maps an op with tag `{merge_tag}_i` to a symbolzied `PhasedXZGate(xi,zi,ai)`."""
        tags: set[Hashable] = set()
        if preserve_tags:
            tags = set(op.tags)
        the_merge_tag = merge_tags.intersection(op.tags)
        if not the_merge_tag:
            return op.with_tags(*tags)
        sid = the_merge_tag.pop().split("_")[-1]
        phxz_params = {
            "x_exponent": sympy.Symbol(f"x{sid}"),
            "z_exponent": sympy.Symbol(f"z{sid}"),
            "axis_phase_exponent": sympy.Symbol(f"a{sid}"),
        }
        if merge_tag:  # tag the merged symbolize gate with the merge_tag.
            tags.add(merge_tag)
        return ops.PhasedXZGate(**phxz_params).on(*op.qubits).with_tags(*tags)

    # Temprory tags used during the process of circuit transformation.
    tmp_tags = merge_tags.union([symbolized_single_tag])
    if not merge_tag:
        tmp_tags.update([_merge_tag])
    new_circuit = align.align_right(
        remove_tags(
            transformer_primitives.map_operations(
                merged_circuits[0].freeze(), _map_func, deep=context.deep if context else False
            ),
            target_tags=tmp_tags,
        )
    )

    # Step 3, get N sets of parameterizations as new_sweep.
    new_sweep = _parameterize_merged_circuits(
        merged_circuits, merge_tags, phxz_symbols, remaining_symbols, sweep
    )

    return new_circuit.unfreeze(copy=False), new_sweep
