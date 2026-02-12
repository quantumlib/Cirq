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

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import cast, TYPE_CHECKING

from cirq import circuits, ops, protocols
from cirq.study.resolver import ParamResolver
from cirq.study.sweeps import dict_to_zip_sweep, ListSweep, ProductOrZipSweepLike, Sweep, Zip
from cirq.transformers import (
    align,
    merge_k_qubit_gates,
    symbolize,
    tag_transformers,
    transformer_api,
    transformer_primitives,
)
from cirq.transformers.analytical_decompositions import single_qubit_decompositions

if TYPE_CHECKING:
    import sympy

    import cirq


@transformer_api.transformer
def merge_single_qubit_gates_to_phased_x_and_z(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    atol: float = 1e-8,
) -> cirq.Circuit:
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

    def rewriter(op: cirq.CircuitOperation) -> cirq.OP_TREE:
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
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    merge_tags_fn: Callable[[cirq.CircuitOperation], list[Hashable]] | None = None,
    atol: float = 1e-8,
) -> cirq.Circuit:
    """Replaces runs of single qubit rotations with a single optional `cirq.PhasedXZGate`.

    Specifically, any run of non-parameterized single-qubit unitaries will be replaced by an
    optional PhasedXZ.

    Args:
        circuit: Input circuit to transform. It will not be modified.
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        merge_tags_fn: A callable returns the tags to be added to the merged operation.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """

    def rewriter(circuit_op: cirq.CircuitOperation) -> cirq.OP_TREE:
        u = protocols.unitary(circuit_op)
        if protocols.num_qubits(circuit_op) == 0:
            return ops.GlobalPhaseGate(u[0, 0]).on()
        gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(u, atol) or ops.I
        phxz_op = gate.on(circuit_op.qubits[0])
        return phxz_op.with_tags(*merge_tags_fn(circuit_op)) if merge_tags_fn else phxz_op

    return merge_k_qubit_gates.merge_k_qubit_unitaries(
        circuit, k=1, context=context, rewriter=rewriter
    )


@transformer_api.transformer
def merge_single_qubit_moments_to_phxz(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    atol: float = 1e-8,
) -> cirq.Circuit:
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

    def can_merge_moment(m: cirq.Moment):
        return all(
            protocols.num_qubits(op) <= 1
            and protocols.has_unitary(op)
            and tags_to_ignore.isdisjoint(op.tags)
            for op in m
        )

    def merge_func(m1: cirq.Moment, m2: cirq.Moment) -> cirq.Moment | None:
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
        # Transfer global phase
        for op in m1.operations + m2.operations:
            if protocols.num_qubits(op) == 0:
                ret_ops.append(op)
        return circuits.Moment(ret_ops)

    return transformer_primitives.merge_moments(
        circuit,
        merge_func,
        deep=context.deep if context else False,
        tags_to_ignore=tuple(tags_to_ignore),
    ).unfreeze(copy=False)


def _sweep_on_symbols(sweep: Sweep, symbols: set[sympy.Symbol]) -> Sweep:
    new_resolvers: list[cirq.ParamResolver] = []
    for resolver in sweep:
        param_dict: cirq.ParamMappingType = {s: resolver.value_of(s) for s in symbols}
        new_resolvers.append(ParamResolver(param_dict))
    return ListSweep(new_resolvers)


def _calc_phxz_sweeps(
    symbolized_circuit: cirq.Circuit, resolved_circuits: list[cirq.Circuit]
) -> Sweep:
    """Returns the phxz sweep of the symbolized_circuit on resolved_circuits.

    Raises:
        ValueError: Structural mismatch: A `resolved_circuit` contains an unexpected gate type.
            Expected a `PhasedXZGate` or `IdentityGate` at a position corresponding to a
            symbolic `PhasedXZGate` in the `symbolized_circuit`.
    """

    def _extract_axz(op: ops.Operation | None) -> tuple[float, float, float]:
        if not op or not op.gate or not isinstance(op.gate, ops.IdentityGate | ops.PhasedXZGate):
            raise ValueError(f"Expect a PhasedXZGate or IdentityGate on op {op}.")
        if isinstance(op.gate, ops.IdentityGate):
            return 0.0, 0.0, 0.0  # Identity gate's a, x, z in PhasedXZ
        return op.gate.axis_phase_exponent, op.gate.x_exponent, op.gate.z_exponent

    values_by_params: dict[sympy.Symbol, tuple[float, ...]] = {}
    for mid, moment in enumerate(symbolized_circuit):
        for op in moment.operations:
            if op.gate and isinstance(op.gate, ops.PhasedXZGate) and protocols.is_parameterized(op):
                sa, sx, sz = op.gate.axis_phase_exponent, op.gate.x_exponent, op.gate.z_exponent
                values_by_params[sa], values_by_params[sx], values_by_params[sz] = zip(
                    *[_extract_axz(c[mid].operation_at(op.qubits[0])) for c in resolved_circuits]
                )

    return dict_to_zip_sweep(cast(ProductOrZipSweepLike, values_by_params))


def merge_single_qubit_gates_to_phxz_symbolized(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    sweep: Sweep,
    atol: float = 1e-8,
) -> tuple[cirq.Circuit, Sweep]:
    """Merges consecutive single qubit gates as PhasedXZ Gates. Symbolizes if any of
      the consecutive gates is symbolized.

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
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        sweep: Sweep of the symbols in the input circuit. An updated Sweep will be returned
            based on the transformation.
        atol: Absolute tolerance to angle error. Larger values allow more negligible gates to be
            dropped, smaller values increase accuracy.

    Returns:
        Copy of the transformed input circuit.
    """
    deep = context.deep if context else False

    # Tag symbolized single-qubit op.
    symbolized_single_tag = "_tmp_symbolize_tag"

    circuit_tagged = transformer_primitives.map_operations(
        circuit,
        lambda op, _: (
            op.with_tags(symbolized_single_tag)
            if protocols.is_parameterized(op) and len(op.qubits) == 1
            else op
        ),
        deep=deep,
    )

    # Step 0, isolate single qubit symbols and resolve the circuit on them.
    single_qubit_gate_symbols: set[sympy.Symbol] = set().union(
        *[
            protocols.parameter_symbols(op) if symbolized_single_tag in op.tags else set()
            for op in circuit_tagged.all_operations()
        ]
    )
    # Remaining symbols, e.g., 2 qubit gates' symbols. Sweep of those symbols keeps unchanged.
    remaining_symbols: set[sympy.Symbol] = set(
        protocols.parameter_symbols(circuit) - single_qubit_gate_symbols
    )
    # If all single qubit gates are not parameterized, call the non-parameterized version of
    # the transformer.
    if not single_qubit_gate_symbols:
        return (merge_single_qubit_gates_to_phxz(circuit, context=context, atol=atol), sweep)
    sweep_of_single: Sweep = _sweep_on_symbols(sweep, single_qubit_gate_symbols)
    # Get all resolved circuits from all sets of resolvers in sweep_of_single.
    resolved_circuits = [
        protocols.resolve_parameters(circuit_tagged, resolver) for resolver in sweep_of_single
    ]

    # Step 1, merge single qubit gates per resolved circuit, preserving
    #  the symbolized_single_tag to indicate the operator is a merged one.
    merged_circuits: list[cirq.Circuit] = [
        merge_single_qubit_gates_to_phxz(
            c,
            context=context,
            merge_tags_fn=lambda circuit_op: (
                [symbolized_single_tag]
                if any(
                    symbolized_single_tag in set(op.tags)
                    for op in circuit_op.circuit.all_operations()
                )
                else []
            ),
            atol=atol,
        )
        for c in resolved_circuits
    ]

    # Step 2, get the new symbolized circuit by symbolizing on indexed symbolized_single_tag.
    new_circuit = tag_transformers.remove_tags(  # remove the temp tags used to track merges
        symbolize.symbolize_single_qubit_gates_by_indexed_tags(
            tag_transformers.index_tags(  # index all 1-qubit-ops merged from ops with symbols
                merged_circuits[0],
                context=transformer_api.TransformerContext(deep=deep),
                target_tags={symbolized_single_tag},
            ),
            symbolize_tag=symbolize.SymbolizeTag(prefix=symbolized_single_tag),
        ),
        remove_if=lambda tag: str(tag).startswith(symbolized_single_tag),
    )

    # Step 3, get N sets of parameterizations as new_sweep.
    if remaining_symbols:
        new_sweep: Sweep = Zip(
            _calc_phxz_sweeps(new_circuit, merged_circuits),  # phxz sweeps
            _sweep_on_symbols(sweep, remaining_symbols),  # remaining sweeps
        )
    else:
        new_sweep = _calc_phxz_sweeps(new_circuit, merged_circuits)

    return align.align_right(new_circuit), new_sweep
