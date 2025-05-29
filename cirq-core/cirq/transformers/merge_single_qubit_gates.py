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

from typing import Callable, cast, Hashable, List, Tuple, TYPE_CHECKING

import sympy

from cirq import circuits, ops, protocols
from cirq.study.resolver import ParamResolver
from cirq.study.sweeps import dict_to_zip_sweep, ListSweep, ProductOrZipSweepLike, Sweep, Zip
from cirq.transformers import (
    align,
    merge_k_qubit_gates,
    transformer_api,
    transformer_primitives,
    symbolize,
    tag_transformers,
)
from cirq.transformers.analytical_decompositions import single_qubit_decompositions

if TYPE_CHECKING:
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
    merge_tags_fn: Callable[[cirq.CircuitOperation], List[Hashable]] | None = None,
    atol: float = 1e-8,
) -> 'cirq.Circuit':
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

    def rewriter(circuit_op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
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
            protocols.num_qubits(op) == 1
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
        return circuits.Moment(ret_ops)

    return transformer_primitives.merge_moments(
        circuit,
        merge_func,
        deep=context.deep if context else False,
        tags_to_ignore=tuple(tags_to_ignore),
    ).unfreeze(copy=False)


def _all_tags_startswith(circuit: cirq.AbstractCircuit, startswith: str):
    tag_set: set[Hashable] = set()
    for op in circuit.all_operations():
        for tag in op.tags:
            if str(tag).startswith(startswith):
                tag_set.add(tag)
    return tag_set


def _sweep_on_symbols(sweep: Sweep, symbols: set[sympy.Symbol]) -> Sweep:
    new_resolvers: List[cirq.ParamResolver] = []
    for resolver in sweep:
        param_dict: 'cirq.ParamMappingType' = {s: resolver.value_of(s) for s in symbols}
        new_resolvers.append(ParamResolver(param_dict))
    return ListSweep(new_resolvers)


def _parameterize_phxz_in_circuits(
    circuit_list: List['cirq.Circuit'],
    merge_tag_prefix: str,
    phxz_symbols: set[sympy.Symbol],
    remaining_symbols: set[sympy.Symbol],
    sweep: Sweep,
) -> Sweep:
    """Parameterizes the circuits and returns a new sweep."""
    values_by_params: dict[str, List[float]] = {**{str(s): [] for s in phxz_symbols}}

    for circuit in circuit_list:
        for op in circuit.all_operations():
            the_merge_tag: str | None = None
            for tag in op.tags:
                if str(tag).startswith(merge_tag_prefix):
                    the_merge_tag = str(tag)
            if not the_merge_tag:
                continue
            sid = the_merge_tag.rsplit("_", maxsplit=-1)[-1]
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

    return Zip(
        dict_to_zip_sweep(cast(ProductOrZipSweepLike, values_by_params)),
        _sweep_on_symbols(sweep, remaining_symbols),
    )


def merge_single_qubit_gates_to_phxz_symbolized(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    sweep: Sweep,
    atol: float = 1e-8,
) -> Tuple[cirq.Circuit, Sweep]:
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
        sweep: Sweep of the symbols in the input circuit, updated Sweep will be returned
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
    # If all single qubit gates are not parameterized, call the nonparamerized version of
    # the transformer.
    if not single_qubit_gate_symbols:
        return (merge_single_qubit_gates_to_phxz(circuit, context=context, atol=atol), sweep)
    sweep_of_single: Sweep = _sweep_on_symbols(sweep, single_qubit_gate_symbols)
    # Get all resolved circuits from all sets of resolvers in sweep_of_single.
    resolved_circuits = [
        protocols.resolve_parameters(circuit_tagged, resolver) for resolver in sweep_of_single
    ]

    # Step 1, merge single qubit gates per resolved circuit, preserving
    #  the symbolized_single_tag with indexes.
    merged_circuits: List['cirq.Circuit'] = []
    for resolved_circuit in resolved_circuits:
        merged_circuit = tag_transformers.index_tags(
            merge_single_qubit_gates_to_phxz(
                resolved_circuit,
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
            ),
            context=transformer_api.TransformerContext(deep=deep),
            target_tags={symbolized_single_tag},
        )
        merged_circuits.append(merged_circuit)

    if not all(
        _all_tags_startswith(merged_circuits[0], startswith=symbolized_single_tag)
        == _all_tags_startswith(merged_circuit, startswith=symbolized_single_tag)
        for merged_circuit in merged_circuits
    ):
        raise RuntimeError("Different resolvers in sweep resulted in different merged structures.")

    # Step 2, get the new symbolized circuit by symbolization on indexed symbolized_single_tag.
    new_circuit = align.align_right(
        tag_transformers.remove_tags(
            symbolize.symbolize_single_qubit_gates_by_indexed_tags(
                merged_circuits[0],
                symbolize_tag=symbolize.SymbolizeTag(prefix=symbolized_single_tag),
            ),
            remove_if=lambda tag: str(tag).startswith(symbolized_single_tag),
        )
    )

    # Step 3, get N sets of parameterizations as new_sweep.
    phxz_symbols: set[sympy.Symbol] = set().union(
        *[
            set(
                [sympy.Symbol(tag.replace(f"{symbolized_single_tag}_", s)) for s in ["x", "z", "a"]]
            )
            for tag in _all_tags_startswith(merged_circuits[0], startswith=symbolized_single_tag)
        ]
    )
    # Remaining symbols, e.g., 2 qubit gates' symbols. Sweep of those symbols keeps unchanged.
    remaining_symbols: set[sympy.Symbol] = set(
        protocols.parameter_symbols(circuit) - single_qubit_gate_symbols
    )
    new_sweep = _parameterize_phxz_in_circuits(
        merged_circuits, symbolized_single_tag, phxz_symbols, remaining_symbols, sweep
    )

    return new_circuit, new_sweep
