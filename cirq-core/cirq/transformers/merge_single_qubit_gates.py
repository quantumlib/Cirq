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

import itertools
import warnings
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import sympy

from cirq import circuits, ops, protocols
from cirq.study.sweeps import Points, Sweep, Zip
from cirq.transformers import merge_k_qubit_gates, transformer_api, transformer_primitives, align
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
    atol: float = 1e-8,
) -> 'cirq.Circuit':
    """Replaces runs of single qubit rotations with a single optional `cirq.PhasedXZGate`.

    Specifically, any run of non-parameterized single-qubit unitaries will be replaced by an
    optional PhasedXZ.

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
        gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(u, atol)
        return gate(op.qubits[0]) if gate else []

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


def _values_of_sweep(sweep: Sweep, key: str | sympy.Symbol):
    p = sympy.Symbol(key) if isinstance(key, str) else key
    return [resolver.value_of(p) for resolver in sweep]


@transformer_api.transformer
def merge_single_qubit_gates_to_phxz_symbolized(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    sweep: Sweep,
    atol: float = 1e-8,
) -> Tuple['cirq.Circuit', Sweep]:
    """Merge consecutive single qubit gates as PhasedXZ Gates. Symbolize if any of the consecutive gates is symbolized.

    Example:
        # pylint: disable=line-too-long
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> c = cirq.Circuit(cirq.X(q0),cirq.CZ(q0,q1)**sympy.Symbol("cz_exp"),cirq.Y(q0)**sympy.Symbol("y_exp"),cirq.X(q0))
        >>> print(c)
        0: ───X───@──────────Y^y_exp───X───
                  │
        1: ───────@^cz_exp─────────────────
        >>> new_circuit, new_sweep = cirq.merge_single_qubit_gates_to_phxz_symbolized(\
                c, sweep=cirq.Points(key="cz_exp", points=[0, 1]) * cirq.Points(key="y_exp", points=[0, 1])\
            )
        >>> print(new_circuit)
        0: ───PhXZ(a=-1,x=1,z=0)───@──────────PhXZ(a=a0,x=x0,z=z0)───
                                   │
        1: ────────────────────────@^cz_exp──────────────────────────
        >>> print(new_sweep)
        cirq.Points('z0', [0, -1.0, 0, -1.0]) + cirq.Points('x0', [1, 0.0, 1, 0.0]) + cirq.Points('a0', [-1.0, -0.5, -1.0, -0.5]) + cirq.Points('cz_exp', [0, 0, 1, 1])
        # pylint: disable=line-too-long

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

    if not protocols.is_parameterized(circuit):
        warnings.warn(
            "Expect parameterized circuits. "
            "Please use cirq.merge_single_qubit_gates_to_phxz instead.",
            UserWarning,
        )
        return merge_single_qubit_gates_to_phxz(circuit, context=context, atol=atol)

    # Tag symbolized single qubit op.
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

    # Symbols of the single qubit symbolized ops.
    single_qubit_gate_symbols: set[sympy.Symbol] = set().union(
        *[
            protocols.parameter_symbols(op) if symbolized_single_tag in op.tags else set()
            for op in circuit_tagged.all_operations()
        ]
    )
    # Remaing symbols, e.g., 2 qubit gates' symbols. Sweep of those symbols keeps unchanged.
    remaining_symbols = protocols.parameter_symbols(circuit) - single_qubit_gate_symbols

    sweep_of_single: Sweep = Zip(
        *[Points(key=k, points=_values_of_sweep(sweep, k)) for k in single_qubit_gate_symbols]
    )

    # Get all resolved circuits from all sets of resolvers in sweep.
    resolved_circuits = [
        protocols.resolve_parameters(circuit_tagged, resolver) for resolver in sweep_of_single
    ]

    # Store the number of merges for all set of resolvers,
    # it should be the same for all resolved circuits.
    merge_counts: list[int] = []
    merged_circuits = []
    phxz_tag_prefix = "_phxz"
    tag_iter: itertools.count

    def rewriter(circuit_op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        nonlocal tag_iter
        tag: Optional[str] = None
        u = protocols.unitary(circuit_op)
        if protocols.num_qubits(circuit_op) == 0:
            return ops.GlobalPhaseGate(u[0, 0]).on()
        for op in circuit_op.circuit.all_operations():
            if symbolized_single_tag in op.tags:
                # Record parameterizations info via tags.
                tag = f"{phxz_tag_prefix}_{next(tag_iter)}"
                break
        gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(u, atol) or ops.I
        op = gate.on(circuit_op.qubits[0])
        if not gate:
            return []
        return op.with_tags(tag) if tag else op

    for resolved_circuit in resolved_circuits:
        tag_iter = itertools.count(start=0, step=1)
        merged_circuits.append(
            merge_k_qubit_gates.merge_k_qubit_unitaries(
                resolved_circuit, k=1, context=context, rewriter=rewriter
            )
        )
        merge_counts.append(next(tag_iter))

    if not all(count == merge_counts[0] for count in merge_counts):
        raise RuntimeError("Different resolvers in sweep result different merged strcuture.")

    # Get the output circuit from the first resolved circuits.
    merge_tags: set[str] = {f"{phxz_tag_prefix}_{i}" for i in range(merge_counts[0])}
    new_symbols: set[str] = set().union(
        *[{f"x{i}", f"z{i}", f"a{i}"} for i in range(merge_counts[0])]
    )

    def _map_func(op: 'cirq.Operation', _):
        """Maps op with tag `_phxz_i` to a symbolzied `PhasedXZGate(xi,zi,ai)`"""
        the_merge_tag = merge_tags.intersection(op.tags)
        if len(the_merge_tag) == 0:
            return op
        if len(the_merge_tag) > 1:
            raise RuntimeError("Multiple merge tags found.")
        sid = the_merge_tag.pop().split("_")[-1]
        phxz_params = {
            "x_exponent": sympy.Symbol(f"x{sid}"),
            "z_exponent": sympy.Symbol(f"z{sid}"),
            "axis_phase_exponent": sympy.Symbol(f"a{sid}"),
        }
        return ops.PhasedXZGate(**phxz_params).on(*op.qubits)

    output_circuit: 'cirq.Circuit' = align.align_right(
        transformer_primitives.map_operations(merged_circuits[0].freeze(), _map_func, deep=deep)
    )

    values_by_params: Dict[str, List[float]] = {
        **{s: [] for s in new_symbols},  # New symbols introduced in merging
        **{
            s: _values_of_sweep(sweep, s) for s in remaining_symbols
        },  # Existing symbols in ops that are not merged, e.g., symbols in 2 qubit gates.
    }

    # Get parameterization for the merged phxz gates.
    for merged_circuit in merged_circuits:
        for op in merged_circuit.all_operations():
            the_merge_tag = merge_tags.intersection(op.tags)
            if len(the_merge_tag) == 0:
                continue
            if len(the_merge_tag) > 1:
                raise RuntimeError("Multiple merge tags found.")
            sid = the_merge_tag.pop().split("_")[-1]
            x, z, a = 0.0, 0.0, 0.0  # Identity gate's parameters.
            if isinstance(op.gate, ops.PhasedXZGate):
                x, z, a = op.gate.x_exponent, op.gate.z_exponent, op.gate.axis_phase_exponent
            elif op.gate is not ops.I:
                raise RuntimeError(
                    f"Expect the merged gate to be a PhasedXZGate or IdentityGate. But got {op.gate}."
                )
            values_by_params[f"x{sid}"].append(x)
            values_by_params[f"z{sid}"].append(z)
            values_by_params[f"a{sid}"].append(a)

    new_sweep: Sweep = Zip(
        *[Points(key=key, points=values) for key, values in values_by_params.items()]
    )

    return output_circuit.unfreeze(copy=False), new_sweep
