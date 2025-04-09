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
from typing import Dict, Hashable, List, Optional, Tuple, TYPE_CHECKING

import sympy

from cirq import circuits, ops, protocols
from cirq.study.sweeps import Points, Sweep, Zip
from cirq.transformers import align, merge_k_qubit_gates, transformer_api, transformer_primitives
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


# ----------------------------------------------------------------------
# Impl merge_single_qubit_gates_to_phxz_symbolized: Start
# ----------------------------------------------------------------------


def _values_of_sweep(sweep: Sweep, key: str | sympy.Symbol):
    p = sympy.Symbol(key) if isinstance(key, str) else key
    return [resolver.value_of(p) for resolver in sweep]


def _merge_single_qubit_gates_to_circuit_op_symbolized(
    resolved_circuits: List['cirq.AbstractCircuit'],
    symbolized_single_tag: str,
    context: Optional['cirq.TransformerContext'],
    atol: float,
) -> Tuple[List['cirq.Circuit'], frozenset[str], frozenset[str]]:
    """Helper function to merge single qubit ops of resolved circuits to ops of CircuitOperation
      type using merge_k_qubit_unitaries.

    Args:
        resolved_circuits: A list of circuits where symbols have been replaced with concrete values.
        symbolized_single_tag: The tag applied to single-qubit operations that originally
          contained symbols before parameterizations.

    Returns:
        Tuple of merge counts, merged circuits, and merge tags.
    """
    merge_counts: list[int] = []  # number of merges per resolved_circuit
    merged_circuits: list['cirq.Circuit'] = []
    tag_iter: itertools.count
    phxz_tag_prefix = "_phxz"

    def rewriter(circuit_op: 'cirq.CircuitOperation') -> 'cirq.OP_TREE':
        nonlocal tag_iter
        tag: Optional[str] = None

        u = protocols.unitary(circuit_op)
        if protocols.num_qubits(circuit_op) == 0:
            return ops.GlobalPhaseGate(u[0, 0]).on()
        # If any of the op in the merged circuit_op is a symbolized single qubit gate,
        # tag the merged phxz gate with next tag id, for further parameterization references.
        for op in circuit_op.circuit.all_operations():
            if symbolized_single_tag in op.tags:
                tag = f"{phxz_tag_prefix}_{next(tag_iter)}"
                break
        gate = single_qubit_decompositions.single_qubit_matrix_to_phxz(u, atol) or ops.I
        op = gate.on(circuit_op.qubits[0])
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
        raise RuntimeError("Different resolvers in sweep resulted in different merged structures.")

    merge_tags: frozenset[str] = frozenset(
        {f"{phxz_tag_prefix}_{i}" for i in range(merge_counts[0])}
    )
    new_symbols: frozenset[str] = frozenset(
        set().union(*[{f"x{i}", f"z{i}", f"a{i}"} for i in range(merge_counts[0])])
    )

    return merged_circuits, merge_tags, new_symbols


def _get_merge_tag_id(merge_tags: frozenset[str], op_tags: Tuple[Hashable, ...]) -> Optional[str]:
    """Extract the id `i` from the merge tag `_phxz_i` if it exists."""
    the_merge_tag: set[str] = set(merge_tags.intersection(op_tags))
    if len(the_merge_tag) == 0:
        return None
    if len(the_merge_tag) > 1:
        raise RuntimeError("Multiple merge tags found.")
    return the_merge_tag.pop().split("_")[-1]


def _map_merged_ops_to_symbolized_phxz(
    circuit: 'cirq.Circuit', merge_tags: frozenset[str], deep: bool
) -> 'cirq.Circuit':
    """Maps merged operations (tagged with merge_tags) in the circuit to symbolized PhasedXZGates.

    Args:
        circuit: Circuit with merge tags to be mapped.
        merge_tags: The set of tags used to identify the merged PhasedXZ gates that need to be
            symbolized.
        deep: Whether to perform the mapping recursively within CircuitOperations.

    Returns:
        A new circuit where tagged PhasedXZ gates are replaced by symbolized versions.
    """

    # Map merged ops to `PhasedXZGate(xi,zi,ai)` based on the tag "_phxz_i".
    def _map_func(op: 'cirq.Operation', _):
        """Maps an op with tag `_phxz_i` to a symbolzied `PhasedXZGate(xi,zi,ai)`"""
        sid = _get_merge_tag_id(merge_tags, op.tags)
        if sid is None:
            return op
        phxz_params = {
            "x_exponent": sympy.Symbol(f"x{sid}"),
            "z_exponent": sympy.Symbol(f"z{sid}"),
            "axis_phase_exponent": sympy.Symbol(f"a{sid}"),
        }
        return ops.PhasedXZGate(**phxz_params).on(*op.qubits)

    return align.align_right(
        transformer_primitives.map_operations(circuit.freeze(), _map_func, deep=deep)
    )


def _parameterize_merged_circuits(
    merged_circuits: List['cirq.Circuit'],
    merge_tags: frozenset[str],
    new_symbols: frozenset[str],
    remaining_symbols: frozenset[str],
    sweep: Sweep,
) -> Sweep:
    """Parameterizes the merged circuits and returns a new sweep."""
    values_by_params: Dict[str, List[float]] = {
        **{s: [] for s in new_symbols},  # New symbols introduced during merging
        **{
            s: _values_of_sweep(sweep, s) for s in remaining_symbols
        },  # Existing symbols in ops that were not merged, e.g., symbols in 2-qubit gates.
    }

    for merged_circuit in merged_circuits:
        for op in merged_circuit.all_operations():
            sid = _get_merge_tag_id(merge_tags, op.tags)
            if sid is None:
                continue
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
        return merge_single_qubit_gates_to_phxz(circuit, context=context, atol=atol), sweep
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

    # Step 1, merge single qubit gates of resolved circuits using merge_k_qubit_unitaries.
    merged_circuits, merge_tags, new_symbols = _merge_single_qubit_gates_to_circuit_op_symbolized(
        resolved_circuits, symbolized_single_tag, context, atol
    )

    # Step 2, get the new symbolzied circuit as new_sweep by mapping merged operations.
    new_circuit = _map_merged_ops_to_symbolized_phxz(merged_circuits[0], merge_tags, deep)

    # Step 3, get N sets of parameterizations as new_sweep.
    new_sweep = _parameterize_merged_circuits(
        merged_circuits, merge_tags, new_symbols, remaining_symbols, sweep
    )

    return new_circuit.unfreeze(copy=False), new_sweep


# ----------------------------------------------------------------------
# Impl merge_single_qubit_gates_to_phxz_symbolized: End
# ----------------------------------------------------------------------
