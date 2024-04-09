# Copyright 2023 The Cirq Developers
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

from typing import Dict, Optional, Set, Tuple, TYPE_CHECKING

from cirq import circuits, ops

if TYPE_CHECKING:
    import cirq


def _get_qubit_mapping_first_and_last_moment(
    circuit: 'cirq.AbstractCircuit',
) -> Dict['cirq.Qid', Tuple[int, int]]:
    """Computes `(first_moment_idx, last_moment_idx)` tuple for each qubit in the input circuit.

    Args:
        circuit: An input cirq circuit to analyze.

    Returns:
        A dict mapping each qubit `q` in the input circuit to a tuple of integers
        `(first_moment_idx, last_moment_idx)` where
         - first_moment_idx: Index of leftmost moment containing an operation that acts on `q`.
         - last_moment_idx: Index of rightmost moment containing an operation that acts on `q`.
    """
    ret = {q: (len(circuit), 0) for q in circuit.all_qubits()}
    for i, moment in enumerate(circuit):
        for q in moment.qubits:
            ret[q] = (min(ret[q][0], i), max(ret[q][1], i))
    return ret


def _is_temp(q: 'cirq.Qid') -> bool:
    return isinstance(q, (ops.CleanQubit, ops.BorrowableQubit))


def map_clean_and_borrowable_qubits(
    circuit: 'cirq.AbstractCircuit', *, qm: Optional['cirq.QubitManager'] = None
) -> 'cirq.Circuit':
    """Uses `qm: QubitManager` to map all `CleanQubit`/`BorrowableQubit`s to system qubits.

    `CleanQubit` and `BorrowableQubit` are internal qubit types that are used as placeholder qubits
    to record a clean / dirty ancilla allocation request.

    This transformer uses the `QubitManager` provided in the input to:
     - Allocate clean ancilla qubits by delegating to `qm.qalloc` for all `CleanQubit`s.
     - Allocate dirty qubits for all `BorrowableQubit` types via the following two steps:
         1. First analyse the input circuit and check if there are any suitable system qubits
            that can be borrowed, i.e. ones which do not have any overlapping operations
            between circuit[start_index : end_index] where `(start_index, end_index)` is the
            lifespan of temporary borrowable qubit under consideration. If yes, borrow the system
            qubits to replace the temporary `BorrowableQubit`.
         2. If no system qubits can be borrowed, delegate the request to `qm.qborrow`.

    Notes:
        1. The borrow protocol can be made more efficient by also considering the newly
    allocated clean ancilla qubits in step-1 before delegating to `qm.borrow`, but this
    optimization is left as a future improvement.
        2. As of now, the transformer does not preserve moment structure and defaults to
    inserting all mapped operations in a resulting circuit using EARLIEST strategy. The reason
    is that preserving moment structure forces additional constraints on the qubit allocation
    strategy (i.e. if two operations `op1` and `op2` are in the same moment, then we cannot
    reuse ancilla across `op1` and `op2`). We leave it up to the user to force such constraints
    using the qubit manager instead of making it part of the transformer.
        3. However, for borrowable system qubits managed by the transformer, we do not reuse qubits
    within the same moment.
        4. Since this is not implemented using the cirq transformers infrastructure, we currently
    do not support recursive mapping within sub-circuits and that is left as a future TODO.

    Args:
        circuit: Input `cirq.Circuit` containing temporarily allocated
            `CleanQubit`/`BorrowableQubit`s.
        qm: An instance of `cirq.QubitManager` specifying the strategy to use for allocating /
            / deallocating new ancilla qubits to replace the temporary qubits.

    Returns:
        An updated `cirq.Circuit` with all `CleanQubit`/`BorrowableQubit` mapped to either existing
        system qubits or new ancilla qubits allocated using the `qm` qubit manager.
    """
    if qm is None:
        qm = ops.GreedyQubitManager(prefix="ancilla")

    allocated_qubits = {q for q in circuit.all_qubits() if _is_temp(q)}
    qubits_lifespan = _get_qubit_mapping_first_and_last_moment(circuit)
    all_qubits = frozenset(circuit.all_qubits() - allocated_qubits)
    trivial_map = {q: q for q in all_qubits}
    # `allocated_map` maintains the mapping of all temporary qubits seen so far, mapping each of
    # them to either a newly allocated managed ancilla or an existing borrowed system qubit.
    allocated_map: Dict['cirq.Qid', 'cirq.Qid'] = {}
    to_free: Set['cirq.Qid'] = set()
    last_op_idx = -1

    def map_func(op: 'cirq.Operation', idx: int) -> 'cirq.OP_TREE':
        nonlocal last_op_idx, to_free
        assert isinstance(qm, ops.QubitManager)

        for q in sorted(to_free):
            is_managed_qubit = allocated_map[q] not in all_qubits
            if idx > last_op_idx or is_managed_qubit:
                # is_managed_qubit: if `q` is mapped to a newly allocated qubit managed by the qubit
                #   manager, we can free it immediately after the previous operation ends. This
                #   assumes that a managed qubit is not considered by the transformer as part of
                #   borrowing qubits (first point of the notes above).
                # idx > last_op_idx: if `q` is mapped to a system qubit, which is not managed by the
                #   qubit manager, we free it only at the end of the moment.
                if is_managed_qubit:
                    qm.qfree([allocated_map[q]])
                allocated_map.pop(q)
                to_free.remove(q)

        last_op_idx = idx

        # To check borrowable qubits, we manually manage only the original system qubits
        # that are not managed by the qubit manager. If any of the system qubits cannot be
        # borrowed, we defer to the qubit manager to allocate a new clean qubit for us.
        # This is a heuristic and can be improved by also checking if any allocated but not
        # yet freed managed qubit can be borrowed for the shorter scope, but we ignore the
        # optimization for the sake of simplicity here.
        borrowable_qubits = set(all_qubits) - set(allocated_map.values())

        op_temp_qubits = (q for q in op.qubits if _is_temp(q))
        for q in op_temp_qubits:
            # Get the lifespan of this temporarily allocated ancilla qubit `q`.
            st, en = qubits_lifespan[q]
            assert st <= idx <= en
            if en == idx:
                # Mark that this temporarily allocated qubit can be freed after this moment ends.
                to_free.add(q)
            if q in allocated_map or st < idx:
                # The qubit already has a mapping iff we have seen it before.
                assert st < idx and q in allocated_map
                # This line is actually covered by
                # `test_map_clean_and_borrowable_qubits_deallocates_only_once` but pytest-cov seems
                # to not recognize it and hence the pragma: no cover.
                continue  # pragma: no cover

            # This is the first time we are seeing this temporary qubit and need to find a mapping.
            if isinstance(q, ops.CleanQubit):
                # Allocate a new clean qubit if `q` using the qubit manager.
                allocated_map[q] = qm.qalloc(1)[0]
            elif isinstance(q, ops.BorrowableQubit):
                # For each of the system qubits that can be borrowed, check whether they have a
                # conflicting operation in the range [st, en]; which is the scope for which the
                # borrower needs the borrowed qubit for.
                start_frontier = {q: st for q in borrowable_qubits}
                end_frontier = {q: en + 1 for q in borrowable_qubits}
                ops_in_between = circuit.findall_operations_between(start_frontier, end_frontier)
                # Filter the set of borrowable qubits which do not have any conflicting operations.
                filtered_borrowable_qubits = borrowable_qubits - set(
                    q for _, op in ops_in_between for q in op.qubits
                )
                if filtered_borrowable_qubits:
                    # Allocate a borrowable qubit and remove it from the pool of available qubits.
                    allocated_map[q] = min(filtered_borrowable_qubits)
                    borrowable_qubits.remove(allocated_map[q])
                else:
                    # Use the qubit manager to get a new borrowable qubit, since we couldn't find
                    # one from the original system qubits.
                    allocated_map[q] = qm.qborrow(1)[0]
            else:
                assert False, f"Unknown temporary qubit type {q}"

        # Return the transformed operation / decomposed op-tree.
        return op.transform_qubits({**allocated_map, **trivial_map})

    return circuits.Circuit(map_func(op, idx) for idx, m in enumerate(circuit) for op in m)
