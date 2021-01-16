# Copyright 2021 The Cirq Developers
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

"""An optimization pass that factors the circuit into non-entangled subcircuits."""
from typing import Mapping, List, FrozenSet, Dict, Iterable

from cirq import CircuitOperation, Moment, Qid, Operation, MeasurementGate
from cirq.circuits import AbstractCircuit, FrozenCircuit


# todo: rewrite optimizers in functional style


def factor_circuit(circuit: AbstractCircuit) -> FrozenCircuit:
    new_circuit: List[Iterable[List[Iterable[Operation]]]] = []
    subcircuits: Mapping[FrozenSet[Qid], List[Iterable[Operation]]] = {
        frozenset([q]): [] for q in circuit.all_qubits()
    }
    for moment in circuit.moments:
        subcircuits = _get_new_subcircuits(moment.operations, subcircuits)
        _add_moment(new_circuit, moment.operations, subcircuits)
        subcircuits = _remove_measurements(moment.operations, subcircuits)
    return _to_circuit(new_circuit)


def _get_new_subcircuits(
    operations: Iterable[Operation],
    current_subcircuits: Mapping[FrozenSet[Qid], List[Iterable[Operation]]],
) -> Mapping[FrozenSet[Qid], List[Iterable[Operation]]]:
    new_subcircuits: Dict[FrozenSet[Qid], List[Iterable[Operation]]] = dict(current_subcircuits)
    current_qids = {qubit: k for k in current_subcircuits.keys() for qubit in k}
    for op in operations:
        distinct_sets: FrozenSet[FrozenSet[Qid]] = frozenset(
            [current_qids[qubit] for qubit in op.qubits]
        )
        if len(distinct_sets) > 1:
            new_set = frozenset.union(*distinct_sets)
            new_subcircuits[new_set] = []
            for old_set in distinct_sets:
                new_subcircuits.pop(old_set)
    return new_subcircuits


def _remove_measurements(
    operations: Iterable[Operation],
    current_subcircuits: Mapping[FrozenSet[Qid], List[Iterable[Operation]]],
):
    new_subcircuits: Dict[FrozenSet[Qid], List[Iterable[Operation]]] = dict(current_subcircuits)
    current_qids = {qubit: k for k in current_subcircuits.keys() for qubit in k}
    for op in operations:
        if isinstance(op.gate, MeasurementGate):
            for qubit in op.qubits:
                new_subcircuits.pop(current_qids[qubit])
                current_qids[qubit] = current_qids[qubit].difference([qubit])
                new_subcircuits[current_qids[qubit]] = []
                new_subcircuits[frozenset([qubit])] = []
                current_qids[qubit] = frozenset([qubit])
    return new_subcircuits


# circuit is a list of moments
# moment is an iterator of subcircuits
# subcircuit is a list of submoments
# submoment is an iterator of operations
def _add_moment(
    new_circuit: List[Iterable[List[Iterable[Operation]]]],
    operations: Iterable[Operation],
    new_subcircuits: Mapping[FrozenSet[Qid], List[Iterable[Operation]]],
):

    new_ops = [circuit for circuit in new_subcircuits.values() if len(circuit) == 0]
    new_circuit.append(new_ops)

    for qubits, subcircuit in new_subcircuits.items():
        relevant_operations = filter(lambda op: next(iter(op.qubits)) in qubits, operations)
        subcircuit.append(list(relevant_operations))


def _to_subcircuit(submoments: Iterable[Iterable[Operation]]):
    circuit = FrozenCircuit([Moment(submoment) for submoment in submoments])
    return CircuitOperation(circuit)


def _to_moment(subcircuits: Iterable[Iterable[Iterable[Operation]]]):
    return Moment([_to_subcircuit(subcircuit) for subcircuit in subcircuits])


def _to_circuit(moments: Iterable[Iterable[Iterable[Iterable[Operation]]]]) -> FrozenCircuit:
    return FrozenCircuit([_to_moment(moment) for moment in moments])
