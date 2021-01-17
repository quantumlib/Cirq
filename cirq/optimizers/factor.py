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
from typing import Mapping, List, FrozenSet, Iterable

from cirq import CircuitOperation, Moment, Qid, Operation, MeasurementGate
from cirq.circuits import AbstractCircuit, FrozenCircuit

TempSubcircuit = List[List[Operation]]
TempMoment = List[TempSubcircuit]
TempCircuit = List[TempMoment]
EntanglementSet = FrozenSet[Qid]


def factor_circuit(circuit: AbstractCircuit) -> FrozenCircuit:
    new_circuit: TempCircuit = []
    subcircuits: Mapping[EntanglementSet, TempSubcircuit] = {
        frozenset([q]): [] for q in circuit.all_qubits()
    }
    for moment in circuit.moments:
        subcircuits = _with_new_entanglements(subcircuits, moment.operations)
        _add_moment(new_circuit, moment.operations, subcircuits)
        subcircuits = _with_measurements_removed(subcircuits, moment.operations)
    return _to_circuit(new_circuit)


def _with_new_entanglements(
    current_subcircuits: Mapping[EntanglementSet, TempSubcircuit],
    operations: Iterable[Operation],
) -> Mapping[EntanglementSet, TempSubcircuit]:
    new_subcircuits = dict(current_subcircuits)
    entanglement_sets = {qubit: k for k in current_subcircuits.keys() for qubit in k}
    for op in operations:
        distinct_sets = frozenset(entanglement_sets[qubit] for qubit in op.qubits)
        if len(distinct_sets) > 1:
            new_set = frozenset.union(*distinct_sets)
            new_subcircuits[new_set] = []
            for old_set in distinct_sets:
                new_subcircuits.pop(old_set)
    return new_subcircuits


def _with_measurements_removed(
    current_subcircuits: Mapping[EntanglementSet, TempSubcircuit],
    operations: Iterable[Operation],
):
    new_subcircuits = dict(current_subcircuits)
    entanglement_sets = {qubit: k for k in current_subcircuits.keys() for qubit in k}
    for op in operations:
        if isinstance(op.gate, MeasurementGate):
            for qubit in op.qubits:
                old_entanglement_set = entanglement_sets[qubit]
                new_subcircuits.pop(old_entanglement_set)
                new_entanglement_set = old_entanglement_set.difference([qubit])
                for other in new_entanglement_set:
                    entanglement_sets[other] = new_entanglement_set
                new_subcircuits[new_entanglement_set] = []
                singleton_entanglement_set = frozenset([qubit])
                new_subcircuits[singleton_entanglement_set] = []
                entanglement_sets[qubit] = singleton_entanglement_set
    return new_subcircuits


def _add_moment(
    new_circuit: TempCircuit,
    operations: Iterable[Operation],
    new_subcircuits: Mapping[EntanglementSet, TempSubcircuit],
):
    new_ops = [circuit for circuit in new_subcircuits.values() if len(circuit) == 0]
    new_circuit.append(new_ops)
    for qubits, subcircuit in new_subcircuits.items():
        relevant_operations = filter(lambda op: next(iter(op.qubits)) in qubits, operations)
        subcircuit.append(list(relevant_operations))


def _to_subcircuit(submoments: TempSubcircuit) -> CircuitOperation:
    circuit = FrozenCircuit([Moment(submoment) for submoment in submoments])
    return CircuitOperation(circuit)


def _to_moment(subcircuits: TempMoment) -> Moment:
    return Moment([_to_subcircuit(subcircuit) for subcircuit in subcircuits])


def _to_circuit(moments: TempCircuit) -> FrozenCircuit:
    return FrozenCircuit([_to_moment(moment) for moment in moments])
