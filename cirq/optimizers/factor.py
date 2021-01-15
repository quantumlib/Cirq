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
from typing import Mapping, List, FrozenSet, Iterator

from cirq import CircuitOperation, Moment, Qid, Operation
from cirq.circuits import AbstractCircuit, FrozenCircuit


# todo: rewrite optimizers in functional style


def factor_circuit(circuit: AbstractCircuit):
    new_circuit: List[Iterator[List[Iterator[Operation]]]] = []
    subcircuits = {frozenset([q]): [] for q in circuit.all_qubits()}
    print()
    for moment in circuit.moments:
        subcircuits = _get_new_subcircuits(moment.operations, subcircuits)
        print(f"new_subcircuits.count:{len(subcircuits)}")
        _add_moment(new_circuit, moment, subcircuits)
    return _to_circuit(new_circuit)


def _get_new_subcircuits(
    operations: Iterator[Operation],
    current_subcircuits: Mapping[FrozenSet[Qid], List[Iterator[Operation]]],
):
    new_subcircuits: dict[FrozenSet[Qid], List[Moment]] = dict(current_subcircuits)
    current_qids = {qubit: k for k in current_subcircuits.keys() for qubit in k}
    for op in operations:
        # todo: measurement gate
        distinct_sets: FrozenSet[FrozenSet[Qid]] = frozenset(
            [current_qids[qubit] for qubit in op.qubits]
        )
        if len(distinct_sets) > 1:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            new_set = frozenset.union(*distinct_sets)
            new_subcircuits[new_set] = []
            for old_set in distinct_sets:
                new_subcircuits.pop(old_set)
    return new_subcircuits


# circuit is a list of moments
# moment is an iterator of subcircuits
# subcircuit is a list of submoments
# submoment is an iterator of operations
def _add_moment(
    new_circuit: List[Iterator[List[Iterator[Operation]]]],
    old_moment: Moment,
    new_subcircuits: Mapping[FrozenSet[Qid], List[Iterator[Operation]]],
):

    new_ops = [circuit for circuit in new_subcircuits.values() if len(circuit) == 0]
    new_circuit.append(new_ops)

    for qubits, subcircuit in new_subcircuits.items():
        operations = filter(lambda op: next(iter(op.qubits)) in qubits, old_moment.operations)
        subcircuit.append(Moment(operations))


def _to_subcircuit(submoments: Iterator[Iterator[Operation]]):
    print()
    print(f"submoments.count:{len(submoments)}")
    circuit = FrozenCircuit([Moment(submoment) for submoment in submoments])
    print("circuit")
    print(circuit)
    op = CircuitOperation(circuit)
    print("op")
    print(op)
    return op


def _to_moment(subcircuits: Iterator[Iterator[Iterator[Operation]]]):
    print(f"subcircuits.count:{len(subcircuits)}")
    return Moment([_to_subcircuit(subcircuit) for subcircuit in subcircuits])


def _to_circuit(moments: Iterator[Iterator[Iterator[Iterator[Operation]]]]):
    print(f"moments.count:{len(moments)}")
    return FrozenCircuit([_to_moment(moment) for moment in moments])
