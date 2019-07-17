# Copyright 2018 The Cirq Developers
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

from typing import cast, Dict, Iterable, Sequence, Tuple, TypeVar, Union

import abc

from cirq import circuits, ops, optimizers, protocols, value


LogicalIndex = TypeVar('LogicalIndex', int, ops.Qid)
LogicalIndexSequence = Union[Sequence[int], Sequence[ops.Qid]]
LogicalGates = Dict[Tuple[LogicalIndex, ...], ops.Gate]
LogicalMappingKey = TypeVar('LogicalMappingKey', bound=ops.Qid)
LogicalMapping = Dict[LogicalMappingKey, LogicalIndex]


class PermutationGate(ops.Gate, metaclass=abc.ABCMeta):
    """A permutation gate indicates a change in the mapping from qubits to
    logical indices.

    Args:
        swap_gate: the gate that swaps the indices mapped to by a pair of
            qubits (e.g. SWAP or fermionic swap).
    """

    def __init__(self, num_qubits: int, swap_gate: ops.Gate=ops.SWAP) -> None:
        self._num_qubits = num_qubits
        self.swap_gate = swap_gate

    def num_qubits(self) -> int:
        return self._num_qubits

    @abc.abstractmethod
    def permutation(self) -> Dict[int, int]:
        """permutation = {i: s[i]} indicates that the i-th element is mapped to
        the s[i]-th element."""

    def update_mapping(self, mapping: Dict[ops.Qid, LogicalIndex],
                       keys: Sequence[ops.Qid]
                       ) -> None:
        """Updates a mapping (in place) from qubits to logical indices.

        Args:
            mapping: The mapping to update.
            keys: The qubits acted on by the gate.
        """
        permutation = self.permutation()
        indices = tuple(permutation.keys())
        new_keys = [keys[permutation[i]] for i in indices]
        old_elements = [mapping[keys[i]] for i in indices]
        mapping.update(zip(new_keys, old_elements))

    @staticmethod
    def validate_permutation(permutation: Dict[int, int],
                             n_elements: int=None) -> None:
        if not permutation:
            return
        if set(permutation.values()) != set(permutation):
            raise IndexError('key and value sets must be the same.')
        if min(permutation) < 0:
            raise IndexError('keys of the permutation must be non-negative.')
        if n_elements is not None:
            if max(permutation) >= n_elements:
                raise IndexError('key is out of bounds.')

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> Tuple[str, ...]:
        if args.known_qubit_count is None:
            return NotImplemented
        permutation = self.permutation()
        arrow = '↦' if args.use_unicode_characters else '->'
        wire_symbols = tuple(str(i) + arrow + str(permutation.get(i, i))
                        for i in range(self.num_qubits()))
        return wire_symbols


class MappingDisplayGate(ops.Gate):
    """Displays the indices mapped to a set of wires."""

    def __init__(self, indices):
        self.indices = tuple(indices)
        self._num_qubits = len(self.indices)

    def num_qubits(self) -> int:
        return self._num_qubits

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                              ) -> protocols.CircuitDiagramInfo:
        wire_symbols = tuple('' if i is None else str(i) for i in self.indices)
        return protocols.CircuitDiagramInfo(wire_symbols, connected=False)


def display_mapping(circuit: circuits.Circuit,
                    initial_mapping: LogicalMapping) -> None:
    """Inserts display gates between moments to indicate the mapping throughout
    the circuit."""
    qubits = sorted(circuit.all_qubits())
    mapping = initial_mapping.copy()

    old_moments = circuit._moments
    gate = MappingDisplayGate(mapping.get(q) for q in qubits)
    new_moments = [ops.Moment([gate(*qubits)])]
    for moment in old_moments:
        new_moments.append(moment)
        update_mapping(mapping, moment)
        gate = MappingDisplayGate(mapping.get(q) for q in qubits)
        new_moments.append(ops.Moment([gate(*qubits)]))

    circuit._moments = new_moments


@value.value_equality
class SwapPermutationGate(PermutationGate):
    """Generic swap gate."""

    def __init__(self, swap_gate: ops.Gate=ops.SWAP):
        super().__init__(2, swap_gate)

    def permutation(self) -> Dict[int, int]:
        return {0: 1, 1: 0}

    def _decompose_(
            self, qubits: Sequence[ops.Qid]) -> ops.OP_TREE:
        yield self.swap_gate(*qubits)

    def __repr__(self):
        return ('cirq.contrib.acquaintance.SwapPermutationGate(' +
                ('' if self.swap_gate == ops.SWAP else repr(self.swap_gate)) +
                ')')

    def _value_equality_values_(self):
        return (self.swap_gate,)


def _canonicalize_permutation(permutation: Dict[int, int]) -> Dict[int, int]:
    return {i: j for i, j in permutation.items() if i != j}


@value.value_equality(unhashable=True)
class LinearPermutationGate(PermutationGate):
    """A permutation gate that decomposes a given permutation using a linear
        sorting network."""

    def __init__(self,
                 num_qubits: int,
                 permutation: Dict[int, int],
                 swap_gate: ops.Gate=ops.SWAP
                 ) -> None:
        """Initializes a linear permutation gate.

        Args:
            permutation: The permutation effected by the gate.
            swap_gate: The swap gate used in decompositions.
        """
        super().__init__(num_qubits, swap_gate)
        PermutationGate.validate_permutation(permutation, num_qubits)
        self._permutation = permutation

    def permutation(self) -> Dict[int, int]:
        return self._permutation

    def _decompose_(self, qubits: Sequence[ops.Qid]) -> ops.OP_TREE:
        swap_gate = SwapPermutationGate(self.swap_gate)
        n_qubits = len(qubits)
        mapping = {i: self._permutation.get(i, i) for i in range(n_qubits)}
        for layer_index in range(n_qubits):
            for i in range(layer_index % 2, n_qubits - 1, 2):
                if mapping[i] > mapping[i + 1]:
                    yield swap_gate(*qubits[i:i+2])
                    mapping[i], mapping[i+1] = mapping[i+1], mapping[i]

    def __repr__(self):
        return ('cirq.contrib.acquaintance.LinearPermutationGate('
                '{!r}, {!r}, {!r})'.format(
                self.num_qubits(), self._permutation, self.swap_gate))

    def _value_equality_values_(self):
        return (tuple(sorted((i, j) for i, j in self._permutation.items()
                if i != j)), self.swap_gate)

    def __bool__(self):
        return bool(_canonicalize_permutation(self._permutation))

    def __pow__(self, exponent):
        if exponent == 1:
            return self
        if exponent == -1:
            return LinearPermutationGate(
                self._num_qubits, {v: k for k, v in self._permutation.items()},
                self.swap_gate)
        return NotImplemented


def update_mapping(mapping: Dict[ops.Qid, LogicalIndex],
                   operations: ops.OP_TREE
                   ) -> None:
    """Updates a mapping (in place) from qubits to logical indices according to
    a set of permutation gates. Any gates other than permutation gates are
    ignored.

    Args:
        mapping: The mapping to update.
        operations: The operations to update according to.
    """
    for op in ops.flatten_op_tree(operations):
        if (isinstance(op, ops.GateOperation) and
            isinstance(op.gate, PermutationGate)):
            op.gate.update_mapping(mapping, op.qubits)


def get_logical_operations(operations: ops.OP_TREE,
                           initial_mapping: Dict[ops.Qid, ops.Qid]
                          ) -> Iterable[ops.Operation]:
    mapping = initial_mapping.copy()
    for op in cast(Iterable[ops.Operation], ops.flatten_op_tree(operations)):
        if (isinstance(op, ops.GateOperation) and
                isinstance(op.gate, PermutationGate)):
            op.gate.update_mapping(mapping, op.qubits)
        else:
            yield op.transform_qubits(mapping.__getitem__)


class ExpandPermutationGates(optimizers.ExpandComposite):
    """Decomposes any permutation gates other SwapPermutationGate."""
    def __init__(self):
        circuits.PointOptimizer.__init__(self)

        self.no_decomp = lambda op: (not all(
                [isinstance(op, ops.GateOperation),
                 isinstance(op.gate, PermutationGate),
                 not isinstance(op.gate, SwapPermutationGate)]))

expand_permutation_gates = ExpandPermutationGates()


def return_to_initial_mapping(circuit: circuits.Circuit,
                              swap_gate: ops.Gate = ops.SWAP) -> None:
    qubits = sorted(circuit.all_qubits())
    n_qubits = len(qubits)

    mapping = {q: i for i, q in enumerate(qubits)}
    update_mapping(mapping, circuit.all_operations())

    permutation = {i: mapping[q] for i, q in enumerate(qubits)}
    returning_permutation_op = LinearPermutationGate(n_qubits, permutation,
                                                     swap_gate)(*qubits)
    circuit.append(returning_permutation_op)


def uses_consistent_swap_gate(circuit: circuits.Circuit,
                              swap_gate: ops.Gate) -> bool:
    for op in circuit.all_operations():
        if (isinstance(op, ops.GateOperation) and
                isinstance(op.gate, PermutationGate)):
            if op.gate.swap_gate != swap_gate:
                return False
    return True
