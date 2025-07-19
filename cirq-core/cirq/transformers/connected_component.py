# Copyright 2025 The Cirq Developers
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

"""Defines a connected component of operations, to be used in merge transformers."""

from __future__ import annotations

from typing import Callable, cast, Sequence, TYPE_CHECKING

from cirq import ops, protocols

if TYPE_CHECKING:
    import cirq


class Component:
    """Internal representation for a connected component of operations.

    It uses the disjoint-set data structure to implement merge efficiently.
    Additional merge conditions can be added by deriving from the Component
    class and overriding the merge function (see ComponentWithOps and
    ComponentWithCircuitOp) below.
    """

    # Properties for the disjoint set data structure
    parent: Component | None = None
    rank: int = 0

    # True if the component can be merged
    is_mergeable: bool

    # Circuit moment containing the component
    moment: int
    # Union of all op qubits in the component
    qubits: frozenset[cirq.Qid]
    # Union of all measurement keys in the component
    mkeys: frozenset[cirq.MeasurementKey]
    # Union of all control keys in the component
    ckeys: frozenset[cirq.MeasurementKey]
    # Initial operation in the component
    op: cirq.Operation

    def __init__(self, op: cirq.Operation, moment: int, is_mergeable=True):
        """Initializes a singleton component."""
        self.is_mergeable = is_mergeable
        self.moment = moment
        self.qubits = frozenset(op.qubits)
        self.mkeys = protocols.measurement_key_objs(op)
        self.ckeys = protocols.control_keys(op)
        self.op = op

    def find(self) -> Component:
        """Finds the component representative."""

        root = self
        while root.parent is not None:
            root = root.parent
        x = self
        while x != root:
            parent = x.parent
            x.parent = root
            x = cast(Component, parent)
        return root

    def merge(self, c: Component, merge_left=True) -> Component | None:
        """Attempts to merge two components.

        We assume the following is true whenever merge is called:
            - if merge_left = True then c.qubits are a subset of self.qubits
            - if merge_left = False then self.qubits are a subset of c.qubits

        If merge_left is True, c is merged into this component, and the representative
        will keep this moment and qubits. If merge_left is False, this component is
        merged into c, and the representative will keep c's moment and qubits.

        Args:
            c: other component to merge
            merge_left: True to keep self's data for the merged component, False to
                keep c's data for the merged component.

        Returns:
            None, if the components can't be merged.
            Otherwise the new component representative.
        """
        x = self.find()
        y = c.find()

        if not x.is_mergeable or not y.is_mergeable:
            return None

        if x == y:
            return x

        if x.rank < y.rank:
            if merge_left:
                # As y will be the new representative, copy moment and qubits from x
                y.moment = x.moment
                y.qubits = x.qubits
            x, y = y, x
        elif not merge_left:
            # As x will be the new representative, copy moment and qubits from y
            x.moment = y.moment
            x.qubits = y.qubits

        y.parent = x
        if x.rank == y.rank:
            x.rank += 1

        x.mkeys = x.mkeys.union(y.mkeys)
        x.ckeys = x.ckeys.union(y.ckeys)
        return x


class ComponentWithOps(Component):
    """Component that keeps track of operations.

    Encapsulates a method can_merge that is used to decide if two components
    can be merged.
    """

    # List of all operations in the component
    ops: list[cirq.Operation]

    # Method to decide if two components can be merged based on their operations
    can_merge: Callable[[Sequence[cirq.Operation], Sequence[cirq.Operation]], bool]

    def __init__(
        self,
        op: cirq.Operation,
        moment: int,
        can_merge: Callable[[Sequence[cirq.Operation], Sequence[cirq.Operation]], bool],
        is_mergeable=True,
    ):
        super().__init__(op, moment, is_mergeable)
        self.ops = [op]
        self.can_merge = can_merge

    def merge(self, c: Component, merge_left=True) -> Component | None:
        """Attempts to merge two components.

        Returns:
            None if can_merge is False, otherwise the new representative.
                The representative will have ops = a.ops + b.ops.
        """
        x = cast(ComponentWithOps, self.find())
        y = cast(ComponentWithOps, c.find())

        if x == y:
            return x

        if not x.is_mergeable or not y.is_mergeable or not x.can_merge(x.ops, y.ops):
            return None

        root = cast(ComponentWithOps, super(ComponentWithOps, x).merge(y, merge_left))
        root.ops = x.ops + y.ops
        # Clear the ops list in the non-representative set to avoid memory consumption
        if x != root:
            x.ops = []
        else:
            y.ops = []
        return root


class ComponentWithCircuitOp(Component):
    """Component that keeps track of operations as a CircuitOperation.

    Encapsulates a method merge_func that is used to merge two components.
    """

    # CircuitOperation containing all the operations in the component,
    # or a single Operation if the component is a singleton
    circuit_op: cirq.Operation

    merge_func: Callable[[ops.Operation, ops.Operation], ops.Operation | None]

    def __init__(
        self,
        op: cirq.Operation,
        moment: int,
        merge_func: Callable[[ops.Operation, ops.Operation], ops.Operation | None],
        is_mergeable=True,
    ):
        super().__init__(op, moment, is_mergeable)
        self.circuit_op = op
        self.merge_func = merge_func

    def merge(self, c: Component, merge_left=True) -> Component | None:
        """Attempts to merge two components.

        Returns:
            None if merge_func returns None, otherwise the new representative.
        """
        x = cast(ComponentWithCircuitOp, self.find())
        y = cast(ComponentWithCircuitOp, c.find())

        if x == y:
            return x

        if not x.is_mergeable or not y.is_mergeable:
            return None

        new_op = x.merge_func(x.circuit_op, y.circuit_op)
        if not new_op:
            return None

        root = cast(ComponentWithCircuitOp, super(ComponentWithCircuitOp, x).merge(y, merge_left))

        root.circuit_op = new_op
        # The merge_func can be arbitrary, so we need to recompute the component properties
        root.qubits = frozenset(new_op.qubits)
        root.mkeys = protocols.measurement_key_objs(new_op)
        root.ckeys = protocols.control_keys(new_op)

        # Clear the circuit op in the non-representative set to avoid memory consumption
        if x != root:
            del x.circuit_op
        else:
            del y.circuit_op
        return root


class ComponentFactory:
    """Factory for components."""

    is_mergeable: Callable[[cirq.Operation], bool]

    def __init__(self, is_mergeable: Callable[[cirq.Operation], bool]):
        self.is_mergeable = is_mergeable

    def new_component(self, op: cirq.Operation, moment: int, is_mergeable=True) -> Component:
        return Component(op, moment, self.is_mergeable(op) and is_mergeable)


class ComponentWithOpsFactory(ComponentFactory):
    """Factory for components with operations."""

    can_merge: Callable[[Sequence[cirq.Operation], Sequence[cirq.Operation]], bool]

    def __init__(
        self,
        is_mergeable: Callable[[cirq.Operation], bool],
        can_merge: Callable[[Sequence[cirq.Operation], Sequence[cirq.Operation]], bool],
    ):
        super().__init__(is_mergeable)
        self.can_merge = can_merge

    def new_component(self, op: cirq.Operation, moment: int, is_mergeable=True) -> Component:
        return ComponentWithOps(op, moment, self.can_merge, self.is_mergeable(op) and is_mergeable)


class ComponentWithCircuitOpFactory(ComponentFactory):
    """Factory for components with operations as CircuitOperation."""

    merge_func: Callable[[ops.Operation, ops.Operation], ops.Operation | None]

    def __init__(
        self,
        is_mergeable: Callable[[cirq.Operation], bool],
        merge_func: Callable[[ops.Operation, ops.Operation], ops.Operation | None],
    ):
        super().__init__(is_mergeable)
        self.merge_func = merge_func

    def new_component(self, op: cirq.Operation, moment: int, is_mergeable=True) -> Component:
        return ComponentWithCircuitOp(
            op, moment, self.merge_func, self.is_mergeable(op) and is_mergeable
        )
