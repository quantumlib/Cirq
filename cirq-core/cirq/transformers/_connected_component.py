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

from collections.abc import Callable, Sequence
from typing import cast, TYPE_CHECKING

from scipy.cluster.hierarchy import DisjointSet

from cirq import ops, protocols

if TYPE_CHECKING:
    import cirq


class Component:
    """Internal representation for a connected component of operations."""

    # Circuit moment containing the component
    moment_id: int
    # Union of all op qubits in the component
    qubits: frozenset[cirq.Qid]
    # Union of all measurement keys in the component
    mkeys: frozenset[cirq.MeasurementKey]
    # Union of all control keys in the component
    ckeys: frozenset[cirq.MeasurementKey]
    # Initial operation in the component
    op: cirq.Operation

    # True if the component can be merged with other components
    is_mergeable: bool

    def __init__(self, op: cirq.Operation, moment_id: int, is_mergeable=True):
        """Initializes a singleton component."""
        self.op = op
        self.is_mergeable = is_mergeable
        self.moment_id = moment_id
        self.qubits = frozenset(op.qubits)
        self.mkeys = protocols.measurement_key_objs(op)
        self.ckeys = protocols.control_keys(op)


class ComponentWithOps(Component):
    """Component that keeps track of operations."""

    # List of all operations in the component
    ops: list[cirq.Operation]

    def __init__(self, op: cirq.Operation, moment_id: int, is_mergeable=True):
        super().__init__(op, moment_id, is_mergeable)
        self.ops = [op]


class ComponentWithCircuitOp(Component):
    """Component that keeps track of operations as a CircuitOperation."""

    # CircuitOperation containing all the operations in the component,
    # or a single Operation if the component is a singleton
    circuit_op: cirq.Operation

    def __init__(self, op: cirq.Operation, moment_id: int, is_mergeable=True):
        super().__init__(op, moment_id, is_mergeable)
        self.circuit_op = op


class ComponentSet:
    """Represents a set of mergeable components of operations."""

    _comp_type: type[Component]

    _disjoint_set: DisjointSet

    # Callable to decide if a component is mergeable
    _is_mergeable: Callable[[cirq.Operation], bool]

    # List of components in creation order
    _components: list[Component]

    def __init__(self, is_mergeable: Callable[[cirq.Operation], bool]):
        self._is_mergeable = is_mergeable
        self._disjoint_set = DisjointSet()
        self._components = []
        self._comp_type = Component

    def new_component(self, op: cirq.Operation, moment_id: int, is_mergeable=True) -> Component:
        """Creates a new component and adds it to the set."""
        c = self._comp_type(op, moment_id, is_mergeable and self._is_mergeable(op))
        self._disjoint_set.add(c)
        self._components.append(c)
        return c

    def components(self) -> list[Component]:
        """Returns the initial components in creation order."""
        return self._components

    def find(self, x: Component) -> Component:
        """Finds the representative for a merged component."""
        return self._disjoint_set[x]

    def merge(self, x: Component, y: Component, merge_left=True) -> Component | None:
        """Attempts to merge two components.

        If merge_left is True, y is merged into x, and the representative will keep
        y's moment. If merge_left is False, x is merged into y, and the representative
        will keep y's moment.

        Args:
            x: First component to merge.
            y: Second component to merge.
            merge_left: True to keep x's moment for the merged component, False to
                keep y's moment for the merged component.

        Returns:
            None, if the components can't be merged.
            Otherwise the new component representative.
        """
        x = self._disjoint_set[x]
        y = self._disjoint_set[y]

        if not x.is_mergeable or not y.is_mergeable:
            return None

        if not self._disjoint_set.merge(x, y):
            return x

        root = self._disjoint_set[x]
        root.moment_id = x.moment_id if merge_left else y.moment_id
        root.qubits = x.qubits.union(y.qubits)
        root.mkeys = x.mkeys.union(y.mkeys)
        root.ckeys = x.ckeys.union(y.ckeys)

        return root


class ComponentWithOpsSet(ComponentSet):
    """Represents a set of mergeable components, where each component tracks operations."""

    # Callable that returns if two components can be merged based on their operations
    _can_merge: Callable[[Sequence[cirq.Operation], Sequence[cirq.Operation]], bool]

    def __init__(
        self,
        is_mergeable: Callable[[cirq.Operation], bool],
        can_merge: Callable[[Sequence[cirq.Operation], Sequence[cirq.Operation]], bool],
    ):
        super().__init__(is_mergeable)
        self._can_merge = can_merge
        self._comp_type = ComponentWithOps

    def merge(self, x: Component, y: Component, merge_left=True) -> Component | None:
        """Attempts to merge two components.

        Returns:
            None if can_merge is False or the merge doesn't succeed, otherwise the
                new representative. The representative will have ops = x.ops + y.ops.
        """
        x = cast(ComponentWithOps, self._disjoint_set[x])
        y = cast(ComponentWithOps, self._disjoint_set[y])

        if x is y:
            return x

        if not x.is_mergeable or not y.is_mergeable or not self._can_merge(x.ops, y.ops):
            return None

        root = cast(ComponentWithOps, super().merge(x, y, merge_left))
        root.ops = x.ops + y.ops
        # Clear the ops list in the non-representative component to avoid duplication
        other = y if x is root else x
        other.ops = []
        return root


class ComponentWithCircuitOpSet(ComponentSet):
    """Represents a set of mergeable components, with operations as a CircuitOperation."""

    # Callable that merges CircuitOperations from two components
    _merge_func: Callable[[ops.Operation, ops.Operation], ops.Operation | None]

    def __init__(
        self,
        is_mergeable: Callable[[cirq.Operation], bool],
        merge_func: Callable[[ops.Operation, ops.Operation], ops.Operation | None],
    ):
        super().__init__(is_mergeable)
        self._merge_func = merge_func
        self._comp_type = ComponentWithCircuitOp

    def merge(self, x: Component, y: Component, merge_left=True) -> Component | None:
        """Attempts to merge two components.

        Returns:
            None if merge_func returns None or the merge doesn't succeed,
                otherwise the new representative.
        """
        x = cast(ComponentWithCircuitOp, self._disjoint_set[x])
        y = cast(ComponentWithCircuitOp, self._disjoint_set[y])

        if x is y:
            return x

        if not x.is_mergeable or not y.is_mergeable:
            return None

        new_op = self._merge_func(x.circuit_op, y.circuit_op)
        if not new_op:
            return None

        root = cast(ComponentWithCircuitOp, super().merge(x, y, merge_left))

        root.circuit_op = new_op
        # The merge_func can be arbitrary, so we need to recompute the component properties
        root.qubits = frozenset(new_op.qubits)
        root.mkeys = protocols.measurement_key_objs(new_op)
        root.ckeys = protocols.control_keys(new_op)

        return root
