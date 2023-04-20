# Copyright 2020 The Cirq Developers
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

"""Transformer pass to repack circuits avoiding simultaneous operations with different classes."""

import itertools
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
)

from cirq import _import, circuits, ops, protocols
from cirq.transformers import transformer_api

drop_empty_moments = _import.LazyLoader('drop_empty_moments', globals(), 'cirq.transformers')

if TYPE_CHECKING:
    import cirq

# A function that decides based on an operation
# whether it belongs to a class or not
Classifier = Callable[['cirq.Operation'], bool]

# Any of the possible operation categories that we can stratify on.
Category = Union[
    'cirq.Gate', 'cirq.Operation', Type['cirq.Gate'], Type['cirq.Operation'], Classifier
]


# Type signature of a stratifying method within this file.
StratifyMethod = Callable[
    [circuits.AbstractCircuit, Sequence[Classifier], 'cirq.TransformerContext'], circuits.Circuit,
]


@transformer_api.transformer(add_deep_support=True)
def stratified_circuit(
    circuit: 'cirq.AbstractCircuit',
    *,
    method: str = "static",
    context: Optional['cirq.TransformerContext'] = None,
    categories: Iterable[Category] = (),
) -> 'cirq.Circuit':
    """Repacks avoiding simultaneous operations with different classes.

    This transforms the given circuit to ensure that no operations of different categories are
    found in the same moment. Makes no optimality guarantees.
    Tagged Operations marked with any of `context.tags_to_ignore` will be treated as a separate
    category will be left in their original moments without stratification.

    Args:
        circuit: The circuit whose operations should be re-arranged. Will not be modified.
        method: A choice of stratifying method.  May be one of "static" or "dynamic".
        context: `cirq.TransformerContext` storing common configurable options for transformers.
        categories: A list of classifiers picking out certain operations. There are several ways
            to specify a classifier. You can pass in a gate instance (e.g. `cirq.X`),
            a gate type (e.g. `cirq.XPowGate`), an operation instance
            (e.g. `cirq.X(cirq.LineQubit(0))`), an operation type (e.g.`cirq.CircuitOperation`),
            or an arbitrary operation predicate (e.g. `lambda op: len(op.qubits) == 2`).

    Returns:
        A copy of the original circuit, but with re-arranged operations.

    Raises:
        ValueError: If 'method' is not equal to "static" or "dynamic".
    """
    if method not in ["static", "dynamic"]:
        raise ValueError(f"Unrecognized stratifying method: {method}.")

    context = context or transformer_api.TransformerContext()

    # Convert categories into classifier functions.
    classifiers = _get_classifiers(circuit, categories)

    if method == "static":
        return _statically_stratify_circuit(circuit, classifiers, context)
    return _dynamically_stratify_circuit(circuit, classifiers, context)


def _optimize_statifying_direction(stratify_method: StratifyMethod) -> StratifyMethod:
    """Decorator to optimize over stratifying a circuit left-to-right vs. right-to-left."""

    def optimized_stratifying_method(
        circuit: 'cirq.AbstractCircuit',
        classifiers: Sequence[Classifier],
        context: 'cirq.TransformerContext',
    ) -> 'cirq.Circuit':
        forward_circuit = stratify_method(circuit, classifiers, context)
        backward_circuit = stratify_method(circuit[::-1], classifiers, context)
        if len(forward_circuit) <= len(backward_circuit):
            return forward_circuit
        return backward_circuit[::-1]

    return optimized_stratifying_method


@_optimize_statifying_direction
def _statically_stratify_circuit(
    circuit: 'cirq.AbstractCircuit',
    classifiers: Sequence[Classifier],
    context: 'cirq.TransformerContext',
) -> 'cirq.Circuit':
    """A "static" stratifying method that:
    - Enforces that a fixed stratification structure, e.g. moments of type [A, B, C, A, B, C, ...].
    - Places each operation into the earliest moment that can accomodate it.
    - Optimizes over the order of the classifiers, returning the shortest circuit found.

    Args:
        circuit: The circuit to break out into homogeneous moments. Will not be edited.
        classifiers: A list of rules to align the circuit. Must be exhaustive, i.e. all operations
            will be caught by one of the processors.
        context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
        The stratified circuit.
    """
    smallest_depth = protocols.num_qubits(circuit) * len(circuit) + 1
    shortest_stratified_circuit = circuits.Circuit()
    for ordered_classifiers in itertools.permutations(classifiers):
        solution = _statically_stratify_circuit_without_optimization(
            circuit, classifiers=ordered_classifiers, context=context
        )
        if len(solution) < smallest_depth:
            shortest_stratified_circuit = solution
            smallest_depth = len(solution)
    return shortest_stratified_circuit


def _statically_stratify_circuit_without_optimization(
    circuit: 'cirq.AbstractCircuit',
    classifiers: Sequence[Classifier],
    context: 'cirq.TransformerContext',
) -> 'cirq.Circuit':
    """Helper function for '_statically_stratify_circuit'.

    Stratifies a circuit without optimizing over the order of classifiers.
    """
    num_classes = len(classifiers) + 1  # include one "extra" category for ignored operations
    new_moments: List[List['cirq.Operation']] = []

    # Keep track of the the latest time index for each qubit, measurement key, and control key.
    qubit_time_index: Dict['cirq.Qid', int] = {}
    measurement_time_index: Dict['cirq.MeasurementKey', int] = {}
    control_time_index: Dict['cirq.MeasurementKey', int] = {}

    # The minimum time index for operations with a tag in context.tags_to_ignore.
    last_ignored_ops_time_index = 0

    for moment in circuit:
        # Identify the new time indices that operations should be moved into.
        ignored_ops = []
        op_time_indices = {}
        for op in moment:

            # Identify the earliest moment that can accommodate this op.
            min_time_index_for_op = circuits.circuit.get_earliest_accommodating_moment_index(
                op, qubit_time_index, measurement_time_index, control_time_index
            )

            # Identify the "class" of this operation (by index).
            ignored_op = any(tag in op.tags for tag in context.tags_to_ignore)
            if not ignored_op:
                op_class = _get_op_class(op, classifiers)
            else:
                op_class = len(classifiers)
                ignored_ops.append(op)
                min_time_index_for_op = max(min_time_index_for_op, last_ignored_ops_time_index + 1)

            # Identify the time index to place this operation into.
            time_index = (min_time_index_for_op // num_classes) * num_classes + op_class
            if time_index < min_time_index_for_op:
                time_index += num_classes
            op_time_indices[op] = time_index

        # Assign ignored operations to the same moment.
        if ignored_ops:
            last_ignored_ops_time_index = max(op_time_indices[op] for op in ignored_ops)
            for op in ignored_ops:
                op_time_indices[op] = last_ignored_ops_time_index

        # Move the operations into their assigned moments.
        for op, time_index in op_time_indices.items():
            if time_index >= len(new_moments):
                new_moments += [[] for _ in range(num_classes)]
            new_moments[time_index].append(op)

            # Update qubit, measurment key, and control key moments.
            for qubit in op.qubits:
                qubit_time_index[qubit] = time_index
            for key in protocols.measurement_key_objs(op):
                measurement_time_index[key] = time_index
            for key in protocols.control_keys(op):
                control_time_index[key] = time_index

    return circuits.Circuit(circuits.Moment(moment) for moment in new_moments if moment)


@_optimize_statifying_direction
def _dynamically_stratify_circuit(
    circuit: 'cirq.AbstractCircuit',
    classifiers: Sequence[Classifier],
    context: 'cirq.TransformerContext',
) -> 'cirq.Circuit':
    """A "dynamic" stratifying method that:
    - Iterates over all operations in topological order.
    - Creates new moments on an as-needed basis.
    - Advances moments up/forward if and when possible to absorb a new operation.

    Most of the complexity of adding operations to moments is offloaded to the _Strata class, while
    this method mostly deals with handling ignored operations (based on `context.tags_to_ignore`).

    Args:
        circuit: The circuit to break out into homogeneous moments. Will not be edited.
        classifiers: A list of rules to align the circuit. Must be exhaustive, i.e. all operations
            will be caught by one of the processors.
        context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
        The stratified circuit.
    """
    # Initialize a _Strata object and add operations to it incrementally.
    strata = _Strata(classifiers)
    for moment in circuit:
        ignored_ops = []
        for op in moment:
            if not any(tag in op.tags for tag in context.tags_to_ignore):
                strata.add(op)
            else:
                ignored_ops.append(op)
        if ignored_ops:
            class_index = hash(tuple(ignored_ops))  # assign a unique "class" to these ignored_ops
            strata.append_stratum(class_index, *ignored_ops)
    return strata.to_circuit()


def _get_classifiers(
    circuit: 'cirq.AbstractCircuit', categories: Iterable[Category]
) -> List[Classifier]:
    """Convert a collection of categories into a list of classifiers.

    The returned list of classifiers is:
    - Exhaustive, meaning every operation in the circuit is classified by at least one classifier.
    - Minimal, meaning unused classifiers are forgotten.
    """
    # Convert all categories into classifiers, and make the list exhaustive by adding a dummy
    # classifier for otherwise unclassified ops.
    classifiers = [_category_to_classifier(cat) for cat in categories] + [_dummy_classifier]

    # Figure out which classes are actually used in the circuit.
    class_is_used = [False for _ in classifiers]
    for op in circuit.all_operations():
        class_is_used[_get_op_class(op, classifiers)] = True
        if all(class_is_used):
            break

    # Return only the classifiers that are used.
    return [classifier for classifier, is_used in zip(classifiers, class_is_used) if is_used]


# No type for `category` because mypy does not keep the return type when
# returning a callback.
def _category_to_classifier(category) -> Classifier:
    """Normalizes the given category into a classifier function."""
    if isinstance(category, ops.Gate):
        return lambda op: op.gate == category
    if isinstance(category, ops.Operation):
        return lambda op: op == category
    elif isinstance(category, type) and issubclass(category, ops.Gate):
        return lambda op: isinstance(op.gate, category)
    elif isinstance(category, type) and issubclass(category, ops.Operation):
        return lambda op: isinstance(op, category)
    elif callable(category):
        return lambda op: category(op)
    else:
        raise TypeError(
            f'Unrecognized classifier type '
            f'{type(category)} ({category!r}).\n'
            f'Expected a cirq.Gate, cirq.Operation, '
            f'Type[cirq.Gate], Type[cirq.Operation], '
            f'or Callable[[cirq.Operation], bool].'
        )


def _dummy_classifier(op: 'cirq.Operation') -> bool:
    """Dummy classifier, used to "complete" a collection of classifiers and make it exhaustive."""
    return False  # coverage: ignore


def _get_op_class(op: 'cirq.Operation', classifiers: Sequence[Classifier]) -> int:
    """Get the "class" of an operator, by index."""
    for class_index, classifier in enumerate(classifiers):
        if classifier is _dummy_classifier:
            dummy_classifier_index = class_index
        elif classifier(op):
            return class_index
    # If we got this far, the operation did not match any "actual" classifier,
    # so return the index of the dummy classifer.
    try:
        return dummy_classifier_index
    except NameError:
        raise ValueError(f"Operation {op} not identified by any classifier")


####################################################################################################
# stratifying data structures


class _Stratum:
    """A custom cirq.Moment that additionally keeps track of:
    - The time_index that it should occupy in a circuit.
    - An integer "class_index" that identifies the "type" of operations in this _Stratum.
    """

    def __init__(self, time_index: int, class_index: int, *ops: ops.Operation) -> None:
        """Initialize a _Stratum with a tentative time_index and a permanent class_index."""
        self.time_index = time_index
        self._class_index = class_index
        self._ops = list(ops)
        self._qubits = set(qubit for op in ops for qubit in op.qubits)

    @property
    def qubits(self) -> Set['cirq.Qid']:
        return self._qubits

    @property
    def class_index(self) -> int:
        return self._class_index

    def to_moment(self) -> circuits.Moment:
        """Convert this _Stratum into a Moment."""
        return circuits.Moment(self._ops)

    def add(self, op: ops.Operation) -> None:
        """Add an operation to this stratum.

        WARNING: For performance reasons, this method does not check whether this stratum can
            accomodate the given op.  Add operations at your own peril!
        """
        self._ops.append(op)
        self._qubits |= set(op.qubits)


class _Strata:
    """A data structure to organize a collection of strata ('_Stratum's).

    The naming and language in this class imagine that strata are organized into a vertical stack,
    with time "increasing" as you go "up".  That is, if stratum A precedes stratum B (i.e.,
    A.time_index < B.time_index), then stratum A is said to be "below" stratum B, and stratum B is
    said to be "above" stratum A.

    In accordance with this metaphor, we build a '_Strata_ object by adding operations to the stack
    of strata "from above".
    """

    def __init__(self, classifiers: Sequence[Classifier]) -> None:
        self._classifiers = classifiers
        self._strata: List[_Stratum] = []

        # maps from a qubit, measurement key, or control key to the highest stratum that adresses it
        self._qubit_floor: Dict['cirq.Qid', _Stratum] = {}
        self._mkey_floor: Dict['cirq.MeasurementKey', _Stratum] = {}
        self._ckey_floor: Dict['cirq.MeasurementKey', _Stratum] = {}

        # map from a stratum to its index in self._strata
        self._stratum_index: Dict[_Stratum, int] = {}

    def to_circuit(self) -> circuits.Circuit:
        return circuits.Circuit(stratum.to_moment() for stratum in self._strata)

    def add(self, op: ops.Operation) -> None:
        """Add an operation to the lowest stratum possible.

        Strategy:
        (1) Find the "op_floor" stratum, i.e., the highest stratum that collides with the op.
        (2) Try to find the lowest stratum that
            (a) is below the op_floor,
            (b) can accomodate the op, and
            (c) can be moved up above the op_floor (without violating causality).
            If such a "below_stratum" exists, move it above the op_floor add the op to it.
        (3) If no below_stratum exists, find the lowest stratum above the op_floor that can
            accomodate the op, and add the op to this "above_stratum".
        (4) If no above_stratum exists either, add the op to a new stratum above everything.
        """
        op_class = _get_op_class(op, self._classifiers)
        op_floor = self._get_op_floor(op)

        op_is_merged = self._merge_into_below_stratum(op_class, op_floor, op)
        if not op_is_merged:
            op_is_merged = self._merge_into_above_stratum(op_class, op_floor, op)
        if not op_is_merged:
            self.append_stratum(op_class, op)

    def _get_op_floor(self, op: ops.Operation) -> Optional[_Stratum]:
        """Get the highest stratum that collides with this op, if there is any."""
        op_qubits = op.qubits
        op_mkeys = protocols.measurement_key_objs(op)
        op_ckeys = protocols.control_keys(op)

        # Identify strata that collide with this op.
        colliding_strata: List[Optional[_Stratum]] = []
        if op_qubits:
            colliding_strata.extend([self._qubit_floor.get(qubit) for qubit in op_qubits])
        if op_mkeys:
            colliding_strata.extend([self._mkey_floor.get(key) for key in op_mkeys])
            colliding_strata.extend([self._ckey_floor.get(key) for key in op_mkeys])
        if op_ckeys:
            colliding_strata.extend([self._mkey_floor.get(key) for key in op_ckeys])

        # Return the highest stratum, if any.
        return max(
            (stratum for stratum in colliding_strata if stratum),
            key=lambda stratum: stratum.time_index,
            default=None,
        )

    # TODO: properly deal with measurement/control keys
    def _merge_into_below_stratum(
        self, op_class: int, op_floor: Optional[_Stratum], op: ops.Operation
    ) -> bool:
        """Try to merge the given operation into the lowest stratum that:
            (a) is below the op_floor,
            (b) can accomodate an op on the given qubits, and
            (c) can be moved up above the op_floor (without violating causality).
        If such a stratum exists, raise it above the op_floor, add the op to it, and return True.
        Reurn False otherwise.
        """
        if op_floor is None:
            # There is no floor, so there are no strata below the floor.
            return False

        below_stratum = None  # initialize the null hypothesis that no below_stratum exists

        # Keep track of qubits in the past light cone of the op, which block a candidate
        # below_stratum from being able to move up above the op_floor.
        past_light_cone_qubits = set(op.qubits)

        # Starting from the op_floor, look down/backwards for a candidate below_stratum.
        op_floor_index = self._stratum_index[op_floor]
        for stratum in self._strata[op_floor_index::-1]:
            if stratum.class_index != op_class:
                # This stratum cannot accomodate the op, but it might be in op's past light cone.
                if not stratum.qubits.isdisjoint(past_light_cone_qubits):
                    past_light_cone_qubits |= stratum.qubits
            else:
                if stratum.qubits.isdisjoint(past_light_cone_qubits):
                    # This stratum can accomodate the op, so it is a candidate below_stratum.
                    below_stratum = stratum
                else:
                    # This stratum collides with the op's past light cone.  Corrolaries:
                    # (1) This stratum cannot accomodate this op (obvious).
                    # (2) No lower stratum can be a candiate below_stratum (less obvious).
                    # Hand-wavy proof by contradiction for claim 2:
                    # (a) Assume there exists a lower stratum is a candidate for the below_stratum,
                    #     which means that it does not collide with this op's past light cone.
                    # (b) In particular, the lower stratum does not collide with *this* stratum's
                    #     past light cone, so it can be moved up and merged into this stratum.
                    # (c) That contradicts the incremental construction of _Strata, which would
                    #     have moved the lower stratum up to absorb ops in this stratum when those
                    #     ops were added to this _Strata object (self).
                    # Altogether, our search for a below_stratum is done, so we can stop our
                    # backwards search through self._strata.
                    break

        if below_stratum is None:
            # No candidate below_stratum was found :(
            return False

        # Move the below_stratum above the op_floor and add the op to it
        if op_floor is not None:
            self._move_stratum_above_floor(below_stratum, op_floor)
        below_stratum.add(op)

        self._update_floors(op, below_stratum)
        return True

    # TODO: properly deal with measurement/control keys
    def _move_stratum_above_floor(self, below_stratum: _Stratum, op_floor: _Stratum) -> None:
        """Move a below_stratum up above the op_floor."""
        # Identify the indices of a few relevant strata.
        op_floor_index = self._stratum_index[op_floor]
        above_floor_index = op_floor_index + 1  # hack around flake8 false positive (E203)
        below_stratum_index = self._stratum_index[below_stratum]

        # Identify all strata in the future light cone of the below_stratum.  When we move the
        # below_stratum up above the op_floor, we need to likewise shift all of these strata up in
        # order to preserve causal structure.
        light_cone_strata = [below_stratum]
        light_cone_qubits = below_stratum.qubits

        # Keep track of "spectator" strata that are currently above the below_stratum, but are not
        # in its future light cone.
        spectator_strata = []

        start = below_stratum_index + 1  # hack around flake8 false positive (E203)
        for stratum in self._strata[start:above_floor_index]:
            if not stratum.qubits.isdisjoint(light_cone_qubits):
                # This stratum is in the future light cone of the below_stratum.
                light_cone_strata.append(stratum)
                light_cone_qubits |= stratum.qubits

            else:
                spectator_strata.append(stratum)

                # The light cone strata are going to be moved above this spectator stratum.
                # Shift the indices of strata accordingly.
                self._stratum_index[stratum] -= len(light_cone_strata)
                for stratum in light_cone_strata:
                    self._stratum_index[stratum] += 1

        # Shift the entire light cone forward, so that the below_stratum lies above the op_floor.
        # Also shift everything above the op_floor forward by the same amount to ensure that it
        # still lies above the below_stratum.
        strata_to_shift = light_cone_strata + self._strata[above_floor_index:]
        time_index_shift = self._strata[op_floor_index].time_index - below_stratum.time_index + 1
        for stratum in strata_to_shift:
            stratum.time_index += time_index_shift

        # Sort all strata by their time_index.
        self._strata[below_stratum_index:] = spectator_strata + strata_to_shift

    def _merge_into_above_stratum(
        self, op_class: int, op_floor: Optional[_Stratum], op: ops.Operation
    ) -> bool:
        """Try to merge the given operation into an existing stratum above the op_floor.
        Return True iff a suitable stratum is found and the op is merged.
        """
        start = self._stratum_index[op_floor] + 1 if op_floor is not None else 0
        for stratum in self._strata[start:]:
            if stratum.class_index == op_class and stratum.qubits.isdisjoint(op.qubits):
                stratum.add(op)
                self._update_floors(op, stratum)
                return True
        return False

    def append_stratum(self, op_class: int, *ops: ops.Operation) -> None:
        """Add the given operations to a new stratum above all other strata."""
        time_index = self._strata[-1].time_index + 1 if self._strata else 0
        stratum = _Stratum(time_index, op_class, *ops)
        self._strata.append(stratum)
        self._stratum_index[stratum] = len(self._strata) - 1
        for op in ops:
            self._update_floors(op, stratum)

    def _update_floors(self, op: ops.Operation, stratum: _Stratum) -> None:
        """Update the "floors" for the qubits, measurment keys, and control keys of an operation.
        The "floor" of, say, a qubit is the highest stratum that addresses it.
        """
        self._qubit_floor.update({qubit: stratum for qubit in op.qubits})
        self._mkey_floor.update({key: stratum for key in protocols.measurement_key_objs(op)})
        self._ckey_floor.update({key: stratum for key in protocols.control_keys(op)})
