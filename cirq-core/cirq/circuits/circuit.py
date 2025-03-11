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

"""The circuit data structure.

Circuits consist of a list of Moments, each Moment made up of a set of
Operations. Each Operation is a Gate that acts on some Qubits, for a given
Moment the Operations must all act on distinct Qubits.
"""

import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from types import NotImplementedType
from typing import (
    AbstractSet,
    Any,
    Callable,
    cast,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableSequence,
    Optional,
    overload,
    Sequence,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import Self

import networkx
import numpy as np

import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.moment import Moment
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.protocols import circuit_diagram_info_protocol

if TYPE_CHECKING:
    import cirq


_TGate = TypeVar('_TGate', bound='cirq.Gate')
_MOMENT_OR_OP = Union['cirq.Moment', 'cirq.Operation']

CIRCUIT_TYPE = TypeVar('CIRCUIT_TYPE', bound='AbstractCircuit')
document(
    CIRCUIT_TYPE,
    """Type variable for an AbstractCircuit.

    This can be used when writing generic functions that operate on circuits.
    For example, suppose we define the following function:

        def foo(circuit: CIRCUIT_TYPE) -> CIRCUIT_TYPE:
            ...

    This lets the typechecker know that this function takes any kind of circuit
    and returns the same type, that is, if passed a `cirq.Circuit` it will return
    `cirq.Circuit`, and similarly if passed `cirq.FrozenCircuit` it will return
    `cirq.FrozenCircuit`. This is particularly useful for things like the
    transformer API, since it can preserve more type information than if we typed
    the function as taking and returning `cirq.AbstractCircuit`.
    """,
)

_INT_TYPE = Union[int, np.integer]


class Alignment(enum.Enum):
    """Alignment option for combining/zipping two circuits together.

    Args:
        LEFT: Stop when left ends are lined up.
        RIGHT: Stop when right ends are lined up.
        FIRST: Stop the first time left ends are lined up or right ends are lined up.
    """

    LEFT = 1
    RIGHT = 2
    FIRST = 3

    def __repr__(self) -> str:
        return f'cirq.Alignment.{self.name}'


class AbstractCircuit(abc.ABC):
    """The base class for Circuit-like objects.

    A circuit-like object must have a list of moments (which can be empty).

    These methods return information about the circuit, and can be called on
    either Circuit or FrozenCircuit objects:

    *   next_moment_operating_on
    *   prev_moment_operating_on
    *   next_moments_operating_on
    *   operation_at
    *   all_qubits
    *   all_operations
    *   findall_operations
    *   findall_operations_between
    *   findall_operations_until_blocked
    *   findall_operations_with_gate_type
    *   reachable_frontier_from
    *   has_measurements
    *   are_all_matches_terminal
    *   are_all_measurements_terminal
    *   unitary
    *   final_state_vector
    *   to_text_diagram
    *   to_text_diagram_drawer
    *   qid_shape
    *   all_measurement_key_names
    *   to_quil
    *   to_qasm
    *   save_qasm
    *   get_independent_qubit_sets
    """

    @classmethod
    def from_moments(cls: Type[CIRCUIT_TYPE], *moments: Optional['cirq.OP_TREE']) -> CIRCUIT_TYPE:
        """Create a circuit from moment op trees.

        Args:
            *moments: Op trees for each moment, which can be one of the following:
                - Moment: will be included directly in the new circuit.
                - AbstractCircuit: will be frozen, wrapped in a CircuitOperation,
                    and included in its own moment in the new circuit.
                - None: will be skipped and omitted from the circuit. This can be
                    used to include or skip a moment based on a conditional, for example.
                - Other OP_TREE: will be passed to `cirq.Moment` to create a new moment
                    which is then included in the new circuit. Note that in this
                    case we have the normal restriction that operations in a
                    moment must be applied to disjoint sets of qubits.
        """
        return cls._from_moments(cls._make_moments(moments))

    @staticmethod
    def _make_moments(moments: Iterable[Optional['cirq.OP_TREE']]) -> Iterator['cirq.Moment']:
        for m in moments:
            if isinstance(m, Moment):
                yield m
            elif isinstance(m, AbstractCircuit):
                yield Moment(m.freeze().to_op())
            elif m is not None:
                yield Moment(m)

    @classmethod
    @abc.abstractmethod
    def _from_moments(cls: Type[CIRCUIT_TYPE], moments: Iterable['cirq.Moment']) -> CIRCUIT_TYPE:
        """Create a circuit from moments.

        This must be implemented by subclasses. It provides a more efficient way
        to construct a circuit instance since we already have the moments and so
        can skip the analysis required to implement various insert strategies.

        Args:
            moments: Moments of the circuit.
        """

    @property
    @abc.abstractmethod
    def moments(self) -> Sequence['cirq.Moment']:
        pass

    @abc.abstractmethod
    def freeze(self) -> 'cirq.FrozenCircuit':
        """Creates a FrozenCircuit from this circuit.

        If 'self' is a FrozenCircuit, the original object is returned.
        """

    @abc.abstractmethod
    def unfreeze(self, copy: bool = True) -> 'cirq.Circuit':
        """Creates a Circuit from this circuit.

        Args:
            copy: If True and 'self' is a Circuit, returns a copy that circuit.
        """

    def __bool__(self) -> bool:
        return bool(self.moments)

    def __eq__(self, other) -> bool:
        if not isinstance(other, AbstractCircuit):
            return NotImplemented
        return other is self or (
            len(self.moments) == len(other.moments)
            and all(m0 == m1 for m0, m1 in zip(self.moments, other.moments))
        )

    def _approx_eq_(self, other: Any, atol: float) -> bool:
        """See `cirq.protocols.SupportsApproximateEquality`."""
        if not isinstance(other, AbstractCircuit):
            return NotImplemented
        return other is self or cirq.protocols.approx_eq(
            tuple(self.moments), tuple(other.moments), atol=atol
        )

    def __ne__(self, other) -> bool:
        return not self == other

    def __len__(self) -> int:
        return len(self.moments)

    def __iter__(self) -> Iterator['cirq.Moment']:
        return iter(self.moments)

    def _decompose_(self) -> 'cirq.OP_TREE':
        """See `cirq.SupportsDecompose`."""
        return self.all_operations()

    # pylint: disable=function-redefined
    @overload
    def __getitem__(self, key: int) -> 'cirq.Moment':
        pass

    @overload
    def __getitem__(self, key: Tuple[int, 'cirq.Qid']) -> 'cirq.Operation':
        pass

    @overload
    def __getitem__(self, key: Tuple[int, Iterable['cirq.Qid']]) -> 'cirq.Moment':
        pass

    @overload
    def __getitem__(self, key: slice) -> Self:
        pass

    @overload
    def __getitem__(self, key: Tuple[slice, 'cirq.Qid']) -> Self:
        pass

    @overload
    def __getitem__(self, key: Tuple[slice, Iterable['cirq.Qid']]) -> Self:
        pass

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._from_moments(self.moments[key])
        if hasattr(key, '__index__'):
            return self.moments[key]
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('If key is tuple, it must be a pair.')
            moment_idx, qubit_idx = key
            # moment_idx - int or slice; qubit_idx - Qid or Iterable[Qid].
            selected_moments = self.moments[moment_idx]
            if isinstance(selected_moments, Moment):
                return selected_moments[qubit_idx]
            if isinstance(qubit_idx, ops.Qid):
                qubit_idx = [qubit_idx]
            return self._from_moments(moment[qubit_idx] for moment in selected_moments)

        raise TypeError('__getitem__ called with key not of type slice, int, or tuple.')

    # pylint: enable=function-redefined

    def __str__(self) -> str:
        return self.to_text_diagram()

    def _repr_args(self) -> str:
        args = []
        if self.moments:
            args.append(_list_repr_with_indented_item_lines(self.moments))
        return f'{", ".join(args)}'

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f'cirq.{cls_name}({self._repr_args()})'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Print ASCII diagram in Jupyter."""
        cls_name = self.__class__.__name__
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text(f'{cls_name}(...)')
        else:
            p.text(self.to_text_diagram())

    def _repr_html_(self) -> str:
        """Print ASCII diagram in Jupyter notebook without wrapping lines."""
        return (
            '<pre style="overflow: auto; white-space: pre;">'
            + html.escape(self.to_text_diagram())
            + '</pre>'
        )

    def _first_moment_operating_on(
        self, qubits: Iterable['cirq.Qid'], indices: Iterable[int]
    ) -> Optional[int]:
        qubits = frozenset(qubits)
        for m in indices:
            if self._has_op_at(m, qubits):
                return m
        return None

    def next_moment_operating_on(
        self,
        qubits: Iterable['cirq.Qid'],
        start_moment_index: int = 0,
        max_distance: Optional[int] = None,
    ) -> Optional[int]:
        """Finds the index of the next moment that touches the given qubits.

        Args:
            qubits: We're looking for operations affecting any of these qubits.
            start_moment_index: The starting point of the search.
            max_distance: The number of moments (starting from the start index
                and moving forward) to check. Defaults to no limit.

        Returns:
            None if there is no matching moment, otherwise the index of the
            earliest matching moment.

        Raises:
          ValueError: negative max_distance.
        """
        max_circuit_distance = len(self.moments) - start_moment_index
        if max_distance is None:
            max_distance = max_circuit_distance
        elif max_distance < 0:
            raise ValueError(f'Negative max_distance: {max_distance}')
        else:
            max_distance = min(max_distance, max_circuit_distance)

        return self._first_moment_operating_on(
            qubits, range(start_moment_index, start_moment_index + max_distance)
        )

    def next_moments_operating_on(
        self, qubits: Iterable['cirq.Qid'], start_moment_index: int = 0
    ) -> Dict['cirq.Qid', int]:
        """Finds the index of the next moment that touches each qubit.

        Args:
            qubits: The qubits to find the next moments acting on.
            start_moment_index: The starting point of the search.

        Returns:
            The index of the next moment that touches each qubit. If there
            is no such moment, the next moment is specified as the number of
            moments in the circuit. Equivalently, can be characterized as one
            plus the index of the last moment after start_moment_index
            (inclusive) that does *not* act on a given qubit.
        """
        next_moments = {}
        for q in qubits:
            next_moment = self.next_moment_operating_on([q], start_moment_index)
            next_moments[q] = len(self.moments) if next_moment is None else next_moment
        return next_moments

    def prev_moment_operating_on(
        self,
        qubits: Sequence['cirq.Qid'],
        end_moment_index: Optional[int] = None,
        max_distance: Optional[int] = None,
    ) -> Optional[int]:
        """Finds the index of the previous moment that touches the given qubits.

        Args:
            qubits: We're looking for operations affecting any of these qubits.
            end_moment_index: The moment index just after the starting point of
                the reverse search. Defaults to the length of the list of
                moments.
            max_distance: The number of moments (starting just before from the
                end index and moving backward) to check. Defaults to no limit.

        Returns:
            None if there is no matching moment, otherwise the index of the
            latest matching moment.

        Raises:
            ValueError: negative max_distance.
        """
        if end_moment_index is None:
            end_moment_index = len(self.moments)

        if max_distance is None:
            max_distance = len(self.moments)
        elif max_distance < 0:
            raise ValueError(f'Negative max_distance: {max_distance}')
        else:
            max_distance = min(end_moment_index, max_distance)

        # Don't bother searching indices past the end of the list.
        if end_moment_index > len(self.moments):
            d = end_moment_index - len(self.moments)
            end_moment_index -= d
            max_distance -= d
        if max_distance <= 0:
            return None

        return self._first_moment_operating_on(
            qubits, (end_moment_index - k - 1 for k in range(max_distance))
        )

    def reachable_frontier_from(
        self,
        start_frontier: Dict['cirq.Qid', int],
        *,
        is_blocker: Callable[['cirq.Operation'], bool] = lambda op: False,
    ) -> Dict['cirq.Qid', int]:
        """Determines how far can be reached into a circuit under certain rules.

        The location L = (qubit, moment_index) is *reachable* if and only if the
        following all hold true:

        - There is not a blocking operation covering L.
        -  At least one of the following holds:
            - qubit is in start frontier and moment_index =
                max(start_frontier[qubit], 0).
            - There is no operation at L and prev(L) = (qubit,
                moment_index-1) is reachable.
            - There is an (non-blocking) operation P covering L such that
                (q', moment_index - 1) is reachable for every q' on which P
                acts.

        An operation in moment moment_index is blocking if at least one of the
        following hold:

        - `is_blocker` returns a truthy value.
        - The operation acts on a qubit not in start_frontier.
        - The operation acts on a qubit q such that start_frontier[q] >
            moment_index.

        In other words, the reachable region extends forward through time along
        each qubit in start_frontier until it hits a blocking operation. Any
        location involving a qubit not in start_frontier is unreachable.

        For each qubit q in `start_frontier`, the reachable locations will
        correspond to a contiguous range starting at start_frontier[q] and
        ending just before some index end_q. The result of this method is a
        dictionary, and that dictionary maps each qubit q to its end_q.

        Examples:

        If `start_frontier` is

        ```
        {
            cirq.LineQubit(0): 6,
            cirq.LineQubit(1): 2,
            cirq.LineQubit(2): 2
        }
        ```

        then the reachable wire locations in the following circuit are
        highlighted with '█' characters:

        ```

                0   1   2   3   4   5   6   7   8   9   10  11  12  13
            0: ───H───@─────────────────█████████████████████─@───H───
                      │                                       │
            1: ───────@─██H███@██████████████████████─@───H───@───────
                              │                       │
            2: ─────────██████@███H██─@───────@───H───@───────────────
                                      │       │
            3: ───────────────────────@───H───@───────────────────────
        ```

        And the computed `end_frontier` is

        ```
        {
            cirq.LineQubit(0): 11,
            cirq.LineQubit(1): 9,
            cirq.LineQubit(2): 6,
        }
        ```

        Note that the frontier indices (shown above the circuit) are
        best thought of (and shown) as happening *between* moment indices.

        If we specify a blocker as follows:

        ```
        is_blocker=lambda: op == cirq.CZ(cirq.LineQubit(1),
                                         cirq.LineQubit(2))
        ```

        and use this `start_frontier`:

        ```
        {
            cirq.LineQubit(0): 0,
            cirq.LineQubit(1): 0,
            cirq.LineQubit(2): 0,
            cirq.LineQubit(3): 0,
        }
        ```

        Then this is the reachable area:

        ```

                0   1   2   3   4   5   6   7   8   9   10  11  12  13
            0: ─██H███@██████████████████████████████████████─@───H───
                      │                                       │
            1: ─██████@███H██─@───────────────────────@───H───@───────
                              │                       │
            2: ─█████████████─@───H───@───────@───H───@───────────────
                                      │       │
            3: ─█████████████████████─@───H───@───────────────────────

        ```

        and the computed `end_frontier` is:

        ```
        {
            cirq.LineQubit(0): 11,
            cirq.LineQubit(1): 3,
            cirq.LineQubit(2): 3,
            cirq.LineQubit(3): 5,
        }
        ```

        Args:
            start_frontier: A starting set of reachable locations.
            is_blocker: A predicate that determines if operations block
                reachability. Any location covered by an operation that causes
                `is_blocker` to return True is considered to be an unreachable
                location.

        Returns:
            An end_frontier dictionary, containing an end index for each qubit q
            mapped to a start index by the given `start_frontier` dictionary.

            To determine if a location (q, i) was reachable, you can use
            this expression:

                q in start_frontier and start_frontier[q] <= i < end_frontier[q]

            where i is the moment index, q is the qubit, and end_frontier is the
            result of this method.
        """
        active: Set['cirq.Qid'] = set()
        end_frontier = {}
        queue = BucketPriorityQueue[ops.Operation](drop_duplicate_entries=True)

        def enqueue_next(qubit: 'cirq.Qid', moment: int) -> None:
            next_moment = self.next_moment_operating_on([qubit], moment)
            if next_moment is None:
                end_frontier[qubit] = max(len(self), start_frontier[qubit])
                if qubit in active:
                    active.remove(qubit)
            else:
                next_op = self.operation_at(qubit, next_moment)
                assert next_op is not None
                queue.enqueue(next_moment, next_op)

        for start_qubit, start_moment in start_frontier.items():
            enqueue_next(start_qubit, start_moment)

        while queue:
            cur_moment, cur_op = queue.dequeue()
            for q in cur_op.qubits:
                if (
                    q in start_frontier
                    and cur_moment >= start_frontier[q]
                    and q not in end_frontier
                ):
                    active.add(q)

            continue_past = (
                cur_op is not None and active.issuperset(cur_op.qubits) and not is_blocker(cur_op)
            )
            if continue_past:
                for q in cur_op.qubits:
                    enqueue_next(q, cur_moment + 1)
            else:
                for q in cur_op.qubits:
                    if q in active:
                        end_frontier[q] = cur_moment
                        active.remove(q)

        return end_frontier

    def findall_operations_between(
        self,
        start_frontier: Dict['cirq.Qid', int],
        end_frontier: Dict['cirq.Qid', int],
        omit_crossing_operations: bool = False,
    ) -> List[Tuple[int, 'cirq.Operation']]:
        """Finds operations between the two given frontiers.

        If a qubit is in `start_frontier` but not `end_frontier`, its end index
        defaults to the end of the circuit. If a qubit is in `end_frontier` but
        not `start_frontier`, its start index defaults to the start of the
        circuit. Operations on qubits not mentioned in either frontier are not
        included in the results.

        Args:
            start_frontier: Just before where to start searching for operations,
                for each qubit of interest. Start frontier indices are
                inclusive.
            end_frontier: Just before where to stop searching for operations,
                for each qubit of interest. End frontier indices are exclusive.
            omit_crossing_operations: Determines whether or not operations that
                cross from a location between the two frontiers to a location
                outside the two frontiers are included or excluded. (Operations
                completely inside are always included, and operations completely
                outside are always excluded.)

        Returns:
            A list of tuples. Each tuple describes an operation found between
            the two frontiers. The first item of each tuple is the index of the
            moment containing the operation, and the second item is the
            operation itself. The list is sorted so that the moment index
            increases monotonically.
        """
        result = BucketPriorityQueue[ops.Operation](drop_duplicate_entries=True)

        involved_qubits = set(start_frontier.keys()) | set(end_frontier.keys())
        # Note: only sorted to ensure a deterministic result ordering.
        for q in sorted(involved_qubits):
            for i in range(start_frontier.get(q, 0), end_frontier.get(q, len(self))):
                op = self.operation_at(q, i)
                if op is None:
                    continue
                if omit_crossing_operations and not involved_qubits.issuperset(op.qubits):
                    continue
                result.enqueue(i, op)

        return list(result)

    def findall_operations_until_blocked(
        self,
        start_frontier: Dict['cirq.Qid', int],
        is_blocker: Callable[['cirq.Operation'], bool] = lambda op: False,
    ) -> List[Tuple[int, 'cirq.Operation']]:
        """Finds all operations until a blocking operation is hit.

        An operation is considered blocking if both of the following hold:

        - It is in the 'light cone' of start_frontier.
        - `is_blocker` returns a truthy value, or it acts on a blocked qubit

        Every qubit acted on by a blocking operation is thereafter itself
        blocked.

        The notion of reachability here differs from that in
        reachable_frontier_from in two respects:

        - An operation is not considered blocking only because it is in a
            moment before the start_frontier of one of the qubits on which it
            acts.
        - Operations that act on qubits not in start_frontier are not
            automatically blocking.

        For every (moment_index, operation) returned:

        - moment_index >= min((start_frontier[q] for q in operation.qubits
            if q in start_frontier), default=0)
        - set(operation.qubits).intersection(start_frontier)

        Below are some examples, where on the left the opening parentheses show
        `start_frontier` and on the right are the operations included (with
        their moment indices) in the output. `F` and `T` indicate that
        `is_blocker` return `False` or `True`, respectively, when applied to
        the gates; `M` indicates that it doesn't matter.

        ```
            ─(─F───F───────    ┄(─F───F─)┄┄┄┄┄
               │   │              │   │
            ─(─F───F───T─── => ┄(─F───F─)┄┄┄┄┄
                       │                  ┊
            ───────────T───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


            ───M─────(─F───    ┄┄┄┄┄┄┄┄┄(─F─)┄┄
               │       │          ┊       │
            ───M───M─(─F───    ┄┄┄┄┄┄┄┄┄(─F─)┄┄
                   │        =>        ┊
            ───────M───M───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
                       │                  ┊
            ───────────M───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


            ───M─(─────M───     ┄┄┄┄┄()┄┄┄┄┄┄┄┄
               │       │           ┊       ┊
            ───M─(─T───M───     ┄┄┄┄┄()┄┄┄┄┄┄┄┄
                   │        =>         ┊
            ───────T───M───     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
                       │                   ┊
            ───────────M───     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


            ─(─F───F───    ┄(─F───F─)┄
               │   │    =>    │   │
            ───F─(─F───    ┄(─F───F─)┄


            ─(─F───────────    ┄(─F─)┄┄┄┄┄┄┄┄┄
               │                  │
            ───F───F───────    ┄(─F─)┄┄┄┄┄┄┄┄┄
                   │        =>        ┊
            ───────F───F───    ┄┄┄┄┄┄┄┄┄(─F─)┄
                       │                  │
            ─(─────────F───    ┄┄┄┄┄┄┄┄┄(─F─)┄
        ```

        Args:
            start_frontier: A starting set of reachable locations.
            is_blocker: A predicate that determines if operations block
                reachability. Any location covered by an operation that causes
                `is_blocker` to return True is considered to be an unreachable
                location.

        Returns:
            A list of tuples. Each tuple describes an operation found between
            the start frontier and a blocking operation. The first item of
            each tuple is the index of the moment containing the operation,
            and the second item is the operation itself.

        """
        op_list: List[Tuple[int, ops.Operation]] = []
        if not start_frontier:
            return op_list
        start_index = min(start_frontier.values())
        blocked_qubits: Set[cirq.Qid] = set()
        for index, moment in enumerate(self[start_index:], start_index):
            active_qubits = set(q for q, s in start_frontier.items() if s <= index)
            for op in moment.operations:
                if is_blocker(op) or blocked_qubits.intersection(op.qubits):
                    blocked_qubits.update(op.qubits)
                elif active_qubits.intersection(op.qubits):
                    op_list.append((index, op))
            if blocked_qubits.issuperset(start_frontier):
                break
        return op_list

    def operation_at(self, qubit: 'cirq.Qid', moment_index: int) -> Optional['cirq.Operation']:
        """Finds the operation on a qubit within a moment, if any.

        Args:
            qubit: The qubit to check for an operation on.
            moment_index: The index of the moment to check for an operation
                within. Allowed to be beyond the end of the circuit.

        Returns:
            None if there is no operation on the qubit at the given moment, or
            else the operation.
        """
        if not 0 <= moment_index < len(self.moments):
            return None
        return self.moments[moment_index].operation_at(qubit)

    def findall_operations(
        self, predicate: Callable[['cirq.Operation'], bool]
    ) -> Iterable[Tuple[int, 'cirq.Operation']]:
        """Find the locations of all operations that satisfy a given condition.

        This returns an iterator of (index, operation) tuples where each
        operation satisfies op_cond(operation) is truthy. The indices are
        in order of the moments and then order of the ops within that moment.

        Args:
            predicate: A method that takes an Operation and returns a Truthy
                value indicating the operation meets the find condition.

        Returns:
            An iterator (index, operation)'s that satisfy the op_condition.
        """
        for index, moment in enumerate(self.moments):
            for op in moment.operations:
                if predicate(op):
                    yield index, op

    def findall_operations_with_gate_type(
        self, gate_type: Type[_TGate]
    ) -> Iterable[Tuple[int, 'cirq.GateOperation', _TGate]]:
        """Find the locations of all gate operations of a given type.

        Args:
            gate_type: The type of gate to find, e.g. XPowGate or
                MeasurementGate.

        Returns:
            An iterator (index, operation, gate)'s for operations with the given
            gate type.
        """
        result = self.findall_operations(lambda operation: isinstance(operation.gate, gate_type))
        for index, op in result:
            gate_op = cast(ops.GateOperation, op)
            yield index, gate_op, cast(_TGate, gate_op.gate)

    def has_measurements(self):
        """Returns whether or not this circuit has measurements.

        Returns: True if `cirq.is_measurement(self)` is True otherwise False.
        """
        return protocols.is_measurement(self)

    def _is_measurement_(self) -> bool:
        return any(protocols.is_measurement(op) for op in self.all_operations())

    def are_all_measurements_terminal(self) -> bool:
        """Whether all measurement gates are at the end of the circuit.

        Returns: True iff no measurement is followed by a gate.
        """
        return self.are_all_matches_terminal(protocols.is_measurement)

    def are_all_matches_terminal(self, predicate: Callable[['cirq.Operation'], bool]) -> bool:
        """Check whether all of the ops that satisfy a predicate are terminal.

        This method will transparently descend into any CircuitOperations this
        circuit contains; as a result, it will misbehave if the predicate
        refers to CircuitOperations. See the tests for an example of this.

        Args:
            predicate: A predicate on ops.Operations which is being checked.

        Returns:
            Whether or not all `Operation` s in a circuit that satisfy the
            given predicate are terminal. Also checks within any CircuitGates
            the circuit may contain.
        """
        from cirq.circuits import CircuitOperation

        if not all(
            self.next_moment_operating_on(op.qubits, i + 1) is None
            for (i, op) in self.findall_operations(predicate)
            if not isinstance(op.untagged, CircuitOperation)
        ):
            return False

        for i, moment in enumerate(self.moments):
            for op in moment.operations:
                circuit = getattr(op.untagged, 'circuit', None)
                if circuit is None:
                    continue
                if not circuit.are_all_matches_terminal(predicate):
                    return False
                if i < len(self.moments) - 1 and not all(
                    self.next_moment_operating_on(op.qubits, i + 1) is None
                    for _, op in circuit.findall_operations(predicate)
                ):
                    return False
        return True

    def are_any_measurements_terminal(self) -> bool:
        """Whether any measurement gates are at the end of the circuit.

        Returns: True iff some measurements are not followed by a gate.
        """
        return self.are_any_matches_terminal(protocols.is_measurement)

    def are_any_matches_terminal(self, predicate: Callable[['cirq.Operation'], bool]) -> bool:
        """Check whether any of the ops that satisfy a predicate are terminal.

        This method will transparently descend into any CircuitOperations this
        circuit contains; as a result, it will misbehave if the predicate
        refers to CircuitOperations. See the tests for an example of this.

        Args:
            predicate: A predicate on ops.Operations which is being checked.

        Returns:
            Whether or not any `Operation` s in a circuit that satisfy the
            given predicate are terminal. Also checks within any CircuitGates
            the circuit may contain.
        """
        from cirq.circuits import CircuitOperation

        if any(
            self.next_moment_operating_on(op.qubits, i + 1) is None
            for (i, op) in self.findall_operations(predicate)
            if not isinstance(op.untagged, CircuitOperation)
        ):
            return True

        for i, moment in reversed(list(enumerate(self.moments))):
            for op in moment.operations:
                circuit = getattr(op.untagged, 'circuit', None)
                if circuit is None:
                    continue
                if not circuit.are_any_matches_terminal(predicate):
                    continue
                if i == len(self.moments) - 1 or any(
                    self.next_moment_operating_on(op.qubits, i + 1) is None
                    for _, op in circuit.findall_operations(predicate)
                ):
                    return True
        return False

    def _has_op_at(self, moment_index: int, qubits: Iterable['cirq.Qid']) -> bool:
        return 0 <= moment_index < len(self.moments) and self.moments[moment_index].operates_on(
            qubits
        )

    def all_qubits(self) -> FrozenSet['cirq.Qid']:
        """Returns the qubits acted upon by Operations in this circuit.

        Returns: FrozenSet of `cirq.Qid` objects acted on by all operations
            in this circuit.
        """
        return frozenset(q for m in self.moments for q in m.qubits)

    def all_operations(self) -> Iterator['cirq.Operation']:
        """Returns an iterator over the operations in the circuit.

        Returns: Iterator over `cirq.Operation` elements found in this circuit.
        """
        return (op for moment in self for op in moment.operations)

    def map_operations(self, func: Callable[['cirq.Operation'], 'cirq.OP_TREE']) -> Self:
        """Applies the given function to all operations in this circuit.

        Args:
            func: a mapping function from operations to OP_TREEs.

        Returns:
            A circuit with the same basic structure as the original, but with
            each operation `op` replaced with `func(op)`.
        """

        def map_moment(moment: 'cirq.Moment') -> 'cirq.Circuit':
            """Apply func to expand each op into a circuit, then zip up the circuits."""
            return Circuit.zip(*[Circuit(func(op)) for op in moment])

        return self._from_moments(m for moment in self for m in map_moment(moment))

    def qid_shape(
        self, qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT
    ) -> Tuple[int, ...]:
        """Get the qubit shapes of all qubits in this circuit.

        Returns: A tuple containing the dimensions (shape) of all qudits
            found in this circuit according to `qubit_order`.
        """
        qids = ops.QubitOrder.as_qubit_order(qubit_order).order_for(self.all_qubits())
        return protocols.qid_shape(qids)

    def all_measurement_key_objs(self) -> FrozenSet['cirq.MeasurementKey']:
        return frozenset(
            key for op in self.all_operations() for key in protocols.measurement_key_objs(op)
        )

    def _measurement_key_objs_(self) -> FrozenSet['cirq.MeasurementKey']:
        """Returns the set of all measurement keys in this circuit.

        Returns: FrozenSet of `cirq.MeasurementKey` objects that are
            in this circuit.
        """
        return self.all_measurement_key_objs()

    def all_measurement_key_names(self) -> FrozenSet[str]:
        """Returns the set of all measurement key names in this circuit.

        Returns: FrozenSet of strings that are the measurement key
            names in this circuit.
        """
        return frozenset(
            key for op in self.all_operations() for key in protocols.measurement_key_names(op)
        )

    def _measurement_key_names_(self) -> FrozenSet[str]:
        return self.all_measurement_key_names()

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]):
        return self._from_moments(
            protocols.with_measurement_key_mapping(moment, key_map) for moment in self.moments
        )

    def _with_key_path_(self, path: Tuple[str, ...]):
        return self._from_moments(protocols.with_key_path(moment, path) for moment in self.moments)

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
        return self._from_moments(
            protocols.with_key_path_prefix(moment, prefix) for moment in self.moments
        )

    def _with_rescoped_keys_(
        self, path: Tuple[str, ...], bindable_keys: FrozenSet['cirq.MeasurementKey']
    ):
        moments = []
        for moment in self.moments:
            new_moment = protocols.with_rescoped_keys(moment, path, bindable_keys)
            moments.append(new_moment)
            bindable_keys |= protocols.measurement_key_objs(new_moment)
        return self._from_moments(moments)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self.qid_shape()

    def _has_unitary_(self) -> bool:
        if not self.are_all_measurements_terminal():
            return False

        unitary_ops = protocols.decompose(
            self.all_operations(),
            keep=protocols.has_unitary,
            intercepting_decomposer=_decompose_measurement_inversions,
            on_stuck_raise=None,
        )
        return all(protocols.has_unitary(e) for e in unitary_ops)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        """Converts the circuit into a unitary matrix, if possible.

        If the circuit contains any non-terminal measurements, the conversion
        into a unitary matrix fails (i.e. returns NotImplemented). Terminal
        measurements are ignored when computing the unitary matrix. The unitary
        matrix is the product of the unitary matrix of all operations in the
        circuit (after expanding them to apply to the whole system).
        """
        if not self._has_unitary_():
            return NotImplemented
        return self.unitary(ignore_terminal_measurements=True)

    def unitary(
        self,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        qubits_that_should_be_present: Iterable['cirq.Qid'] = (),
        ignore_terminal_measurements: bool = True,
        dtype: Type[np.complexfloating] = np.complex128,
    ) -> np.ndarray:
        """Converts the circuit into a unitary matrix, if possible.

        Returns the same result as `cirq.unitary`, but provides more options.

        Args:
            qubit_order: Determines how qubits are ordered when passing matrices
                into np.kron.
            qubits_that_should_be_present: Qubits that may or may not appear
                in operations within the circuit, but that should be included
                regardless when generating the matrix.
            ignore_terminal_measurements: When set, measurements at the end of
                the circuit are ignored instead of causing the method to
                fail.
            dtype: The numpy dtype for the returned unitary. Defaults to
                np.complex128. Specifying np.complex64 will run faster at the
                cost of precision. `dtype` must be a complex np.dtype, unless
                all operations in the circuit have unitary matrices with
                exclusively real coefficients (e.g. an H + TOFFOLI circuit).

        Returns:
            A (possibly gigantic) 2d numpy array corresponding to a matrix
            equivalent to the circuit's effect on a quantum state.

        Raises:
            ValueError: The circuit contains measurement gates that are not
                ignored.
            TypeError: The circuit contains gates that don't have a known
                unitary matrix, e.g. gates parameterized by a Symbol.
        """

        if not ignore_terminal_measurements and any(
            protocols.is_measurement(op) for op in self.all_operations()
        ):
            raise ValueError('Circuit contains a measurement.')

        if not self.are_all_measurements_terminal():
            raise ValueError('Circuit contains a non-terminal measurement.')

        qs = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            self.all_qubits().union(qubits_that_should_be_present)
        )

        # Force qubits to have dimension at least 2 for backwards compatibility.
        qid_shape = self.qid_shape(qubit_order=qs)
        side_len = np.prod(qid_shape, dtype=np.int64)

        state = qis.eye_tensor(qid_shape, dtype=dtype)

        result = _apply_unitary_circuit(self, state, qs, dtype)
        return result.reshape((side_len, side_len))

    def _has_superoperator_(self) -> bool:
        """Returns True if self has superoperator representation."""
        return all(m._has_superoperator_() for m in self)

    def _superoperator_(self) -> np.ndarray:
        """Compute superoperator matrix for quantum channel specified by this circuit."""
        all_qubits = self.all_qubits()
        n = len(all_qubits)
        if n > 10:
            raise ValueError(f"{n} > 10 qubits is too many to compute superoperator")

        circuit_superoperator = np.eye(4**n)
        for moment in self:
            full_moment = moment.expand_to(all_qubits)
            moment_superoperator = full_moment._superoperator_()
            circuit_superoperator = moment_superoperator @ circuit_superoperator
        return circuit_superoperator

    def final_state_vector(
        self,
        *,
        initial_state: 'cirq.STATE_VECTOR_LIKE' = 0,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        ignore_terminal_measurements: bool = False,
        dtype: Type[np.complexfloating] = np.complex128,
        param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        """Returns the state vector resulting from acting operations on a state.

        This is equivalent to calling cirq.final_state_vector with the same
        arguments and this circuit as the "program".

        Args:
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state. Otherwise  if this
                is a np.ndarray it is the full initial state. In this case it
                must be the correct size, be normalized (an L2 norm of 1), and
                be safely castable to an appropriate dtype for the simulator.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            qubits_that_should_be_present: Qubits that may or may not appear
                in operations within the circuit, but that should be included
                regardless when generating the matrix.
            ignore_terminal_measurements: When set, measurements at the end of
                the circuit are ignored instead of causing the method to
                fail. Defaults to False.
            dtype: The `numpy.dtype` used by the simulation. Typically one of
                `numpy.complex64` or `numpy.complex128`.
            param_resolver: Parameters to run with the program.
            seed: The random seed to use for this simulator.

        Returns:
            The state vector resulting from applying the given unitary
            operations to the desired initial state. Specifically, a numpy
            array containing the amplitudes in np.kron order, where the
            order of arguments to kron is determined by the qubit order
            argument (which defaults to just sorting the qubits that are
            present into an ascending order).

        Raises:
            ValueError: If the program doesn't have a well defined final state
                because it has non-unitary gates.
        """
        from cirq.sim.mux import final_state_vector

        return final_state_vector(
            self,
            initial_state=initial_state,
            param_resolver=param_resolver,
            qubit_order=qubit_order,
            ignore_terminal_measurements=ignore_terminal_measurements,
            dtype=dtype,
            seed=seed,
        )

    def to_text_diagram(
        self,
        *,
        use_unicode_characters: bool = True,
        transpose: bool = False,
        include_tags: bool = True,
        precision: Optional[int] = 3,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ) -> str:
        """Returns text containing a diagram describing the circuit.

        Args:
            use_unicode_characters: Determines if unicode characters are
                allowed (as opposed to ascii-only diagrams).
            transpose: Arranges qubit wires vertically instead of horizontally.
            include_tags: Whether tags on TaggedOperations should be printed
            precision: Number of digits to display in text diagram
            qubit_order: Determines how qubits are ordered in the diagram.

        Returns:
            The text diagram.
        """
        diagram = self.to_text_diagram_drawer(
            use_unicode_characters=use_unicode_characters,
            include_tags=include_tags,
            precision=precision,
            qubit_order=qubit_order,
            transpose=transpose,
        )

        return diagram.render(
            crossing_char=(None if use_unicode_characters else ('-' if transpose else '|')),
            horizontal_spacing=1 if transpose else 3,
            use_unicode_characters=use_unicode_characters,
        )

    def to_text_diagram_drawer(
        self,
        *,
        use_unicode_characters: bool = True,
        qubit_namer: Optional[Callable[['cirq.Qid'], str]] = None,
        transpose: bool = False,
        include_tags: bool = True,
        draw_moment_groups: bool = True,
        precision: Optional[int] = 3,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        get_circuit_diagram_info: Optional[
            Callable[['cirq.Operation', 'cirq.CircuitDiagramInfoArgs'], 'cirq.CircuitDiagramInfo']
        ] = None,
    ) -> 'cirq.TextDiagramDrawer':
        """Returns a TextDiagramDrawer with the circuit drawn into it.

        Args:
            use_unicode_characters: Determines if unicode characters are
                allowed (as opposed to ascii-only diagrams).
            qubit_namer: Names qubits in diagram. Defaults to using _circuit_diagram_info_ or str.
            transpose: Arranges qubit wires vertically instead of horizontally.
            include_tags: Whether to include tags in the operation.
            draw_moment_groups: Whether to draw moment symbol or not
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the diagram.
            get_circuit_diagram_info: Gets circuit diagram info. Defaults to
                protocol with fallback.

        Returns:
            The TextDiagramDrawer instance.
        """
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(self.all_qubits())
        cbits = tuple(
            sorted(
                set(key for op in self.all_operations() for key in protocols.control_keys(op)),
                key=str,
            )
        )
        labels = qubits + cbits
        label_map = {labels[i]: i for i in range(len(labels))}

        def default_namer(label_entity):
            info = protocols.circuit_diagram_info(label_entity, default=None)
            qubit_name = info.wire_symbols[0] if info else str(label_entity)
            return qubit_name + ('' if transpose else ': ')

        if qubit_namer is None:
            qubit_namer = default_namer
        diagram = TextDiagramDrawer()
        diagram.write(0, 0, '')
        for label_entity, i in label_map.items():
            name = (
                qubit_namer(label_entity)
                if isinstance(label_entity, ops.Qid)
                else default_namer(label_entity)
            )
            diagram.write(0, i, name)
        first_annotation_row = max(label_map.values(), default=0) + 1

        if any(isinstance(op.gate, cirq.GlobalPhaseGate) for op in self.all_operations()):
            diagram.write(0, max(label_map.values(), default=0) + 1, 'global phase:')
            first_annotation_row += 1

        moment_groups: List[Tuple[int, int]] = []
        for moment in self.moments:
            _draw_moment_in_diagram(
                moment=moment,
                use_unicode_characters=use_unicode_characters,
                label_map=label_map,
                out_diagram=diagram,
                precision=precision,
                moment_groups=moment_groups,
                get_circuit_diagram_info=get_circuit_diagram_info,
                include_tags=include_tags,
                first_annotation_row=first_annotation_row,
                transpose=transpose,
            )

        w = diagram.width()
        for i in label_map.values():
            diagram.horizontal_line(i, 0, w, doubled=not isinstance(labels[i], ops.Qid))

        if moment_groups and draw_moment_groups:
            _draw_moment_groups_in_diagram(moment_groups, use_unicode_characters, diagram)

        if transpose:
            diagram = diagram.transpose()

        return diagram

    def _is_parameterized_(self) -> bool:
        return any(protocols.is_parameterized(op) for op in self.all_operations())

    def _parameter_names_(self) -> AbstractSet[str]:
        return {name for op in self.all_operations() for name in protocols.parameter_names(op)}

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> Self:
        changed = False
        resolved_moments: List['cirq.Moment'] = []
        for moment in self:
            resolved_moment = protocols.resolve_parameters(moment, resolver, recursive)
            if resolved_moment is not moment:
                changed = True
            resolved_moments.append(resolved_moment)
        if not changed:
            return self
        return self._from_moments(resolved_moments)

    def _qasm_(self, args: Optional['cirq.QasmArgs'] = None) -> str:
        if args is None:
            output = self._to_qasm_output()
        else:
            output = self._to_qasm_output(precision=args.precision, version=args.version)
        return str(output)

    def _to_qasm_output(
        self,
        header: Optional[str] = None,
        precision: int = 10,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        version: str = '2.0',
    ) -> 'cirq.QasmOutput':
        """Returns a QASM object equivalent to the circuit.

        Args:
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
            version:  Version of OpenQASM to render as output.  Defaults
                to OpenQASM 2.0.  For OpenQASM 3.0, set this to '3.0'.
        """
        if header is None:
            header = f'Generated from Cirq v{cirq._version.__version__}'
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(self.all_qubits())
        return QasmOutput(
            operations=self.all_operations(),
            qubits=qubits,
            header=header,
            precision=precision,
            version=version,
        )

    def to_qasm(
        self,
        header: Optional[str] = None,
        precision: int = 10,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
        version: str = '2.0',
    ) -> str:
        """Returns QASM equivalent to the circuit.

        Args:
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
            version: Version of OpenQASM to output.  Defaults to OpenQASM 2.0.
                Specify '3.0' if OpenQASM 3.0 is desired.
        """

        return str(self._to_qasm_output(header, precision, qubit_order, version))

    def save_qasm(
        self,
        file_path: Union[str, bytes, int],
        header: Optional[str] = None,
        precision: int = 10,
        qubit_order: 'cirq.QubitOrderOrList' = ops.QubitOrder.DEFAULT,
    ) -> None:
        """Save a QASM file equivalent to the circuit.

        Args:
            file_path: The location of the file where the qasm will be written.
            header: A multi-line string that is placed in a comment at the top
                of the QASM. Defaults to a cirq version specifier.
            precision: Number of digits to use when representing numbers.
            qubit_order: Determines how qubits are ordered in the QASM
                register.
        """
        self._to_qasm_output(header, precision, qubit_order).save(file_path)

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['moments'])

    @classmethod
    def _from_json_dict_(cls, moments, **kwargs):
        return cls(moments, strategy=InsertStrategy.EARLIEST)

    def zip(
        *circuits: 'cirq.AbstractCircuit', align: Union['cirq.Alignment', str] = Alignment.LEFT
    ) -> 'cirq.AbstractCircuit':
        """Combines operations from circuits in a moment-by-moment fashion.

        Moment k of the resulting circuit will have all operations from moment
        k of each of the given circuits.

        When the given circuits have different lengths, the shorter circuits are
        implicitly padded with empty moments. This differs from the behavior of
        python's built-in zip function, which would instead truncate the longer
        circuits.

        The zipped circuits can't have overlapping operations occurring at the
        same moment index.

        Args:
            *circuits: The circuits to merge together.
            align: The alignment for the zip, see `cirq.Alignment`.

        Returns:
            The merged circuit.

        Raises:
            ValueError: If the zipped circuits have overlapping operations occurring
                at the same moment index.

        Examples:

        >>> import cirq
        >>> a, b, c, d = cirq.LineQubit.range(4)
        >>> circuit1 = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b))
        >>> circuit2 = cirq.Circuit(cirq.X(c), cirq.Y(c), cirq.Z(c))
        >>> circuit3 = cirq.Circuit(cirq.Moment(), cirq.Moment(cirq.S(d)))
        >>> print(circuit1.zip(circuit2))
        0: ───H───@───────
                  │
        1: ───────X───────
        <BLANKLINE>
        2: ───X───Y───Z───
        >>> print(circuit1.zip(circuit2, circuit3))
        0: ───H───@───────
                  │
        1: ───────X───────
        <BLANKLINE>
        2: ───X───Y───Z───
        <BLANKLINE>
        3: ───────S───────
        >>> print(cirq.Circuit.zip(circuit3, circuit2, circuit1))
        0: ───H───@───────
                  │
        1: ───────X───────
        <BLANKLINE>
        2: ───X───Y───Z───
        <BLANKLINE>
        3: ───────S───────
        """
        n = max([len(c) for c in circuits], default=0)

        if isinstance(align, str):
            align = Alignment[align.upper()]

        result = cirq.Circuit()
        for k in range(n):
            try:
                if align == Alignment.LEFT:
                    moment = cirq.Moment(c[k] for c in circuits if k < len(c))
                else:
                    moment = cirq.Moment(c[len(c) - n + k] for c in circuits if len(c) - n + k >= 0)
                result.append(moment)
            except ValueError as ex:
                raise ValueError(
                    f"Overlapping operations between zipped circuits at moment index {k}.\n{ex}"
                ) from ex
        return result

    def concat_ragged(
        *circuits: 'cirq.AbstractCircuit', align: Union['cirq.Alignment', str] = Alignment.LEFT
    ) -> 'cirq.AbstractCircuit':
        """Concatenates circuits, overlapping them if possible due to ragged edges.

        Starts with the first circuit (index 0), then iterates over the other
        circuits while folding them in. To fold two circuits together, they
        are placed one after the other and then moved inward until just before
        their operations would collide. If any of the circuits do not share
        qubits and so would not collide, the starts or ends of the circuits will
        be aligned, according to the given align parameter.

        Beware that this method is *not* associative. For example:

        >>> a, b = cirq.LineQubit.range(2)
        >>> A = cirq.Circuit(cirq.H(a))
        >>> B = cirq.Circuit(cirq.H(b))
        >>> f = cirq.Circuit.concat_ragged
        >>> f(f(A, B), A) == f(A, f(B, A))
        False
        >>> len(f(f(f(A, B), A), B)) == len(f(f(A, f(B, A)), B))
        False

        Args:
            *circuits: The circuits to concatenate.
            align: When to stop when sliding the circuits together.
                'left': Stop when the starts of the circuits align.
                'right': Stop when the ends of the circuits align.
                'first': Stop the first time either the starts or the ends align. Circuits
                    are never overlapped more than needed to align their starts (in case
                    the left circuit is smaller) or to align their ends (in case the right
                    circuit is smaller)

        Returns:
            The concatenated and overlapped circuit.
        """
        if len(circuits) == 0:
            return Circuit()
        n_acc = len(circuits[0])

        if isinstance(align, str):
            align = Alignment[align.upper()]

        # Allocate a buffer large enough to append and prepend all the circuits.
        pad_len = sum(len(c) for c in circuits) - n_acc
        buffer: MutableSequence['cirq.Moment'] = [cirq.Moment()] * (pad_len * 2 + n_acc)

        # Put the initial circuit in the center of the buffer.
        offset = pad_len
        buffer[offset : offset + n_acc] = circuits[0].moments

        # Accumulate all the circuits into the buffer.
        for k in range(1, len(circuits)):
            offset, n_acc = _concat_ragged_helper(offset, n_acc, buffer, circuits[k].moments, align)

        return cirq.Circuit(buffer[offset : offset + n_acc])

    def get_independent_qubit_sets(self) -> List[Set['cirq.Qid']]:
        """Divide circuit's qubits into independent qubit sets.

        Independent qubit sets are the qubit sets such that there are
        no entangling gates between qubits belonging to different sets.
        If this is not possible, a sequence with a single factor (the whole set of
        circuit's qubits) is returned.

        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> circuit = cirq.Circuit()
        >>> circuit.append(cirq.Moment(cirq.H(q2)))
        >>> circuit.append(cirq.Moment(cirq.CZ(q0,q1)))
        >>> circuit.append(cirq.H(q0))
        >>> print(circuit)
        0: ───────@───H───
                  │
        1: ───────@───────
        <BLANKLINE>
        2: ───H───────────
        >>> [sorted(qs) for qs in circuit.get_independent_qubit_sets()]
        [[cirq.LineQubit(0), cirq.LineQubit(1)], [cirq.LineQubit(2)]]

        Returns:
            The list of independent qubit sets.

        """
        uf = networkx.utils.UnionFind(self.all_qubits())
        for op in self.all_operations():
            if len(op.qubits) > 1:
                uf.union(*op.qubits)
        return sorted([qs for qs in uf.to_sets()], key=min)

    def factorize(self) -> Iterable[Self]:
        """Factorize circuit into a sequence of independent circuits (factors).

        Factorization is possible when the circuit's qubits can be divided
        into two or more independent qubit sets. Preserves the moments from
        the original circuit.
        If this is not possible, returns the set consisting of the single
        circuit (this one).

        >>> q0, q1, q2 = cirq.LineQubit.range(3)
        >>> circuit = cirq.Circuit()
        >>> circuit.append(cirq.Moment(cirq.H(q2)))
        >>> circuit.append(cirq.Moment(cirq.CZ(q0,q1)))
        >>> circuit.append(cirq.H(q0))
        >>> print(circuit)
        0: ───────@───H───
                  │
        1: ───────@───────
        <BLANKLINE>
        2: ───H───────────
        >>> for i, f in enumerate(circuit.factorize()):
        ...     print("Factor {}".format(i))
        ...     print(f)
        ...
        Factor 0
        0: ───────@───H───
                  │
        1: ───────@───────
        Factor 1
        2: ───H───────────

        Returns:
            The sequence of circuits, each including only the qubits from one
            independent qubit set.

        """

        qubit_factors = self.get_independent_qubit_sets()
        if len(qubit_factors) == 1:
            return (self,)
        # Note: In general, Moment.__getitem__ returns all operations on which
        # any of the qubits operate. However, in this case we know that all of
        # the qubits from one factor belong to a specific independent qubit set.
        # This makes it possible to create independent circuits based on these
        # moments.
        return (self._from_moments(m[qubits] for m in self.moments) for qubits in qubit_factors)

    def _control_keys_(self) -> FrozenSet['cirq.MeasurementKey']:
        controls = frozenset(k for op in self.all_operations() for k in protocols.control_keys(op))
        return controls - protocols.measurement_key_objs(self)


def _overlap_collision_time(
    c1: Sequence['cirq.Moment'], c2: Sequence['cirq.Moment'], align: 'cirq.Alignment'
) -> int:
    # Tracks the first used moment index for each qubit in c2.
    # Tracks the complementary last used moment index for each qubit in c1.
    seen_times: Dict['cirq.Qid', int] = {}

    # Start scanning from end of first and start of second.
    if align == Alignment.LEFT:
        upper_bound = len(c1)
    elif align == Alignment.RIGHT:
        upper_bound = len(c2)
    elif align == Alignment.FIRST:
        upper_bound = min(len(c1), len(c2))
    else:
        raise NotImplementedError(f"Unrecognized alignment: {align}")

    t = 0
    while t < upper_bound:
        if t < len(c2):
            for op in c2[t]:
                for q in op.qubits:
                    # Record time but check if qubit already seen on other side.
                    k2 = seen_times.setdefault(q, t)
                    if k2 < 0:
                        # Use this qubit collision to bound the collision time.
                        upper_bound = min(upper_bound, t + ~k2)
        if t < len(c1):
            for op in c1[-1 - t]:
                for q in op.qubits:
                    # Record time but check if qubit already seen on other side.
                    # Note t is bitwise complemented to pack in left-vs-right origin data.
                    k2 = seen_times.setdefault(q, ~t)
                    if k2 >= 0:
                        # Use this qubit collision to bound the collision time.
                        upper_bound = min(upper_bound, t + k2)
        t += 1
    return upper_bound


def _concat_ragged_helper(
    c1_offset: int,
    n1: int,
    buf: MutableSequence['cirq.Moment'],
    c2: Sequence['cirq.Moment'],
    align: 'cirq.Alignment',
) -> Tuple[int, int]:
    n2 = len(c2)
    shift = _overlap_collision_time(buf[c1_offset : c1_offset + n1], c2, align)
    c2_offset = c1_offset + n1 - shift
    for k in range(n2):
        buf[k + c2_offset] = (buf[k + c2_offset] or Moment()) + c2[k]
    return min(c1_offset, c2_offset), max(n1, n2, n1 + n2 - shift)


class Circuit(AbstractCircuit):
    """A mutable list of groups of operations to apply to some qubits.

    Methods returning information about the circuit (inherited from
    AbstractCircuit):

    *   next_moment_operating_on
    *   earliest_available_moment
    *   prev_moment_operating_on
    *   next_moments_operating_on
    *   operation_at
    *   all_qubits
    *   all_operations
    *   findall_operations
    *   findall_operations_between
    *   findall_operations_until_blocked
    *   findall_operations_with_gate_type
    *   reachable_frontier_from
    *   has_measurements
    *   are_all_matches_terminal
    *   are_all_measurements_terminal
    *   unitary
    *   final_state_vector
    *   to_text_diagram
    *   to_text_diagram_drawer
    *   qid_shape
    *   all_measurement_key_names
    *   to_quil
    *   to_qasm
    *   save_qasm
    *   get_independent_qubit_sets

    Methods for mutation:

    *   insert
    *   append
    *   insert_into_range
    *   clear_operations_touching
    *   batch_insert
    *   batch_remove
    *   batch_insert_into
    *   insert_at_frontier

    Circuits can also be iterated over,

    ```
        for moment in circuit:
            ...
    ```

    and sliced,

    *   `circuit[1:3]` is a new Circuit made up of two moments, the first being
            `circuit[1]` and the second being `circuit[2]`;
    *   `circuit[:, qubit]` is a new Circuit with the same moments, but with
            only those operations which act on the given Qubit;
    *   `circuit[:, qubits]`, where 'qubits' is list of Qubits, is a new Circuit
            with the same moments, but only with those operations which touch
            any of the given qubits;
    *   `circuit[1:3, qubit]` is equivalent to `circuit[1:3][:, qubit]`;
    *   `circuit[1:3, qubits]` is equivalent to `circuit[1:3][:, qubits]`;

    and concatenated,

    *    `circuit1 + circuit2` is a new Circuit made up of the moments in
            circuit1 followed by the moments in circuit2;

    and multiplied by an integer,

    *    `circuit * k` is a new Circuit made up of the moments in circuit repeated
            k times.

    and mutated,
    *    `circuit[1:7] = [Moment(...)]`

    and factorized,
    *   `circuit.factorize()` returns a sequence of Circuits which represent
            independent 'factors' of the original Circuit.
    """

    def __init__(
        self, *contents: 'cirq.OP_TREE', strategy: 'cirq.InsertStrategy' = InsertStrategy.EARLIEST
    ) -> None:
        """Initializes a circuit.

        Args:
            contents: The initial list of moments and operations defining the
                circuit. You can also pass in operations, lists of operations,
                or generally anything meeting the `cirq.OP_TREE` contract.
                Non-moment entries will be inserted according to the specified
                insertion strategy.
            strategy: When initializing the circuit with operations and moments
                from `contents`, this determines how the operations are packed
                together. This option does not affect later insertions into the
                circuit.
        """
        self._placement_cache: Optional[_PlacementCache] = _PlacementCache()
        self._moments: List['cirq.Moment'] = []

        # Implementation note: the following cached properties are set lazily and then
        # invalidated and reset to None in `self._mutated()`, which is called any time
        # `self._moments` is changed.
        self._all_qubits: Optional[FrozenSet['cirq.Qid']] = None
        self._frozen: Optional['cirq.FrozenCircuit'] = None
        self._is_measurement: Optional[bool] = None
        self._is_parameterized: Optional[bool] = None
        self._parameter_names: Optional[AbstractSet[str]] = None
        if not contents:
            return
        flattened_contents = tuple(ops.flatten_to_ops_or_moments(contents))
        if all(isinstance(c, Moment) for c in flattened_contents):
            self._placement_cache = None
            self._moments[:] = cast(Iterable[Moment], flattened_contents)
            return
        with _compat.block_overlapping_deprecation('.*'):
            if strategy == InsertStrategy.EARLIEST:
                self._load_contents_with_earliest_strategy(flattened_contents)
            else:
                self.append(flattened_contents, strategy=strategy)

    def _mutated(self, *, preserve_placement_cache=False) -> None:
        """Clear cached properties in response to this circuit being mutated."""
        self._all_qubits = None
        self._frozen = None
        self._is_measurement = None
        self._is_parameterized = None
        self._parameter_names = None
        if not preserve_placement_cache:
            self._placement_cache = None

    @classmethod
    def _from_moments(cls, moments: Iterable['cirq.Moment']) -> 'Circuit':
        new_circuit = Circuit()
        new_circuit._moments[:] = moments
        new_circuit._placement_cache = None
        return new_circuit

    def _load_contents_with_earliest_strategy(self, contents: 'cirq.OP_TREE'):
        """Optimized algorithm to load contents quickly.

        The default algorithm appends operations one-at-a-time, letting them
        fall back until they encounter a moment they cannot commute with. This
        is slow because it requires re-checking for conflicts at each moment.

        Here, we instead keep track of the greatest moment that contains each
        qubit, measurement key, and control key, and append the operation to
        the moment after the maximum of these. This avoids having to check each
        moment.

        Args:
            contents: The initial list of moments and operations defining the
                circuit. You can also pass in operations, lists of operations,
                or generally anything meeting the `cirq.OP_TREE` contract.
                Non-moment entries will be inserted according to the EARLIEST
                insertion strategy.
        """
        # PlacementCache holds dicts from the qubit/key to the greatest moment index that has it.
        placement_cache = cast(_PlacementCache, self._placement_cache)

        # We also maintain the dict from moment index to moments/ops that go into it, for use when
        # building the actual moments at the end.
        op_lists_by_index: Dict[int, List['cirq.Operation']] = defaultdict(list)
        moments_by_index: Dict[int, 'cirq.Moment'] = {}

        # "mop" means current moment-or-operation
        for mop in ops.flatten_to_ops_or_moments(contents):
            # Identify the index of the moment to place this `mop` into.
            placement_index = placement_cache.append(mop)
            if isinstance(mop, Moment):
                moments_by_index[placement_index] = mop
            else:
                op_lists_by_index[placement_index].append(mop)

        # Finally, once everything is placed, we can construct and append the actual moments for
        # each index.
        for i in range(placement_cache._length):
            if i in moments_by_index:
                self._moments.append(moments_by_index[i].with_operations(op_lists_by_index[i]))
            else:
                self._moments.append(Moment(op_lists_by_index[i]))

    def __copy__(self) -> 'cirq.Circuit':
        return self.copy()

    def freeze(self) -> 'cirq.FrozenCircuit':
        """Gets a frozen version of this circuit.

        Repeated calls to `.freeze()` will return the same FrozenCircuit
        instance as long as this circuit is not mutated.
        """
        from cirq.circuits.frozen_circuit import FrozenCircuit

        if self._frozen is None:
            self._frozen = FrozenCircuit.from_moments(*self._moments)
        return self._frozen

    def unfreeze(self, copy: bool = True) -> 'cirq.Circuit':
        return self.copy() if copy else self

    def all_qubits(self) -> FrozenSet['cirq.Qid']:
        if self._all_qubits is None:
            self._all_qubits = super().all_qubits()
        return self._all_qubits

    def _is_measurement_(self) -> bool:
        if self._is_measurement is None:
            self._is_measurement = super()._is_measurement_()
        return self._is_measurement

    def _is_parameterized_(self) -> bool:
        if self._is_parameterized is None:
            self._is_parameterized = super()._is_parameterized_()
        return self._is_parameterized

    def _parameter_names_(self) -> AbstractSet[str]:
        if self._parameter_names is None:
            self._parameter_names = super()._parameter_names_()
        return self._parameter_names

    def copy(self) -> 'Circuit':
        """Return a copy of this circuit."""
        copied_circuit = Circuit()
        copied_circuit._moments = self._moments[:]
        copied_circuit._placement_cache = None
        return copied_circuit

    # pylint: disable=function-redefined
    @overload
    def __setitem__(self, key: int, value: 'cirq.Moment'):
        pass

    @overload
    def __setitem__(self, key: slice, value: Iterable['cirq.Moment']):
        pass

    def __setitem__(self, key, value):
        if isinstance(key, int) and not isinstance(value, Moment):
            raise TypeError('Can only assign Moments into Circuits.')

        if isinstance(key, slice):
            value = list(value)
            if any(not isinstance(v, Moment) for v in value):
                raise TypeError('Can only assign Moments into Circuits.')

        self._moments[key] = value
        self._mutated()

    # pylint: enable=function-redefined

    def __delitem__(self, key: Union[int, slice]):
        del self._moments[key]
        self._mutated()

    def __iadd__(self, other):
        self.append(other)
        return self

    def __add__(self, other):
        if not isinstance(other, (ops.Operation, Iterable)):
            return NotImplemented

        result = self.copy()
        return result.__iadd__(other)

    def __radd__(self, other):
        # The Circuit + Circuit case is handled by __add__
        if not isinstance(other, (ops.Operation, Iterable)):
            return NotImplemented
        # Auto wrap OP_TREE inputs into a circuit.
        result = self.copy()
        result._moments[:0] = Circuit(other)._moments
        return result

    # Needed for numpy to handle multiplication by np.int64 correctly.
    __array_priority__ = 10000

    def __imul__(self, repetitions: _INT_TYPE):
        if not isinstance(repetitions, (int, np.integer)):
            return NotImplemented
        self._moments *= int(repetitions)
        self._mutated()
        return self

    def __mul__(self, repetitions: _INT_TYPE):
        if not isinstance(repetitions, (int, np.integer)):
            return NotImplemented
        return Circuit(self._moments * int(repetitions))

    def __rmul__(self, repetitions: _INT_TYPE):
        if not isinstance(repetitions, (int, np.integer)):
            return NotImplemented
        return self * int(repetitions)

    def __pow__(self, exponent: int) -> 'cirq.Circuit':
        """A circuit raised to a power, only valid for exponent -1, the inverse.

        This will fail if anything other than -1 is passed to the Circuit by
        returning NotImplemented.  Otherwise this will return the inverse
        circuit, which is the circuit with its moment order reversed and for
        every moment all the moment's operations are replaced by its inverse.
        If any of the operations do not support inverse, NotImplemented will be
        returned.
        """
        if exponent != -1:
            return NotImplemented
        inv_moments = []
        for moment in self[::-1]:
            inv_moment = cirq.inverse(moment, default=NotImplemented)
            if inv_moment is NotImplemented:
                return NotImplemented
            inv_moments.append(inv_moment)

        return cirq.Circuit(inv_moments)

    __hash__ = None  # type: ignore

    def concat_ragged(
        *circuits: 'cirq.AbstractCircuit', align: Union['cirq.Alignment', str] = Alignment.LEFT
    ) -> 'cirq.Circuit':
        return AbstractCircuit.concat_ragged(*circuits, align=align).unfreeze(copy=False)

    concat_ragged.__doc__ = AbstractCircuit.concat_ragged.__doc__

    def zip(
        *circuits: 'cirq.AbstractCircuit', align: Union['cirq.Alignment', str] = Alignment.LEFT
    ) -> 'cirq.Circuit':
        return AbstractCircuit.zip(*circuits, align=align).unfreeze(copy=False)

    zip.__doc__ = AbstractCircuit.zip.__doc__

    def transform_qubits(
        self, qubit_map: Union[Dict['cirq.Qid', 'cirq.Qid'], Callable[['cirq.Qid'], 'cirq.Qid']]
    ) -> 'cirq.Circuit':
        """Returns the same circuit, but with different qubits.

        This function will return a new `Circuit` with the same gates but
        with qubits mapped according to the argument.

        For example, the following will translate LineQubits to GridQubits:

        >>> grid_qubits = cirq.GridQubit.square(2)
        >>> line_qubits = cirq.LineQubit.range(4)
        >>> circuit = cirq.Circuit([cirq.H(q) for q in line_qubits])
        >>> circuit.transform_qubits(lambda q : grid_qubits[q.x])
        cirq.Circuit([
            cirq.Moment(
                cirq.H(cirq.GridQubit(0, 0)),
                cirq.H(cirq.GridQubit(0, 1)),
                cirq.H(cirq.GridQubit(1, 0)),
                cirq.H(cirq.GridQubit(1, 1)),
            ),
        ])

        Args:
            qubit_map: A function or a dict mapping each current qubit into a desired
                new qubit.

        Returns:
            The receiving circuit but with qubits transformed by the given
                function.

        Raises:
            TypeError: If `qubit_function` is not a function or a dict.
        """
        if callable(qubit_map):
            transform = qubit_map
        elif isinstance(qubit_map, dict):
            transform = lambda q: qubit_map.get(q, q)
        else:
            raise TypeError('qubit_map must be a function or dict mapping qubits to qubits.')

        op_list = [
            Moment(operation.transform_qubits(transform) for operation in moment.operations)
            for moment in self._moments
        ]

        return Circuit(op_list)

    def earliest_available_moment(
        self, op: 'cirq.Operation', *, end_moment_index: Optional[int] = None
    ) -> int:
        """Finds the index of the earliest (i.e. left most) moment which can accommodate `op`.

        Note that, unlike `circuit.prev_moment_operating_on`, this method also takes care of
        implicit dependencies between measurements and classically controlled operations (CCO)
        that depend on the results of those measurements. Therefore, using this method, a CCO
        `op` would not be allowed to move left past a measurement it depends upon.

        Args:
            op: Operation for which the earliest moment that can accommodate it needs to be found.
            end_moment_index: The moment index just after the starting point of the reverse search.
                Defaults to the length of the list of moments.

        Returns:
            Index of the earliest matching moment. Returns `end_moment_index` if no moment on left
            is available.
        """
        if end_moment_index is None:
            end_moment_index = len(self.moments)
        last_available = end_moment_index
        k = end_moment_index
        op_control_keys = protocols.control_keys(op)
        op_measurement_keys = protocols.measurement_key_objs(op)
        op_qubits = op.qubits
        while k > 0:
            k -= 1
            moment = self._moments[k]
            if moment.operates_on(op_qubits):
                return last_available
            moment_measurement_keys = moment._measurement_key_objs_()
            if (
                not op_measurement_keys.isdisjoint(moment_measurement_keys)
                or not op_control_keys.isdisjoint(moment_measurement_keys)
                or not moment._control_keys_().isdisjoint(op_measurement_keys)
            ):
                return last_available
            if self._can_add_op_at(k, op):
                # Note: Remove the if condition after `self._device` is gone and move the method to
                # `cirq.AbstractDevice`.
                last_available = k
        return last_available

    def _can_add_op_at(self, moment_index: int, operation: 'cirq.Operation') -> bool:
        if not 0 <= moment_index < len(self._moments):
            return True

        return not self._moments[moment_index].operates_on(operation.qubits)

    def insert(
        self,
        index: int,
        moment_or_operation_tree: 'cirq.OP_TREE',
        strategy: 'cirq.InsertStrategy' = InsertStrategy.EARLIEST,
    ) -> int:
        """Inserts operations into the circuit.

        Operations are inserted into the moment specified by the index and
        'InsertStrategy'.
        Moments within the operation tree are inserted intact.

        Args:
            index: The index to insert all the operations at.
            moment_or_operation_tree: The moment or operation tree to insert.
            strategy: How to pick/create the moment to put operations into.

        Returns:
            The insertion index that will place operations just after the
            operations that were inserted by this method.

        Raises:
            ValueError: Bad insertion strategy.
        """
        # limit index to 0..len(self._moments), also deal with indices smaller 0
        k = max(min(index if index >= 0 else len(self._moments) + index, len(self._moments)), 0)
        if strategy != InsertStrategy.EARLIEST or k != len(self._moments):
            self._placement_cache = None
        mops = list(ops.flatten_to_ops_or_moments(moment_or_operation_tree))
        if self._placement_cache:
            batches = [mops]  # Any grouping would work here; this just happens to be the fastest.
        elif strategy is InsertStrategy.NEW:
            batches = [[mop] for mop in mops]  # Each op goes into its own moment.
        else:
            batches = list(_group_into_moment_compatible(mops))
        for batch in batches:
            # Insert a moment if inline/earliest and _any_ op in the batch requires it.
            if (
                not self._placement_cache
                and not isinstance(batch[0], Moment)
                and strategy in (InsertStrategy.INLINE, InsertStrategy.EARLIEST)
                and not all(
                    (strategy is InsertStrategy.EARLIEST and self._can_add_op_at(k, op))
                    or (k > 0 and self._can_add_op_at(k - 1, op))
                    for op in cast(List['cirq.Operation'], batch)
                )
            ):
                self._moments.insert(k, Moment())
                if strategy is InsertStrategy.INLINE:
                    k += 1
            max_p = 0
            for moment_or_op in batch:
                # Determine Placement
                if self._placement_cache:
                    p = self._placement_cache.append(moment_or_op)
                elif isinstance(moment_or_op, Moment):
                    p = k
                elif strategy in (InsertStrategy.NEW, InsertStrategy.NEW_THEN_INLINE):
                    self._moments.insert(k, Moment())
                    p = k
                elif strategy is InsertStrategy.INLINE:
                    p = k - 1
                else:  # InsertStrategy.EARLIEST:
                    p = self.earliest_available_moment(moment_or_op, end_moment_index=k)
                # Place
                if isinstance(moment_or_op, Moment):
                    self._moments.insert(p, moment_or_op)
                elif p == len(self._moments):
                    self._moments.append(Moment(moment_or_op))
                else:
                    self._moments[p] = self._moments[p].with_operation(moment_or_op)
                # Iterate
                max_p = max(p, max_p)
                if strategy is InsertStrategy.NEW_THEN_INLINE:
                    strategy = InsertStrategy.INLINE
                    k += 1
            k = max(k, max_p + 1)
        self._mutated(preserve_placement_cache=True)
        return k

    def insert_into_range(self, operations: 'cirq.OP_TREE', start: int, end: int) -> int:
        """Writes operations inline into an area of the circuit.

        Args:
            start: The start of the range (inclusive) to write the
                given operations into.
            end: The end of the range (exclusive) to write the given
                operations into. If there are still operations remaining,
                new moments are created to fit them.
            operations: An operation or tree of operations to insert.

        Returns:
            An insertion index that will place operations after the operations
            that were inserted by this method.

        Raises:
            IndexError: Bad inline_start and/or inline_end.
        """
        if not 0 <= start <= end <= len(self):
            raise IndexError(f'Bad insert indices: [{start}, {end})')

        flat_ops = list(ops.flatten_to_ops(operations))

        i = start
        op_index = 0
        while op_index < len(flat_ops):
            op = flat_ops[op_index]
            while i < end and self._moments[i].operates_on(op.qubits):
                i += 1
            if i >= end:
                break

            self._moments[i] = self._moments[i].with_operation(op)
            op_index += 1
        self._mutated()

        if op_index >= len(flat_ops):
            return end

        return self.insert(end, flat_ops[op_index:])

    def _push_frontier(
        self,
        early_frontier: Dict['cirq.Qid', int],
        late_frontier: Dict['cirq.Qid', int],
        update_qubits: Optional[Iterable['cirq.Qid']] = None,
    ) -> Tuple[int, int]:
        """Inserts moments to separate two frontiers.

        After insertion n_new moments, the following holds:
           for q in late_frontier:
               early_frontier[q] <= late_frontier[q] + n_new
           for q in update_qubits:
               early_frontier[q] the identifies the same moment as before
                   (but whose index may have changed if this moment is after
                   those inserted).

        Args:
            early_frontier: The earlier frontier. For qubits not in the later
                frontier, this is updated to account for the newly inserted
                moments.
            late_frontier: The later frontier. This is not modified.
            update_qubits: The qubits for which to update early_frontier to
                account for the newly inserted moments.

        Returns:
            (index at which new moments were inserted, how many new moments
            were inserted) if new moments were indeed inserted. (0, 0)
            otherwise.
        """
        if update_qubits is None:
            update_qubits = set(early_frontier).difference(late_frontier)
        n_new_moments = (
            max(early_frontier.get(q, 0) - late_frontier[q] for q in late_frontier)
            if late_frontier
            else 0
        )
        if n_new_moments > 0:
            insert_index = min(late_frontier.values())
            self._moments[insert_index:insert_index] = [Moment()] * n_new_moments
            self._mutated()
            for q in update_qubits:
                if early_frontier.get(q, 0) > insert_index:
                    early_frontier[q] += n_new_moments
            return insert_index, n_new_moments
        return (0, 0)

    def _insert_operations(
        self, operations: Sequence['cirq.Operation'], insertion_indices: Sequence[int]
    ) -> None:
        """Inserts operations at the specified moments. Appends new moments if
        necessary.

        Args:
            operations: The operations to insert.
            insertion_indices: Where to insert them, i.e. operations[i] is
                inserted into moments[insertion_indices[i].

        Raises:
            ValueError: operations and insert_indices have different lengths.

        NB: It's on the caller to ensure that the operations won't conflict
        with operations already in the moment or even each other.
        """
        if len(operations) != len(insertion_indices):
            raise ValueError('operations and insertion_indices must have the same length.')
        self._moments += [Moment() for _ in range(1 + max(insertion_indices) - len(self))]
        self._mutated()
        moment_to_ops: Dict[int, List['cirq.Operation']] = defaultdict(list)
        for op_index, moment_index in enumerate(insertion_indices):
            moment_to_ops[moment_index].append(operations[op_index])
        for moment_index, new_ops in moment_to_ops.items():
            self._moments[moment_index] = self._moments[moment_index].with_operations(*new_ops)

    def insert_at_frontier(
        self,
        operations: 'cirq.OP_TREE',
        start: int,
        frontier: Optional[Dict['cirq.Qid', int]] = None,
    ) -> Dict['cirq.Qid', int]:
        """Inserts operations inline at frontier.

        Args:
            operations: The operations to insert.
            start: The moment at which to start inserting the operations.
            frontier: frontier[q] is the earliest moment in which an operation
                acting on qubit q can be placed.

        Raises:
            ValueError: If the frontier given is after start.
        """
        if frontier is None:
            frontier = defaultdict(lambda: 0)
        flat_ops = tuple(ops.flatten_to_ops(operations))
        if not flat_ops:
            return frontier
        qubits = set(q for op in flat_ops for q in op.qubits)
        if any(frontier[q] > start for q in qubits):
            raise ValueError(
                'The frontier for qubits on which the operations'
                'to insert act cannot be after start.'
            )

        next_moments = self.next_moments_operating_on(qubits, start)

        insertion_indices, _ = _pick_inserted_ops_moment_indices(flat_ops, start, frontier)

        self._push_frontier(frontier, next_moments)

        self._insert_operations(flat_ops, insertion_indices)

        return frontier

    def batch_remove(self, removals: Iterable[Tuple[int, 'cirq.Operation']]) -> None:
        """Removes several operations from a circuit.

        Args:
            removals: A sequence of (moment_index, operation) tuples indicating
                operations to delete from the moments that are present. All
                listed operations must actually be present or the edit will
                fail (without making any changes to the circuit).

        Raises:
            ValueError: One of the operations to delete wasn't present to start with.
            IndexError: Deleted from a moment that doesn't exist.
        """
        copy = self.copy()
        for i, op in removals:
            if op not in copy._moments[i].operations:
                raise ValueError(f"Can't remove {op} @ {i} because it doesn't exist.")
            copy._moments[i] = Moment(
                old_op for old_op in copy._moments[i].operations if op != old_op
            )
        self._moments = copy._moments
        self._mutated()

    def batch_replace(
        self, replacements: Iterable[Tuple[int, 'cirq.Operation', 'cirq.Operation']]
    ) -> None:
        """Replaces several operations in a circuit with new operations.

        Args:
            replacements: A sequence of (moment_index, old_op, new_op) tuples
                indicating operations to be replaced in this circuit. All "old"
                operations must actually be present or the edit will fail
                (without making any changes to the circuit).

        Raises:
            ValueError: One of the operations to replace wasn't present to start with.
            IndexError: Replaced in a moment that doesn't exist.
        """
        copy = self.copy()
        for i, op, new_op in replacements:
            if op not in copy._moments[i].operations:
                raise ValueError(f"Can't replace {op} @ {i} because it doesn't exist.")
            copy._moments[i] = Moment(
                old_op if old_op != op else new_op for old_op in copy._moments[i].operations
            )
        self._moments = copy._moments
        self._mutated()

    def batch_insert_into(self, insert_intos: Iterable[Tuple[int, 'cirq.OP_TREE']]) -> None:
        """Inserts operations into empty spaces in existing moments.

        If any of the insertions fails (due to colliding with an existing
        operation), this method fails without making any changes to the circuit.

        Args:
            insert_intos: A sequence of (moment_index, new_op_tree)
                pairs indicating a moment to add new operations into.

        Raises:
            ValueError: One of the insertions collided with an existing
                operation.
            IndexError: Inserted into a moment index that doesn't exist.
        """
        copy = self.copy()
        for i, insertions in insert_intos:
            copy._moments[i] = copy._moments[i].with_operations(insertions)
        self._moments = copy._moments
        self._mutated()

    def batch_insert(self, insertions: Iterable[Tuple[int, 'cirq.OP_TREE']]) -> None:
        """Applies a batched insert operation to the circuit.

        Transparently handles the fact that earlier insertions may shift
        the index that later insertions should occur at. For example, if you
        insert an operation at index 2 and at index 4, but the insert at index 2
        causes a new moment to be created, then the insert at "4" will actually
        occur at index 5 to account for the shift from the new moment.

        All insertions are done with the strategy `cirq.InsertStrategy.EARLIEST`.

        When multiple inserts occur at the same index, the gates from the later
        inserts end up before the gates from the earlier inserts (exactly as if
        you'd called list.insert several times with the same index: the later
        inserts shift the earliest inserts forward).

        Args:
            insertions: A sequence of (insert_index, operations) pairs
                indicating operations to add into the circuit at specific
                places.
        """
        # Work on a copy in case validation fails halfway through.
        copy = self.copy()
        shift = 0
        # Note: python `sorted` is guaranteed to be stable. This matters.
        insertions = sorted(insertions, key=lambda e: e[0])
        groups = _group_until_different(insertions, key=lambda e: e[0], val=lambda e: e[1])
        for i, group in groups:
            insert_index = i + shift
            next_index = copy.insert(insert_index, reversed(group), InsertStrategy.EARLIEST)
            if next_index > insert_index:
                shift += next_index - insert_index
        self._moments = copy._moments
        self._mutated()

    def append(
        self,
        moment_or_operation_tree: 'cirq.OP_TREE',
        strategy: 'cirq.InsertStrategy' = InsertStrategy.EARLIEST,
    ) -> None:
        """Appends operations onto the end of the circuit.

        Moments within the operation tree are appended intact.

        Args:
            moment_or_operation_tree: The moment or operation tree to append.
            strategy: How to pick/create the moment to put operations into.
        """
        self.insert(len(self._moments), moment_or_operation_tree, strategy)

    def clear_operations_touching(
        self, qubits: Iterable['cirq.Qid'], moment_indices: Iterable[int]
    ):
        """Clears operations that are touching given qubits at given moments.

        Args:
            qubits: The qubits to check for operations on.
            moment_indices: The indices of moments to check for operations
                within.
        """
        qubits = frozenset(qubits)
        for k in moment_indices:
            if 0 <= k < len(self._moments):
                self._moments[k] = self._moments[k].without_operations_touching(qubits)
        self._mutated()

    @property
    def moments(self) -> Sequence['cirq.Moment']:
        return self._moments

    def with_noise(self, noise: 'cirq.NOISE_MODEL_LIKE') -> 'cirq.Circuit':
        """Make a noisy version of the circuit.

        Args:
            noise: The noise model to use.  This describes the kind of noise to
                add to the circuit.

        Returns:
            A new circuit with the same moment structure but with new moments
            inserted where needed when more than one noisy operation is
            generated for an input operation.  Emptied moments are removed.
        """
        noise_model = devices.NoiseModel.from_noise_model_like(noise)
        qubits = sorted(self.all_qubits())
        c_noisy = Circuit()
        for op_tree in noise_model.noisy_moments(self, qubits):
            # Keep moments aligned
            c_noisy += Circuit(op_tree)
        return c_noisy


def _pick_inserted_ops_moment_indices(
    operations: Sequence['cirq.Operation'],
    start: int = 0,
    frontier: Optional[Dict['cirq.Qid', int]] = None,
) -> Tuple[Sequence[int], Dict['cirq.Qid', int]]:
    """Greedily assigns operations to moments.

    Args:
        operations: The operations to assign to moments.
        start: The first moment to consider assignment to.
        frontier: The first moment to which an operation acting on a qubit
            can be assigned. Updated in place as operations are assigned.

    Returns:
        The frontier giving the index of the moment after the last one to
        which an operation that acts on each qubit is assigned. If a
        frontier was specified as an argument, this is the same object.
    """
    if frontier is None:
        frontier = defaultdict(lambda: 0)
    moment_indices = []
    for op in operations:
        op_start = max(start, max((frontier[q] for q in op.qubits), default=0))
        moment_indices.append(op_start)
        for q in op.qubits:
            frontier[q] = max(frontier[q], op_start + 1)

    return moment_indices, frontier


def _get_moment_annotations(moment: 'cirq.Moment') -> Iterator['cirq.Operation']:
    for op in moment.operations:
        if op.qubits:
            continue
        op = op.untagged
        if isinstance(op.gate, ops.GlobalPhaseGate):
            continue
        if isinstance(op, CircuitOperation):
            for m in op.circuit:
                yield from _get_moment_annotations(m)
        else:
            yield op


def _draw_moment_annotations(
    *,
    moment: 'cirq.Moment',
    col: int,
    use_unicode_characters: bool,
    label_map: Dict['cirq.LabelEntity', int],
    out_diagram: 'cirq.TextDiagramDrawer',
    precision: Optional[int],
    get_circuit_diagram_info: Callable[
        ['cirq.Operation', 'cirq.CircuitDiagramInfoArgs'], 'cirq.CircuitDiagramInfo'
    ],
    include_tags: bool,
    first_annotation_row: int,
    transpose: bool,
):
    for k, annotation in enumerate(_get_moment_annotations(moment)):
        args = protocols.CircuitDiagramInfoArgs(
            known_qubits=(),
            known_qubit_count=0,
            use_unicode_characters=use_unicode_characters,
            label_map=label_map,
            precision=precision,
            include_tags=include_tags,
            transpose=transpose,
        )
        info = get_circuit_diagram_info(annotation, args)
        symbols = info._wire_symbols_including_formatted_exponent(args)
        text = symbols[0] if symbols else str(annotation)
        out_diagram.force_vertical_padding_after(first_annotation_row + k - 1, 0)
        out_diagram.write(col, first_annotation_row + k, text)


def _draw_moment_in_diagram(
    *,
    moment: 'cirq.Moment',
    use_unicode_characters: bool,
    label_map: Dict['cirq.LabelEntity', int],
    out_diagram: 'cirq.TextDiagramDrawer',
    precision: Optional[int],
    moment_groups: List[Tuple[int, int]],
    get_circuit_diagram_info: Optional[
        Callable[['cirq.Operation', 'cirq.CircuitDiagramInfoArgs'], 'cirq.CircuitDiagramInfo']
    ],
    include_tags: bool,
    first_annotation_row: int,
    transpose: bool,
):
    if get_circuit_diagram_info is None:
        get_circuit_diagram_info = circuit_diagram_info_protocol._op_info_with_fallback
    x0 = out_diagram.width()

    non_global_ops = [op for op in moment.operations if op.qubits]

    max_x = x0
    for op in non_global_ops:
        qubits = tuple(op.qubits)
        cbits = tuple(protocols.measurement_keys_touched(op) & label_map.keys())
        labels = qubits + cbits
        indices = [label_map[label] for label in labels]
        y1 = min(indices)
        y2 = max(indices)

        # Find an available column.
        x = x0
        while any(out_diagram.content_present(x, y) for y in range(y1, y2 + 1)):
            out_diagram.force_horizontal_padding_after(x, 0)
            x += 1

        args = protocols.CircuitDiagramInfoArgs(
            known_qubits=op.qubits,
            known_qubit_count=len(op.qubits),
            use_unicode_characters=use_unicode_characters,
            label_map=label_map,
            precision=precision,
            include_tags=include_tags,
            transpose=transpose,
        )
        info = get_circuit_diagram_info(op, args)

        # Draw vertical line linking the gate's qubits.
        if y2 > y1 and info.connected:
            out_diagram.vertical_line(x, y1, y2, doubled=len(cbits) != 0)

        # Print gate qubit labels.
        symbols = info._wire_symbols_including_formatted_exponent(
            args,
            preferred_exponent_index=max(range(len(labels)), key=lambda i: label_map[labels[i]]),
        )
        for s, q in zip(symbols, labels):
            out_diagram.write(x, label_map[q], s)

        if x > max_x:
            max_x = x

    _draw_moment_annotations(
        moment=moment,
        use_unicode_characters=use_unicode_characters,
        col=x0,
        label_map=label_map,
        out_diagram=out_diagram,
        precision=precision,
        get_circuit_diagram_info=get_circuit_diagram_info,
        include_tags=include_tags,
        first_annotation_row=first_annotation_row,
        transpose=transpose,
    )

    global_phase, tags = _get_global_phase_and_tags_for_ops(moment)

    # Print out global phase, unless it's 1 (phase of 0pi) or it's the only op.
    if global_phase and (global_phase != 1 or not non_global_ops):
        desc = _formatted_phase(global_phase, use_unicode_characters, precision)
        if desc:
            y = max(label_map.values(), default=0) + 1
            if tags and include_tags:
                desc = desc + f"[{', '.join(map(str, tags))}]"
            out_diagram.write(x0, y, desc)

    if not non_global_ops:
        out_diagram.write(x0, 0, '')

    # Group together columns belonging to the same Moment.
    if moment.operations and max_x > x0:
        moment_groups.append((x0, max_x))


def _get_global_phase_and_tags_for_op(op: 'cirq.Operation') -> Tuple[Optional[complex], List[Any]]:
    if isinstance(op.gate, ops.GlobalPhaseGate):
        return complex(op.gate.coefficient), list(op.tags)
    elif isinstance(op.untagged, CircuitOperation):
        op_phase, op_tags = _get_global_phase_and_tags_for_ops(op.untagged.circuit.all_operations())
        return op_phase, list(op.tags) + op_tags
    return None, []


def _get_global_phase_and_tags_for_ops(op_list: Any) -> Tuple[Optional[complex], List[Any]]:
    global_phase: Optional[complex] = None
    tags: List[Any] = []
    for op in op_list:
        op_phase, op_tags = _get_global_phase_and_tags_for_op(op)
        if op_phase:
            if global_phase is None:
                global_phase = complex(1)
            global_phase *= op_phase
        if op_tags:
            tags.extend(op_tags)
    return global_phase, tags


def _formatted_phase(coefficient: complex, unicode: bool, precision: Optional[int]) -> str:
    h = math.atan2(coefficient.imag, coefficient.real) / math.pi
    unit = 'π' if unicode else 'pi'
    if h == 1:
        return unit
    return f'{{:.{precision}}}'.format(h) + unit


def _draw_moment_groups_in_diagram(
    moment_groups: List[Tuple[int, int]],
    use_unicode_characters: bool,
    out_diagram: 'cirq.TextDiagramDrawer',
):
    out_diagram.insert_empty_rows(0)
    h = out_diagram.height()

    # Insert columns starting from the back since the insertion
    # affects subsequent indices.
    for x1, x2 in reversed(moment_groups):
        out_diagram.insert_empty_columns(x2 + 1)
        out_diagram.force_horizontal_padding_after(x2, 0)
        out_diagram.insert_empty_columns(x1)
        out_diagram.force_horizontal_padding_after(x1, 0)
        x2 += 2
        for x in range(x1, x2):
            out_diagram.force_horizontal_padding_after(x, 0)

        for y in [0, h]:
            out_diagram.horizontal_line(y, x1, x2)
        out_diagram.vertical_line(x1, 0, 0.5)
        out_diagram.vertical_line(x2, 0, 0.5)
        out_diagram.vertical_line(x1, h, h - 0.5)
        out_diagram.vertical_line(x2, h, h - 0.5)

    # Rounds up to 1 when horizontal, down to 0 when vertical.
    # (Matters when transposing.)
    out_diagram.force_vertical_padding_after(0, 0.5)
    out_diagram.force_vertical_padding_after(h - 1, 0.5)


def _apply_unitary_circuit(
    circuit: 'cirq.AbstractCircuit',
    state: np.ndarray,
    qubits: Tuple['cirq.Qid', ...],
    dtype: Type[np.complexfloating],
) -> np.ndarray:
    """Applies a circuit's unitary effect to the given vector or matrix.

    This method assumes that the caller wants to ignore measurements.

    Args:
        circuit: The circuit to simulate. All operations must have a known
            matrix or decompositions leading to known matrices. Measurements
            are allowed to be in the circuit, but they will be ignored.
        state: The initial state tensor (i.e. superposition or unitary matrix).
            This is what will be left-multiplied by the circuit's effective
            unitary. If this is a state vector, it must have shape
            (2,) * num_qubits. If it is a unitary matrix it should have shape
            (2,) * (2*num_qubits).
        qubits: The qubits in the state tensor. Determines which axes operations
            apply to. An operation targeting the k'th qubit in this list will
            operate on the k'th axis of the state tensor.
        dtype: The numpy dtype to use for applying the unitary. Must be a
            complex dtype.

    Returns:
        The left-multiplied state tensor.
    """
    buffer = np.empty_like(state)

    def on_stuck(bad_op):
        return TypeError(f'Operation without a known matrix or decomposition: {bad_op!r}')

    unitary_ops = protocols.decompose(
        circuit.all_operations(),
        keep=protocols.has_unitary,
        intercepting_decomposer=_decompose_measurement_inversions,
        on_stuck_raise=on_stuck,
    )

    result = protocols.apply_unitaries(
        unitary_ops, qubits, protocols.ApplyUnitaryArgs(state, buffer, range(len(qubits)))
    )
    assert result is not None, "apply_unitaries() should raise TypeError instead"
    return result


def _decompose_measurement_inversions(op: 'cirq.Operation') -> 'cirq.OP_TREE':
    if isinstance(op.gate, ops.MeasurementGate):
        return [ops.X(q) for q, b in zip(op.qubits, op.gate.invert_mask) if b]
    return NotImplemented


def _list_repr_with_indented_item_lines(items: Sequence[Any]) -> str:
    block = '\n'.join([repr(op) + ',' for op in items])
    indented = '    ' + '\n    '.join(block.split('\n'))
    return f'[\n{indented}\n]'


_TIn = TypeVar('_TIn')
_TOut = TypeVar('_TOut')
_TKey = TypeVar('_TKey')


@overload
def _group_until_different(
    items: Iterable[_TIn], key: Callable[[_TIn], _TKey]
) -> Iterable[Tuple[_TKey, List[_TIn]]]:
    pass


@overload
def _group_until_different(
    items: Iterable[_TIn], key: Callable[[_TIn], _TKey], val: Callable[[_TIn], _TOut]
) -> Iterable[Tuple[_TKey, List[_TOut]]]:
    pass


def _group_until_different(items: Iterable[_TIn], key: Callable[[_TIn], _TKey], val=lambda e: e):
    """Groups runs of items that are identical according to a keying function.

    Args:
        items: The items to group.
        key: If two adjacent items produce the same output from this function,
            they will be grouped.
        val: Maps each item into a value to put in the group. Defaults to the
            item itself.

    Examples:
        _group_until_different(range(11), key=is_prime) yields
            (False, [0, 1])
            (True, [2, 3])
            (False, [4])
            (True, [5])
            (False, [6])
            (True, [7])
            (False, [8, 9, 10])

    Yields:
        Tuples containing the group key and item values.
    """
    return ((k, [val(i) for i in v]) for (k, v) in itertools.groupby(items, key))


def _group_into_moment_compatible(inputs: Sequence[_MOMENT_OR_OP]) -> Iterator[List[_MOMENT_OR_OP]]:
    """Groups sequential ops into those that can coexist in a single moment.

    This function will go through the input sequence in order, emitting lists of sequential
    operations that can go into a single moment. It does not try to rearrange the elements or try
    to move them to open slots in earlier moments; it simply processes them in order and outputs
    them. i.e. the output, if flattened, will equal the input.

    Actual Moments in the input will always be emitted by themselves as a single-element list.

    Examples:
        [X(a), X(b), X(a)] -> [[X(a), X(b)], [X(a)]]
        [X(a), X(a), X(b)] -> [[X(a)], [X(a), X(b)]]
        [X(a), Moment(X(b)), X(c)] -> [[X(a)], [Moment(X(b))], [X(c)]]
    """
    batch: List[_MOMENT_OR_OP] = []
    batch_qubits: Set['cirq.Qid'] = set()
    for mop in inputs:
        is_moment = isinstance(mop, cirq.Moment)
        if (is_moment and batch) or not batch_qubits.isdisjoint(mop.qubits):
            yield batch
            batch = []
            batch_qubits.clear()
        if is_moment:
            yield [mop]
            continue
        batch.append(mop)
        batch_qubits.update(mop.qubits)
    if batch:
        yield batch


def get_earliest_accommodating_moment_index(
    moment_or_operation: _MOMENT_OR_OP,
    qubit_indices: Dict['cirq.Qid', int],
    mkey_indices: Dict['cirq.MeasurementKey', int],
    ckey_indices: Dict['cirq.MeasurementKey', int],
    length: Optional[int] = None,
) -> int:
    """Get the index of the earliest moment that can accommodate the given moment or operation.

    Updates the dictionaries keeping track of the last moment index addressing a given qubit,
    measurement key, and control key.

    Args:
        moment_or_operation: The moment operation in question.
        qubit_indices: A dictionary mapping qubits to the latest moments that address them.
        mkey_indices: A dictionary mapping measurement keys to the latest moments that address them.
        ckey_indices: A dictionary mapping control keys to the latest moments that address them.
        length: The length of the circuit that we are trying to insert a moment or operation into.
            Should probably be equal to the maximum of the values in `qubit_indices`,
            `mkey_indices`, and `ckey_indices`.

    Returns:
        The integer index of the earliest moment that can accommodate the given moment or operation.
    """
    mop_qubits = moment_or_operation.qubits
    mop_mkeys = protocols.measurement_key_objs(moment_or_operation)
    mop_ckeys = protocols.control_keys(moment_or_operation)

    if isinstance(moment_or_operation, Moment):
        # For consistency with `Circuit.append`, moments always get placed at the end of a circuit.
        if length is not None:
            last_conflict = length - 1
        else:
            last_conflict = max(
                [*qubit_indices.values(), *mkey_indices.values(), *ckey_indices.values(), -1]
            )

    else:
        # We start by searching for the `latest_conflict` moment index, which we will increment by
        # `1` to identify the earliest moment that *does not* conflict with the given operation.
        # The `latest_conflict` is initialized to `-1` before searching for later conflicting
        # moments.
        last_conflict = -1

        # Look for the maximum conflict; i.e. a moment that has a qubit the same as one of this op's
        # qubits, that has a measurement or control key the same as one of this op's measurement
        # keys, or that has a measurement key the same as one of this op's control keys. (Control
        # keys alone can commute past each other). The `ifs` are logically unnecessary but seem to
        # make this slightly faster.
        if mop_qubits:
            last_conflict = max(
                last_conflict, *[qubit_indices.get(qubit, -1) for qubit in mop_qubits]
            )
        if mop_mkeys:
            last_conflict = max(last_conflict, *[mkey_indices.get(key, -1) for key in mop_mkeys])
            last_conflict = max(last_conflict, *[ckey_indices.get(key, -1) for key in mop_mkeys])
        if mop_ckeys:
            last_conflict = max(last_conflict, *[mkey_indices.get(key, -1) for key in mop_ckeys])

    # The index of the moment to place this moment or operation ("mop") into.
    mop_index = last_conflict + 1

    # Update our dicts with data from this `mop` placement. Note `mop_index` will always be greater
    # than the existing value for all of these, by construction.
    for qubit in mop_qubits:
        qubit_indices[qubit] = mop_index
    for key in mop_mkeys:
        mkey_indices[key] = mop_index
    for key in mop_ckeys:
        ckey_indices[key] = mop_index

    return mop_index


class _PlacementCache:
    """Maintains qubit and cbit indices for quick op placement.

    Here, we keep track of the greatest moment that contains each qubit,
    measurement key, and control key, and append operations to the moment after
    the maximum of these. This avoids having to iterate backwards, checking
    each moment one at a time.

    It is only valid for `append` operations, and if any other insert strategy
    is used, or if any operation is added to the circuit without notifying the
    cache, then the cache must be invalidated for the circuit or rebuilt from
    scratch. Future improvements may ease this restriction.
    """

    def __init__(self) -> None:
        # These are dicts from the qubit/key to the greatest moment index that has it.
        self._qubit_indices: Dict['cirq.Qid', int] = {}
        self._mkey_indices: Dict['cirq.MeasurementKey', int] = {}
        self._ckey_indices: Dict['cirq.MeasurementKey', int] = {}

        # For keeping track of length of the circuit thus far.
        self._length = 0

    def append(self, moment_or_operation: _MOMENT_OR_OP) -> int:
        """Find placement for moment/operation and update cache.

        Determines the placement index of the provided operation, assuming
        EARLIEST (append) strategy, and assuming that the internal cache
        correctly represents the circuit. It then updates the cache and returns
        the placement index.

        Args:
            moment_or_operation: The moment or operation to append.

        Returns:
            The index at which the moment/operation should be placed.
        """
        # Identify the index of the moment to place this into.
        index = get_earliest_accommodating_moment_index(
            moment_or_operation,
            self._qubit_indices,
            self._mkey_indices,
            self._ckey_indices,
            self._length,
        )
        self._length = max(self._length, index + 1)
        return index
