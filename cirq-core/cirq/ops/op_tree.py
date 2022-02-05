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

"""A recursive type describing trees of operations, and utility methods for it.
"""

from typing import Callable, Iterable, Iterator, NoReturn, Union, TYPE_CHECKING
from typing_extensions import Protocol

from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops.raw_types import Operation

if TYPE_CHECKING:
    import cirq

moment = LazyLoader("moment", globals(), "cirq.circuits.moment")


class OpTree(Protocol):
    """The recursive type consumed by circuit builder methods.

    An OpTree is a type protocol, satisfied by anything that can be recursively
    flattened into Operations. We also define the Union type OP_TREE which
    can be an OpTree or just a single Operation.

    For example:
    - An Operation is an OP_TREE all by itself.
    - A list of operations is an OP_TREE.
    - A list of tuples of operations is an OP_TREE.
    - A list with a mix of operations and lists of operations is an OP_TREE.
    - A generator yielding operations is an OP_TREE.

    Note: once mypy supports recursive types this could be defined as an alias:

    OP_TREE = Union[Operation, Iterable['OP_TREE']]

    See: https://github.com/python/mypy/issues/731
    """

    def __iter__(self) -> Iterator[Union[Operation, 'OpTree']]:
        pass


OP_TREE = Union[Operation, OpTree]
document(
    OP_TREE,  # type: ignore
    """An operation or nested collections of operations.

    Here are some examples of things that can be given to a method that takes a
    `cirq.OP_TREE` argument:

    - A single operation (a `cirq.Operation`).
    - A list of operations (a `List[cirq.Operation]`).
    - A list of lists of operations (a `List[List[cirq.Operation]]`).
    - A list mixing operations and generators of operations
        (a `List[Union[cirq.Operation, Iterator[cirq.Operation]]]`).
    - Generally anything that can be iterated, and its items iterated, and
        so forth recursively until a bottom layer of operations is found.
    """,
)


def flatten_op_tree(
    root: OP_TREE, preserve_moments: bool = False
) -> Iterator[Union[Operation, 'cirq.Moment']]:
    """Performs an in-order iteration of the operations (leaves) in an OP_TREE.

    Args:
        root: The operation or tree of operations to iterate.
        preserve_moments: Whether to yield Moments intact instead of
            flattening them

    Yields:
        Operations from the tree.

    Raises:
        TypeError: root isn't a valid OP_TREE.
    """
    if preserve_moments:
        return flatten_to_ops_or_moments(root)
    else:
        return flatten_to_ops(root)


def flatten_to_ops(root: OP_TREE) -> Iterator[Operation]:
    """Performs an in-order iteration of the operations (leaves) in an OP_TREE.

    Args:
        root: The operation or tree of operations to iterate.

    Yields:
        Operations or moments from the tree.

    Raises:
        TypeError: root isn't a valid OP_TREE.
    """
    if isinstance(root, Operation):
        yield root
    elif isinstance(root, Iterable) and not isinstance(root, str):
        for subtree in root:
            yield from flatten_to_ops(subtree)
    else:
        _bad_op_tree(root)


def flatten_to_ops_or_moments(root: OP_TREE) -> Iterator[Union[Operation, 'cirq.Moment']]:
    """Performs an in-order iteration OP_TREE, yielding ops and moments.

    Args:
        root: The operation or tree of operations to iterate.

    Yields:
        Operations or moments from the tree.

    Raises:
        TypeError: root isn't a valid OP_TREE.
    """
    if isinstance(root, (Operation, moment.Moment)):
        yield root
    elif isinstance(root, Iterable) and not isinstance(root, str):
        for subtree in root:
            yield from flatten_to_ops_or_moments(subtree)
    else:
        _bad_op_tree(root)


def transform_op_tree(
    root: OP_TREE,
    op_transformation: Callable[[Operation], OP_TREE] = lambda e: e,
    iter_transformation: Callable[[Iterable[OP_TREE]], OP_TREE] = lambda e: e,
    preserve_moments: bool = False,
) -> OP_TREE:
    """Maps transformation functions onto the nodes of an OP_TREE.

    Args:
        root: The operation or tree of operations to transform.
        op_transformation: How to transform the operations (i.e. leaves).
        iter_transformation: How to transform the iterables (i.e. internal
            nodes).
        preserve_moments: Whether to leave Moments alone. If True, the
            transformation functions will not be applied to Moments or the
            operations within them.

    Returns:
        A transformed operation tree.

    Raises:
        TypeError: root isn't a valid OP_TREE.
    """
    if isinstance(root, Operation):
        return op_transformation(root)

    if preserve_moments and isinstance(root, moment.Moment):
        return root

    if isinstance(root, Iterable) and not isinstance(root, str):
        return iter_transformation(
            transform_op_tree(subtree, op_transformation, iter_transformation, preserve_moments)
            for subtree in root
        )

    _bad_op_tree(root)


def freeze_op_tree(root: OP_TREE) -> OP_TREE:
    """Replaces all iterables in the OP_TREE with tuples.

    Args:
        root: The operation or tree of operations to freeze.

    Returns:
        An OP_TREE with the same operations and branching structure, but where
        all internal nodes are tuples instead of arbitrary iterables.
    """
    return transform_op_tree(root, iter_transformation=tuple)


def _bad_op_tree(root: OP_TREE) -> NoReturn:
    raise TypeError(f'Not an Operation or Iterable: {type(root)} {root}')
