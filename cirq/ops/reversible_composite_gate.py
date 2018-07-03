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

"""Implements the inverse() method of a CompositeGate & ReversibleEffect."""
from typing import TypeVar, Generic, cast

from cirq import abc
from cirq.extension import Extensions
from cirq.ops import gate_features, op_tree, raw_types


def _reverse_operation(operation: raw_types.Operation,
                       extensions: Extensions) -> raw_types.Operation:
    """Returns the inverse of an operation, if possible.

    Args:
        operation: The operation to reverse.

    Returns:
        An operation on the same qubits but with the inverse gate.

    Raises:
        ValueError: The operation's gate isn't reversible.
    """
    gate = extensions.try_cast(operation.gate, gate_features.ReversibleEffect)
    if gate is None:
        raise ValueError('Not reversible: {}'.format(operation))
    return raw_types.Operation(cast(raw_types.Gate, gate.inverse()),
                               operation.qubits)


def inverse_of_invertible_op_tree(root: op_tree.OP_TREE,
                                  extensions: Extensions = None
                                  ) -> op_tree.OP_TREE:
    """Generates OP_TREE inverses.

    Args:
        root: An operation tree containing only invertible operations.
        extensions: For caller-provided implementations of gate inverses.

    Returns:
        An OP_TREE that performs the inverse operation of the given OP_TREE.
    """
    ext = extensions or Extensions()
    return op_tree.transform_op_tree(
        root=root,
        op_transformation=lambda e: _reverse_operation(e, ext),
        iter_transformation=lambda e: reversed(list(e)))


TOriginal = TypeVar('TOriginal', bound='ReversibleCompositeGate')


class ReversibleCompositeGate(gate_features.CompositeGate,
                              gate_features.ReversibleEffect,
                              metaclass=abc.ABCMeta):
    """A composite gate that gets decomposed into reversible gates."""

    def inverse(self: TOriginal
                ) -> '_ReversedReversibleCompositeGate[TOriginal]':
        return _ReversedReversibleCompositeGate(self)


class _ReversedReversibleCompositeGate(Generic[TOriginal],
                                       gate_features.CompositeGate,
                                       gate_features.ReversibleEffect):
    """A reversed reversible composite gate."""

    def __init__(self, forward_form: TOriginal) -> None:
        self.forward_form = forward_form

    def inverse(self) -> TOriginal:
        return self.forward_form

    def default_decompose(self, qubits):
        return inverse_of_invertible_op_tree(
            self.forward_form.default_decompose(qubits))
