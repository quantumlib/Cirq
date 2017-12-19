# Copyright 2017 Google LLC
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

"""Implements the inverse() method of a CompositeGate & ReversibleGate."""

import abc

from cirq.ops import gate_features, op_tree, raw_types


def _reverse_operation(operation: raw_types.Operation) -> raw_types.Operation:
    """Returns the inverse of an operation, if possible.

    Args:
        operation: The operation to reverse.

    Returns:
        An operation on the same qubits but with the inverse gate.

    Raises:
        ValueError: The operation's gate isn't reversible.
    """
    if isinstance(operation.gate, gate_features.ReversibleGate):
        return raw_types.Operation(operation.gate.inverse(), operation.qubits)
    raise ValueError('Not reversible: {}'.format(operation))


def inverse_of_invertable_op_tree(root: op_tree.OP_TREE) -> op_tree.OP_TREE:
    """Generates OP_TREE inverses.

    Args:
        root: An operation tree containing only invertable operations.

    Returns:
        An OP_TREE that performs the inverse operation of the given OP_TREE.
    """
    return op_tree.transform_op_tree(
        root=root,
        op_transformation=_reverse_operation,
        iter_transformation=lambda e: reversed(list(e)))


class ReversibleCompositeGate(gate_features.CompositeGate,
                              gate_features.ReversibleGate,
                              metaclass=abc.ABCMeta):
    """A composite gate that gets decomposed into reversible gates."""

    def inverse(self) -> '_ReversedReversibleCompositeGate':
        return _ReversedReversibleCompositeGate(self)


class _ReversedReversibleCompositeGate(gate_features.CompositeGate,
                                       gate_features.ReversibleGate):
    """A reversed reversible composite gate."""

    def __init__(self, forward_form: ReversibleCompositeGate):
        self.forward_form = forward_form

    def inverse(self) -> ReversibleCompositeGate:
        return self.forward_form

    def default_decompose(self, qubits):
        return inverse_of_invertable_op_tree(
            self.forward_form.default_decompose(qubits))
