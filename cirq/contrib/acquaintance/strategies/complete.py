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

from typing import Sequence

from cirq import circuits, ops

from cirq.contrib.acquaintance.devices import UnconstrainedAcquaintanceDevice
from cirq.contrib.acquaintance.gates import acquaint
from cirq.contrib.acquaintance.mutation_utils import (
    expose_acquaintance_gates, replace_acquaintance_with_swap_network)


def complete_acquaintance_strategy(qubit_order: Sequence[ops.Qid],
                                   acquaintance_size: int=0,
                                   ) -> circuits.Circuit:
    """
    Returns an acquaintance strategy capable of executing a gate corresponding
    to any set of at most acquaintance_size qubits.

    Args:
        qubit_order: The qubits on which the strategy should be defined.
        acquaintance_size: The maximum number of qubits to be acted on by
        an operation.

    Returns:
        An circuit capable of implementing any set of k-local
        operation.
    """
    if acquaintance_size < 0:
        raise ValueError('acquaintance_size must be non-negative.')
    elif acquaintance_size == 0:
        return circuits.Circuit(device=UnconstrainedAcquaintanceDevice)

    if acquaintance_size > len(qubit_order):
        return circuits.Circuit(device=UnconstrainedAcquaintanceDevice)
    if acquaintance_size == len(qubit_order):
        return circuits.Circuit.from_ops(
                acquaint(*qubit_order), device=UnconstrainedAcquaintanceDevice)

    strategy = circuits.Circuit.from_ops(
            (acquaint(q) for q in qubit_order),
            device=UnconstrainedAcquaintanceDevice)
    for size_to_acquaint in range(2, acquaintance_size + 1):
        expose_acquaintance_gates(strategy)
        replace_acquaintance_with_swap_network(
                strategy, qubit_order, size_to_acquaint)
    return strategy
