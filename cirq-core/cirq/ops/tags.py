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

"""Canonical tags for the TaggedOperation class."""

from __future__ import annotations


class VirtualTag:
    """A `cirq.TaggedOperation` tag indicating that the operation is virtual.

    Virtual operations are ones that do not correspond to some physical signal sent
    to the quantum computer. An example of such an operation is a Z rotation gates
    where the gate is not enacted in the circuit, but instead is tracked in software.
    Another example is noise that has been added to a gate to make it appear as
    a noisy gate in a `cirq.NoiseModel`.

    Operations marked with this tag are presumed to have zero duration of their
    own, although they may have a non-zero duration if run in the same Moment
    as a non-virtual operation.
    """

    def __eq__(self, other):
        return isinstance(other, VirtualTag)

    def __str__(self) -> str:
        return '<virtual>'

    def __repr__(self) -> str:
        return 'cirq.VirtualTag()'

    def _json_dict_(self) -> dict[str, str]:
        return {}

    def __hash__(self):
        return hash(VirtualTag)


class RoutingSwapTag:
    """A 'cirq.TaggedOperation' tag indicated that the operation is an inserted SWAP.

    A RoutingSwapTag is meant to be used to distinguish SWAP operations that are inserted during
    a routing procedure and SWAP operations that are part of the original circuit before routing.
    """

    def __eq__(self, other):
        return isinstance(other, RoutingSwapTag)

    def __str__(self) -> str:
        return '<r>'

    def __repr__(self) -> str:
        return 'cirq.RoutingSwapTag()'

    def _json_dict_(self) -> dict[str, str]:
        return {}

    def __hash__(self):
        return hash(RoutingSwapTag)
