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

"""Implements the inverse method of a CompositeOperation & ReversibleEffect."""
from typing import TypeVar, Generic, Any

from cirq import abc, protocols
from cirq.ops import gate_features

TOriginal = TypeVar('TOriginal', bound='ReversibleCompositeGate')


class ReversibleCompositeGate(gate_features.CompositeGate,
                              metaclass=abc.ABCMeta):
    """A composite gate that gets decomposed into reversible gates."""

    def __pow__(self: TOriginal, exponent: Any
                ) -> '_ReversedReversibleCompositeGate[TOriginal]':
        if exponent != -1:
            return NotImplemented
        return _ReversedReversibleCompositeGate(self)


class _ReversedReversibleCompositeGate(Generic[TOriginal],
                                       gate_features.CompositeGate):
    """A reversed reversible composite gate."""

    def __init__(self, forward_form: TOriginal) -> None:
        self.forward_form = forward_form

    def __pow__(self, exponent: Any) -> TOriginal:
        if exponent != -1:
            return NotImplemented
        return self.forward_form

    def default_decompose(self, qubits):
        return protocols.inverse(self.forward_form.default_decompose(qubits))
