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

"""Adds an inverse to gates meeting the `SupportsDecomposeWithQubits` protocol.
"""

from typing import TypeVar, Generic

import abc

from cirq import protocols


TOriginal = TypeVar('TOriginal', bound='ReversibleCompositeGate')


class ReversibleCompositeGate(metaclass=abc.ABCMeta):
    """A composite gate that gets decomposed into reversible gates."""

    def __pow__(self: TOriginal,
                power) -> '_ReversedReversibleCompositeGate[TOriginal]':
        if power != -1:
            return NotImplemented
        return _ReversedReversibleCompositeGate(self)

    @abc.abstractmethod
    def _decompose_(self, qubits):
        pass


class _ReversedReversibleCompositeGate(Generic[TOriginal]):
    """A reversed reversible composite gate."""

    def __init__(self, forward_form: TOriginal) -> None:
        self.forward_form = forward_form

    def __pow__(self, power) -> TOriginal:
        if power != -1:
            return NotImplemented
        return self.forward_form

    def _decompose_(self, qubits):
        return protocols.inverse(protocols.decompose_once_with_qubits(
            self.forward_form, qubits))
