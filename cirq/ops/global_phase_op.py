# Copyright 2019 The Cirq Developers
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
"""A no-qubit global phase operation."""

import numpy as np

from cirq import value, protocols
from cirq.ops import raw_types


@value.value_equality(approximate=True)
class GlobalPhaseOperation(raw_types.Operation):

    def __init__(self, coefficient, atol=1e-8):
        if abs(1 - abs(coefficient)) > atol:
            raise ValueError(
                'Coefficient is not unitary: {!r}'.format(coefficient))
        self.coefficient = coefficient

    @property
    def qubits(self):
        return ()

    def with_qubits(self, *new_qubits):
        if new_qubits:
            raise ValueError(
                '{!r} applies to 0 qubits but new_qubits={!r}.'.format(
                    self, new_qubits))
        return self

    def _value_equality_values_(self):
        return self.coefficient

    def _has_unitary_(self):
        return True

    def __pow__(self, power):
        if isinstance(power, (int, float)):
            return GlobalPhaseOperation(self.coefficient**power)
        return NotImplemented

    def _unitary_(self):
        return np.array([[self.coefficient]])

    def _apply_unitary_(self, args):
        args.target_tensor *= self.coefficient
        return args.target_tensor

    def __str__(self):
        return str(self.coefficient)

    def __repr__(self):
        return 'cirq.GlobalPhaseOperation({!r})'.format(self.coefficient)

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['coefficient'])
