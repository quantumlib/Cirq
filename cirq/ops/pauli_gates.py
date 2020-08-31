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
import abc
from typing import Any, cast, Tuple, TYPE_CHECKING, Union, Dict

from cirq._doc import document
from cirq.ops import common_gates, raw_types, identity
from cirq.type_workarounds import NotImplementedType


if TYPE_CHECKING:
    import cirq
    from cirq.ops.pauli_string import SingleQubitPauliStringGateOperation
    from cirq.value.product_state import (_XEigenState, _YEigenState,
                                          _ZEigenState)  # coverage: ignore


class Pauli(raw_types.Gate, metaclass=abc.ABCMeta):
    """Represents the Pauli gates.

    This is an abstract class with no public subclasses. The only instances
    of private subclasses are the X, Y, or Z Pauli gates defined below.
    """
    _XYZ = None  # type: Tuple[Pauli, Pauli, Pauli]

    @staticmethod
    def by_index(index: int) -> 'Pauli':
        return Pauli._XYZ[index % 3]

    @staticmethod
    def by_relative_index(p: 'Pauli', relative_index: int) -> 'Pauli':
        return Pauli._XYZ[(p._index + relative_index) % 3]

    def __init__(self, index: int, name: str) -> None:
        self._index = index
        self._name = name

    def num_qubits(self):
        return 1

    def _commutes_(self, other: Any,
                   atol: float) -> Union[bool, NotImplementedType, None]:
        if not isinstance(other, Pauli):
            return NotImplemented
        return self is other

    def third(self, second: 'Pauli') -> 'Pauli':
        return Pauli._XYZ[(-self._index - second._index) % 3]

    def relative_index(self, second: 'Pauli') -> int:
        """Relative index of self w.r.t. second in the (X, Y, Z) cycle."""
        return (self._index - second._index + 1) % 3 - 1

    def phased_pauli_product(
            self, other: Union['cirq.Pauli', 'identity.IdentityGate']
    ) -> Tuple[complex, Union['cirq.Pauli', 'identity.IdentityGate']]:
        if self == other:
            return 1, identity.I
        if other is identity.I:
            return 1, self
        return 1j**cast(Pauli, other).relative_index(self), self.third(
            cast(Pauli, other))

    def __gt__(self, other):
        if not isinstance(other, Pauli):
            return NotImplemented
        return (self._index - other._index) % 3 == 1

    def __lt__(self, other):
        if not isinstance(other, Pauli):
            return NotImplemented
        return (other._index - self._index) % 3 == 1

    def on(self, *qubits: 'cirq.Qid') -> 'SingleQubitPauliStringGateOperation':
        """Returns an application of this gate to the given qubits.

        Args:
            *qubits: The collection of qubits to potentially apply the gate to.
        """
        if len(qubits) != 1:
            raise ValueError(
                'Expected a single qubit, got <{!r}>.'.format(qubits))
        from cirq.ops.pauli_string import SingleQubitPauliStringGateOperation
        return SingleQubitPauliStringGateOperation(self, qubits[0])

    @property
    def _canonical_exponent(self):
        """Overrides EigenGate._canonical_exponent in subclasses."""
        return 1

class _PauliX(Pauli, common_gates.XPowGate):

    def __init__(self):
        Pauli.__init__(self, index=0, name='X')
        common_gates.XPowGate.__init__(self, exponent=1.0)

    def __pow__(self: '_PauliX',
                exponent: 'cirq.TParamVal') -> common_gates.XPowGate:
        return common_gates.XPowGate(exponent=exponent)

    def _with_exponent(self: '_PauliX',
                       exponent: 'cirq.TParamVal') -> common_gates.XPowGate:
        return self.__pow__(exponent)

    @classmethod
    def _from_json_dict_(cls, exponent, global_shift, **kwargs):
        assert global_shift == 0
        assert exponent == 1
        return Pauli._XYZ[0]

    @property
    def basis(self: '_PauliX') -> Dict[int, '_XEigenState']:
        from cirq.value.product_state import _XEigenState
        return {
            +1: _XEigenState(+1),
            -1: _XEigenState(-1),
        }


class _PauliY(Pauli, common_gates.YPowGate):

    def __init__(self):
        Pauli.__init__(self, index=1, name='Y')
        common_gates.YPowGate.__init__(self, exponent=1.0)

    def __pow__(self: '_PauliY',
                exponent: 'cirq.TParamVal') -> common_gates.YPowGate:
        return common_gates.YPowGate(exponent=exponent)

    def _with_exponent(self: '_PauliY',
                       exponent: 'cirq.TParamVal') -> common_gates.YPowGate:
        return self.__pow__(exponent)

    @classmethod
    def _from_json_dict_(cls, exponent, global_shift, **kwargs):
        assert global_shift == 0
        assert exponent == 1
        return Pauli._XYZ[1]

    @property
    def basis(self: '_PauliY') -> Dict[int, '_YEigenState']:
        from cirq.value.product_state import _YEigenState
        return {
            +1: _YEigenState(+1),
            -1: _YEigenState(-1),
        }


class _PauliZ(Pauli, common_gates.ZPowGate):

    def __init__(self):
        Pauli.__init__(self, index=2, name='Z')
        common_gates.ZPowGate.__init__(self, exponent=1.0)

    def __pow__(self: '_PauliZ',
                exponent: 'cirq.TParamVal') -> common_gates.ZPowGate:
        return common_gates.ZPowGate(exponent=exponent)

    def _with_exponent(self: '_PauliZ',
                       exponent: 'cirq.TParamVal') -> common_gates.ZPowGate:
        return self.__pow__(exponent)

    @classmethod
    def _from_json_dict_(cls, exponent, global_shift, **kwargs):
        assert global_shift == 0
        assert exponent == 1
        return Pauli._XYZ[2]

    @property
    def basis(self: '_PauliZ') -> Dict[int, '_ZEigenState']:
        from cirq.value.product_state import _ZEigenState
        return {
            +1: _ZEigenState(+1),
            -1: _ZEigenState(-1),
        }


X = _PauliX()
document(
    X, """The Pauli X gate.

    Matrix:

        [[0, 1],
         [1, 0]]
    """)

Y = _PauliY()
document(
    Y, """The Pauli Y gate.

    Matrix:

        [[0, -i],
         [i, 0]]
    """)

Z = _PauliZ()
document(
    Z, """The Pauli Z gate.

    Matrix:

        [[1, 0],
         [0, -1]]
    """)

Pauli._XYZ = (X, Y, Z)
