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
import numbers
from typing import Union, List, Optional, Sequence, TYPE_CHECKING, Any, Tuple, TypeVar, Type, Iterable, SupportsComplex, \
    cast
import abc

from cirq import protocols, linalg, value
from cirq.ops import (raw_types, common_gates, pauli_gates, global_phase_op,
                      pauli_string, gate_operation)
import numpy as np

if TYPE_CHECKING:
    import cirq

# Order is important! Index equals numeric value.
PAULI_CHARS = 'IXZY'
PAULI_GATES: List['cirq.Gate'] = [
    common_gates.I, pauli_gates.X, pauli_gates.Z, pauli_gates.Y
]


TCls = TypeVar('TCls', bound='BaseDensePauliString')


@value.value_equality(approximate=True, distinct_child_types=True)
class BaseDensePauliString(raw_types.Gate, metaclass=abc.ABCMeta):
    """Parent class for `DensePauliString` and `MutableDensePauliString`."""

    I_MASK = 0
    X_MASK = 1
    Z_MASK = 2
    Y_MASK = X_MASK | Z_MASK

    def __init__(self,
                 *,
                 pauli_mask: Union[Iterable[int], np.ndarray],
                 coefficient: Union[int, float, complex] = 1):
        """Initializes a new dense pauli string.

        Args:
            pauli_mask: A compact representation of the Pauli gates to use.
                Each entry in the array is a Pauli, with 0=I, 1=X, 2=Z, 3=Y.
                Must be a 1-dimensional uint8 numpy array containing no value
                larger than 3.
            coefficient: A complex number. Usually +1, -1, 1j, or -1j but other
                values are supported.
        """
        self.pauli_mask = np.asarray(pauli_mask, dtype=np.uint8)
        self.coefficient = complex(coefficient)

    def _value_equality_values_(self):
        # Note: can't use pauli_mask directly.
        # Would cause approx_eq to false positive when atol > 1.
        return self.coefficient, tuple(PAULI_CHARS[p] for p in self.pauli_mask)

    @classmethod
    def one_hot(cls: Type[TCls], *, index: int, length: int,
                pauli: Union[str, 'cirq.Gate']) -> TCls:
        """Creates a dense pauli string with only one non-identity Pauli.

        Args:
            index: The index of the Pauli that is not an identity.
            pauli: The pauli gate to put at the hot index. Can be set to either
                a string ('X', 'Y', 'Z', 'I') or a cirq gate (`cirq.X`,
                `cirq.Y`, `cirq.Z`, or ``cirq.I`).
        """
        mask = np.zeros(length, dtype=np.uint8)
        mask[index] = _pauli_index(pauli)
        concrete_cls = DensePauliString if cls is BaseDensePauliString else cls
        return concrete_cls(pauli_mask=mask)

    @classmethod
    def eye(cls: Type[TCls], length: int) -> TCls:
        """Creates a dense pauli string containing only identity gates.

        Args:
            length: The length of the dense pauli string.
        """
        concrete_cls = DensePauliString if cls is BaseDensePauliString else cls
        return concrete_cls(pauli_mask=np.zeros(length, dtype=np.uint8))

    def _num_qubits_(self):
        return len(self)

    def _has_unitary_(self):
        return not self._is_parameterized_() and (
                abs(abs(self.coefficient) - 1) < 1e-8)

    def _unitary_(self):
        if not self._has_unitary_():
            return NotImplemented
        return self.coefficient * linalg.kron(
            *[protocols.unitary(PAULI_GATES[p]) for p in self.pauli_mask])

    def _apply_unitary_(self, args):
        if not self._has_unitary_():
            return NotImplemented
        from cirq import devices
        qubits = devices.LineQubit.range(len(self))
        return protocols.apply_unitaries(self._decompose_(qubits), qubits, args)

    def _decompose_(self, qubits):
        if not self._has_unitary_():
            return NotImplemented
        result = [
            PAULI_GATES[p].on(q) for p, q in zip(self.pauli_mask, qubits) if p
        ]
        if self.coefficient != 1:
            result.append(global_phase_op.GlobalPhaseOperation(self.coefficient))
        return result

    def _is_parameterized_(self):
        return protocols.is_parameterized(self.coefficient)

    def _resolve_parameters_(self, resolver):
        return self.copy(coefficient=protocols.resolve_parameters(
            self.coefficient, resolver))

    def __pos__(self):
        return self

    def __pow__(self, power):
        if isinstance(power, int):
            i_group = [1, +1j, -1, -1j]
            if self.coefficient in i_group:
                coef = i_group[i_group.index(self.coefficient) * power % 4]
            else:
                coef = self.coefficient**power
            if power % 2 == 0:
                return coef * DensePauliString.eye(len(self))
            return DensePauliString(coefficient=coef,
                                    pauli_mask=self.pauli_mask)
        return NotImplemented

    @classmethod
    def from_text(cls: Type[TCls], text: str) -> TCls:
        # Separate into coefficient and pauli text.
        if '*' in text:
            parts = text.split('*')
            if len(parts) != 2:
                raise ValueError(
                    f'Pauli strings text cannot contain two "*"s.\n'
                    f'text={repr(text)}.')
            coefficient = complex(parts[0])
            pauli_text = parts[1]
        elif text.startswith('+'):
            coefficient = 1
            pauli_text = text[1:]
        elif text.startswith('-'):
            coefficient = -1
            pauli_text = text[1:]
        else:
            coefficient = 1
            pauli_text = text

        pauli_mask = np.zeros(len(pauli_text), dtype=np.uint8)
        for i in range(len(pauli_text)):
            c = pauli_text[i]
            try:
                pauli_mask[i] = PAULI_CHARS.index(c)
            except IndexError:
                raise ValueError(
                    f'Text contains non-Pauli-character {repr(c)} in Pauli '
                    f'part.  Valid Pauli characters are upper case IXYZ.\n'
                    f'text={repr(text)}.')

        concrete_cls = DensePauliString if cls is BaseDensePauliString else cls
        return concrete_cls(coefficient=coefficient, pauli_mask=pauli_mask)

    def __getitem__(self, item):
        if isinstance(item, int):
            return PAULI_GATES[self.pauli_mask[item]]

        if isinstance(item, slice):
            return type(self)(coefficient=1, pauli_mask=self.pauli_mask[item])

        return NotImplemented

    def __len__(self):
        return len(self.pauli_mask)

    def __neg__(self):
        return DensePauliString(coefficient=-self.coefficient,
                                pauli_mask=self.pauli_mask)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return self.__mul__(1 / other)

    def __mul__(self, other):
        if isinstance(other, BaseDensePauliString):
            max_len = max(len(self.pauli_mask), len(other.pauli_mask))
            min_len = min(len(self.pauli_mask), len(other.pauli_mask))
            new_mask = np.zeros(max_len, dtype=np.uint8)
            new_mask[:len(self.pauli_mask)] ^= self.pauli_mask
            new_mask[:len(other.pauli_mask)] ^= other.pauli_mask
            tweak = _vectorized_pauli_mul_phase(self.pauli_mask[:min_len],
                                     other.pauli_mask[:min_len])
            return DensePauliString(pauli_mask=new_mask,
                                    coefficient=self.coefficient *
                                    other.coefficient * tweak)

        if isinstance(other, numbers.Number):
            return DensePauliString(
                pauli_mask=self.pauli_mask,
                coefficient=self.coefficient * complex(cast(SupportsComplex, other)))

        split = _attempt_value_to_pauli_index(other)
        if split is not None:
            p, i = split
            mask = np.copy(self.pauli_mask)
            mask[i] ^= p
            return DensePauliString(pauli_mask=mask,
                                    coefficient=self.coefficient *
                                                _vectorized_pauli_mul_phase(self.pauli_mask[i], p))

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return self.__mul__(other)

        split = _attempt_value_to_pauli_index(other)
        if split is not None:
            p, i = split
            mask = np.copy(self.pauli_mask)
            mask[i] ^= p
            return DensePauliString(pauli_mask=mask,
                                    coefficient=self.coefficient *
                                                _vectorized_pauli_mul_phase(p, self.pauli_mask[i]))

        return NotImplemented

    def tensor_product(self,
                       other: 'BaseDensePauliString') -> 'DensePauliString':
        return DensePauliString(
            coefficient=self.coefficient * other.coefficient,
            pauli_mask=np.concatenate([self.pauli_mask, other.pauli_mask]))

    def __abs__(self):
        return DensePauliString(coefficient=abs(self.coefficient),
                                pauli_mask=self.pauli_mask)

    def sparse(self, qubits: Optional[Sequence['cirq.Qid']] = None
              ) -> 'cirq.PauliString':
        if qubits is None:
            from cirq import devices
            qubits = devices.LineQubit.range(len(self))

        if len(qubits) != len(self):
            raise ValueError('Wrong number of qubits.')

        return pauli_string.PauliString(
            coefficient=self.coefficient,
            qubit_pauli_map={
                q: PAULI_GATES[p] for q, p in zip(qubits, self.pauli_mask) if p
            })

    def __str__(self):
        if self.coefficient == 1:
            coef = '+'
        elif self.coefficient == -1:
            coef = '-'
        else:
            coef = repr(self.coefficient) + '*'
        mask = ''.join(PAULI_CHARS[p] for p in self.pauli_mask)
        return coef + mask

    def _commutes_(self, other):
        if isinstance(other, BaseDensePauliString):
            n = min(len(self.pauli_mask), len(other.pauli_mask))
            phase = _vectorized_pauli_mul_phase(self.pauli_mask[:n], other.pauli_mask[:n])
            return phase == 1 or phase == -1

        # Single qubit Pauli operation.
        split = _attempt_value_to_pauli_index(other)
        if split is not None:
            p1, i = split
            p2 = self.pauli_mask[i]
            return (p1 or p2) == (p2 or p1)

        return NotImplemented

    def frozen(self) -> 'DensePauliString':
        return DensePauliString(coefficient=self.coefficient,
                                pauli_mask=self.pauli_mask)

    def mutable_copy(self) -> 'MutableDensePauliString':
        return MutableDensePauliString(coefficient=self.coefficient,
                                       pauli_mask=np.copy(self.pauli_mask))

    @abc.abstractmethod
    def copy(self,
             coefficient: Optional[complex] = None,
             pauli_mask: Union[None, Iterable[int], np.ndarray] = None) -> 'BaseDensePauliString':
        pass


class DensePauliString(BaseDensePauliString):

    def __init__(self,
                 *,
                 pauli_mask: Union[Iterable[int], np.ndarray],
                 coefficient: Union[int, float, complex] = 1):
        super().__init__(coefficient=coefficient,
                         pauli_mask=np.array(pauli_mask, dtype=np.uint8))

    def frozen(self) -> 'DensePauliString':
        return self

    def copy(self,
             coefficient: Optional[complex] = None,
             pauli_mask: Union[None, Iterable[int], np.ndarray] = None) -> 'DensePauliString':
        if pauli_mask is None and (coefficient is None or
                                   coefficient == self.coefficient):
            return self
        return DensePauliString(
            coefficient=self.coefficient
            if coefficient is None else coefficient,
            pauli_mask=self.pauli_mask if pauli_mask is None else pauli_mask)

    def __repr__(self):
        return f'cirq.DensePauliString.from_text({repr(str(self))})'


@value.value_equality(unhashable=True, approximate=True)
class MutableDensePauliString(BaseDensePauliString):

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if isinstance(value, (str, raw_types.Gate)):
                self.pauli_mask[key] = _pauli_index(value)
                return self

        if isinstance(key, slice):
            if isinstance(value, str):
                value = BaseDensePauliString.from_text(value)
            if isinstance(value, BaseDensePauliString):
                if value.coefficient != 1:
                    raise ValueError(
                        "Can't slice-assign from a PauliProduct whose "
                        "coefficient is not 1.\n"
                        "\nWorkaround: If you just want to ignore the "
                        "coefficient, do `= value[:]` instead of `= value`.")
                self.pauli_mask[key] = value.pauli_mask
                return self

        return NotImplemented

    def __imul__(self, other):
        if isinstance(other, BaseDensePauliString):
            if len(other) > len(self):
                raise ValueError(
                    "The receiving dense pauli string is smaller than "
                    "the dense pauli string being multiplied into it.\n"
                    f"self={repr(self)}\n"
                    f"other={repr(other)}")
            self_mask = self.pauli_mask[:len(other.pauli_mask)]
            self.coefficient *= _vectorized_pauli_mul_phase(self_mask, other.pauli_mask)
            self.coefficient *= other.coefficient
            self_mask ^= other.pauli_mask
            return self

        if isinstance(other, numbers.Number):
            self.coefficient *= complex(cast(SupportsComplex, other))
            return self

        split = _attempt_value_to_pauli_index(other)
        if split is not None:
            p, i = split
            self.coefficient *= _vectorized_pauli_mul_phase(self.pauli_mask[i], p)
            self.pauli_mask[i] ^= p
            return self

        return NotImplemented

    def copy(self,
             coefficient: Optional[complex] = None,
             pauli_mask: Union[None, Iterable[int], np.ndarray] = None
            ) -> 'MutableDensePauliString':
        return MutableDensePauliString(
            coefficient=self.coefficient
            if coefficient is None else coefficient,
            pauli_mask=np.copy(
                self.pauli_mask if pauli_mask is None else pauli_mask))

    def __str__(self):
        return super().__str__() + ' (mutable)'

    def __repr__(self):
        return f'cirq.MutableDensePauliString.from_text' \
               f'({repr(BaseDensePauliString.__str__(self))})'

    @classmethod
    def inline_gaussian_elimination(cls, rows: 'List[MutableDensePauliString]'
                                   ) -> None:
        if not rows:
            return

        def sig(product: 'BaseDensePauliString') -> int:
            t = 0
            for p in product.pauli_mask:
                t |= p
            return t

        height = len(rows)
        width = len(rows[0])
        next_row = 0

        for _ in range(width):
            for pauli_base in [
                    BaseDensePauliString.X_MASK, BaseDensePauliString.Z_MASK
            ]:
                # Locate pivot row.
                for k in range(next_row, height):
                    if sig(rows[k]) & pauli_base:
                        pivot_row = k
                        break
                else:
                    continue

                # Eliminate column entry in other rows.
                for k in range(height):
                    if pivot_row != k and sig(rows[k]) & pauli_base:
                        rows[k].__imul__(rows[pivot_row])

                # Keep it sorted.
                if pivot_row != next_row:
                    rows[next_row], rows[pivot_row] = (rows[pivot_row],
                                                       rows[next_row])

                next_row += 1


def _pauli_index(val: Union[str, 'cirq.Gate']):
    matcher = PAULI_CHARS if isinstance(val, str) else PAULI_GATES
    return matcher.index(val)


def _attempt_value_to_pauli_index(v: Any) -> Optional[Tuple[int, int]]:
    pauli_gate = gate_operation.op_gate_of_type(v, pauli_gates.Pauli)
    if pauli_gate is None:
        return None

    q = v.qubits[0]
    from cirq import  devices
    if not isinstance(q, devices.LineQubit):
        raise ValueError(
            'Got a Pauli operation, but it was applied to a qubit type '
            'other than `cirq.LineQubit` so its dense index is ambiguous.\n'
            f'v={repr(v)}.')
    return PAULI_GATES.index(pauli_gate), q.x


def _vectorized_pauli_mul_phase(lhs: Union[int, np.ndarray],
                                rhs: Union[int, np.ndarray]) -> complex:
    """Computes the leading coefficient of a pauli string multiplication.

    The two inputs must have the same length. They must follow the convention
    that I=0, X=1, Z=2, Y=3 and have no out-of-range values.

    Args:
        lhs: Left hand side `pauli_mask` from `DensePauliString`.
        rhs: Right hand side `pauli_mask` from `DensePauliString`.

    Returns:
        1, 1j, -1, or -1j.
    """

    # Vectorized computation of per-term phase exponents.
    t = np.array(lhs, dtype=np.int8)
    t *= rhs != 0
    t -= rhs * (lhs != 0)
    t += 1
    t %= 3
    t -= 1

    # Result is i raised to the sum of the per-term phase exponents.
    s = np.sum(t, dtype=np.uint8) & 3
    return 1j**s
