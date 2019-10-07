from typing import Union, List, Optional, Sequence, TYPE_CHECKING
import abc

from cirq import protocols, linalg, devices, value
from cirq.ops import raw_types, common_gates, pauli_gates, global_phase_op, pauli_string
import numpy as np

if TYPE_CHECKING:
    import cirq


I_VAL = 0
X_VAL = 1
Z_VAL = 2
Y_VAL = X_VAL | Z_VAL

# Order is important! Index equals numeric value.
PAULI_CHARS = 'IXZY'
PAULI_GATES: List['cirq.Gate'] = [common_gates.I,
                                  pauli_gates.X,
                                  pauli_gates.Z,
                                  pauli_gates.Y]


class BaseDensePauliString(metaclass=abc.ABCMeta):
    def __init__(self,
                 *,
                 pauli_mask: np.ndarray,
                 coefficient: Union[int, float, complex] = 1):
        self.pauli_mask = pauli_mask
        self.coefficient = complex(coefficient)

    def _value_equality_values_(self):
        return self.coefficient, tuple(self.pauli_mask)

    def _value_equality_values_cls_(self):
        return BaseDensePauliString

    def _num_qubits_(self):
        return len(self)

    def _has_unitary_(self):
        return not self._is_parameterized_()

    def _unitary_(self):
        if self._is_parameterized_():
            return NotImplemented
        return linalg.kron(*[protocols.unitary(PAULI_GATES[p])
                             for p in self.pauli_mask])

    def _apply_unitary_(self, args):
        if self._is_parameterized_():
            return NotImplemented
        qubits = devices.LineQubit.range(len(self))
        protocols.apply_unitaries(
            self._decompose_(qubits),
            qubits,
            args)

    def _decompose_(self, qubits):
        result = [
            PAULI_GATES[p].on(q)
            for p, q in zip(self.pauli_mask, qubits)
            if p
        ]
        result.append(global_phase_op.GlobalPhaseOperation(self.coefficient))
        return result

    def _is_parameterized_(self):
        return protocols.is_parameterized(self.coefficient)

    def _resolve_parameters_(self, resolver):
        return self.copy(
            coefficient=protocols.resolve_parameters(
                self.coefficient, resolver))

    @classmethod
    def _parse(cls, text: str):
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

        return cls(coefficient=coefficient, pauli_mask=pauli_mask)

    def __getitem__(self, item):
        if isinstance(item, int):
            return PAULI_GATES[self.pauli_mask[item]]

        if isinstance(item, slice):
            return type(self)(coefficient=1,
                              pauli_mask=self.pauli_mask[item])

        return NotImplemented

    def __len__(self):
        return len(self.pauli_mask)

    def __mul__(self, other):
        if isinstance(other, BaseDensePauliString):
            max_len = max(len(self.pauli_mask), len(other.pauli_mask))
            min_len = min(len(self.pauli_mask), len(other.pauli_mask))
            new_mask = np.zeros(max_len, dtype=np.uint8)
            new_mask[:len(self.pauli_mask)] ^= self.pauli_mask
            new_mask[:len(other.pauli_mask)] ^= other.pauli_mask
            tweak = _pauli_mul_phase(self.pauli_mask[:min_len],
                                     other.pauli_mask[:min_len])
            return DensePauliString(
                pauli_mask=new_mask,
                coefficient=self.coefficient * other.coefficient * tweak)

        if isinstance(other, complex):
            return DensePauliString(
                pauli_mask=self.pauli_mask,
                coefficient=self.coefficient * other)

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, complex):
            return DensePauliString(
                pauli_mask=self.pauli_mask,
                coefficient=self.coefficient * other)

        return NotImplemented

    def tensor_product(self, other: 'BaseDensePauliString'
                       ) -> 'DensePauliString':
        return DensePauliString(
            coefficient=self.coefficient * other.coefficient,
            pauli_mask=np.concatenate([self.pauli_mask, other.pauli_mask]))

    def __abs__(self):
        return DensePauliString(
            coefficient=abs(self.coefficient),
            pauli_mask=self.pauli_mask)

    def sparse(self, qubits: Optional[Sequence['cirq.Qid']] = None) -> 'cirq.PauliString':
        if qubits is None:
            qubits = devices.LineQubit.range(len(self))

        if len(qubits) != len(self):
            raise ValueError('Wrong number of qubits.')

        return pauli_string.PauliString(
            coefficient=self.coefficient,
            qubit_pauli_map={
                q: PAULI_GATES[p]
                for q, p in zip(qubits, self.pauli_mask)
                if p
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
            phase = _pauli_mul_phase(
                self.pauli_mask[:n], other.pauli_mask[:n])
            return phase == 1 or phase == -1

        return NotImplemented

    def frozen_copy(self) -> 'DensePauliString':
        return DensePauliString(
            coefficient=self.coefficient,
            pauli_mask=self.pauli_mask)

    def mutable_copy(self) -> 'MutableDensePauliString':
        return MutableDensePauliString(
            coefficient=self.coefficient,
            pauli_mask=np.copy(self.pauli_mask))

    @abc.abstractmethod
    def copy(self,
             coefficient: Optional[complex] = None,
             pauli_mask: Optional[np.ndarray] = None) -> 'BaseDensePauliString':
        pass


@value.value_equality(manual_cls=True)
class DensePauliString(raw_types.Gate, BaseDensePauliString):
    def __init__(self,
                 *,
                 pauli_mask: np.ndarray,
                 coefficient: Union[int, float, complex] = 1):
        super().__init__(
            coefficient=coefficient,
            pauli_mask=np.copy(pauli_mask))

    @classmethod
    def from_text(cls, text: str) -> 'DensePauliString':
        return cls._parse(text)

    def copy(self,
             coefficient: Optional[complex] = None,
             pauli_mask: Optional[np.ndarray] = None) -> 'DensePauliString':
        if pauli_mask is None and (
                coefficient is None or coefficient == self.coefficient):
            return self
        return DensePauliString(
            coefficient=self.coefficient if coefficient is None else coefficient,
            pauli_mask=self.pauli_mask if pauli_mask is None else pauli_mask)

    def __repr__(self):
        return f'cirq.DensePauliString.from_text({repr(str(self))})'


@value.value_equality(unhashable=True, manual_cls=True)
class MutableDensePauliString(BaseDensePauliString):
    @classmethod
    def from_text(cls, text: str) -> 'MutableDensePauliString':
        return cls._parse(text)

    def __setitem__(self, key, value):
        if isinstance(value, raw_types.Gate):
            self.pauli_mask[key] = PAULI_GATES.index(value)
            return self

        if isinstance(key, slice) and isinstance(value, BaseDensePauliString):
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
            self_mask = self.pauli_mask[:len(other.pauli_mask)]
            self.coefficient *= _pauli_mul_phase(self_mask, other.pauli_mask)
            self.coefficient *= other.coefficient
            self_mask ^= other.pauli_mask
            return self

        if isinstance(other, complex):
            self.coefficient *= other
            return self

        return NotImplemented

    def copy(self,
             coefficient: Optional[complex] = None,
             pauli_mask: Optional[np.ndarray] = None) -> 'MutableDensePauliString':
        return MutableDensePauliString(
            coefficient=self.coefficient if coefficient is None else coefficient,
            pauli_mask=np.copy(self.pauli_mask if pauli_mask is None else pauli_mask))

    def __str__(self):
        return super().__str__() + ' (mutable)'

    def __repr__(self):
        return f'cirq.MutableDensePauliString.from_text({repr(str(self))})'

    @classmethod
    def inline_gaussian_elimination(
            cls, rows: 'List[MutableDensePauliString]') -> None:
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

        for col in range(width):
            for pauli_base in [X_VAL, Z_VAL]:
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
                    rows[next_row], rows[pivot_row] = (
                        rows[pivot_row], rows[next_row])

                next_row += 1


def _pauli_mul_phase(lhs: np.ndarray, rhs: np.ndarray) -> complex:
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
    t = lhs * (rhs != 0)
    t -= rhs * (lhs != 0)
    t += 1
    t %= 3
    t -= 1

    # Result is i raised to the sum of the per-term phase exponents.
    s = np.sum(t, dtype=np.uint8) & 3
    return 1j**s
