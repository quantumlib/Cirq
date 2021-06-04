from collections import defaultdict
import functools
import math
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
import sympy.parsing.sympy_parser as sympy_parser

import cirq
from cirq import value
from cirq.ops import raw_types


class HamiltonianPolynomial:
    """A container class of Boolean function as equation (2) of [1]

    It is essentially a polynomial of Pauli Zs on different qubits. For example, this object could
    represent the polynomial 0.5*I - 0.5*Z_1*Z_2 and it would be stored inside this object as:
    self._hamiltonians = {(): 0.5, (1, 2): 0.5}.

    While this object can represent any polynomial of Pauli Zs, in this file, it will be used to
    represent a Boolean operation which has a unique representation as a polynomial. This object
    only handle basic operation on the polynomials (e.g. multiplication). The construction of a
    polynomial from a Boolean is performed by _build_hamiltonian_from_boolean().

    References:
    [1] On the representation of Boolean and real functions as Hamiltonians for quantum computing
        by Stuart Hadfield, https://arxiv.org/pdf/1804.09130.pdf
    [2] https://www.youtube.com/watch?v=AOKM9BkweVU is a useful intro
    [3] https://github.com/rsln-s/IEEE_QW_2020/blob/master/Slides.pdf
    [4] Efficient quantum circuits for diagonal unitaries without ancillas by Jonathan Welch, Daniel
        Greenbaum, Sarah Mostame, Alán Aspuru-Guzik, https://arxiv.org/abs/1306.3991
    """

    def __init__(self, hamiltonians: DefaultDict[Tuple[int, ...], float]):
        # The representation is Tuple[int, ...] to weights. The tuple contains the integers of
        # where Z_i is present. For example, Z_0.Z_3 would be (0, 3), and I is the empty tuple.
        self._hamiltonians: DefaultDict[Tuple[int, ...], float] = defaultdict(
            float, {h: w for h, w in hamiltonians.items() if math.fabs(w) > 1e-12}
        )

    @property
    def hamiltonians(self) -> DefaultDict[Tuple[int, ...], float]:
        return self._hamiltonians

    def __repr__(self):
        # For run-to-run identicalness, we sort the keys lexicographically.
        formatted_terms = [
            f"{self._hamiltonians[h]:.2f}*{'*'.join('Z_%d' % d for d in h) if h else 'I'}"
            for h in sorted(self._hamiltonians)
        ]
        return "; ".join(formatted_terms)

    def __add__(self, other: 'HamiltonianPolynomial') -> 'HamiltonianPolynomial':
        return self._signed_add(other, 1.0)

    def __sub__(self, other: 'HamiltonianPolynomial') -> 'HamiltonianPolynomial':
        return self._signed_add(other, -1.0)

    def _signed_add(self, other: 'HamiltonianPolynomial', sign: float) -> 'HamiltonianPolynomial':
        hamiltonians: DefaultDict[Tuple[int, ...], float] = self._hamiltonians.copy()
        for h, w in other.hamiltonians.items():
            hamiltonians[h] += sign * w
        return HamiltonianPolynomial(hamiltonians)

    def __rmul__(self, other: float) -> 'HamiltonianPolynomial':
        return HamiltonianPolynomial(
            defaultdict(float, {k: other * w for k, w in self._hamiltonians.items()})
        )

    def __mul__(self, other: 'HamiltonianPolynomial') -> 'HamiltonianPolynomial':
        hamiltonians: DefaultDict[Tuple[int, ...], float] = defaultdict(float, {})
        for h1, w1 in self._hamiltonians.items():
            for h2, w2 in other.hamiltonians.items():
                # Since we represent the Hamilonians using the indices of the Z_i, when we multiply
                # two Hamiltionians, it's equivalent to doing an XOR of the two sets. For example,
                # if h_A = Z_1 . Z_2 and h_B = Z_1 . Z_3 then the product is:
                # h_A . h_B = Z_1 . Z_2 . Z_1 . Z_3 = Z_1 . Z_1 . Z_2 . Z_3 = Z_2 . Z_3
                # and thus, it is represented by (2, 3). In sort, we do the XOR / symmetric
                # difference of the two tuples.
                h = tuple(sorted(set(h1).symmetric_difference(h2)))
                w = w1 * w2
                hamiltonians[h] += w
        return HamiltonianPolynomial(hamiltonians)

    @staticmethod
    def O() -> 'HamiltonianPolynomial':
        return HamiltonianPolynomial(defaultdict(float, {}))

    @staticmethod
    def I() -> 'HamiltonianPolynomial':
        return HamiltonianPolynomial(defaultdict(float, {(): 1.0}))

    @staticmethod
    def Z(i: int) -> 'HamiltonianPolynomial':
        return HamiltonianPolynomial(defaultdict(float, {(i,): 1.0}))


def _build_hamiltonian_from_boolean(
    boolean_expr: Expr, name_to_id: Dict[str, int]
) -> HamiltonianPolynomial:
    """Builds the Hamiltonian representation of Boolean expression as per [1]:

    Args:
        boolean_expr: A Sympy expression containing symbols and Boolean operations
        name_to_id: A dictionary from symbol name to an integer, typically built by calling
            get_name_to_id().

    Return:
        The HamiltonianPolynomial that represents the Boolean expression.
    """
    if isinstance(boolean_expr, Symbol):
        # Table 1 of [1], entry for 'x' is '1/2.I - 1/2.Z'
        i = name_to_id[boolean_expr.name]
        return 0.5 * HamiltonianPolynomial.I() - 0.5 * HamiltonianPolynomial.Z(i)

    if isinstance(boolean_expr, (And, Not, Or, Xor)):
        sub_hamiltonians = [
            _build_hamiltonian_from_boolean(sub_boolean_expr, name_to_id)
            for sub_boolean_expr in boolean_expr.args
        ]
        # We apply the equalities of theorem 1 of [1].
        if isinstance(boolean_expr, And):
            hamiltonian = HamiltonianPolynomial.I()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian * sub_hamiltonian
        elif isinstance(boolean_expr, Not):
            assert len(sub_hamiltonians) == 1
            hamiltonian = HamiltonianPolynomial.I() - sub_hamiltonians[0]
        elif isinstance(boolean_expr, Or):
            hamiltonian = HamiltonianPolynomial.O()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian + sub_hamiltonian - hamiltonian * sub_hamiltonian
        elif isinstance(boolean_expr, Xor):
            hamiltonian = HamiltonianPolynomial.O()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian + sub_hamiltonian - 2.0 * hamiltonian * sub_hamiltonian
        return hamiltonian

    raise ValueError(f'Unsupported type: {type(boolean_expr)}')


def _gray_code_comparator(k1: Tuple[int, ...], k2: Tuple[int, ...], flip: bool = False) -> int:
    """Compares two Gray-encoded binary numbers.

    Args:
        k1: A tuple of ints, representing the bits that are one. For example, 6 would be (1, 2).
        k2: The second number, represented similarly as k1.
        flip: Whether to flip the comparison.

    Returns:
        -1 if k1 < k2 (or +1 if flip is true)
        0 if k1 == k2
        +1 if k1 > k2 (or -1 if flip is true)
    """
    max_1 = k1[-1] if k1 else -1
    max_2 = k2[-1] if k2 else -1
    if max_1 != max_2:
        return -1 if (max_1 < max_2) ^ flip else 1
    if max_1 == -1:
        return 0
    return _gray_code_comparator(k1[0:-1], k2[0:-1], not flip)


def _get_gates_from_hamiltonians(
    hamiltonian_polynomial_list: List[HamiltonianPolynomial],
    qubits: Sequence['cirq.Qid'],
    theta: float,
    ladder_target: bool = False,
):
    """Builds a circuit according to [1].

    Args:
        hamiltonian_polynomial_list: the list of Hamiltonians, typically built by calling
            _build_hamiltonian_from_boolean().
        qubits: The list of qubits corresponding to the variables.
        theta: A single float scaling the rotations.
        ladder_target: Whether to use convention of figure 7a or 7b.

    Yield:
        Gates that are the decomposition of the Hamiltonian.
    """
    combined = sum(hamiltonian_polynomial_list, HamiltonianPolynomial.O())

    # Here we follow improvements of [4] cancelling out the CNOTs. The first step is to order by
    # Gray code so that as few as possible gates are changed.
    sorted_hs = sorted(
        list(combined.hamiltonians.keys()), key=functools.cmp_to_key(_gray_code_comparator)
    )

    def _simplify_cnots(cnots):
        _control = 0
        _target = 1

        while True:
            # As per equations 9 and 10 of [4], if all the targets (resp. controls) are the same,
            # the cnots commute. Further, if the control (resp. targets) are the same, the cnots
            # can be simplified away:
            # CNOT(i, j) @ CNOT(k, j) = CNOT(k, j) @ CNOT(i, j)
            # CNOT(i, k) @ CNOT(i, j) = CNOT(i, j) @ CNOT(i, k)
            found_simplification = False
            for x, y in [(_control, _target), (_target, _control)]:
                i = 0
                qubit_to_index: Dict[int, int] = {}
                for j in range(1, len(cnots)):
                    if cnots[i][x] != cnots[j][x]:
                        # The targets (resp. control) don't match, so we reset the search.
                        i = j
                        qubit_to_index = {cnots[j][y]: j}
                        continue

                    if cnots[j][y] in qubit_to_index:
                        k = qubit_to_index[cnots[j][y]]
                        # The controls (resp. targets) are the same, so we can simplify away.
                        cnots = [cnots[n] for n in range(len(cnots)) if n != j and n != k]
                        found_simplification = True
                        break

                    qubit_to_index[cnots[j][y]] = j

            if found_simplification:
                continue

            # Here we apply the simplification of equation 11. Note that by flipping the control
            # and target qubits, we have an equally valid identity:
            # CNOT(i, j) @ CNOT(j, k) == CNOT(j, k) @ CNOT(i, k) @ CNOT(i, j)
            # CNOT(j, i) @ CNOT(k, j) == CNOT(k, j) @ CNOT(k, i) @ CNOT(j, i)
            for x, y in [(_control, _target), (_target, _control)]:
                # We investigate potential pivots sequentially.
                for j in range(1, len(cnots) - 1):
                    # First, we look back for as long as the targets (resp. triggers) are the same.
                    # They all commute, so all are potential candidates for being simplified.
                    common_A: Dict[int, int] = {}
                    for i in range(j - 1, -1, -1):
                        if cnots[i][y] != cnots[j][y]:
                            break
                        # We take a note of the trigger (resp. target).
                        common_A[cnots[i][x]] = i

                    # Next, we look forward for as long as the triggers (resp. targets) are the
                    # same. They all commute, so all are potential candidates for being simplified.
                    common_B: Dict[int, int] = {}
                    for k in range(j + 1, len(cnots)):
                        if cnots[j][x] != cnots[k][x]:
                            break
                        # We take a note of the target (resp. trigger).
                        common_B[cnots[k][y]] = k

                    # Among all the candidates, find if they have a match.
                    keys = common_A.keys() & common_B.keys()
                    for key in keys:
                        assert common_A[key] != common_B[key]
                        # We perform the swap which removes the pivot.
                        new_idx: List[int] = (
                            [idx for idx in range(0, j) if idx != common_A[key]]
                            + [common_B[key], common_A[key]]
                            + [idx for idx in range(j + 1, len(cnots)) if idx != common_B[key]]
                        )
                        # Since we removed the pivot, the length should be one fewer.
                        assert len(new_idx) == len(cnots) - 1
                        cnots = [cnots[idx] for idx in new_idx]
                        found_simplification = True
                        break

                    if found_simplification:
                        break
                if found_simplification:
                    break

            if found_simplification:
                continue
            break

        return cnots

    def _apply_cnots(prevh: Tuple[int, ...], currh: Tuple[int, ...]):
        # This function applies in sequence the CNOTs from prevh and then currh. However, given
        # that the h are sorted in Gray ordering and that some cancel each other, we can reduce
        # the number of gates. See [4] for more details.

        cnots: List[Tuple[int, int]] = []

        if ladder_target:
            cnots.extend((prevh[i], prevh[i + 1]) for i in reversed(range(len(prevh) - 1)))
            cnots.extend((currh[i], currh[i + 1]) for i in range(len(currh) - 1))
        else:
            cnots.extend((prevh[i], prevh[-1]) for i in range(len(prevh) - 1))
            cnots.extend((currh[i], currh[-1]) for i in range(len(currh) - 1))

        cnots = _simplify_cnots(cnots)

        for gate in (cirq.CNOT(qubits[c], qubits[t]) for c, t in cnots):
            yield gate

    previous_h: Tuple[int, ...] = ()
    for h in sorted_hs:
        w = combined.hamiltonians[h]

        yield _apply_cnots(previous_h, h)

        if len(h) >= 1:
            yield cirq.Rz(rads=(theta * w)).on(qubits[h[-1]])

        previous_h = h

    # Flush the last CNOTs.
    yield _apply_cnots(previous_h, ())


@value.value_equality
class BooleanHamiltonianOperation(raw_types.Gate):
    """A gate that applies a Hamiltonian from a set of Boolean functions."""

    def __init__(
        self,
        boolean_strs: Sequence[str],
        theta: float,
        ladder_target: bool,
        symbol_names: Optional[Sequence[str]] = None,
    ):
        """
        Builds an BooleanHamiltonianOperation.

        For each element of a sequence of Boolean expressions, the code first transforms it into a
        polynomial of Pauli Zs that represent that particular expression. Then, we sum all the
        polynomials, thus making a function that goes from a series to Boolean inputs to an integer
        that is the number of Boolean expressions that are true.

        For example, if we were using this gate for the max-cut problem that is typically used to
        demonstrate the QAOA algorithm, there would be one Boolean expression per edge. Each
        Boolean expression would be true iff the vertices on that are in different cuts (i.e. it's)
        an XOR.

        Then, we compute exp(j * theta * polynomial), which is unitary because the polynomial is
        Hermitian.

        Args:
            boolean_strs: The list of Sympy-parsable Boolean expressions.
            theta: The list of thetas to scale the Hamiltonian.
            ladder_target: Whether to use convention of figure 7a or 7b.
            symbol_names: A list of symbol names that should be a superset of all the symbols used.
        """
        self._boolean_strs: Sequence[str] = boolean_strs
        self._theta: float = theta
        self._ladder_target: bool = ladder_target
        self._symbol_names: Optional[Sequence[str]] = symbol_names

        boolean_exprs = [sympy_parser.parse_expr(boolean_str) for boolean_str in boolean_strs]
        name_to_id = BooleanHamiltonianOperation.get_name_to_id(boolean_exprs, symbol_names)
        self._hamiltonian_polynomial_list = [
            _build_hamiltonian_from_boolean(boolean_expr, name_to_id)
            for boolean_expr in boolean_exprs
        ]
        self._num_qubits = len(name_to_id)

    def num_qubits(self) -> int:
        return self._num_qubits

    @staticmethod
    def get_name_to_id(
        boolean_exprs: Sequence[Expr], symbol_names: Optional[Sequence[str]] = None
    ) -> Dict[str, int]:
        """Maps the variables to a unique integer.

        Args:
            boolean_expr: A Sympy expression containing symbols and Boolean operations
            symbol_names: A list of symbol names that should be a superset of all the symbols used.

        Return:
            A dictionary of string (the variable name) to a unique integer.
        """

        required_symbol_names = {
            symbol.name for boolean_expr in boolean_exprs for symbol in boolean_expr.free_symbols
        }
        # For run-to-run identicalness, we sort the symbol name lexicographically.
        if symbol_names:
            for symb in required_symbol_names:
                if symb not in symbol_names:
                    raise ValueError(f'Missing required symbol: {symb}')
            return {symb: i for i, symb in enumerate(sorted(symbol_names))}
        else:
            return {symb: i for i, symb in enumerate(sorted(required_symbol_names))}

    def _value_equality_values_(self):
        return self._boolean_strs, self._theta, self._ladder_target

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'boolean_strs': self._boolean_strs,
            'theta': self._theta,
            'ladder_target': self._ladder_target,
            'symbol_names': self._symbol_names,
        }

    @classmethod
    def _from_json_dict_(cls, boolean_strs, theta, ladder_target, symbol_names, **kwargs):
        return cls(boolean_strs, theta, ladder_target, symbol_names)

    def _decompose_(self, qubits: Sequence['cirq.Qid']):
        yield _get_gates_from_hamiltonians(
            self._hamiltonian_polynomial_list, qubits, self._theta, self._ladder_target
        )
