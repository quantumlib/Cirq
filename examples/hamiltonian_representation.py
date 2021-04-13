from collections import defaultdict
import functools
import math
from typing import DefaultDict, Dict, List, Sequence, Tuple

from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
import sympy.parsing.sympy_parser as sympy_parser

import cirq


class HamiltonianPolynomial:
    """A container class of Boolean function as equation (2) or [1]

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
            f"{self._hamiltonians[h]:.2f}.{'.'.join('Z_%d' % d for d in h) if h else 'I'}"
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


def build_hamiltonian_from_boolean(
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
            build_hamiltonian_from_boolean(sub_boolean_expr, name_to_id)
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


def get_name_to_id(boolean_exprs: Sequence[Expr]) -> Dict[str, int]:
    """Maps the variables to a unique integer.

    Args:
        boolean_expr: A Sympy expression containing symbols and Boolean operations

    Return:
        A dictionary of string (the variable name) to a unique integer.
    """

    # For run-to-run identicalness, we sort the symbol name lexicographically.
    symbol_names = sorted(
        {symbol.name for boolean_expr in boolean_exprs for symbol in boolean_expr.free_symbols}
    )
    return {symbol_name: i for i, symbol_name in enumerate(symbol_names)}


def _gray_code_comparator(k1, k2, flip=False):
    max_1 = k1[-1] if k1 else -1
    max_2 = k2[-1] if k2 else -1
    if max_1 != max_2:
        return -1 if (max_1 < max_2) ^ flip else 1
    if max_1 == -1:
        return 0
    return _gray_code_comparator(k1[0:-1], k2[0:-1], not flip)


def build_circuit_from_hamiltonians(
    hamiltonian_polynomial_list: List[HamiltonianPolynomial],
    qubits: List[cirq.NamedQubit],
    theta: float,
) -> cirq.Circuit:
    """Builds a circuit according to [1].

    Args:
        hamiltonian_polynomial_list: the list of Hamiltonians, typically built by calling
            build_hamiltonian_from_boolean().
        qubits: The list of qubits corresponding to the variables.
        theta: A single float scaling the rotations.

    Return:
        A dictionary of string (the variable name) to a unique integer.
    """
    combined = sum(hamiltonian_polynomial_list, HamiltonianPolynomial.O())

    circuit = cirq.Circuit()

    # Here we follow improvements of [4] cancelling out the CNOTs. The first step is to order by
    # Gray code so that as few as possible gates are changed.
    sorted_hs = sorted(
        list(combined.hamiltonians.keys()), key=functools.cmp_to_key(_gray_code_comparator)
    )

    def _apply_cnots(previous_h, h):
        # This function applies in sequence the CNOTs from previous_h and then h. However, given
        # that the h are sorted in Gray ordering and that some cancel each other, we can reduce the
        # number of gates. See [4] for more details.

        # We first test whether the rotation is applied on the same qubit.
        last_qubit_is_same = previous_h and h and previous_h[-1] == h[-1]
        if last_qubit_is_same:
            # Instead of applying previous_h and then h, we just apply the symmetric difference of
            # the two CNOTs.
            h_diff = tuple(sorted(set(previous_h).symmetric_difference(h)))
            h_diff += (h[-1],)
            circuit.append([cirq.CNOT(qubits[c], qubits[h_diff[-1]]) for c in h_diff[0:-1]])
        else:
            # This is the fall-back, where we just apply the CNOTs without cancellations.
            circuit.append([cirq.CNOT(qubits[c], qubits[previous_h[-1]]) for c in previous_h[0:-1]])
            circuit.append([cirq.CNOT(qubits[c], qubits[h[-1]]) for c in h[0:-1]])

    previous_h: Tuple[int, ...] = ()
    for h in sorted_hs:
        w = combined.hamiltonians[h]

        _apply_cnots(previous_h, h)

        if len(h) >= 1:
            circuit.append(cirq.Rz(rads=(theta * w)).on(qubits[h[-1]]))

        previous_h = h

    # Flush the last CNOTs.
    _apply_cnots(previous_h, ())

    return circuit


def build_circuit_from_boolean_expressions(boolean_exprs: Sequence[Expr], theta: float):
    """Wrappers of all the functions to go from Boolean expressions to circuit.

    Args:
        boolean_exprs: The list of Sympy Boolean expressions.
        theta: The list of thetas to scale the

    Return:
        A dictionary of string (the variable name) to a unique integer.
    """
    booleans = [sympy_parser.parse_expr(boolean_expr) for boolean_expr in boolean_exprs]
    name_to_id = get_name_to_id(booleans)

    hamiltonian_polynomial_list = [
        build_hamiltonian_from_boolean(boolean, name_to_id) for boolean in booleans
    ]

    qubits = [cirq.NamedQubit(name) for name in name_to_id.keys()]
    circuit = cirq.Circuit()
    circuit += build_circuit_from_hamiltonians(hamiltonian_polynomial_list, qubits, theta)

    return circuit, qubits
