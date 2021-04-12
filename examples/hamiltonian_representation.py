import math
from typing import Dict, List, Sequence, Tuple

from sympy.logic.boolalg import And, Not, Or, Xor
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.parsing.sympy_parser import parse_expr

import cirq

# References:
# [1] On the representation of Boolean and real functions as Hamiltonians for quantum computing
#     by Stuart Hadfield, https://arxiv.org/pdf/1804.09130.pdf
# [2] https://www.youtube.com/watch?v=AOKM9BkweVU is a useful intro
# [3] https://github.com/rsln-s/IEEE_QW_2020/blob/master/Slides.pdf


class HamiltonianList:
    """A container class of Boolean function as equation (2) or [1]"""

    def __init__(self, hamiltonians: Dict[Tuple[int, ...], float]):
        # The representation is Tuple[int, ...] to weights. The tuple contains the integers of
        # where Z_i is present. For example, Z_0.Z_3 would be (0, 3), and I is the empty tuple.
        self._hamiltonians = {h: w for h, w in hamiltonians.items() if math.fabs(w) > 1e-12}

    @property
    def hamiltonians(self):
        return self._hamiltonians

    def __str__(self):
        # For run-to-run identicalness, we sort the keys lexicographically.
        return "; ".join(
            f"{self._hamiltonians[h]:.2f}.{'.'.join('Z_%d' % d for d in h) if h else 'I'}"
            for h in sorted(self._hamiltonians)
        )

    def __add__(self, other: 'HamiltonianList') -> 'HamiltonianList':
        return self._signed_add(other, 1.0)

    def __sub__(self, other: 'HamiltonianList') -> 'HamiltonianList':
        return self._signed_add(other, -1.0)

    def _signed_add(self, other: 'HamiltonianList', sign: float) -> 'HamiltonianList':
        hamiltonians = self._hamiltonians.copy()
        for h, w in other.hamiltonians.items():
            if h not in hamiltonians:
                hamiltonians[h] = 0
            hamiltonians[h] += sign * w
        return HamiltonianList(hamiltonians)

    def __rmul__(self, other: float) -> 'HamiltonianList':
        return HamiltonianList({k: other * w for k, w in self._hamiltonians.items()})

    def __mul__(self, other: 'HamiltonianList') -> 'HamiltonianList':
        hamiltonians = {}
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
                if h not in hamiltonians:
                    hamiltonians[h] = 0
                hamiltonians[h] += w
        return HamiltonianList(hamiltonians)

    @staticmethod
    def O() -> 'HamiltonianList':
        return HamiltonianList({})

    @staticmethod
    def I() -> 'HamiltonianList':
        return HamiltonianList({(): 1.0})

    @staticmethod
    def Z(i: int) -> 'HamiltonianList':
        return HamiltonianList({(i,): 1.0})


def build_hamiltonian_from_boolean(
    boolean_expr: Expr, name_to_id: Dict[str, int]
) -> HamiltonianList:
    """Builds the Hamiltonian representation of Boolean expression as per [1]:

    Args:
        boolean_expr: A Sympy expression containing symbols and Boolean operations
        name_to_id: A dictionary from symbol name to an integer, typically built by calling
            get_name_to_id().

    Return:
        The HamiltonianList that represents the Boolean expression.
    """
    if isinstance(boolean_expr, Symbol):
        # Table 1 of [1], entry for 'x' is '1/2.I - 1/2.Z'
        i = name_to_id[boolean_expr.name]
        return 0.5 * HamiltonianList.I() - 0.5 * HamiltonianList.Z(i)

    if isinstance(boolean_expr, (And, Not, Or, Xor)):
        sub_hamiltonians = [
            build_hamiltonian_from_boolean(sub_boolean_expr, name_to_id)
            for sub_boolean_expr in boolean_expr.args
        ]
        # We apply the equalities of theorem 1 of [1].
        if isinstance(boolean_expr, And):
            hamiltonian = HamiltonianList.I()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian * sub_hamiltonian
        elif isinstance(boolean_expr, Not):
            assert len(sub_hamiltonians) == 1
            hamiltonian = HamiltonianList.I() - sub_hamiltonians[0]
        elif isinstance(boolean_expr, Or):
            hamiltonian = HamiltonianList.O()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian + sub_hamiltonian - hamiltonian * sub_hamiltonian
        elif isinstance(boolean_expr, Xor):
            hamiltonian = HamiltonianList.O()
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


def build_circuit_from_hamiltonians(
    hamiltonian_lists: List[HamiltonianList], qubits: List[cirq.Qid], theta: float
) -> cirq.Circuit:
    """Builds a circuit according to [1].

    Args:
        hamiltonian_lists: the list of Hamiltonians, typically built by calling
            build_hamiltonian_from_boolean().
        qubits: The list of qubits corresponding to the variables.
        theta: A single float scaling the rotations.

    Return:
        A dictionary of string (the variable name) to a unique integer.
    """
    circuit = cirq.Circuit()
    for hamiltonian_list in hamiltonian_lists:
        for h, w in hamiltonian_list.hamiltonians.items():
            circuit.append([cirq.CNOT(qubits[c], qubits[h[0]]) for c in h[1:]])

            if len(h) >= 1:
                circuit.append(cirq.Rz(rads=(theta * w)).on(qubits[h[0]]))

            circuit.append([cirq.CNOT(qubits[c], qubits[h[0]]) for c in h[1:]])

    return circuit


def build_circuit_from_boolean_expressions(boolean_exprs: Sequence[Expr], theta: float):
    """Wrappers of all the functions to go from Boolean expressions to circuit.

    Args:
        boolean_exprs: The list of Sympy Boolean expressions.
        theta: The list of thetas to scale the

    Return:
        A dictionary of string (the variable name) to a unique integer.
    """
    booleans = [parse_expr(boolean_expr) for boolean_expr in boolean_exprs]
    name_to_id = get_name_to_id(booleans)

    hamiltonians = [build_hamiltonian_from_boolean(boolean, name_to_id) for boolean in booleans]

    qubits = [cirq.NamedQubit(name) for name in name_to_id.keys()]
    circuit = cirq.Circuit()
    circuit += build_circuit_from_hamiltonians(hamiltonians, qubits, theta)

    return circuit, qubits
