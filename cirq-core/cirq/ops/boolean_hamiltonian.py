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


def _Hamiltonian_O():
    return 0.0 * cirq.PauliString({})


def _Hamiltonian_I():
    return cirq.PauliString({})


def _Hamiltonian_Z(qubit):
    return cirq.PauliString({qubit: cirq.Z})


def _build_hamiltonian_from_boolean(
    boolean_expr: Expr, qubit_map: Dict[str, 'cirq.Qid']
) -> 'cirq.PauliString':
    """Builds the Hamiltonian representation of Boolean expression as per [1]:

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

    Args:
        boolean_expr: A Sympy expression containing symbols and Boolean operations
        qubit_map: map of string (boolean variable name) to qubit.

    Return:
        The HamiltonianPolynomial that represents the Boolean expression.
    """
    if isinstance(boolean_expr, Symbol):
        # Table 1 of [1], entry for 'x' is '1/2.I - 1/2.Z'
        return 0.5 * _Hamiltonian_I() - 0.5 * _Hamiltonian_Z(qubit_map[boolean_expr.name])

    if isinstance(boolean_expr, (And, Not, Or, Xor)):
        sub_hamiltonians = [
            _build_hamiltonian_from_boolean(sub_boolean_expr, qubit_map)
            for sub_boolean_expr in boolean_expr.args
        ]
        # We apply the equalities of theorem 1 of [1].
        if isinstance(boolean_expr, And):
            hamiltonian = _Hamiltonian_I()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian * sub_hamiltonian
        elif isinstance(boolean_expr, Not):
            assert len(sub_hamiltonians) == 1
            hamiltonian = _Hamiltonian_I() - sub_hamiltonians[0]
        elif isinstance(boolean_expr, Or):
            hamiltonian = _Hamiltonian_O()
            for sub_hamiltonian in sub_hamiltonians:
                hamiltonian = hamiltonian + sub_hamiltonian - hamiltonian * sub_hamiltonian
        elif isinstance(boolean_expr, Xor):
            hamiltonian = _Hamiltonian_O()
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
    hamiltonian_polynomial_list: List['cirq.PauliSum'],
    qubit_map: Dict[str, 'cirq.Qid'],
    theta: float,
    ladder_target: bool = False,
):
    """Builds a circuit according to [1].

    Args:
        hamiltonian_polynomial_list: the list of Hamiltonians, typically built by calling
            _build_hamiltonian_from_boolean().
        qubit_map: map of string (boolean variable name) to qubit.
        theta: A single float scaling the rotations.
        ladder_target: Whether to use convention of figure 7a or 7b.

    Yield:
        Gates that are the decomposition of the Hamiltonian.
    """
    combined: 'cirq.PauliSum' = sum(hamiltonian_polynomial_list, _Hamiltonian_O())

    qubit_names = sorted(qubit_map.keys())
    qubits = [qubit_map[name] for name in qubit_names]
    qubit_indices = {qubit: i for i, qubit in enumerate(qubits)}

    hamiltonians = {}
    for pauli_string in combined:
        w = pauli_string.coefficient.real
        qubit_idx = tuple(sorted(qubit_indices[qubit] for qubit in pauli_string.qubits))
        hamiltonians[qubit_idx] = w

    # Here we follow improvements of [4] cancelling out the CNOTs. The first step is to order by
    # Gray code so that as few as possible gates are changed.
    sorted_hs = sorted(list(hamiltonians.keys()), key=functools.cmp_to_key(_gray_code_comparator))

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
        w = hamiltonians[h]

        yield _apply_cnots(previous_h, h)

        if len(h) >= 1:
            yield cirq.Rz(rads=(theta * w)).on(qubits[h[-1]])

        previous_h = h

    # Flush the last CNOTs.
    yield _apply_cnots(previous_h, ())


@value.value_equality
class BooleanHamiltonian(raw_types.Operation):
    """A gate that applies a Hamiltonian from a set of Boolean functions."""

    def __init__(
        self,
        qubit_map: Dict[str, 'cirq.Qid'],
        boolean_strs: Sequence[str],
        theta: float,
        ladder_target: bool,
    ):
        """
        Builds an BooleanHamiltonian.

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
            qubit_map: map of string (boolean variable name) to qubit.
            theta: The list of thetas to scale the Hamiltonian.
            ladder_target: Whether to use convention of figure 7a or 7b.
        """
        self._qubit_map: Dict[str, 'cirq.Qid'] = qubit_map
        self._boolean_strs: Sequence[str] = boolean_strs
        self._theta: float = theta
        self._ladder_target: bool = ladder_target

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'BooleanHamiltonian':
        return BooleanHamiltonian(
            {q.name(): q for q in new_qubits},
            self._boolean_strs,
            self._theta,
            self._ladder_target,
        )

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        return tuple(self._qubit_map.values())

    def num_qubits(self) -> int:
        return len(self._qubit_map)

    def _value_equality_values_(self):
        return self._qubit_map, self._boolean_strs, self._theta, self._ladder_target

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'qubit_map': self._qubit_map,
            'boolean_strs': self._boolean_strs,
            'theta': self._theta,
            'ladder_target': self._ladder_target,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_map, boolean_strs, theta, ladder_target, **kwargs):
        return cls(qubit_map, boolean_strs, theta, ladder_target)

    def _decompose_(self):
        boolean_exprs = [sympy_parser.parse_expr(boolean_str) for boolean_str in self._boolean_strs]
        hamiltonian_polynomial_list = [
            _build_hamiltonian_from_boolean(boolean_expr, self._qubit_map)
            for boolean_expr in boolean_exprs
        ]
        yield _get_gates_from_hamiltonians(
            hamiltonian_polynomial_list, self._qubit_map, self._theta, self._ladder_target
        )
