# Copyright 2021 The Cirq Developers
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
"""Represents Boolean functions as a series of CNOT and rotation gates. The Boolean functions are
passed as Sympy expressions and then turned into an optimized set of gates.

References:
[1] On the representation of Boolean and real functions as Hamiltonians for quantum computing
    by Stuart Hadfield, https://arxiv.org/pdf/1804.09130.pdf
[2] https://www.youtube.com/watch?v=AOKM9BkweVU is a useful intro
[3] https://github.com/rsln-s/IEEE_QW_2020/blob/master/Slides.pdf
[4] Efficient Quantum Circuits for Diagonal Unitaries Without Ancillas by Jonathan Welch, Daniel
    Greenbaum, Sarah Mostame, and Alán Aspuru-Guzik, https://arxiv.org/abs/1306.3991
"""
import functools
import itertools
from typing import Any, Dict, Generator, List, Sequence, Tuple

import sympy.parsing.sympy_parser as sympy_parser

import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString


@value.value_equality
class BooleanHamiltonianGate(raw_types.Gate):
    """A gate that represents a Hamiltonian from a set of Boolean functions."""

    def __init__(
        self,
        parameter_names: Sequence[str],
        boolean_strs: Sequence[str],
        theta: float,
    ):
        """Builds a BooleanHamiltonianGate.

        For each element of a sequence of Boolean expressions, the code first transforms it into a
        polynomial of Pauli Zs that represent that particular expression. Then, we sum all the
        polynomials, thus making a function that goes from a series to Boolean inputs to an integer
        that is the number of Boolean expressions that are true.

        For example, if we were using this gate for the unweighted max-cut problem that is typically
        used to demonstrate the QAOA algorithm, there would be one Boolean expression per edge. Each
        Boolean expression would be true iff the vertices on that are in different cuts (i.e. it's)
        an XOR.

        Then, we compute exp(-j * theta * polynomial), which is unitary because the polynomial is
        Hermitian.

        Args:
            parameter_names: The names of the inputs to the expressions.
            boolean_strs: The list of Sympy-parsable Boolean expressions.
            theta: The evolution time (angle) for the Hamiltonian
        """
        self._parameter_names: Sequence[str] = parameter_names
        self._boolean_strs: Sequence[str] = boolean_strs
        self._theta: float = theta

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,) * len(self._parameter_names)

    def _value_equality_values_(self) -> Any:
        return self._parameter_names, self._boolean_strs, self._theta

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'parameter_names': self._parameter_names,
            'boolean_strs': self._boolean_strs,
            'theta': self._theta,
        }

    @classmethod
    def _from_json_dict_(
        cls, parameter_names, boolean_strs, theta, **kwargs
    ) -> 'cirq.BooleanHamiltonianGate':
        return cls(parameter_names, boolean_strs, theta)

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        qubit_map = dict(zip(self._parameter_names, qubits))
        boolean_exprs = [sympy_parser.parse_expr(boolean_str) for boolean_str in self._boolean_strs]
        hamiltonian_polynomial_list = [
            PauliSum.from_boolean_expression(boolean_expr, qubit_map)
            for boolean_expr in boolean_exprs
        ]

        return _get_gates_from_hamiltonians(hamiltonian_polynomial_list, qubit_map, self._theta)

    def _has_unitary_(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            f'cirq.BooleanHamiltonianGate('
            f'parameter_names={self._parameter_names!r}, '
            f'boolean_strs={self._boolean_strs!r}, '
            f'theta={self._theta!r})'
        )


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


def _simplify_commuting_cnots(
    cnots: List[Tuple[int, int]], flip_control_and_target: bool
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Attempts to commute CNOTs and remove cancelling pairs.

    Commutation relations are based on 9 (flip_control_and_target=False) or 10
    (flip_control_target=True) of [4]:
    When flip_control_target=True:

         CNOT(j, i) @ CNOT(j, k) = CNOT(j, k) @ CNOT(j, i)
    ───X───────       ───────X───
       │                     │
    ───@───@───   =   ───@───@───
           │             │
    ───────X───       ───X───────

    When flip_control_target=False:

    CNOT(i, j) @ CNOT(k, j) = CNOT(k, j) @ CNOT(i, j)
    ───@───────       ───────@───
       │                     │
    ───X───X───   =   ───X───X───
           │             │
    ───────@───       ───@───────

    Args:
        cnots: A list of CNOTS, encoded as integer tuples (control, target). The code does not make
            any assumption as to the order of the CNOTs, but it is likely to work better if its
            inputs are from Gray-sorted Hamiltonians. Regardless of the order of the CNOTs, the
            code is conservative and should be robust to mis-ordered inputs with the only side
            effect being a lack of simplification.
        flip_control_and_target: Whether to flip control and target.

    Returns:
        A tuple containing a Boolean that tells whether a simplification has been performed and the
        CNOT list, potentially simplified, encoded as integer tuples (control, target).
    """

    target, control = (0, 1) if flip_control_and_target else (1, 0)

    to_remove = set()
    qubit_to_index: List[Tuple[int, Dict[int, int]]] = []
    for j in range(len(cnots)):
        if not qubit_to_index or cnots[j][target] != qubit_to_index[-1][0]:
            # The targets (resp. control) don't match, so we create a new dict.
            qubit_to_index.append((cnots[j][target], {cnots[j][control]: j}))
            continue

        if cnots[j][control] in qubit_to_index[-1][1]:
            k = qubit_to_index[-1][1].pop(cnots[j][control])
            # The controls (resp. targets) are the same, so we can simplify away.
            to_remove.update([k, j])
            if not qubit_to_index[-1][1]:
                qubit_to_index.pop()
        else:
            qubit_to_index[-1][1][cnots[j][control]] = j

    cnots = [cnot for i, cnot in enumerate(cnots) if i not in to_remove]
    return bool(to_remove), cnots


def _simplify_cnots_triplets(
    cnots: List[Tuple[int, int]], flip_control_and_target: bool
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Simplifies CNOT pairs according to equation 11 of [4].

    CNOT(i, j) @ CNOT(j, k) == CNOT(j, k) @ CNOT(i, k) @ CNOT(i, j)
    ───@───────       ───────@───@───
       │                     │   │
    ───X───@───   =   ───@───┼───X───
           │             │   │
    ───────X───       ───X───X───────

    Args:
        cnots: A list of CNOTS, encoded as integer tuples (control, target).
        flip_control_and_target: Whether to flip control and target.

    Returns:
        A tuple containing a Boolean that tells whether a simplification has been performed and the
        CNOT list, potentially simplified, encoded as integer tuples (control, target).
    """
    target, control = (0, 1) if flip_control_and_target else (1, 0)

    # We investigate potential pivots sequentially.
    for j in range(1, len(cnots) - 1):
        # First, we look back for as long as the controls (resp. targets) are the same.
        # They all commute, so all are potential candidates for being simplified.
        # prev_match_index is qubit to index in `cnots` array.
        prev_match_index: Dict[int, int] = {}
        for i in range(j - 1, -1, -1):
            # These CNOTs have the same target (resp. control) and though they are not candidates
            # for simplification, since they commute, we can keep looking for candidates.
            if cnots[i][target] == cnots[j][target]:
                continue
            if cnots[i][control] != cnots[j][control]:
                break
            # We take a note of the control (resp. target).
            prev_match_index[cnots[i][target]] = i

        # Next, we look forward for as long as the targets (resp. controls) are the
        # same. They all commute, so all are potential candidates for being simplified.
        # post_match_index is qubit to index in `cnots` array.
        post_match_index: Dict[int, int] = {}
        for k in range(j + 1, len(cnots)):
            # These CNOTs have the same control (resp. target) and though they are not candidates
            # for simplification, since they commute, we can keep looking for candidates.
            if cnots[j][control] == cnots[k][control]:
                continue
            if cnots[j][target] != cnots[k][target]:
                break
            # We take a note of the target (resp. control).
            post_match_index[cnots[k][control]] = k

        # Among all the candidates, find if they have a match.
        keys = prev_match_index.keys() & post_match_index.keys()
        for key in keys:
            # We perform the swap which removes the pivot.
            new_idx: List[int] = (
                # Anything strictly before the pivot that is not the CNOT to swap.
                [idx for idx in range(0, j) if idx != prev_match_index[key]]
                # The two swapped CNOTs.
                + [post_match_index[key], prev_match_index[key]]
                # Anything after the pivot that is not the CNOT to swap.
                + [idx for idx in range(j + 1, len(cnots)) if idx != post_match_index[key]]
            )
            # Since we removed the pivot, the length should be one fewer.
            cnots = [cnots[idx] for idx in new_idx]
            # TODO(#4532): Speed up code by not returning early.
            return True, cnots

    return False, cnots


def _simplify_cnots(cnots: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Takes a series of CNOTs and tries to applies rule to cancel out gates.

    Algorithm based on "Efficient quantum circuits for diagonal unitaries without ancillas" by
    Jonathan Welch, Daniel Greenbaum, Sarah Mostame, Alán Aspuru-Guzik
    https://arxiv.org/abs/1306.3991

    Args:
        cnots: A list of CNOTs represented as tuples of integer (control, target).

    Returns:
        The simplified list of CNOTs, encoded as integer tuples (control, target).
    """

    found_simplification = True
    while found_simplification:
        for simplify_fn, flip_control_and_target in itertools.product(
            [_simplify_commuting_cnots, _simplify_cnots_triplets], [False, True]
        ):
            found_simplification, cnots = simplify_fn(cnots, flip_control_and_target)
            if found_simplification:
                break

    return cnots


def _get_gates_from_hamiltonians(
    hamiltonian_polynomial_list: List['cirq.PauliSum'],
    qubit_map: Dict[str, 'cirq.Qid'],
    theta: float,
) -> Generator['cirq.Operation', None, None]:
    """Builds a circuit according to [1].

    Args:
        hamiltonian_polynomial_list: the list of Hamiltonians, typically built by calling
            PauliSum.from_boolean_expression().
        qubit_map: map of string (boolean variable name) to qubit.
        theta: A single float scaling the rotations.
    Yields:
        Gates that are the decomposition of the Hamiltonian.
    """
    combined = sum(hamiltonian_polynomial_list, PauliSum.from_pauli_strings(PauliString({})))

    qubit_names = sorted(qubit_map.keys())
    qubits = [qubit_map[name] for name in qubit_names]
    qubit_indices = {qubit: i for i, qubit in enumerate(qubits)}

    hamiltonians = {}
    for pauli_string in combined:
        w = pauli_string.coefficient.real
        qubit_idx = tuple(sorted(qubit_indices[qubit] for qubit in pauli_string.qubits))
        hamiltonians[qubit_idx] = w

    def _apply_cnots(prevh: Tuple[int, ...], currh: Tuple[int, ...]):
        cnots: List[Tuple[int, int]] = []

        cnots.extend((prevh[i], prevh[-1]) for i in range(len(prevh) - 1))
        cnots.extend((currh[i], currh[-1]) for i in range(len(currh) - 1))

        cnots = _simplify_cnots(cnots)

        for gate in (cirq.CNOT(qubits[c], qubits[t]) for c, t in cnots):
            yield gate

    sorted_hamiltonian_keys = sorted(
        hamiltonians.keys(), key=functools.cmp_to_key(_gray_code_comparator)
    )

    previous_h: Tuple[int, ...] = ()
    for h in sorted_hamiltonian_keys:
        w = hamiltonians[h]
        yield _apply_cnots(previous_h, h)

        if len(h) >= 1:
            yield cirq.Rz(rads=(theta * w)).on(qubits[h[-1]])

        previous_h = h

    # Flush the last CNOTs.
    yield _apply_cnots(previous_h, ())
