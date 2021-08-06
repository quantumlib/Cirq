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
"""

from typing import cast, Any, Dict, Generator, List, Sequence, Tuple

import sympy.parsing.sympy_parser as sympy_parser

import cirq
from cirq import value
from cirq.ops import raw_types
from cirq.ops.linear_combinations import PauliSum, PauliString


@value.value_equality
class BooleanHamiltonian(raw_types.Operation):
    """An operation that represents a Hamiltonian from a set of Boolean functions."""

    def __init__(
        self,
        qubit_map: Dict[str, 'cirq.Qid'],
        boolean_strs: Sequence[str],
        theta: float,
    ):
        """Builds a BooleanHamiltonian.

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
            boolean_strs: The list of Sympy-parsable Boolean expressions.
            qubit_map: map of string (boolean variable name) to qubit.
            theta: The evolution time (angle) for the Hamiltonian
        """
        self._qubit_map: Dict[str, 'cirq.Qid'] = qubit_map
        self._boolean_strs: Sequence[str] = boolean_strs
        self._theta: float = theta

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'BooleanHamiltonian':
        return BooleanHamiltonian(
            {cast(cirq.NamedQubit, q).name: q for q in new_qubits},
            self._boolean_strs,
            self._theta,
        )

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        return tuple(self._qubit_map.values())

    def num_qubits(self) -> int:
        return len(self._qubit_map)

    def _value_equality_values_(self):
        return self._qubit_map, self._boolean_strs, self._theta

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'qubit_map': self._qubit_map,
            'boolean_strs': self._boolean_strs,
            'theta': self._theta,
        }

    @classmethod
    def _from_json_dict_(cls, qubit_map, boolean_strs, theta, **kwargs):
        return cls(qubit_map, boolean_strs, theta)

    def _decompose_(self):
        boolean_exprs = [sympy_parser.parse_expr(boolean_str) for boolean_str in self._boolean_strs]
        hamiltonian_polynomial_list = [
            PauliSum.from_boolean_expression(boolean_expr, self._qubit_map)
            for boolean_expr in boolean_exprs
        ]

        return _get_gates_from_hamiltonians(
            hamiltonian_polynomial_list, self._qubit_map, self._theta
        )


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

        # TODO(tonybruguier): At this point, some CNOT gates can be cancelled out according to:
        # "Efficient quantum circuits for diagonal unitaries without ancillas" by Jonathan Welch,
        # Daniel Greenbaum, Sarah Mostame, Alán Aspuru-Guzik
        # https://arxiv.org/abs/1306.3991

        for gate in (cirq.CNOT(qubits[c], qubits[t]) for c, t in cnots):
            yield gate

    previous_h: Tuple[int, ...] = ()
    for h, w in hamiltonians.items():
        yield _apply_cnots(previous_h, h)

        if len(h) >= 1:
            yield cirq.Rz(rads=(theta * w)).on(qubits[h[-1]])

        previous_h = h

    # Flush the last CNOTs.
    yield _apply_cnots(previous_h, ())
