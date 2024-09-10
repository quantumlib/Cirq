# Copyright 2022 The Cirq Developers
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

import math
from typing import Any, cast, Dict, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

from sympy.combinatorics import GrayCode

from cirq import value
from cirq.ops import common_gates, pauli_gates, raw_types

if TYPE_CHECKING:
    import cirq


def _prob_to_angle(prob):
    # From equation 13 of the paper or in the write up. Note that atan(sqrt(x / (1 - x))) =
    # asin(sqrt(x)) and some of the references use the asin.
    return 2.0 * math.asin(math.sqrt(prob))


def _generate_gate_set_for_arc_prob(target, params, cond_probs):
    # Here we deviate slightly from the original paper, by using Gray coding as described in:
    # [arXiv:1306.3991](https://arxiv.org/abs/1306.3991){:.external}
    #
    # The goal is to reduce the total number of gates, but the math is unchanged.

    graycode = GrayCode(len(params))
    previous_binary = '1' * (2 ** len(params))

    for notted_binary in graycode.generate_gray():
        # We get the NOT of the code because we want to start with 111...1 so that we don't need
        # to have X gates on all qubits from the begining. This does not change the output and is
        # simply an optimization.
        binary = ''.join('1' if bit == '0' else '0' for bit in notted_binary)

        # TODO(tonybruguier): Further reduce the number of gates in case the prob is 0.0. This
        # would mean skipping a Gray code, so the accounting of the X gates must be done
        # carefully.
        for bit, previous_bit, param in zip(binary, previous_binary, params):
            if bit != previous_bit:
                yield pauli_gates.X(param)

        yield common_gates.ry(_prob_to_angle(cond_probs[int(binary, 2)])).on(target).controlled_by(
            *params
        )

        previous_binary = binary

    for previous_bit, param in zip(previous_binary, params):
        if previous_bit == '0':
            yield pauli_gates.X(param)


def _generate_got_set_for_init_prob(qubit, init_prob):
    if init_prob is not None:
        yield common_gates.ry(_prob_to_angle(init_prob)).on(qubit)


@value.value_equality
class BayesianNetworkGate(raw_types.Gate):
    """A gate that represents a Bayesian network.

    This class implements Quantum Bayesian Networks as described in:
    [arXiv:2004.14803](https://arxiv.org/abs/2004.14803){:.external}

    In addition, these write ups could be helpful:
    [towardsdatascience1](
        https://towardsdatascience.com/create-a-quantum-bayesian-network-d26d7c5c4217){:.external}
    [towardsdatascience2](
        https://towardsdatascience.com/how-to-create-a-quantum-bayesian-network-5b011914b03e)
        {:.external}

    In order to reduce the number of gates, the code uses Gray coding, as describe in a separate
    paper for another type of gates:
    [arXiv:1306.3991](https://arxiv.org/abs/1306.3991){:.external}

    Note that Bayesian networks are directed acyclic graphs, but the present class does not handle
    any of the graph properties. Instead, it narrowly focuses only on the quantum implementation,
    and only receives as inputs base Python objects, not a graph. It is incumbent on the user to
    make sure the input of the gates indeed represent a Bayesian network.
    """

    def __init__(
        self,
        init_probs: List[Tuple[str, Optional[float]]],
        arc_probs: List[Tuple[str, Tuple[str], List[float]]],
    ):
        """Builds a BayesianNetworkGate.

        The network is specified by the two types of probabilitites: The probabilitites for the
        independent variables, and the probabilitites for the dependent ones.

        For example, we could have two independent variables, q0 and q1, and one dependent variable,
        q2. The independent variables could be defined as p(q0 = 1) and p(q1 = 1). The dependence
        could be defined as p(q2 = 1 | q0, q1) for the four values that (q0, q1) can take.

        In this case, the input arguments would be:
        init_prob = [
            ('q0', 0.123)   # Indicates that p(q0 = 1) = 0.123
            ('q1', 0.456)   # Indicates that p(q1 = 1) = 0.456
            ('q2', None)    # Indicates that q2 is a dependent variable
        ]
        arc_probs = [
            ('q2', ('q0', 'q1'), [0.1, 0.2, 0.3, 0.4])
                # Indicates that p(q2 = 1 | q0 = 0 and q1 = 0) = 0.1
                # Indicates that p(q2 = 1 | q0 = 0 and q1 = 1) = 0.2
                # Indicates that p(q2 = 1 | q0 = 1 and q1 = 0) = 0.3
                # Indicates that p(q2 = 1 | q0 = 1 and q1 = 1) = 0.4
        ]

        By convention, all the probabilties are for the variable being equal to 1 and the
        probability of being equal to zero can be inferred. In the example above, we thus have:
        p(q2 = 0 | q0 = 1 and q1 = 0) = 1.0 - p(q2 = 1 | q0 = 1 and q1 = 0) = 0.7

        Note that there is NO checking that the chain of probability creates a directed acyclic
        graph. In particular, the order of the elements in arc_probs matters. Also, if you want to
        specify the dependent probabilities outside of this gate, you can mark all the variables as
        dependent in init_probs.

        init_prob: A list of tuples, each representing a single variable. The first element of the
            tuples is a string representing the name of the variable. The second element of the
            tuples is either None for dependent variables, or a float representing a probability.

        arc_probs: A list of tuples, each representing a dependence. The first element of the tuples
            is a string representing the name of the variable. The second element of the tuples is
            itself a tuple of n strings, representing the dependence. The third element of the
            tuples is a list of 2**n floats, each representing the probabilities.

        Raises:
            ValueError: If the probabilities are not in [0, 1], or an incorrect number of
            probability is specified, or if the parameter names are no passed as a tuple.
        """
        for _, init_prob in init_probs:
            if init_prob is None:
                continue
            if init_prob < 0.0 or init_prob > 1.0:
                raise ValueError('Initial prob should be between 0 and 1.')
        self._init_probs = init_probs
        for _, params, cond_probs in arc_probs:
            if not isinstance(params, tuple):
                raise ValueError('Conditional prob params must be a tuple.')
            if len(cond_probs) != 2 ** len(params):
                raise ValueError('Incorrect number of conditional probs.')
            for cond_prob in cond_probs:
                if cond_prob < 0.0 or cond_prob > 1.0:
                    raise ValueError('Conditional prob should be between 0 and 1.')
        self._arc_probs = arc_probs

    def _decompose_(self, qubits: Sequence['raw_types.Qid']) -> Iterator['cirq.OP_TREE']:
        parameter_names = [init_prob[0] for init_prob in self._init_probs]
        qubit_map = dict(zip(parameter_names, qubits))

        for param, init_prob in self._init_probs:
            yield _generate_got_set_for_init_prob(qubit_map[param], init_prob)

        for target, params, cond_probs in self._arc_probs:
            yield _generate_gate_set_for_arc_prob(
                qubit_map[target], [qubit_map[param] for param in params], cond_probs
            )

    def _has_unitary_(self) -> bool:
        return True

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,) * len(self._init_probs)

    def _value_equality_values_(self):
        return self._init_probs, self._arc_probs

    def _json_dict_(self) -> Dict[str, Any]:
        return {'init_probs': self._init_probs, 'arc_probs': self._arc_probs}

    @classmethod
    def _from_json_dict_(
        cls,
        init_probs: List[List[Union[str, Optional[float]]]],
        arc_probs: List[List[Union[str, List[str], List[float]]]],
        **kwargs,
    ) -> 'BayesianNetworkGate':
        converted_init_probs = cast(
            List[Tuple[str, Optional[float]]],
            [(param, init_prob) for param, init_prob in init_probs],
        )
        converted_cond_probs = cast(
            List[Tuple[str, Tuple[str], List[float]]],
            [(target, tuple(params), cond_probs) for target, params, cond_probs in arc_probs],
        )
        return cls(converted_init_probs, converted_cond_probs)

    def __repr__(self) -> str:
        return (
            f'cirq.BayesianNetworkGate('
            f'init_probs={self._init_probs!r}, '
            f'arc_probs={self._arc_probs!r})'
        )
