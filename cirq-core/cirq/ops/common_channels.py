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

"""Quantum channels that are commonly used in the literature."""

import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

import numpy as np

from cirq import protocols, value
from cirq.linalg import transformations
from cirq.ops import raw_types, common_gates, pauli_gates, identity

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class AsymmetricDepolarizingChannel(raw_types.Gate):
    r"""A channel that depolarizes asymmetrically along different directions.

    This channel applies one of $4^n$ disjoint possibilities: nothing (the
    identity channel) or one of the $4^n - 1$ pauli gates.

    This channel evolves a density matrix via

    $$
    \sum_i p_i P_i \rho P_i
    $$

    where i varies from $0$ to $4^n-1$ and $P_i$ represents n-qubit Pauli operator
    (including identity). The input $\rho$ is the density matrix before the
    depolarization.

    Note: prior to Cirq v0.14, this class contained `num_qubits` as a property.
    This violates the contract of `cirq.Gate` so it was removed.  One can
    instead get the number of qubits by calling the method `num_qubits`.
    """

    def __init__(
        self,
        p_x: Optional[float] = None,
        p_y: Optional[float] = None,
        p_z: Optional[float] = None,
        error_probabilities: Optional[Dict[str, float]] = None,
        tol: float = 1e-8,
    ) -> None:
        r"""The asymmetric depolarizing channel.

        Args:
            p_x: The probability that a Pauli X and no other gate occurs.
            p_y: The probability that a Pauli Y and no other gate occurs.
            p_z: The probability that a Pauli Z and no other gate occurs.
            error_probabilities: Dictionary of string (Pauli operator) to its
                probability. If the identity is missing from the list, it will
                be added so that the total probability mass is 1.
            tol: The tolerance used making sure the total probability mass is
                equal to 1.

        Examples of calls:
            * Single qubit: AsymmetricDepolarizingChannel(0.2, 0.1, 0.3)
            * Single qubit: AsymmetricDepolarizingChannel(p_z=0.3)
            * Two qubits: AsymmetricDepolarizingChannel(
                                error_probabilities={'XX': 0.2})

        Raises:
            ValueError: if the args or the sum of args are not probabilities.
        """
        if error_probabilities:
            num_qubits = len(list(error_probabilities)[0])
            for k in error_probabilities.keys():
                if not set(k).issubset({'I', 'X', 'Y', 'Z'}):
                    raise ValueError(f"{k} is not made solely of I, X, Y, Z.")
                if len(k) != num_qubits:
                    raise ValueError(f"{k} must have {num_qubits} Pauli gates.")
            for k, v in error_probabilities.items():
                value.validate_probability(v, f"p({k})")
            sum_probs = sum(error_probabilities.values())
            identity = 'I' * num_qubits
            if sum_probs < 1.0 - tol and identity not in error_probabilities:
                error_probabilities[identity] = 1.0 - sum_probs
            elif abs(sum_probs - 1.0) > tol:
                raise ValueError(f"Probabilities do not add up to 1 but to {sum_probs}")
            self._num_qubits = num_qubits
            self._error_probabilities = error_probabilities
        else:
            p_x = 0.0 if p_x is None else p_x
            p_y = 0.0 if p_y is None else p_y
            p_z = 0.0 if p_z is None else p_z

            p_x = value.validate_probability(p_x, 'p_x')
            p_y = value.validate_probability(p_y, 'p_y')
            p_z = value.validate_probability(p_z, 'p_z')
            p_i = 1 - value.validate_probability(p_x + p_y + p_z, 'p_x + p_y + p_z')

            self._num_qubits = 1
            self._error_probabilities = {'I': p_i, 'X': p_x, 'Y': p_y, 'Z': p_z}

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        ps = []
        for pauli in self._error_probabilities:
            Pi = np.identity(1)
            for gate in pauli:
                if gate == 'I':
                    Pi = np.kron(Pi, protocols.unitary(identity.I))
                elif gate == 'X':
                    Pi = np.kron(Pi, protocols.unitary(pauli_gates.X))
                elif gate == 'Y':
                    Pi = np.kron(Pi, protocols.unitary(pauli_gates.Y))
                elif gate == 'Z':
                    Pi = np.kron(Pi, protocols.unitary(pauli_gates.Z))
            ps.append(Pi)
        return tuple(zip(self._error_probabilities.values(), ps))

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._num_qubits, tuple(sorted(self._error_probabilities.items()))

    def __repr__(self) -> str:
        return 'cirq.asymmetric_depolarize(' + f"error_probabilities={self._error_probabilities})"

    def __str__(self) -> str:
        return 'asymmetric_depolarize(' + f"error_probabilities={self._error_probabilities})"

    def _circuit_diagram_info_(
        self, args: 'protocols.CircuitDiagramInfoArgs'
    ) -> Union[str, Iterable[str]]:
        if self._num_qubits == 1:
            if args.precision is not None:
                return (
                    f"A({self.p_x:.{args.precision}g},"
                    + f"{self.p_y:.{args.precision}g},"
                    + f"{self.p_z:.{args.precision}g})"
                )
            return f"A({self.p_x},{self.p_y},{self.p_z})"
        if args.precision is not None:
            error_probabilities = [
                f"{pauli}:{p:.{args.precision}g}" for pauli, p in self._error_probabilities.items()
            ]
        else:
            error_probabilities = [f"{pauli}:{p}" for pauli, p in self._error_probabilities.items()]
        return [f"A({', '.join(error_probabilities)})"] + [
            f'({i})' for i in range(1, self._num_qubits)
        ]

    @property
    def p_i(self) -> float:
        """The probability that an Identity I and no other gate occurs."""
        if self._num_qubits != 1:
            raise ValueError('num_qubits should be 1')
        return self._error_probabilities.get('I', 0.0)

    @property
    def p_x(self) -> float:
        """The probability that a Pauli X and no other gate occurs."""
        if self._num_qubits != 1:
            raise ValueError('num_qubits should be 1')
        return self._error_probabilities.get('X', 0.0)

    @property
    def p_y(self) -> float:
        """The probability that a Pauli Y and no other gate occurs."""
        if self._num_qubits != 1:
            raise ValueError('num_qubits should be 1')
        return self._error_probabilities.get('Y', 0.0)

    @property
    def p_z(self) -> float:
        """The probability that a Pauli Z and no other gate occurs."""
        if self._num_qubits != 1:
            raise ValueError('num_qubits should be 1')
        return self._error_probabilities.get('Z', 0.0)

    @property
    def error_probabilities(self) -> Dict[str, float]:
        """A dictionary from Pauli gates to probability"""
        return self._error_probabilities

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['error_probabilities'])


def asymmetric_depolarize(
    p_x: Optional[float] = None,
    p_y: Optional[float] = None,
    p_z: Optional[float] = None,
    error_probabilities: Optional[Dict[str, float]] = None,
    tol: float = 1e-8,
) -> AsymmetricDepolarizingChannel:
    r"""Returns an `AsymmetricDepolarizingChannel` with the given parameters.

    This channel applies one of $4^n$ disjoint possibilities: nothing (the
    identity channel) or one of the $4^n - 1$ pauli gates.

    This channel evolves a density matrix via

    $$
    \sum_i p_i P_i \rho P_i
    $$

    where i varies from $0$ to $4^n-1$ and $P_i$ represents n-qubit Pauli operator
    (including identity). The input $\rho$ is the density matrix before the
    depolarization.

    Args:
        p_x: The probability that a Pauli X and no other gate occurs.
        p_y: The probability that a Pauli Y and no other gate occurs.
        p_z: The probability that a Pauli Z and no other gate occurs.
        error_probabilities: Dictionary of string (Pauli operator) to its
            probability. If the identity is missing from the list, it will
            be added so that the total probability mass is 1.
        tol: The tolerance used making sure the total probability mass is
            equal to 1.

    Examples of calls:

    *     Single qubit: `AsymmetricDepolarizingChannel(0.2, 0.1, 0.3)`
    *     Single qubit: `AsymmetricDepolarizingChannel(p_z=0.3)`
    *     Two qubits: `AsymmetricDepolarizingChannel(error_probabilities={'XX': 0.2})`

    Raises:
        ValueError: if the args or the sum of the args are not probabilities.
    """
    return AsymmetricDepolarizingChannel(p_x, p_y, p_z, error_probabilities, tol)


@value.value_equality(approximate=True)
class DepolarizingChannel(raw_types.Gate):
    r"""A channel that depolarizes one or several qubits.

    This channel applies one of $4^n$ disjoint possibilities: nothing (the
    identity channel) or one of the $4^n - 1$ pauli gates. The disjoint
    probabilities of the non-identity Pauli gates are all the same,
    $p / (4^n - 1)$, and the identity is done with probability $1 - p$. The
    supplied probability must be a valid probability or else this
    constructor will raise a ValueError.


    This channel evolves a density matrix via

    $$
    \rho \rightarrow (1 - p) \rho + p / (4^n - 1) \sum _i P_i \rho P_i
    $$

    where $P_i$ are the $4^n - 1$ Pauli gates (excluding the identity).
    """

    def __init__(self, p: float, n_qubits: int = 1) -> None:
        """Constructs a depolarization channel on several qubits.

        Args:
            p: The probability that one of the Pauli gates is applied. Each of
                the Pauli gates is applied independently with probability
                $p / (4^n - 1)$.
            n_qubits: the number of qubits.

        Raises:
            ValueError: if p is not a valid probability.
        """

        error_probabilities = {}

        p_depol = p / (4**n_qubits - 1)
        p_identity = 1.0 - p
        for pauli_tuple in itertools.product(['I', 'X', 'Y', 'Z'], repeat=n_qubits):
            pauli_string = ''.join(pauli_tuple)
            if pauli_string == 'I' * n_qubits:
                error_probabilities[pauli_string] = p_identity
            else:
                error_probabilities[pauli_string] = p_depol

        self._p = p
        self._n_qubits = n_qubits

        self._delegate = AsymmetricDepolarizingChannel(error_probabilities=error_probabilities)

    def _qid_shape_(self):
        return (2,) * self._n_qubits

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        return self._delegate._mixture_()

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p, self._n_qubits

    def __repr__(self) -> str:
        if self._n_qubits == 1:
            return f"cirq.depolarize(p={self._p})"
        return f"cirq.depolarize(p={self._p},n_qubits={self._n_qubits})"

    def __str__(self) -> str:
        if self._n_qubits == 1:
            return f"depolarize(p={self._p})"
        return f"depolarize(p={self._p},n_qubits={self._n_qubits})"

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> Tuple[str, ...]:
        result: Tuple[str, ...]
        if args.precision is not None:
            result = (f"D({self._p:.{args.precision}g})",)
        else:
            result = (f"D({self._p})",)
        while len(result) < self.num_qubits():
            result += (f"#{len(result) + 1}",)
        return result

    @property
    def p(self) -> float:
        """The probability that one of the Pauli gates is applied.

        Each of the Pauli gates is applied independently with probability
        $p / (4^n - 1)$, where $n$ is `n_qubits`.
        """
        return self._p

    @property
    def n_qubits(self) -> int:
        """The number of qubits"""
        return self._n_qubits

    def _json_dict_(self) -> Dict[str, Any]:
        if self._n_qubits == 1:
            return protocols.obj_to_dict_helper(self, ['p'])
        return protocols.obj_to_dict_helper(self, ['p', 'n_qubits'])


def depolarize(p: float, n_qubits: int = 1) -> DepolarizingChannel:
    r"""Returns a DepolarizingChannel with given probability of error.

    This channel applies one of $4^n$ disjoint possibilities: nothing (the
    identity channel) or one of the $4^n - 1$ pauli gates. The disjoint
    probabilities of the non-identity Pauli gates are all the same,
    $p / (4^n - 1)$, and the identity is done with probability 1 - p. The
    supplied probability must be a valid probability or else this constructor
    will raise a ValueError.

    This channel evolves a density matrix via

    $$
    \rho \rightarrow (1 - p) \rho + p / (4^n - 1) \sum _i P_i \rho P_i
    $$

    where $P_i$ are the $4^n - 1$ Pauli gates (excluding the identity).

    Args:
        p: The probability that one of the Pauli gates is applied. Each of
            the Pauli gates is applied independently with probability
            $p / (4^n - 1)$, where $n$ is n_qubits.
        n_qubits: The number of qubits.

    Raises:
        ValueError: if p is not a valid probability.
    """
    return DepolarizingChannel(p, n_qubits)


@value.value_equality(approximate=True)
class GeneralizedAmplitudeDampingChannel(raw_types.Gate):
    r"""Dampen qubit amplitudes through non ideal dissipation.

    This channel models the effect of energy dissipation into the environment
    as well as the environment depositing energy into the system.

    Construct a channel to model energy dissipation into the environment
    as well as the environment depositing energy into the system. The
    probabilities with which the energy exchange occur are given by `gamma`,
    and the probability of the environment being not excited is given by
    `p`.

    The stationary state of this channel is the diagonal density matrix
    with probability `p` of being |0⟩ and probability `1-p` of being |1⟩.

    This channel evolves a density matrix via

    $$
    \rho \rightarrow \sum_{i=0}^3 M_i \rho M_i^\dagger
    $$

    with

    $$
    \begin{aligned}
    M_0 =& \sqrt{p} \begin{bmatrix}
                        1 & 0  \\
                        0 & \sqrt{1 - \gamma}
                    \end{bmatrix}
    \\
    M_1 =& \sqrt{p} \begin{bmatrix}
                        0 & \sqrt{\gamma} \\
                        0 & 0
                   \end{bmatrix}
    \\
    M_2 =& \sqrt{1-p} \begin{bmatrix}
                        \sqrt{1-\gamma} & 0 \\
                         0 & 1
                      \end{bmatrix}
    \\
    M_3 =& \sqrt{1-p} \begin{bmatrix}
                         0 & 0 \\
                         \sqrt{\gamma} & 0
                     \end{bmatrix}
    \end{aligned}
    $$
    """

    def __init__(self, p: float, gamma: float) -> None:
        r"""The generalized amplitude damping channel.

        Args:
            p: the probability of the environment being not excited
            gamma: the probability of energy transfer

        Raises:
            ValueError: if gamma or p is not a valid probability.
        """
        self._p = value.validate_probability(p, 'p')
        self._gamma = value.validate_probability(gamma, 'gamma')

    def _num_qubits_(self) -> int:
        return 1

    def _kraus_(self) -> Iterable[np.ndarray]:
        p0 = np.sqrt(self._p)
        p1 = np.sqrt(1.0 - self._p)
        sqrt_g = np.sqrt(self._gamma)
        sqrt_g1 = np.sqrt(1.0 - self._gamma)
        return (
            p0 * np.array([[1.0, 0.0], [0.0, sqrt_g1]]),
            p0 * np.array([[0.0, sqrt_g], [0.0, 0.0]]),
            p1 * np.array([[sqrt_g1, 0.0], [0.0, 1.0]]),
            p1 * np.array([[0.0, 0.0], [sqrt_g, 0.0]]),
        )

    def _has_kraus_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p, self._gamma

    def __repr__(self) -> str:
        return f'cirq.generalized_amplitude_damp(p={self._p!r},gamma={self._gamma!r})'

    def __str__(self) -> str:
        return f'generalized_amplitude_damp(p={self._p!r},gamma={self._gamma!r})'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return f'GAD({f},{f})'.format(self._p, self._gamma)
        return f'GAD({self._p!r},{self._gamma!r})'

    @property
    def p(self) -> float:
        """The probability of the environment being not excited."""
        return self._p

    @property
    def gamma(self) -> float:
        """The probability of energy transfer."""
        return self._gamma

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p', 'gamma'])


def generalized_amplitude_damp(p: float, gamma: float) -> GeneralizedAmplitudeDampingChannel:
    r"""Returns a GeneralizedAmplitudeDampingChannel with probabilities gamma and p.

    This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
          + M_2 \rho M_2^\dagger + M_3 \rho M_3^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \sqrt{p} \begin{bmatrix}
                        1 & 0  \\
                        0 & \sqrt{1 - \gamma}
                   \end{bmatrix}
    \\
    M_1 =& \sqrt{p} \begin{bmatrix}
                        0 & \sqrt{\gamma} \\
                        0 & 0
                   \end{bmatrix}
    \\
    M_2 =& \sqrt{1-p} \begin{bmatrix}
                        \sqrt{1-\gamma} & 0 \\
                         0 & 1
                      \end{bmatrix}
    \\
    M_3 =& \sqrt{1-p} \begin{bmatrix}
                         0 & 0 \\
                         \sqrt{\gamma} & 0
                     \end{bmatrix}
    \end{aligned}
    $$

    Args:
        gamma: the probability of the interaction being dissipative.
        p: the probability of the qubit and environment exchanging energy.

    Raises:
        ValueError: gamma or p is not a valid probability.
    """
    return GeneralizedAmplitudeDampingChannel(p, gamma)


@value.value_equality(approximate=True)
class AmplitudeDampingChannel(raw_types.Gate):
    r"""Dampen qubit amplitudes through dissipation.

    This channel models the effect of energy dissipation to the
    surrounding environment.  The probability of
    energy exchange occurring is given by gamma.

    This channel evolves a density matrix as follows:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \begin{bmatrix}
            1 & 0  \\
            0 & \sqrt{1 - \gamma}
          \end{bmatrix}
    \\
    M_1 =& \begin{bmatrix}
            0 & \sqrt{\gamma} \\
            0 & 0
          \end{bmatrix}
    \end{aligned}
    $$
    """

    def __init__(self, gamma: float) -> None:
        """Construct an amplitude damping channel.

        Args:
            gamma: the probability of the interaction being dissipative.

        Raises:
            ValueError: if gamma is not a valid probability.
        """
        self._gamma = value.validate_probability(gamma, 'gamma')
        self._delegate = GeneralizedAmplitudeDampingChannel(1.0, self._gamma)

    def _num_qubits_(self) -> int:
        return 1

    def _kraus_(self) -> Iterable[np.ndarray]:
        # just return first two kraus ops, we don't care about
        # the last two.
        return list(self._delegate._kraus_())[:2]

    def _has_kraus_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._gamma

    def __repr__(self) -> str:
        return f'cirq.amplitude_damp(gamma={self._gamma!r})'

    def __str__(self) -> str:
        return f'amplitude_damp(gamma={self._gamma!r})'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return f'AD({f})'.format(self._gamma)
        return f'AD({self._gamma!r})'

    @property
    def gamma(self) -> float:
        """The probability of the interaction being dissipative."""
        return self._gamma

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['gamma'])


def amplitude_damp(gamma: float) -> AmplitudeDampingChannel:
    r"""Returns an AmplitudeDampingChannel with the given probability gamma.

    This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \begin{bmatrix}
            1 & 0  \\
            0 & \sqrt{1 - \gamma}
          \end{bmatrix}
    \\
    M_1 =& \begin{bmatrix}
            0 & \sqrt{\gamma} \\
            0 & 0
          \end{bmatrix}
    \end{aligned}
    $$

    Args:
        gamma: the probability of the interaction being dissipative.

    Raises:
        ValueError: if gamma is not a valid probability.
    """
    return AmplitudeDampingChannel(gamma)


@value.value_equality
class ResetChannel(raw_types.Gate):
    r"""Reset a qubit back to its |0⟩ state.

    The reset channel is equivalent to performing an unobserved measurement
    which then controls a bit flip onto the targeted qubit.

    This channel evolves a density matrix as follows:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \begin{bmatrix}
            1 & 0  \\
            0 & 0
          \end{bmatrix}
    \\
    M_1 =& \begin{bmatrix}
            0 & 1 \\
            0 & 0
          \end{bmatrix}
    \end{aligned}
    $$
    """

    def __init__(self, dimension: int = 2) -> None:
        """Construct channel that resets to the zero state.

        Args:
            dimension: Specify this argument when resetting a qudit.  There will
                be `dimension` number of dimension by dimension matrices
                describing the channel, each with a 1 at a different position in
                the top row.
        """
        self._dimension = dimension

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        return True

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        args.validate_version('2.0', '3.0')
        return args.format('reset {0};\n', qubits[0])

    def _qid_shape_(self):
        return (self._dimension,)

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']):
        if len(qubits) != 1:
            return NotImplemented

        from cirq.sim import simulation_state

        if (
            isinstance(sim_state, simulation_state.SimulationState)
            and not sim_state.can_represent_mixed_states
        ):
            result = sim_state._perform_measurement(qubits)[0]
            gate = common_gates.XPowGate(dimension=self.dimension) ** (self.dimension - result)
            protocols.act_on(gate, sim_state, qubits)
            return True

        return NotImplemented

    def _kraus_(self) -> Iterable[np.ndarray]:
        # The first axis is over the list of channel matrices
        channel = np.zeros((self._dimension,) * 3, dtype=np.complex64)
        channel[:, 0, :] = np.eye(self._dimension)
        return channel

    def _apply_channel_(self, args: 'cirq.ApplyChannelArgs'):
        configs = []
        for i in range(self._dimension):
            s1 = transformations._SliceConfig(
                axis=args.left_axes[0], source_index=i, target_index=0
            )
            s2 = transformations._SliceConfig(
                axis=args.right_axes[0], source_index=i, target_index=0
            )
            configs.append(transformations._BuildFromSlicesArgs(slices=(s1, s2), scale=1))
        transformations._build_from_slices(configs, args.target_tensor, out=args.out_buffer)
        return args.out_buffer

    def _has_kraus_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._dimension

    def __repr__(self) -> str:
        if self._dimension == 2:
            return 'cirq.ResetChannel()'
        else:
            return f'cirq.ResetChannel(dimension={self._dimension!r})'

    def __str__(self) -> str:
        return 'reset'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> str:
        return 'R'

    @property
    def dimension(self) -> int:
        """The dimension of the qudit being reset."""
        return self._dimension

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['dimension'])


def reset(qubit: 'cirq.Qid') -> raw_types.Operation:
    """Returns a `cirq.ResetChannel` on the given qubit.

    This can also be used with the alias `cirq.R`.
    """
    return ResetChannel(qubit.dimension).on(qubit)


R = reset


def reset_each(*qubits: 'cirq.Qid') -> List[raw_types.Operation]:
    """Returns a list of `cirq.ResetChannel` instances on the given qubits."""
    return [ResetChannel(q.dimension).on(q) for q in qubits]


@value.value_equality(approximate=True)
class PhaseDampingChannel(raw_types.Gate):
    r"""Dampen qubit phase.

    This channel models phase damping which is the loss of quantum
    information without the loss of energy.

    Construct a channel that enacts a phase damping constant gamma.

    This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \begin{bmatrix}
            1 & 0 \\
            0 & \sqrt{1 - \gamma}
          \end{bmatrix}
    \\
    M_1 =& \begin{bmatrix}
            0 & 0 \\
            0 & \sqrt{\gamma}
          \end{bmatrix}
    \end{aligned}
    $$
    """

    def __init__(self, gamma: float) -> None:
        """Construct a channel that dampens qubit phase.

        Args:
            gamma: The damping constant.
        Raises:
            ValueError: if gamma is not a valid probability.
        """
        self._gamma = value.validate_probability(gamma, 'gamma')

    def _num_qubits_(self) -> int:
        return 1

    def _apply_channel_(self, args: 'cirq.ApplyChannelArgs'):
        if self._gamma == 0:
            return args.target_tensor
        if self._gamma != 1:
            return NotImplemented
        configs = []
        for i in range(2):
            s1 = transformations._SliceConfig(
                axis=args.left_axes[0], source_index=i, target_index=i
            )
            s2 = transformations._SliceConfig(
                axis=args.right_axes[0], source_index=i, target_index=i
            )
            configs.append(transformations._BuildFromSlicesArgs(slices=(s1, s2), scale=1))
        transformations._build_from_slices(configs, args.target_tensor, out=args.out_buffer)
        return args.out_buffer

    def _kraus_(self) -> Iterable[np.ndarray]:
        return (
            np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - self._gamma)]]),
            np.array([[0.0, 0.0], [0.0, np.sqrt(self._gamma)]]),
        )

    def _has_kraus_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._gamma

    def __repr__(self) -> str:
        return f'cirq.phase_damp(gamma={self._gamma!r})'

    def __str__(self) -> str:
        return f'phase_damp(gamma={self._gamma!r})'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return f'PD({f})'.format(self._gamma)
        return f'PD({self._gamma!r})'

    @property
    def gamma(self) -> float:
        """The damping constant."""
        return self._gamma

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['gamma'])


def phase_damp(gamma: float) -> PhaseDampingChannel:
    r"""Creates a PhaseDampingChannel with damping constant gamma.

    This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \begin{bmatrix}
            1 & 0  \\
            0 & \sqrt{1 - \gamma}
          \end{bmatrix}
    \\
    M_1 =& \begin{bmatrix}
            0 & 0 \\
            0 & \sqrt{\gamma}
          \end{bmatrix}
    \end{aligned}
    $$

    Args:
        gamma: The damping constant.

    Raises:
        ValueError: is gamma is not a valid probability.
    """
    return PhaseDampingChannel(gamma)


@value.value_equality(approximate=True)
class PhaseFlipChannel(raw_types.Gate):
    r"""Probabilistically flip the sign of the phase of a qubit.

    This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \sqrt{1 - p} \begin{bmatrix}
                        1 & 0  \\
                        0 & 1
                    \end{bmatrix}
    \\
    M_1 =& \sqrt{p} \begin{bmatrix}
                        1 & 0 \\
                        0 & -1
                    \end{bmatrix}
    \end{aligned}
    $$
    """

    def __init__(self, p: float) -> None:
        """Construct a channel that probabilistically flips the sign of the phase.
        Args:
            p: the probability of a phase flip.

        Raises:
            ValueError: if p is not a valid probability.
        """
        self._p = value.validate_probability(p, 'p')
        self._delegate = AsymmetricDepolarizingChannel(0.0, 0.0, p)

    def _num_qubits_(self) -> int:
        return 1

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        mixture = self._delegate._mixture_()
        # just return identity and z term
        return (mixture[0], mixture[3])

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return f'cirq.phase_flip(p={self._p!r})'

    def __str__(self) -> str:
        return f'phase_flip(p={self._p!r})'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return f'PF({f})'.format(self._p)
        return f'PF({self._p!r})'

    @property
    def p(self) -> float:
        """The probability of a phase flip."""
        return self._p

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p'])


def _phase_flip_Z() -> common_gates.ZPowGate:
    """Returns a cirq.Z which corresponds to a guaranteed phase flip."""
    return common_gates.ZPowGate()


def _phase_flip(p: float) -> PhaseFlipChannel:
    r"""Returns a PhaseFlipChannel that flips a qubit's phase with probability p.

    This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \sqrt{1 - p} \begin{bmatrix}
                        1 & 0  \\
                        0 & 1
                    \end{bmatrix}
    \\
    M_1 =& \sqrt{p} \begin{bmatrix}
                        1 & 0 \\
                        0 & -1
                    \end{bmatrix}
    \end{aligned}
    $$

    Args:
        p: the probability of a phase flip.

    Raises:
        ValueError: if p is not a valid probability.
    """
    return PhaseFlipChannel(p)


def phase_flip(p: Optional[float] = None) -> Union[common_gates.ZPowGate, PhaseFlipChannel]:
    r"""Returns a PhaseFlipChannel that flips a qubit's phase with probability p.

    If `p` is None, return a guaranteed phase flip in the form of a Z operation.

    This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \sqrt{1 - p} \begin{bmatrix}
                        1 & 0  \\
                        0 & 1
                    \end{bmatrix}
    \\
    M_1 =& \sqrt{p} \begin{bmatrix}
                        1 & 0 \\
                        0 & -1
                    \end{bmatrix}
    \end{aligned}
    $$

    Args:
        p: the probability of a phase flip.

    Raises:
        ValueError: if p is not a valid probability.
    """
    if p is None:
        return _phase_flip_Z()

    return _phase_flip(p)


@value.value_equality(approximate=True)
class BitFlipChannel(raw_types.Gate):
    r"""Probabilistically flip a qubit from 1 to 0 state or vice versa.

    Construct a channel that flips a qubit with probability p.

    This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
        M_0 =& \sqrt{1 - p} \begin{bmatrix}
                            1 & 0  \\
                            0 & 1
                       \end{bmatrix}
        \\
        M_1 =& \sqrt{p} \begin{bmatrix}
                            0 & 1 \\
                            1 & 0
                         \end{bmatrix}
        \end{aligned}
    $$
    """

    def __init__(self, p: float) -> None:
        """Construct a channel that probabilistically flips a qubit.

        Args:
            p: the probability of a bit flip.

        Raises:
            ValueError: if p is not a valid probability.
        """
        self._p = value.validate_probability(p, 'p')
        self._delegate = AsymmetricDepolarizingChannel(p, 0.0, 0.0)

    def _num_qubits_(self) -> int:
        return 1

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        mixture = self._delegate._mixture_()
        # just return identity and x term
        return (mixture[0], mixture[1])

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return f'cirq.bit_flip(p={self._p!r})'

    def __str__(self) -> str:
        return f'bit_flip(p={self._p!r})'

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return f'BF({f})'.format(self._p)
        return f'BF({self._p!r})'

    @property
    def p(self) -> float:
        """The probability of a bit flip."""
        return self._p

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p'])


def _bit_flip(p: float) -> BitFlipChannel:
    r"""Construct a BitFlipChannel that flips a qubit state with probability of a flip given by p.

    This channel evolves a density matrix via:

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With:

    $$
    \begin{aligned}
    M_0 =& \sqrt{1-p} \begin{bmatrix}
                        1 & 0 \\
                        0 & 1
                   \end{bmatrix}
    \\
    M_1 =& \sqrt{p} \begin{bmatrix}
                        0 & 1 \\
                        1 & 0
                     \end{bmatrix}
    \end{aligned}
    $$

    Args:
        p: the probability of a bit flip.

    Raises:
        ValueError: if p is not a valid probability.
    """
    return BitFlipChannel(p)


def bit_flip(p: Optional[float] = None) -> Union[common_gates.XPowGate, BitFlipChannel]:
    r"""Construct a BitFlipChannel that flips a qubit state with probability p.

    If p is None, this returns a guaranteed flip in the form of an X operation.

    This channel evolves a density matrix via

    $$
    \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
    $$

    With

    $$
    \begin{aligned}
    M_0 =& \sqrt{1-p} \begin{bmatrix}
                        1 & 0 \\
                        0 & 1
                   \end{bmatrix}
    \\
    M_1 =& \sqrt{p} \begin{bmatrix}
                        0 & 1 \\
                        1 & 0
                     \end{bmatrix}
    \end{aligned}
    $$

    Args:
        p: the probability of a bit flip.

    Raises:
        ValueError: if p is not a valid probability.
    """
    if p is None:
        return pauli_gates.X

    return _bit_flip(p)
