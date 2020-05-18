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

from typing import (Any, Dict, Iterable, Optional, Sequence, Tuple, Union,
                    TYPE_CHECKING)

import numpy as np

from cirq import protocols, value
from cirq.ops import (raw_types, common_gates, pauli_gates, gate_features,
                      identity)

if TYPE_CHECKING:
    import cirq


@value.value_equality
class AsymmetricDepolarizingChannel(gate_features.SingleQubitGate):
    """A channel that depolarizes asymmetrically along different directions."""

    def __init__(self, p_x: float, p_y: float, p_z: float) -> None:
        r"""The asymmetric depolarizing channel.

        This channel applies one of four disjoint possibilities: nothing (the
        identity channel) or one of the three pauli gates. The disjoint
        probabilities of the three gates are p_x, p_y, and p_z and the
        identity is done with probability 1 - p_x - p_y - p_z. The supplied
        probabilities must be valid probabilities and the sum p_x + p_y + p_z
        must be a valid probability or else this constructor will raise a
        ValueError.

        This channel evolves a density matrix via

            $$
            \rho \rightarrow (1 - p_x - p_y - p_z) \rho
                    + p_x X \rho X + p_y Y \rho Y + p_z Z \rho Z
            $$

        Args:
            p_x: The probability that a Pauli X and no other gate occurs.
            p_y: The probability that a Pauli Y and no other gate occurs.
            p_z: The probability that a Pauli Z and no other gate occurs.

        Raises:
            ValueError: if the args or the sum of args are not probabilities.
        """

        self._p_x = value.validate_probability(p_x, 'p_x')
        self._p_y = value.validate_probability(p_y, 'p_y')
        self._p_z = value.validate_probability(p_z, 'p_z')
        self._p_i = 1 - value.validate_probability(p_x + p_y + p_z,
                                                   'p_x + p_y + p_z')

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        return ((self._p_i, protocols.unitary(identity.I)),
                (self._p_x, protocols.unitary(pauli_gates.X)),
                (self._p_y, protocols.unitary(pauli_gates.Y)),
                (self._p_z, protocols.unitary(pauli_gates.Z)))

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p_x, self._p_y, self._p_z

    def __repr__(self) -> str:
        return 'cirq.asymmetric_depolarize(p_x={!r},p_y={!r},p_z={!r})'.format(
            self._p_x, self._p_y, self._p_z
        )

    def __str__(self) -> str:
        return 'asymmetric_depolarize(p_x={!r},p_y={!r},p_z={!r})'.format(
            self._p_x, self._p_y, self._p_z
        )

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'A({},{},{})'.format(f, f, f).format(self._p_x, self._p_y,
                                                        self._p_z)
        return 'A({!r},{!r},{!r})'.format(self._p_x, self._p_y, self._p_z)

    @property
    def p_x(self) -> float:
        """The probability that a Pauli X and no other gate occurs."""
        return self._p_x

    @property
    def p_y(self) -> float:
        """The probability that a Pauli Y and no other gate occurs."""
        return self._p_y

    @property
    def p_z(self) -> float:
        """The probability that a Pauli Z and no other gate occurs."""
        return self._p_z

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p_x', 'p_y', 'p_z'])


def asymmetric_depolarize(
    p_x: float, p_y: float, p_z: float
) -> AsymmetricDepolarizingChannel:
    r"""Returns a AsymmetricDepolarizingChannel with given parameter.

    This channel evolves a density matrix via

        $$
        \rho \rightarrow (1 - p_x - p_y - p_z) \rho
                + p_x X \rho X + p_y Y \rho Y + p_z Z \rho Z
        $$

    Args:
        p_x: The probability that a Pauli X and no other gate occurs.
        p_y: The probability that a Pauli Y and no other gate occurs.
        p_z: The probability that a Pauli Z and no other gate occurs.

    Raises:
        ValueError: if the args or the sum of the args are not probabilities.
    """
    return AsymmetricDepolarizingChannel(p_x, p_y, p_z)


@value.value_equality
class DepolarizingChannel(gate_features.SingleQubitGate):
    """A channel that depolarizes a qubit."""

    def __init__(self, p: float) -> None:
        r"""The symmetric depolarizing channel.

        This channel applies one of four disjoint possibilities: nothing (the
        identity channel) or one of the three pauli gates. The disjoint
        probabilities of the three gates are all the same, p / 3, and the
        identity is done with probability 1 - p. The supplied probability
        must be a valid probability or else this constructor will raise a
        ValueError.

        This channel evolves a density matrix via

            $$
            \rho \rightarrow (1 - p) \rho
                    + (p / 3) X \rho X + (p / 3) Y \rho Y + (p / 3) Z \rho Z
            $$

        Args:
            p: The probability that one of the Pauli gates is applied. Each of
                the Pauli gates is applied independently with probability p / 3.

        Raises:
            ValueError: if p is not a valid probability.
        """

        self._p = p
        self._delegate = AsymmetricDepolarizingChannel(p / 3, p / 3, p / 3)

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        return self._delegate._mixture_()

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return 'cirq.depolarize(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'depolarize(p={!r})'.format(self._p)

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'D({})'.format(f).format(self._p)
        return 'D({!r})'.format(self._p)

    @property
    def p(self) -> float:
        """The probability that one of the Pauli gates is applied.

        Each of the Pauli gates is applied independently with probability p / 3.
        """
        return self._p

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p'])


def depolarize(p: float) -> DepolarizingChannel:
    r"""Returns a DepolarizingChannel with given probability of error.

    This channel applies one of four disjoint possibilities: nothing (the
    identity channel) or one of the three pauli gates. The disjoint
    probabilities of the three gates are all the same, p / 3, and the
    identity is done with probability 1 - p. The supplied probability
    must be a valid probability or else this constructor will raise a
    ValueError.

    This channel evolves a density matrix via

        $$
        \rho \rightarrow (1 - p) \rho
                + (p / 3) X \rho X + (p / 3) Y \rho Y + (p / 3) Z \rho Z
        $$

    Args:
        p: The probability that one of the Pauli gates is applied. Each of
            the Pauli gates is applied independently with probability p / 3.

    Raises:
        ValueError: if p is not a valid probability.
    """
    return DepolarizingChannel(p)


@value.value_equality
class GeneralizedAmplitudeDampingChannel(gate_features.SingleQubitGate):
    """Dampen qubit amplitudes through non ideal dissipation.

    This channel models the effect of energy dissipation into the environment
    as well as the environment depositing energy into the system.
    """

    def __init__(self, p: float, gamma: float) -> None:
        r"""The generalized amplitude damping channel.

        Construct a channel to model energy dissipation into the environment
        as well as the environment depositing energy into the system. The
        probabilities with which the energy exchange occur are given by `gamma`,
        and the probability of the environment being not excited is given by
        `p`.

        The stationary state of this channel is the diagonal density matrix
        with probability `p` of being |0⟩ and probability `1-p` of being |1⟩.

        This channel evolves a density matrix via

            $$
            \rho \rightarrow M_0 \rho M_0^\dagger
                           + M_1 \rho M_1^\dagger
                           + M_2 \rho M_2^\dagger
                           + M_3 \rho M_3^\dagger
            $$

        With

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
            ValueError: if gamma or p is not a valid probability.
        """
        self._gamma = value.validate_probability(gamma, 'gamma')
        self._p = value.validate_probability(p, 'p')

    def _channel_(self) -> Iterable[np.ndarray]:
        p0 = np.sqrt(self._p)
        p1 = np.sqrt(1. - self._p)
        sqrt_g = np.sqrt(self._gamma)
        sqrt_g1 = np.sqrt(1. - self._gamma)
        return (
            p0 * np.array([[1., 0.], [0., sqrt_g1]]),
            p0 * np.array([[0., sqrt_g], [0., 0.]]),
            p1 * np.array([[sqrt_g1, 0.], [0., 1.]]),
            p1 * np.array([[0., 0.], [sqrt_g, 0.]]),
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p, self._gamma

    def __repr__(self) -> str:
        return 'cirq.generalized_amplitude_damp(p={!r},gamma={!r})'.format(
            self._p, self._gamma
        )

    def __str__(self) -> str:
        return 'generalized_amplitude_damp(p={!r},gamma={!r})'.format(
            self._p, self._gamma
        )

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'GAD({},{})'.format(f, f).format(self._p, self._gamma)
        return 'GAD({!r},{!r})'.format(self._p, self._gamma)

    @property
    def p(self) -> float:
        """The probability of the qubit and environment exchanging energy."""
        return self._p

    @property
    def gamma(self) -> float:
        """The probability of the interaction being dissipative."""
        return self._gamma

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p', 'gamma'])


def generalized_amplitude_damp(
    p: float, gamma: float
) -> GeneralizedAmplitudeDampingChannel:
    r"""
    Returns a GeneralizedAmplitudeDampingChannel with the given
    probabilities gamma and p.

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


@value.value_equality
class AmplitudeDampingChannel(gate_features.SingleQubitGate):
    """Dampen qubit amplitudes through dissipation.

    This channel models the effect of energy dissipation to the
    surrounding environment.
    """

    def __init__(self, gamma: float) -> None:
        r"""The amplitude damping channel.

        Construct a channel that dissipates energy. The probability of
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

        Args:
            gamma: the probability of the interaction being dissipative.

        Raises:
            ValueError: is gamma is not a valid probability.
        """
        self._gamma = value.validate_probability(gamma, 'gamma')
        self._delegate = GeneralizedAmplitudeDampingChannel(1.0, self._gamma)

    def _channel_(self) -> Iterable[np.ndarray]:
        # just return first two kraus ops, we don't care about
        # the last two.
        return list(self._delegate._channel_())[:2]

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._gamma

    def __repr__(self) -> str:
        return 'cirq.amplitude_damp(gamma={!r})'.format(self._gamma)

    def __str__(self) -> str:
        return 'amplitude_damp(gamma={!r})'.format(self._gamma)

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'AD({})'.format(f).format(self._gamma)
        return 'AD({!r})'.format(self._gamma)

    @property
    def gamma(self) -> float:
        """The probability of the interaction being dissipative."""
        return self._gamma

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['gamma'])


def amplitude_damp(gamma: float) -> AmplitudeDampingChannel:
    r"""
    Returns an AmplitudeDampingChannel with the given probability gamma.

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
class ResetChannel(gate_features.SingleQubitGate):
    """Reset a qubit back to its |0⟩ state.

    The reset channel is equivalent to performing an unobserved measurement
    which then controls a bit flip onto the targeted qubit.
    """

    def __init__(self, dimension: int = 2) -> None:
        r"""The reset channel.

        Construct a channel that resets the qubit.

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

        Args:
            dimension: Specify this argument when resetting a qudit.  There will
                be `dimension` number of dimension by dimension matrices
                describing the channel, each with a 1 at a different position in
                the top row.
        """
        self._dimension = dimension

    def _qid_shape_(self):
        return (self._dimension,)

    def _act_on_(self, args: Any):
        from cirq import sim

        if isinstance(args, sim.ActOnStateVectorArgs):
            # Do a silent measurement.
            measurements, _ = sim.measure_state_vector(
                args.target_tensor,
                args.axes,
                out=args.target_tensor,
                qid_shape=args.target_tensor.shape)
            result = measurements[0]

            # Use measurement result to zero the qid.
            if result:
                zero = args.subspace_index(0)
                other = args.subspace_index(result)
                args.target_tensor[zero] = args.target_tensor[other]
                args.target_tensor[other] = 0

            return True

        return NotImplemented

    def _channel_(self) -> Iterable[np.ndarray]:
        # The first axis is over the list of channel matrices
        channel = np.zeros((self._dimension,) * 3, dtype=np.complex64)
        channel[:, 0, :] = np.eye(self._dimension)
        return channel

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._dimension

    def __repr__(self) -> str:
        if self._dimension == 2:
            return 'cirq.ResetChannel()'
        else:
            return 'cirq.ResetChannel(dimension={!r})'.format(self._dimension)

    def __str__(self) -> str:
        return 'reset'

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        return 'R'

    @property
    def dimension(self) -> int:
        """The dimension of the qudit being reset."""
        return self._dimension

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['dimension'])


def reset(qubit: 'cirq.Qid') -> raw_types.Operation:
    """Returns a `ResetChannel` on the given qubit.
    """
    return ResetChannel(qubit.dimension).on(qubit)


@value.value_equality
class PhaseDampingChannel(gate_features.SingleQubitGate):
    """Dampen qubit phase.

    This channel models phase damping which is the loss of quantum
    information without the loss of energy.
    """

    def __init__(self, gamma: float) -> None:
        r"""The phase damping channel.

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

        Args:
            gamma: The damping constant.

        Raises:
            ValueError: if gamma is not a valid probability.
        """
        self._gamma = value.validate_probability(gamma, 'gamma')

    def _channel_(self) -> Iterable[np.ndarray]:
        return (
            np.array([[1., 0.], [0., np.sqrt(1. - self._gamma)]]),
            np.array([[0., 0.], [0., np.sqrt(self._gamma)]]),
        )

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._gamma

    def __repr__(self) -> str:
        return 'cirq.phase_damp(gamma={!r})'.format(self._gamma)

    def __str__(self) -> str:
        return 'phase_damp(gamma={!r})'.format(self._gamma)

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'PD({})'.format(f).format(self._gamma)
        return 'PD({!r})'.format(self._gamma)

    @property
    def gamma(self) -> float:
        """The damping constant."""
        return self._gamma

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['gamma'])


def phase_damp(gamma: float) -> PhaseDampingChannel:
    r"""
    Creates a PhaseDampingChannel with damping constant gamma.

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


@value.value_equality
class PhaseFlipChannel(gate_features.SingleQubitGate):
    """Probabilistically flip the sign of the phase of a qubit."""

    def __init__(self, p: float) -> None:
        r"""The phase flip channel.

        Construct a channel to flip the phase with probability p.

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
        self._p = value.validate_probability(p, 'p')
        self._delegate = AsymmetricDepolarizingChannel(0., 0., p)

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        mixture = self._delegate._mixture_()
        # just return identity and z term
        return (mixture[0], mixture[3])

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return 'cirq.phase_flip(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'phase_flip(p={!r})'.format(self._p)

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'PF({})'.format(f).format(self._p)
        return 'PF({!r})'.format(self._p)

    @property
    def p(self) -> float:
        """The probability of a phase flip."""
        return self._p

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p'])


def _phase_flip_Z() -> common_gates.ZPowGate:
    """
    Returns a cirq.Z which corresponds to a guaranteed phase flip.
    """
    return common_gates.ZPowGate()


def _phase_flip(p: float) -> PhaseFlipChannel:
    r"""
    Returns a PhaseFlipChannel that flips a qubit's phase with probability p.

    This channel evolves a density matrix via:

        $$
        \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
        $$

    With:

        $$
        \begin{aligned}
        M_0 =& \sqrt{p} \begin{bmatrix}
                            1 & 0  \\
                            0 & 1
                       \end{bmatrix}
        \\
        M_1 =& \sqrt{1-p} \begin{bmatrix}
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


def phase_flip(
    p: Optional[float] = None
) -> Union[common_gates.ZPowGate, PhaseFlipChannel]:
    r"""
    Returns a PhaseFlipChannel that flips a qubit's phase with probability p
    if p is None, return a guaranteed phase flip in the form of a Z operation.

    This channel evolves a density matrix via:

        $$
        \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
        $$

    With:

        $$
        \begin{aligned}
        M_0 =& \sqrt{p} \begin{bmatrix}
                            1 & 0  \\
                            0 & 1
                       \end{bmatrix}
        \\
        M_1 =& \sqrt{1-p} \begin{bmatrix}
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


@value.value_equality
class BitFlipChannel(gate_features.SingleQubitGate):
    r"""Probabilistically flip a qubit from 1 to 0 state or vice versa."""

    def __init__(self, p: float) -> None:
        r"""The bit flip channel.

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

        Args:
            p: the probability of a bit flip.

        Raises:
            ValueError: if p is not a valid probability.
        """
        self._p = value.validate_probability(p, 'p')
        self._delegate = AsymmetricDepolarizingChannel(p, 0., 0.)

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        mixture = self._delegate._mixture_()
        # just return identity and x term
        return (mixture[0], mixture[1])

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return 'cirq.bit_flip(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'bit_flip(p={!r})'.format(self._p)

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'BF({})'.format(f).format(self._p)
        return 'BF({!r})'.format(self._p)

    @property
    def p(self) -> float:
        """The probability of a bit flip."""
        return self._p

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['p'])


def _bit_flip(p: float) -> BitFlipChannel:
    r"""
    Construct a BitFlipChannel that flips a qubit state
    with probability of a flip given by p.

    This channel evolves a density matrix via:

        $$
        \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
        $$

    With:

        $$
        \begin{aligned}
        M_0 =& \sqrt{p} \begin{bmatrix}
                            1 & 0 \\
                            0 & 1
                       \end{bmatrix}
        \\
        M_1 =& \sqrt{1-p} \begin{bmatrix}
                            0 & 1 \\
                            1 & -0
                         \end{bmatrix}
        \end{aligned}
        $$

    Args:
        p: the probability of a bit flip.

    Raises:
        ValueError: if p is not a valid probability.
    """
    return BitFlipChannel(p)


def bit_flip(
    p: Optional[float] = None
) -> Union[common_gates.XPowGate, BitFlipChannel]:
    r"""
    Construct a BitFlipChannel that flips a qubit state
    with probability of a flip given by p. If p is None, return
    a guaranteed flip in the form of an X operation.

    This channel evolves a density matrix via

        $$
        \rho \rightarrow M_0 \rho M_0^\dagger + M_1 \rho M_1^\dagger
        $$

    With

        $$
        \begin{aligned}
        M_0 =& \sqrt{p} \begin{bmatrix}
                            1 & 0 \\
                            0 & 1
                       \end{bmatrix}
        \\
        M_1 =& \sqrt{1-p} \begin{bmatrix}
                            0 & 1 \\
                            1 & -0
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
