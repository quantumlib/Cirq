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

from typing import Iterable

import numpy as np

from cirq import protocols
from cirq.ops import raw_types


class AsymmetricDepolarizingChannel(raw_types.Gate):
    """A channel that depolarizes asymmetrically along different directions."""

    def __init__(self, p_x: float, p_y: float, p_z: float) -> None:
        """The asymmmetric depolarizing channel.

        This channel applies one of four disjoint possibilities: nothing (the
        identity channel) or one of the three pauli gates. The disjoint
        probabilities of the three gates are p_x, p_y, and p_z and the
        identity is done with probability 1 - p_x - p_y - p_z. The supplied
        probabilities must be valid probabilities and the sum p_x + p_y + p_z
        must be a valid probability or else this constructor will raise a
        ValueError.

        This channel evolves a density matrix via
            \rho -> (1 -p_x + p_y + p_z) \rho
                    + p_x X \rho X + p_y Y \rho Y + p_z Z \rho Z

        Args:
            p_x: The probability that a Pauli X and no other gate occurs.
            p_y: The probability that a Pauli Y and no other gate occurs.
            p_z: The probability that a Pauli Z and no other gate occurs.

        Raises:
            ValueError if the args or the sum of the args are not probabilities.
        """
        def validate_probability(p, p_str):
            if p < 0:
                raise ValueError('{} was less than 0.'.format(p_str))
            elif p > 1:
                raise ValueError('{} was greater than 1.'.format(p_str))
            return p
        self._p_x = validate_probability(p_x, 'p_x')
        self._p_y = validate_probability(p_y, 'p_y')
        self._p_z = validate_probability(p_z, 'p_z')
        self._p_i = 1 - validate_probability(p_x + p_y + p_z, 'p_x + p_y + p_z')

    def _channel_(self) -> Iterable[np.ndarray]:
        return (
            np.sqrt(self._p_i) * np.eye(2),
            np.sqrt(self._p_x) * np.array([[0, 1], [1, 0]]),
            np.sqrt(self._p_y) * np.array([[0, -1j], [1j, 0]]),
            np.sqrt(self._p_z) * np.array([[1, 0], [0, -1]])
        )

    def _eq_tuple(self):
        return (AsymmetricDepolarizingChannel,
                self._p_x, self._p_y, self._p_z)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._eq_tuple() == other._eq_tuple()

    def __ne__(self, other):
        return not self == other

    def __repr__(self) -> str:
        return (
            'cirq.asymmetric_depolarize(p_x={!r},p_y={!r},p_z={!r})'
                .format(self._p_x, self._p_y, self._p_z)
        )

    def __str__(self) -> str:
        return ('asymmetric_depolarize(p_x={!r},p_y={!r},p_z={!r})'
                .format(self._p_x, self._p_y, self._p_z))

    def _circuit_diagram_info_(self,
        args: protocols.CircuitDiagramInfoArgs) -> str:
        return 'A({!r},{!r},{!r})'.format(self._p_x, self._p_y, self._p_z)


def asymmetric_depolarize(
    p_x: float,
    p_y: float,
    p_z: float) -> AsymmetricDepolarizingChannel:
    """Returns a AsymmetricDepolarizingChannel with given parameter.

    This channel evolves a density matrix via
        \rho -> (1 -p_x + p_y + p_z) \rho
                + p_x X \rho X + p_y Y \rho Y + p_z Z \rho Z

    Args:
        p_x: The probability that a Pauli X and no other gate occurs.
        p_y: The probability that a Pauli Y and no other gate occurs.
        p_z: The probability that a Pauli Z and no other gate occurs.

    Raises:
        ValueError if the args or the sum of the args are not probabilities.
    """
    return AsymmetricDepolarizingChannel(p_x, p_y, p_z)


class DepolarizingChannel(raw_types.Gate):
    """A channel that depolarizes a qubit."""

    def __init__(self, p) -> None:
        """The symmetric depolarizing channel.

        This channel applies one of four disjoint possibilities: nothing (the
        identity channel) or one of the three pauli gates. The disjoint
        probabilities of the three gates are all the same, p / 3, and the
        identity is done with probability 1 - p. The supplied probability
        must be a valid probability or else this constructor will raise a
        ValueError.

        This channel evolves a density matrix via
            \rho -> (1 - p) \rho
                    + (p / 3) X \rho X + (p / 3) Y \rho Y + (p / 3) Z \rho Z

        Args:
            p: The probability that one of the Pauli gates is applied. Each of
                the Pauli gates is applied independently with probability p / 3.

        Raises:
            ValueError if p is not a valid probability.
        """

        self._p = p
        self._delegate = AsymmetricDepolarizingChannel(p / 3, p / 3, p /3)

    def _channel_(self) -> Iterable[np.ndarray]:
        return self._delegate._channel_()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._p == other._p

    def __ne__(self, other):
        return not self == other

    def __repr__(self) -> str:
        return 'cirq.depolarize(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'depolarize(p={!r})'.format(self._p)

    def _circuit_diagram_info_(self,
        args: protocols.CircuitDiagramInfoArgs) -> str:
        return 'D({!r})'.format(self._p)


def depolarize(p: float) -> DepolarizingChannel:
    """Returns a DepolarizingChannel with given probability of error.

    This channel applies one of four disjoint possibilities: nothing (the
    identity channel) or one of the three pauli gates. The disjoint
    probabilities of the three gates are all the same, p / 3, and the
    identity is done with probability 1 - p. The supplied probability
    must be a valid probability or else this constructor will raise a
    ValueError.

    This channel evolves a density matrix via
        \rho -> (1 - p) \rho
                + (p / 3) X \rho X + (p / 3) Y \rho Y + (p / 3) Z \rho Z

    Args:
        p: The probability that one of the Pauli gates is applied. Each of
            the Pauli gates is applied independently with probability p / 3.

    Raises:
        ValueError if p is not a valid probability.
    """
    return DepolarizingChannel(p)
