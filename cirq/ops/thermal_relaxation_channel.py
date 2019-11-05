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
"""Module for the thermal relaxation channel."""

from typing import Iterable

import numpy as np

from cirq import protocols, value
from cirq.ops import gate_features


@value.value_equality
class ThermalRelaxationChannel(gate_features.SingleQubitGate):
    """Dampen qubits at varying rates through non ideal disipation.

    This channel models the effect of energy dissipation into the environment
    as well as the environment depositing energy into the system at varying
    rates.
    """

    def __init__(self, p_exchange: float, p_relaxation: float,
                 p_dephasing: float) -> None:
        r"""The thermal relaxation channel.

        Construct a channel to model energy dissipation into the environment
        as well as the environment depositing energy into the system. The
        probabilities with which the energy exchange occur are given by `gamma`
        and `beta` with the probability of the environment being not excited is
        given by `p`.

        The stationary state of this channel is the diagonal density matrix
        with probability `p` of being |0⟩ and probability `1-p` of being |1⟩.

        Recall the kraus opeartors for the `GeneralizedAmplitudeDampingChannel`:

            $$
            \begin{aligned}
            M_0 =& \sqrt{p_exchange} \begin{bmatrix}
                                1 & 0  \\
                                0 & \sqrt{1 - p_relaxation}
                            \end{bmatrix}
            \\
            M_1 =& \sqrt{p_exchange} \begin{bmatrix}
                                0 & \sqrt{p_relaxation} \\
                                0 & 0
                           \end{bmatrix}
            \\
            M_2 =& \sqrt{1-p_exchange} \begin{bmatrix}
                                \sqrt{1-p_relaxation} & 0 \\
                                 0 & 1
                              \end{bmatrix}
            \\
            M_3 =& \sqrt{1-p_exchange} \begin{bmatrix}
                                 0 & 0 \\
                                 \sqrt{p_relaxation} & 0
                             \end{bmatrix}
            \end{aligned}
            $$

        This has a choi representation of:

            $$
             \begin{aligned}
                CHO =&  \begin{bmatrix}
                            1 - (1-p_exchange) p_relaxation & 0 & 0 &
                                \sqrt{1 - p_relaxation} \\
                            0 & (1-p_exchange) p_relaxation 0 & 0 \\
                            0 & 0 & p_exchange p_relaxation & 0 \\
                            \sqrt{1 - p_relaxation} & 0 & 0 &
                                1 - p_exchange p_relaxation
                        \end{bmatrix}
             \end{aligned}
            $$

        Where each 2x2 block can be thought of as how the channel acts on each
        basis element in the space of density matrices.

        For the `ThermalRelaxationChannel` channel we simply substitute
        `p_dephasing` in for `p_relaxation` on the off diagonal elements of the
        above choi matrix and then evolve the channel using the kraus operators
        found by sqrt factorization of the choi matrix.

        Args:
            p_relaxation: the probability of amplitude damping/relaxation
                occuring.
            p_dephasing: the probability of dephasing occuring.
            p_exchange: the probability of the qubit and environment exchanging
                energy.

        Raises:
            ValueError: if p_exchange, p_relaxation or p_dephasing is not a
                valid probability.
            ValueError: if p_exchange, p_relaxation, p_dephasing breaks CP
                condition.
        """

        self._p_relaxation = value.validate_probability(p_relaxation,
                                                        'p_relaxation')
        self._p_dephasing = value.validate_probability(p_dephasing,
                                                       'p_dephasing')
        self._p_exchange = value.validate_probability(p_exchange, 'p_exchange')

        choi = np.array(
            [[
                1. - (1. - p_exchange) * p_relaxation, 0, 0,
                np.sqrt(1. - p_dephasing)
            ], [0, (1. - p_exchange) * p_relaxation, 0, 0],
             [0, 0, p_exchange * p_relaxation, 0],
             [np.sqrt(1. - p_dephasing), 0, 0, 1. - p_exchange * p_relaxation]])

        vals, vecs = np.linalg.eigh(choi)

        # Convert to kraus using choi's theorem on CP maps.
        self._kraus: Iterable[np.ndarray] = ()
        for i in range(len(vals)):
            vals += 1e-9  # fix small roundoffs.
            if vals[i] < 0:
                raise ValueError('Thermal relaxation with '
                                 'p_exchange={}, p_relaxation={}, '
                                 'p_dephasing={} breaks CP'
                                 ' requirement.'.format(p_exchange,
                                                        p_relaxation,
                                                        p_dephasing))
            kraus_i = np.sqrt(vals[i]) * vecs[:, i].reshape((2, 2)).T
            self._kraus += (kraus_i,)

    def _channel_(self) -> Iterable[np.ndarray]:
        return self._kraus

    def _has_channel_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p_exchange, self._p_relaxation, self._p_dephasing

    def __repr__(self) -> str:
        return ('cirq.thermal_relaxation(p_exchange={!r},p_relaxation={!r},'
                'p_dephasing={!r})').format(self._p_exchange,
                                            self._p_relaxation,
                                            self._p_dephasing)

    def __str__(self) -> str:
        return ('thermal_relaxation(p_exchange={!r},p_relaxation={!r},'
                'p_dephasing={!r})').format(self._p_exchange,
                                            self._p_relaxation,
                                            self._p_dephasing)

    def _circuit_diagram_info_(self,
                               args: 'protocols.CircuitDiagramInfoArgs') -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'ThR({},{},{})'.format(f, f,
                                          f).format(self._p_exchange,
                                                    self._p_relaxation,
                                                    self._p_dephasing)
        return 'ThR({!r},{!r},{!r})'.format(self._p_exchange,
                                            self._p_relaxation,
                                            self._p_dephasing)

    @property
    def p_exchange(self) -> float:
        """The probability of the qubit and environment exchanging energy."""
        return self._p_exchange

    @property
    def p_relaxation(self) -> float:
        """The probability of the interaction being dissipative."""
        return self._p_relaxation

    @property
    def p_dephasing(self) -> float:
        """The probability of the interaction being dissipative."""
        return self._p_dephasing

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(
            self, ['p_exchange', 'p_relaxation', 'p_dephasing'])


def thermal_relaxation(p_exchange: float, p_relaxation: float,
                       p_dephasing: float) -> ThermalRelaxationChannel:
    r"""The thermal relaxation channel.

    Construct a channel to model energy dissipation into the environment
    as well as the environment depositing energy into the system. The
    probabilities with which the energy exchange occur are given by `gamma`
    and `beta` with the probability of the environment being not excited is
    given by `p`.

    The stationary state of this channel is the diagonal density matrix
    with probability `p` of being |0⟩ and probability `1-p` of being |1⟩.

    Recall the kraus opeartors for the `GeneralizedAmplitudeDampingChannel`:

        $$
        \begin{aligned}
        M_0 =& \sqrt{p_exchange} \begin{bmatrix}
                            1 & 0  \\
                            0 & \sqrt{1 - p_relaxation}
                        \end{bmatrix}
        \\
        M_1 =& \sqrt{p_exchange} \begin{bmatrix}
                            0 & \sqrt{p_relaxation} \\
                            0 & 0
                       \end{bmatrix}
        \\
        M_2 =& \sqrt{1-p_exchange} \begin{bmatrix}
                            \sqrt{1-p_relaxation} & 0 \\
                             0 & 1
                          \end{bmatrix}
        \\
        M_3 =& \sqrt{1-p_exchange} \begin{bmatrix}
                             0 & 0 \\
                             \sqrt{p_relaxation} & 0
                         \end{bmatrix}
        \end{aligned}
        $$

    This has a choi representation of:

        $$
         \begin{aligned}
            CHO =&  \begin{bmatrix}
                        1 - (1-p_exchange) p_relaxation & 0 & 0 &
                            \sqrt{1 - p_relaxation} \\
                        0 & (1-p_exchange) p_relaxation 0 & 0 \\
                        0 & 0 & p_exchange p_relaxation & 0 \\
                        \sqrt{1 - p_relaxation} & 0 & 0 &
                            1 - p_exchange p_relaxation
                    \end{bmatrix}
         \end{aligned}
        $$

    Where each 2x2 block can be thought of as how the channel acts on each
    basis element in the space of density matrices.

    For the `ThermalRelaxationChannel` channel we simply substitute
    `p_dephasing` in for `p_relaxation` on the off diagonal elements of the
    above choi matrix and then evolve the channel using the kraus operators
    found by sqrt factorization of the choi matrix.

    Args:
        p_relaxation: the probability of amplitude damping/relaxation
            occuring.
        p_dephasing: the probability of dephasing occuring.
        p_exchange: the probability of the qubit and environment exchanging
            energy.

    Raises:
        ValueError: if p_exchange, p_relaxation or p_dephasing is not a valid
            probability.
        ValueError: if p_exchange, p_relaxation, p_dephasing breaks CP
            condition.
    """
    return ThermalRelaxationChannel(p_exchange, p_relaxation, p_dephasing)
