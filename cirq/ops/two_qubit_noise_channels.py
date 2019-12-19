# Copyright 2019 The Cirq Developers
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
import cmath
from typing import Sequence, Tuple, Union, List

import numpy as np

import cirq

from cirq import value, protocols

from cirq._compat import proper_repr
from cirq.ops import gate_features, eigen_gate, raw_types


@cirq.value.value_equality
class TwoQubitAsymmetricDepolarizingChannel(gate_features.TwoQubitGate):
    """A channel to apply when a two qubit gate is applied.
    """

    def __init__(self, p_xi: float, p_yi: float, p_zi: float, p_xx: float,
                 p_yx: float, p_zx: float, p_xy: float, p_yy: float,
                 p_zy: float, p_xz: float, p_yz: float, p_zz: float,
                 p_ix: float, p_iy: float, p_iz: float) -> None:
        r"""The asymmetric depolarizing channel.

            This channel applies one of 16 disjoint possibilities: nothing (the
            identity channel) or a combination of the three pauli gates and the identity on each  of two qubits.
            The disjoint probabilities of the 15 gates are p_xi, p_xx, and p_xy, p_xz, and so on.
            The identity is done with probability 1 - sum(p_jk), where j is the operation on the first qubit and k is the operation on the second.
            The supplied probabilities must be valid probabilities and the sum p_jk
            must be a valid probability or else this constructor will raise a
            ValueError.

            This channel evolves a density matrix via

                $$
                \rho \rightarrow (1 - \Sum_{jk} p_jk) \rho
                        + \Sum_{jk} p_jk j \otimes k
                $$

            Args:
                p_xi: The probability that a Pauli X on qubit 1 and no other gate occurs.
                p_yi: The probability that a Pauli Y on qubit 1 and no other gate occurs.
                p_zi: The probability that a Pauli Z on qubit 1 and no other gate occurs.
                p_xx: The probability that a Pauli X on qubit 1 and a Pauli X on qubit 2 occurs.
                p_yx: The probability that a Pauli Y on qubit 1 and a Pauli Y on qubit 2 occurs.
                p_zx: The probability that a Pauli Z on qubit 1 and a Pauli Z on qubit 2 occurs.
                p_xy: The probability that a Pauli X on qubit 1 and a Pauli X on qubit 2 occurs.
                p_yy: The probability that a Pauli Y on qubit 1 and a Pauli Y on qubit 2 occurs.
                p_zy: The probability that a Pauli Z on qubit 1 and a Pauli Z on qubit 2 occurs.
                p_xz: The probability that a Pauli X on qubit 1 and a Pauli X on qubit 2 occurs.
                p_yz: The probability that a Pauli Y on qubit 1 and a Pauli Y on qubit 2 occurs.
                p_zz: The probability that a Pauli Z on qubit 1 and a Pauli Z on qubit 2 occurs.

            Raises:
                ValueError: if the args or the sum of args are not probabilities.
        """
        self._p_xi = value.validate_probability(p_xi, 'p_xi')
        self._p_yi = value.validate_probability(p_yi, 'p_yi')
        self._p_zi = value.validate_probability(p_zi, 'p_zi')
        self._p_xx = value.validate_probability(p_xx, 'p_xx')
        self._p_yx = value.validate_probability(p_yx, 'p_yx')
        self._p_zx = value.validate_probability(p_zx, 'p_zx')
        self._p_xy = value.validate_probability(p_xy, 'p_xy')
        self._p_yy = value.validate_probability(p_yy, 'p_yy')
        self._p_zy = value.validate_probability(p_zy, 'p_zy')
        self._p_xz = value.validate_probability(p_xz, 'p_xz')
        self._p_yz = value.validate_probability(p_yz, 'p_yz')
        self._p_zz = value.validate_probability(p_zz, 'p_zz')
        self._p_ix = value.validate_probability(p_ix, 'p_ix')
        self._p_iy = value.validate_probability(p_iy, 'p_iy')
        self._p_iz = value.validate_probability(p_iz, 'p_iz')
        self._p_ii = 1 - value.validate_probability(
            p_xi + p_yi + p_zi + p_xx + p_yx + p_zx + p_xy + p_yy + p_zy +
            p_xz + p_yz + p_zz + p_ix + p_iy + p_iz, 'p_ii')

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        return ((self._p_ii, protocols.unitary(cirq.IdentityGate(2))),
                (self._p_xi, protocols.unitary(XIGate())),
                (self._p_xx,
                 protocols.unitary(XXGate())), (self._p_xy,
                                                protocols.unitary(XYGate())),
                (self._p_xz,
                 protocols.unitary(XZGate())), (self._p_yi,
                                                protocols.unitary(YIGate())),
                (self._p_yx,
                 protocols.unitary(YXGate())), (self._p_yy,
                                                protocols.unitary(YYGate())),
                (self._p_yz,
                 protocols.unitary(YZGate())), (self._p_zi,
                                                protocols.unitary(ZIGate())),
                (self._p_zx,
                 protocols.unitary(ZXGate())), (self._p_zy,
                                                protocols.unitary(ZYGate())),
                (self._p_zz,
                 protocols.unitary(ZZGate())), (self._p_ix,
                                                protocols.unitary(IXGate())),
                (self._p_iy,
                 protocols.unitary(IYGate())), (self._p_iz,
                                                protocols.unitary(IZGate())))

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p_xi, self._p_xx, self._p_xy, self._p_xz, \
               self._p_yi, self._p_yx, self._p_yy, self._p_yz, \
               self._p_zi, self._p_zx, self._p_zy, self._p_zz, \
               self._p_iz, self._p_ix, self._p_iy

    def __repr__(self) -> str:
        return 'two_qubit_asymmetric_depolarize' \
               '(p_xi={!r},p_xx={!r},p_xy={!r},p_xz={!r}, ' \
               'p_yi={!r},p_yx={!r},p_yy={!r},p_yz={!r}, ' \
               'p_zi={!r},p_zx={!r},p_zy={!r},p_zz={!r}, ' \
               'p_ix={!r},p_iy={!r},p_iz={!r})'.format(
            self._p_xi, self._p_xx, self._p_xy, self._p_xz,
            self._p_yi, self._p_yx, self._p_yy, self._p_yz,
            self._p_zi, self._p_zx, self._p_zy, self._p_zz,
            self._p_iz, self._p_ix, self._p_iy)

    def __str__(self) -> str:
        return 'two_qubit_asymmetric_depolarize' \
               '(p_xi={!r},p_xx={!r},p_xy={!r},p_xz={!r}, ' \
               'p_yi={!r},p_yx={!r},p_yy={!r},p_yz={!r}, ' \
               'p_zi={!r},p_zx={!r},p_zy={!r},p_zz={!r}, ' \
               'p_ix={!r},p_iy={!r},p_iz={!r})'.format(
            self._p_xi, self._p_xx, self._p_xy, self._p_xz,
            self._p_yi, self._p_yx, self._p_yy, self._p_yz,
            self._p_zi, self._p_zx, self._p_zy, self._p_zz,
            self._p_iz, self._p_ix, self._p_iy)

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs) -> str:
        if args.precision is not None:
            f = '{:.' + str(args.precision) + 'g}'
            return 'A({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})' \
                .format(f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, ).format(
                        self._p_xi, self._p_xx, self._p_xy, self._p_xz,
                        self._p_yi, self._p_yx, self._p_yy, self._p_yz,
                        self._p_zi, self._p_zx, self._p_zy, self._p_zz,
                        self._p_iz, self._p_ix, self._p_iy)
        return 'A({!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r},{!r})'.format(
            self._p_xi, self._p_xx, self._p_xy, self._p_xz, self._p_yi,
            self._p_yx, self._p_yy, self._p_yz, self._p_zi, self._p_zx,
            self._p_zy, self._p_zz, self._p_iz, self._p_ix, self._p_iy)


def two_qubit_asymmetric_depolarize(p_xi: float, p_yi: float, p_zi: float,
                                    p_xx: float, p_yx: float, p_zx: float,
                                    p_xy: float, p_yy: float, p_zy: float,
                                    p_xz: float, p_yz: float, p_zz: float,
                                    p_ix: float, p_iy: float, p_iz: float
                                   ) -> TwoQubitAsymmetricDepolarizingChannel:
    """
    Returns a TwoQubitAsymmetricDepolarisingChannel with the given probabilities.
    :param p_xi: The probability that a Pauli X on qubit 1 and no other gate occurs.
    :param p_yi: The probability that a Pauli Y on qubit 1 and no other gate occurs.
    :param p_zi: The probability that a Pauli Z on qubit 1 and no other gate occurs.
    :param p_xx: The probability that a Pauli X on qubit 1 and a Pauli X on qubit 2 occurs.
    :param p_yx: The probability that a Pauli Y on qubit 1 and a Pauli Y on qubit 2 occurs.
    :param p_zx: The probability that a Pauli Z on qubit 1 and a Pauli Z on qubit 2 occurs.
    :param p_xy: The probability that a Pauli X on qubit 1 and a Pauli X on qubit 2 occurs.
    :param p_yy: The probability that a Pauli Y on qubit 1 and a Pauli Y on qubit 2 occurs.
    :param p_zy: The probability that a Pauli Z on qubit 1 and a Pauli Z on qubit 2 occurs.
    :param p_xz: The probability that a Pauli X on qubit 1 and a Pauli X on qubit 2 occurs.
    :param p_yz: The probability that a Pauli Y on qubit 1 and a Pauli Y on qubit 2 occurs.
    :param p_zz: The probability that a Pauli Z on qubit 1 and a Pauli Z on qubit 2 occurs.
    :return: A two qubit asymmetric depolarising channel with the given probabilities.
    """
    return TwoQubitAsymmetricDepolarizingChannel(p_xi, p_yi, p_zi, p_xx, p_yx,
                                                 p_zx, p_xy, p_yy, p_zy, p_xz,
                                                 p_yz, p_zz, p_ix, p_iy, p_iz)


@cirq.value.value_equality
class TwoQubitDepolarizingChannel(gate_features.TwoQubitGate):

    def __init__(self, p) -> None:
        r"""The symmetric depolarizing channel.
        Applies the two qubit asymmetric depolarising channel with equal probability for each Kraus operator.
        """
        self._p = p
        self._delegate = TwoQubitAsymmetricDepolarizingChannel(
            p / 15, p / 15, p / 15, p / 15, p / 15, p / 15, p / 15, p / 15,
            p / 15, p / 15, p / 15, p / 15, p / 15, p / 15, p / 15)

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        return self._delegate._mixture_()

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def __repr__(self) -> str:
        return 'two_qubit_depolarize(p={!r})'.format(self._p)

    def __str__(self) -> str:
        return 'two_qubit_depolarize(p={!r})'.format(self._p)

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs) -> str:
        return 'D2({!r})'.format(self._p)


def two_qubit_depolarize(p: float) -> TwoQubitDepolarizingChannel:
    r"""The symmetric depolarizing channel.
    Applies the two qubit asymmetric depolarising channel with equal probability for each Kraus operator.
    :param p: The probability of applying a noisy gate.
    :return: The two qubit symmetric depolarising channel.
    """
    return TwoQubitDepolarizingChannel(p)


class XIGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """
    A two qubit gate which rotates the first qubit about the X axis of its Bloch sphere,
    and does nothing to the second qubit.

    The unitary  matrix of XIGate(exponent=t) is:

        [   c·g,       0, -i·g·s,      0],
        [     0,     c·g,      0, -i·g·s],
        [-i·g·s,       0,    c·g,      0],
        [      0, -i·g·s,      0,    c·g]]

    where:
        c = cos(π·t/2)
        s = sin(π·t/2)
        g = exp(i·π·t/2).

    """
    def _eigen_components(self):
        return [(1,
                 np.array([[0.5, 0., -0.5, 0.], [0., 0.5, 0., -0.5],
                           [-0.5, 0., 0.5, 0.], [0., -0.5, 0., 0.5]])),
                (0,
                 np.array([[0.5, 0., 0.5, 0.], [0., 0.5, 0., 0.5],
                           [0.5, 0., 0.5, 0.], [0., 0.5, 0., 0.5]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.X().on(a)
        yield cirq.IdentityGate(1).on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'XI'
        return 'XI**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.XI'
            return '(cirq.XI**{})'.format(proper_repr(self._exponent))
        return ('cirq.XIPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class XXGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the X axis of its Bloch sphere,
       and rotates the second qubit about the X axis of its Bloch sphere by the same amount.

       The unitary  matrix of XXGate(exponent=t) is:

            [[  c^2·g^2, -i·c·g^2·s, -i·c·g^2·s,   -g^2·s^2],
            [-i·c·g^2·s,    c^2·g^2,   -g^2·s^2, -i·c·g^2·s],
            [-i·c·g^2·s,   -g^2·s^2,    c^2·g^2, -i·c·g^2·s],
            [  -g^2·s^2, -i·c·g^2·s, -i·c·g^2·s,    c^2·g^2]]

       where:
           c = cos(π·t/2)
           s = sin(π·t/2)
           g = exp(i·π·t/2).
       """
    def _eigen_components(self):
        return [(1,
                 np.array([[0.5, 0., 0., -0.5], [0., 0.5, -0.5, 0.],
                           [0., -0.5, 0.5, 0.], [-0.5, 0., 0., 0.5]])),
                (0,
                 np.array([[0.5, 0., 0., 0.5], [0., 0.5, 0.5, 0.],
                           [0., 0.5, 0.5, 0.], [0.5, 0., 0., 0.5]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.X().on(a)
        yield cirq.X().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'XX'
        return 'XX**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.XX'
            return '(cirq.XX**{})'.format(proper_repr(self._exponent))
        return ('cirq.XXPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class XYGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the X axis of its Bloch sphere,
       and rotates the second qubit about the Y axis of its Bloch sphere by the same amount.

       The unitary  matrix of XYGate(exponent=t) is:

        [[  c^2·g^2,   -c·g^2·s, -i·c·g^2·s,  i·g^2·s^2],
        [   c·g^2·s,    c^2·g^2, -i·g^2·s^2, -i·c·g^2·s],
        [-i·c·g^2·s,  i·g^2·s^2,    c^2·g^2,   -c·g^2·s],
        [-i·g^2·s^2, -i·c·g^2·s,    c·g^2·s,    c^2·g^2]]

       where:
           c = cos(π·t/2)
           s = sin(π·t/2)
           g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.5j],
                           [0. + 0.j, 0.5 + 0.j, 0. - 0.5j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.5j, 0.5 + 0.j, 0. + 0.j],
                           [0. - 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.j]])),
                (0,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. - 0.5j],
                           [0. + 0.j, 0.5 + 0.j, 0. + 0.5j, 0. + 0.j],
                           [0. + 0.j, 0. - 0.5j, 0.5 + 0.j, 0. + 0.j],
                           [0. + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.j]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.X().on(a)
        yield cirq.Y().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'XY'
        return 'XY**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.XY'
            return '(cirq.XY**{})'.format(proper_repr(self._exponent))
        return ('cirq.XYPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class XZGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the X axis of its Bloch sphere,
       and rotates the second qubit about the Z axis of its Bloch sphere by the same amount.

       The unitary  matrix of XZGate(exponent=t) is:

       [[   c·g,        0, -i·g·s,        0],
        [     0,    c·g^2,      0, -i·g^2·s],
        [-i·g·s,        0,    c·g,        0],
        [     0, -i·g^2·s,      0,    c·g^2]]

       where:
           c = cos(π·t/2)
           s = sin(π·t/2)
           g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0.5, 0., -0.5, 0.], [0., 0.5, 0., 0.5],
                           [-0.5, 0., 0.5, 0.], [0., 0.5, 0., 0.5]])),
                (0.0,
                 np.array([[0.5, 0., 0.5, 0.], [0., 0.5, 0., -0.5],
                           [0.5, 0., 0.5, 0.], [0., -0.5, 0., 0.5]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Y().on(a)
        yield cirq.X().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'XZ'
        return 'XZ**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.XZ'
            return '(cirq.XZ**{})'.format(proper_repr(self._exponent))
        return ('cirq.XZPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class YIGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """
        A two qubit gate which rotates the first qubit about the Y axis of its Bloch sphere,
        and does nothing to the second qubit.

        The unitary  matrix of YIGate(exponent=t) is:

            [c·g,   0, -g·s,    0],
            [  0, c·g,    0, -g·s],
            [g·s,   0,  c·g,    0],
            [  0, g·s,    0,  c·g]]

        where:

            c = cos(π·t/2)
            s = sin(π·t/2)
            g = exp(i·π·t/2).

        """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. + 0.5j, 0. + 0.j],
                           [0. + 0.j, 0.5 + 0.j, 0. + 0.j, 0. + 0.5j],
                           [0. - 0.5j, 0. + 0.j, 0.5 + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. - 0.5j, 0. + 0.j, 0.5 + 0.j]])),
                (0.0,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. - 0.5j, 0. + 0.j],
                           [0. + 0.j, 0.5 + 0.j, 0. + 0.j, 0. - 0.5j],
                           [0. + 0.5j, 0. + 0.j, 0.5 + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.5j, 0. + 0.j, 0.5 + 0.j]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Y().on(a)
        yield cirq.IdentityGate(1).on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'YI'
        return 'YI**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.YI'
            return '(cirq.YI**{})'.format(proper_repr(self._exponent))
        return ('cirq.YIPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class YXGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the Y axis of its Bloch sphere,
       and rotates the second qubit about the X axis of its Bloch sphere by the same amount.

       The unitary  matrix of YXGate(exponent=t) is:

       [[   c^2·g^2, -i·c·g^2·s,   -c·g^2·s,  i·g^2·s^2],
        [-i·c·g^2·s,    c^2·g^2,  i·g^2·s^2,   -c·g^2·s],
        [   c·g^2·s, -i·g^2·s^2,    c^2·g^2, -i·c·g^2·s],
        [-i·g^2·s^2,    c·g^2·s, -i·c·g^2·s,    c^2·g^2]]

       where:
           c = cos(π·t/2)
           s = sin(π·t/2)
           g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.5j],
                           [0. + 0.j, 0.5 + 0.j, 0. + 0.5j, 0. + 0.j],
                           [0. + 0.j, 0. - 0.5j, 0.5 + 0.j, 0. + 0.j],
                           [0. - 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.j]])),
                (0.0,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. - 0.5j],
                           [0. + 0.j, 0.5 + 0.j, 0. - 0.5j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.5j, 0.5 + 0.j, 0. + 0.j],
                           [0. + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.j]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Y().on(a)
        yield cirq.X().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'YX'
        return 'YX**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.YX'
            return '(cirq.YX**{})'.format(proper_repr(self._exponent))
        return ('cirq.YXPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class YYGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the Y axis of its Bloch sphere,
       and rotates the second qubit about the Y axis of its Bloch sphere by the same amount.

       The unitary  matrix of YYGate(exponent=t) is:

       [[c^2·g^2,  -c·g^2·s,  -c·g^2·s, g^2·s^2],
        [ c·g^2·s,  c^2·g^2, -g^2·s^2, -c·g^2·s],
        [ c·g^2·s, -g^2·s^2,  c^2·g^2, -c·g^2·s],
        [g^2·s^2,   c·g^2·s,   c·g^2·s, c^2·g^2]]

       where:
           c = cos(π·t/2)
           s = sin(π·t/2)
           g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0.5 + 0.j],
                           [0. + 0.j, 0.5 + 0.j, -0.5 + 0.j, 0. + 0.j],
                           [0. + 0.j, -0.5 + 0.j, 0.5 + 0.j, 0. + 0.j],
                           [0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0.5 + 0.j]])),
                (0.0,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. + 0.j, -0.5 + 0.j],
                           [0. + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0. + 0.j],
                           [0. + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0. + 0.j],
                           [-0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0.5 + 0.j]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Y().on(a)
        yield cirq.Y().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'YY'
        return 'YY**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.YY'
            return '(cirq.YY**{})'.format(proper_repr(self._exponent))
        return ('cirq.YYPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class YZGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the Y axis of its Bloch sphere,
       and rotates the second qubit about the Z axis of its Bloch sphere by the same amount.

       The unitary  matrix of YZGate(exponent=t) is:

       [[c·g,     0, -g·s,      0],
        [  0, c·g^2,    0, -g^2·s],
        [g·s,     0,  c·g,      0],
        [  0, g^2·s,    0,  c·g^2]]

       where:
           c = cos(π·t/2)
           s = sin(π·t/2)
           g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. + 0.5j, 0. + 0.j],
                           [0. + 0.j, 0.5 + 0.j, 0. + 0.j, 0. - 0.5j],
                           [0. - 0.5j, 0. + 0.j, 0.5 + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.5j, 0. + 0.j, 0.5 + 0.j]])),
                (0.0,
                 np.array([[0.5 + 0.j, 0. + 0.j, 0. - 0.5j, 0. + 0.j],
                           [0. + 0.j, 0.5 + 0.j, 0. + 0.j, 0. + 0.5j],
                           [0. + 0.5j, 0. + 0.j, 0.5 + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. - 0.5j, 0. + 0.j, 0.5 + 0.j]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Y().on(a)
        yield cirq.Z().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'YZ'
        return 'YZ**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.YZ'
            return '(cirq.YZ**{})'.format(proper_repr(self._exponent))
        return ('cirq.YZPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class ZIGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the Z axis of its Bloch sphere,
   and does nothing to the second qubit.

   The unitary  matrix of ZIGate(exponent=t) is:

        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, g, 0],
        [0, 0, 0, g]]

   where:

       g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])),
                (0.0,
                 np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.],
                           [0., 0., 0., 0.]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Z().on(a)
        yield cirq.IdentityGate(1).on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'ZI'
        return 'ZI**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ZI'
            return '(cirq.ZI**{})'.format(proper_repr(self._exponent))
        return ('cirq.ZIPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class ZXGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the Z axis of its Bloch sphere,
       and rotates the second qubit about the X axis of its Bloch sphere by the same amount.

       The unitary  matrix of ZXGate(exponent=t) is:

        [[   c·g, -i·g·s,        0,        0],
         [-i·g·s,    c·g,        0,        0],
         [     0,      0,    c·g^2, -i·g^2·s],
         [     0,      0, -i·g^2·s,    c·g^2]]

       where:
           c = cos(π·t/2)
           s = sin(π·t/2)
           g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0.5, -0.5, 0., 0.], [-0.5, 0.5, 0., 0.],
                           [0., 0., 0.5, 0.5], [0., 0., 0.5, 0.5]])),
                (0.0,
                 np.array([[0.5, 0.5, 0., 0.], [0.5, 0.5, 0., 0.],
                           [0., 0., 0.5, -0.5], [0., 0., -0.5, 0.5]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Z().on(a)
        yield cirq.X().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'ZX'
        return 'ZX**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ZX'
            return '(cirq.ZX**{})'.format(proper_repr(self._exponent))
        return ('cirq.ZXPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class ZYGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the Z axis of its Bloch sphere,
       and rotates the second qubit about the Y axis of its Bloch sphere by the same amount.

       The unitary  matrix of ZYGate(exponent=t) is:

        [[c·g, -g·s,     0,      0],
         [g·s,  c·g,     0,      0],
         [  0,    0, c·g^2, -g^2·s],
         [  0,    0, g^2·s,  c·g^2]]

       where:
           c = cos(π·t/2)
           s = sin(π·t/2)
           g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0.5 + 0.j, 0. + 0.5j, 0. + 0.j, 0. + 0.j],
                           [0. - 0.5j, 0.5 + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 0.5 + 0.j, 0. - 0.5j],
                           [0. + 0.j, 0. + 0.j, 0. + 0.5j, 0.5 + 0.j]])),
                (0.0,
                 np.array([[0.5 + 0.j, 0. - 0.5j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.5j, 0.5 + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 0.5 + 0.j, 0. + 0.5j],
                           [0. + 0.j, 0. + 0.j, 0. - 0.5j, 0.5 + 0.j]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Z().on(a)
        yield cirq.Y().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'ZY'
        return 'ZY**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ZY'
            return '(cirq.ZY**{})'.format(proper_repr(self._exponent))
        return ('cirq.ZYPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class ZZGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which rotates the first qubit about the Z axis of its Bloch sphere,
       and rotates the second qubit about the Z axis of its Bloch sphere by the same amount.

       The unitary  matrix of ZZGate(exponent=t) is:

            [[1, 0, 0,   0],
             [0, g, 0,   0],
             [0, 0, g,   0],
             [0, 0, 0, g^2]]

       where:

           g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.],
                           [0., 0., 0., 0.]])),
                (0.0,
                 np.array([[1., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.],
                           [0., 0., 0., 1.]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.Z().on(a)
        yield cirq.Z().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'ZZ'
        return 'ZZ**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ZZ'
            return '(cirq.ZZ**{})'.format(proper_repr(self._exponent))
        return ('cirq.ZZPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class IXGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which does nothing to the first qubit and
    rotates the second qubit about the X axis of its Bloch sphere,

   The unitary  matrix of IXGate(exponent=t) is:

        [   c·g,      0, -i·g·s,      0],
        [     0,    c·g,      0, -i·g·s],
        [-i·g·s,      0,    c·g,      0],
        [     0, -i·g·s,      0,    c·g]]

   where:

        c = cos(π·t/2)
        s = sin(π·t/2)
        g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0.5, -0.5, 0., 0.], [-0.5, 0.5, 0., 0.],
                           [0., 0., 0.5, -0.5], [0., 0., -0.5, 0.5]])),
                (0.0,
                 np.array([[0.5, 0.5, 0., 0.], [0.5, 0.5, 0., 0.],
                           [0., 0., 0.5, 0.5], [0., 0., 0.5, 0.5]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.IdentityGate(1).on(a)
        yield cirq.X().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'IX'
        return 'IX**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.IX'
            return '(cirq.IX**{})'.format(proper_repr(self._exponent))
        return ('cirq.IXPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class IYGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which does nothing to the first qubit and
        rotates the second qubit about the Y axis of its Bloch sphere,

       The unitary  matrix of IYGate(exponent=t) is:

            [c·g,   0, -g·s,    0],
            [  0, c·g,    0, -g·s],
            [g·s,   0,  c·g,    0],
            [  0, g·s,    0,  c·g]]

       where:

            c = cos(π·t/2)
            s = sin(π·t/2)
            g = exp(i·π·t/2).
        """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0.5 + 0.j, 0. + 0.5j, 0. + 0.j, 0. + 0.j],
                           [0. - 0.5j, 0.5 + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 0.5 + 0.j, 0. + 0.5j],
                           [0. + 0.j, 0. + 0.j, 0. - 0.5j, 0.5 + 0.j]])),
                (0.0,
                 np.array([[0.5 + 0.j, 0. - 0.5j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.5j, 0.5 + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 0.5 + 0.j, 0. - 0.5j],
                           [0. + 0.j, 0. + 0.j, 0. + 0.5j, 0.5 + 0.j]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.IdentityGate(1).on(a)
        yield cirq.Y().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'IY'
        return 'IY**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.IY'
            return '(cirq.IY**{})'.format(proper_repr(self._exponent))
        return ('cirq.IYPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


class IZGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """A two qubit gate which does nothing to the first qubit and
       rotates the second qubit about the Z axis of its Bloch sphere,

        The unitary  matrix of IZGate(exponent=t) is:

           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, g, 0],
           [0, 0, 0, g]]

        where:

           g = exp(i·π·t/2).
    """
    def _eigen_components(self):
        return [(1.0,
                 np.array([[0., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 0.],
                           [0., 0., 0., 1.]])),
                (0.0,
                 np.array([[1., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 1., 0.],
                           [0., 0., 0., 0.]]))]

    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.IdentityGate(1).on(a)
        yield cirq.Z().on(b)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'IZ'
        return 'IZ**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.IZ'
            return '(cirq.IZ**{})'.format(proper_repr(self._exponent))
        return ('cirq.IZPowGate(exponent={}, '
                'global_shift={!r})').format(proper_repr(self._exponent),
                                             self._global_shift)

    def on(self, *args: raw_types.Qid,
           **kwargs: raw_types.Qid) -> raw_types.Operation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))
