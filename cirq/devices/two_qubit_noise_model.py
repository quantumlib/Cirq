import cmath
from typing import Sequence, Tuple, Union, List

import numpy as np

import cirq
from cirq import value, protocols

from cirq._compat import proper_repr
from cirq.ops import gate_features, eigen_gate, raw_types


class TwoQubitNoiseModel(cirq.NoiseModel):
    """
    The two qubit noise channel applies a different noise channel to the circuit
    dependent on whether the gate applied is applied to one or two qubits.
    Can be used to apply the same symmetric depolarising channel with different depolarising probabilities,
    or a different type of channel altogether, dependent on the gate applied.
    """
    def __init__(self, single_qubit_noise_gate: cirq.Gate,
                 two_qubit_noise_gate: cirq.Gate):
        if single_qubit_noise_gate.num_qubits() != 1:
            raise ValueError('The noise gate provided to single_qubit_noise_gate has number of qubits != 1.')
        if two_qubit_noise_gate.num_qubits() != 2:
            raise ValueError('The noise gate provided to two_qubit_noise_gate has number of qubits != 2.')
        self.single_qubit_noise_gate = single_qubit_noise_gate
        self.two_qubit_noise_gate = two_qubit_noise_gate

    def noisy_operation(self, operation: cirq.Operation
                       ) -> Tuple[cirq.Operation, Union[List[cirq.GateOperation], cirq.GateOperation]]:
        """
        Checks if the gate in the operation is a one- or two- qubit gate,
        and applies self.single_qubit_noise_gate or self.two_qubit_noise_gate appropriately.
        If the operation has > 2 qubits, applies the single qubit noise gate to all.
        :param operation: The operation to apply noise to
        :return: The supplied operation and the noise gate(s) to apply.
        """
        n_qubits = len(operation.qubits)
        if n_qubits == 1:
            return operation, self.single_qubit_noise_gate(operation.qubits[0])
        elif n_qubits == 2:
            return operation, self.two_qubit_noise_gate(operation.qubits[0],
                                                        operation.qubits[1])
        else:
            return operation, [
                self.single_qubit_noise_gate(q) for q in operation.qubits
            ]


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
    """
    return TwoQubitAsymmetricDepolarizingChannel(p_xi, p_yi, p_zi, p_xx, p_yx,
                                                 p_zx, p_xy, p_yy, p_zy, p_xz,
                                                 p_yz, p_zz, p_ix, p_iy, p_iz)


class XIGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):

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


def construct_eigen_space(mat):
    eig_vals, eig_vecs = np.linalg.eig(mat)
    degenerative = np.unique(eig_vals)
    tuples = []
    for val in degenerative:
        index = np.where(eig_vals == val)[0]
        proj = np.zeros_like(mat)
        for i in index:
            proj = np.add(
                proj,
                np.einsum('i,j->ij', eig_vecs[:, i], np.conj(eig_vecs[:, i])))
        phase = cmath.phase(complex(val)) / np.pi
        tuples.append((phase, proj))
    return tuples
