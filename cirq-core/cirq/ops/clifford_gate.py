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

from typing import Any, cast, Dict, NamedTuple, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import numpy as np

from cirq import protocols, value, linalg
from cirq._doc import document
from cirq.ops import common_gates, gate_features, named_qubit, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq

PauliTransform = NamedTuple('PauliTransform', [('to', Pauli), ('flip', bool)])
document(PauliTransform, """+X, -X, +Y, -Y, +Z, or -Z.""")


def _to_pauli_transform(matrix: np.ndarray) -> Optional[PauliTransform]:
    """Converts matrix to PauliTransform.

    If matrix is not ±Pauli matrix, returns None.
    """
    for pauli in Pauli._XYZ:
        p = protocols.unitary(pauli)
        if np.allclose(matrix, p):
            return PauliTransform(pauli, False)
        if np.allclose(matrix, -p):
            return PauliTransform(pauli, True)
    return None


def _pretend_initialized() -> 'SingleQubitCliffordGate':
    # HACK: This is a workaround to fool mypy and pylint into correctly handling
    # class fields that can't be initialized until after the class is defined.
    pass


def _validate_map_input(
    required_transform_count: int,
    pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]],
    x_to: Optional[Tuple[Pauli, bool]],
    y_to: Optional[Tuple[Pauli, bool]],
    z_to: Optional[Tuple[Pauli, bool]],
) -> Dict[Pauli, PauliTransform]:
    if pauli_map_to is None:
        xyz_to = {pauli_gates.X: x_to, pauli_gates.Y: y_to, pauli_gates.Z: z_to}
        pauli_map_to = {cast(Pauli, p): trans for p, trans in xyz_to.items() if trans is not None}
    elif x_to is not None or y_to is not None or z_to is not None:
        raise ValueError(
            '{} can take either pauli_map_to or a combination'
            ' of x_to, y_to, and z_to but both were given'
        )
    if len(pauli_map_to) != required_transform_count:
        raise ValueError(
            'Method takes {} transform{} but {} {} given'.format(
                required_transform_count,
                '' if required_transform_count == 1 else 's',
                len(pauli_map_to),
                'was' if len(pauli_map_to) == 1 else 'were',
            )
        )
    if len(set((to for to, _ in pauli_map_to.values()))) != len(pauli_map_to):
        raise ValueError('A rotation cannot map two Paulis to the same')
    return {frm: PauliTransform(to, flip) for frm, (to, flip) in pauli_map_to.items()}


@value.value_equality
class SingleQubitCliffordGate(gate_features.SingleQubitGate):
    """Any single qubit Clifford rotation."""

    I = _pretend_initialized()
    H = _pretend_initialized()
    X = _pretend_initialized()
    Y = _pretend_initialized()
    Z = _pretend_initialized()
    X_sqrt = _pretend_initialized()
    Y_sqrt = _pretend_initialized()
    Z_sqrt = _pretend_initialized()
    X_nsqrt = _pretend_initialized()
    Y_nsqrt = _pretend_initialized()
    Z_nsqrt = _pretend_initialized()

    def __init__(
        self,
        *,
        _rotation_map: Dict[Pauli, PauliTransform],
        _inverse_map: Dict[Pauli, PauliTransform],
    ) -> None:
        self._rotation_map = _rotation_map
        self._inverse_map = _inverse_map

    @staticmethod
    def from_xz_map(
        x_to: Tuple[Pauli, bool], z_to: Tuple[Pauli, bool]
    ) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the specified transforms.
        The Y transform is derived from the X and Z.

        Args:
            x_to: Which Pauli to transform X to and if it should negate.
            z_to: Which Pauli to transform Z to and if it should negate.
        """
        return SingleQubitCliffordGate.from_double_map(x_to=x_to, z_to=z_to)

    @staticmethod
    def from_single_map(
        pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]] = None,
        *,
        x_to: Optional[Tuple[Pauli, bool]] = None,
        y_to: Optional[Tuple[Pauli, bool]] = None,
        z_to: Optional[Tuple[Pauli, bool]] = None,
    ) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        The arguments are exclusive, only one may be specified.

        Args:
            pauli_map_to: A dictionary with a single key value pair describing
                the transform.
            x_to: The transform from cirq.X
            y_to: The transform from cirq.Y
            z_to: The transform from cirq.Z
        """
        rotation_map = _validate_map_input(1, pauli_map_to, x_to=x_to, y_to=y_to, z_to=z_to)
        ((trans_from, (trans_to, flip)),) = tuple(rotation_map.items())
        if trans_from == trans_to:
            trans_from2 = Pauli.by_relative_index(trans_to, 1)  # 1 or 2 work
            trans_to2 = Pauli.by_relative_index(trans_from, 1)
            flip2 = False
        else:
            trans_from2 = trans_to
            trans_to2 = trans_from
            flip2 = not flip
        rotation_map[trans_from2] = PauliTransform(trans_to2, flip2)
        return SingleQubitCliffordGate.from_double_map(
            cast(Dict[Pauli, Tuple[Pauli, bool]], rotation_map)
        )

    @staticmethod
    def from_double_map(
        pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]] = None,
        *,
        x_to: Optional[Tuple[Pauli, bool]] = None,
        y_to: Optional[Tuple[Pauli, bool]] = None,
        z_to: Optional[Tuple[Pauli, bool]] = None,
    ) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        Either pauli_map_to or two of (x_to, y_to, z_to) may be specified.

        Args:
            pauli_map_to: A dictionary with two key value pairs describing
                two transforms.
            x_to: The transform from cirq.X
            y_to: The transform from cirq.Y
            z_to: The transform from cirq.Z
        """
        rotation_map = _validate_map_input(2, pauli_map_to, x_to=x_to, y_to=y_to, z_to=z_to)
        (from1, trans1), (from2, trans2) = tuple(rotation_map.items())
        from3 = from1.third(from2)
        to3 = trans1.to.third(trans2.to)
        flip3 = trans1.flip ^ trans2.flip ^ ((from1 < from2) != (trans1.to < trans2.to))
        rotation_map[from3] = PauliTransform(to3, flip3)
        inverse_map = {to: PauliTransform(frm, flip) for frm, (to, flip) in rotation_map.items()}
        return SingleQubitCliffordGate(_rotation_map=rotation_map, _inverse_map=inverse_map)

    @staticmethod
    def from_pauli(pauli: Pauli, sqrt: bool = False) -> 'SingleQubitCliffordGate':
        prev_pauli = Pauli.by_relative_index(pauli, -1)
        next_pauli = Pauli.by_relative_index(pauli, 1)
        if sqrt:
            rotation_map = {
                prev_pauli: PauliTransform(next_pauli, True),
                pauli: PauliTransform(pauli, False),
                next_pauli: PauliTransform(prev_pauli, False),
            }
        else:
            rotation_map = {
                prev_pauli: PauliTransform(prev_pauli, True),
                pauli: PauliTransform(pauli, False),
                next_pauli: PauliTransform(next_pauli, True),
            }
        inverse_map = {to: PauliTransform(frm, flip) for frm, (to, flip) in rotation_map.items()}
        return SingleQubitCliffordGate(_rotation_map=rotation_map, _inverse_map=inverse_map)

    @staticmethod
    def from_quarter_turns(pauli: Pauli, quarter_turns: int) -> 'SingleQubitCliffordGate':
        quarter_turns = quarter_turns % 4
        if quarter_turns == 0:
            return SingleQubitCliffordGate.I
        if quarter_turns == 1:
            return SingleQubitCliffordGate.from_pauli(pauli, True)
        if quarter_turns == 2:
            return SingleQubitCliffordGate.from_pauli(pauli)

        return SingleQubitCliffordGate.from_pauli(pauli, True) ** -1

    @staticmethod
    def from_unitary(u: np.ndarray) -> Optional['SingleQubitCliffordGate']:
        """Creates Clifford gate with given unitary (up to global phase).

        Args:
            u: 2x2 unitary matrix of a Clifford gate.

        Returns:
            SingleQubitCliffordGate, whose matrix is equal to given matrix (up
            to global phase), or `None` if `u` is not a matrix of a single-qubit
            Clifford gate.
        """
        if u.shape != (2, 2) or not linalg.is_unitary(u):
            return None
        x = protocols.unitary(pauli_gates.X)
        z = protocols.unitary(pauli_gates.Z)
        x_to = _to_pauli_transform(u @ x @ u.conj().T)
        z_to = _to_pauli_transform(u @ z @ u.conj().T)
        if x_to is None or z_to is None:
            return None
        return SingleQubitCliffordGate.from_double_map({pauli_gates.X: x_to, pauli_gates.Z: z_to})

    def transform(self, pauli: Pauli) -> PauliTransform:
        return self._rotation_map[pauli]

    def to_phased_xz_gate(self) -> phased_x_z_gate.PhasedXZGate:
        """Convert this gate to a PhasedXZGate instance.

        The rotation can be categorized by {axis} * {degree}:
            * Identity: I
            * {x, y, z} * {90, 180, 270}  --- {X, Y, Z} + 6 Quarter turn gates
            * {+/-xy, +/-yz, +/-zx} * 180  --- 6 Hadamard-like gates
            * {middle point of xyz in 4 Quadrant} * {120, 240} --- swapping axis
        note 1 + 9 + 6 + 8 = 24 in total.

        To associate with Clifford Tableau, it can also be grouped by 4:
            * {I,X,Y,Z} is [[1 0], [0, 1]]
            * {+/- X_sqrt, 2 Hadamard-like gates acting on the YZ plane} is [[1, 0], [1, 1]]
            * {+/- Z_sqrt, 2 Hadamard-like gates acting on the XY plane} is [[1, 1], [0, 1]]
            * {+/- Y_sqrt, 2 Hadamard-like gates acting on the XZ plane} is [[0, 1], [1, 0]]
            * {middle point of xyz in 4 Quadrant} * 120 is [[0, 1], [1, 1]]
            * {middle point of xyz in 4 Quadrant} * 240 is [[1, 1], [1, 0]]
        """
        x_to = self._rotation_map[pauli_gates.X]
        z_to = self._rotation_map[pauli_gates.Z]
        flip_index = int(z_to.flip) * 2 + int(x_to.flip)
        a, x, z = 0.0, 0.0, 0.0

        if (x_to.to, z_to.to) == (pauli_gates.X, pauli_gates.Z):
            # I, Z, X, Y cases
            to_phased_xz = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.5, 1.0, 0.0)]
            a, x, z = to_phased_xz[flip_index]
        elif (x_to.to, z_to.to) == (pauli_gates.X, pauli_gates.Y):
            # +/- X_sqrt, 2 Hadamard-like gates acting on the YZ plane
            a = 0.0
            x = 0.5 if x_to.flip ^ z_to.flip else -0.5
            z = 1.0 if x_to.flip else 0.0
        elif (x_to.to, z_to.to) == (pauli_gates.Z, pauli_gates.X):
            # +/- Y_sqrt, 2 Hadamard-like gates acting on the XZ plane
            a = 0.5
            x = 0.5 if x_to.flip else -0.5
            z = 0.0 if x_to.flip ^ z_to.flip else 1.0
        elif (x_to.to, z_to.to) == (pauli_gates.Y, pauli_gates.Z):
            # +/- Z_sqrt, 2 Hadamard-like gates acting on the XY plane
            to_phased_xz = [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.25, 1.0, 0.0), (-0.25, 1.0, 0.0)]
            a, x, z = to_phased_xz[flip_index]
        elif (x_to.to, z_to.to) == (pauli_gates.Z, pauli_gates.Y):
            # axis swapping rotation -- (312) permutation
            a = 0.5
            x = 0.5 if x_to.flip else -0.5
            z = 0.5 if x_to.flip ^ z_to.flip else -0.5
        else:
            # axis swapping rotation -- (231) permutation.
            # This should be the only cases left.
            assert (x_to.to, z_to.to) == (pauli_gates.Y, pauli_gates.X)
            a = 0.0
            x = -0.5 if x_to.flip ^ z_to.flip else 0.5
            z = -0.5 if x_to.flip else 0.5

        return phased_x_z_gate.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)

    def _value_equality_values_(self):
        return (
            self.transform(pauli_gates.X),
            self.transform(pauli_gates.Y),
            self.transform(pauli_gates.Z),
        )

    def __pow__(self, exponent) -> 'SingleQubitCliffordGate':
        if exponent == 0.5 or exponent == -0.5:
            return SQRT_EXP_MAP[exponent][self]
        if exponent != -1:
            return NotImplemented

        return SingleQubitCliffordGate(
            _rotation_map=self._inverse_map, _inverse_map=self._rotation_map
        )

    def _commutes_(self, other: Any, atol: float) -> Union[bool, NotImplementedType]:
        if isinstance(other, SingleQubitCliffordGate):
            return self.commutes_with_single_qubit_gate(other)
        if isinstance(other, Pauli):
            return self.commutes_with_pauli(other)
        return NotImplemented

    def commutes_with_single_qubit_gate(self, gate: 'SingleQubitCliffordGate') -> bool:
        """Tests if the two circuits would be equivalent up to global phase:
        --self--gate-- and --gate--self--"""
        for pauli0 in (pauli_gates.X, pauli_gates.Z):
            pauli1, flip1 = self.transform(cast(Pauli, pauli0))
            pauli2, flip2 = gate.transform(cast(Pauli, pauli1))
            pauli3, flip3 = self._inverse_map[pauli2]
            pauli4, flip4 = gate._inverse_map[pauli3]
            if pauli4 != pauli0 or (flip1 ^ flip2 ^ flip3 ^ flip4):
                return False
        return True

    def commutes_with_pauli(self, pauli: Pauli) -> bool:
        to, flip = self.transform(pauli)
        return to == pauli and not flip

    def merged_with(self, second: 'SingleQubitCliffordGate') -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate such that the circuits
            --output-- and --self--second--
        are equivalent up to global phase."""
        x_intermediate_pauli, x_flip1 = self.transform(pauli_gates.X)
        x_final_pauli, x_flip2 = second.transform(x_intermediate_pauli)
        z_intermediate_pauli, z_flip1 = self.transform(pauli_gates.Z)
        z_final_pauli, z_flip2 = second.transform(z_intermediate_pauli)
        return SingleQubitCliffordGate.from_xz_map(
            (x_final_pauli, x_flip1 ^ x_flip2), (z_final_pauli, z_flip1 ^ z_flip2)
        )

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        mat = np.eye(2)
        qubit = named_qubit.NamedQubit('arbitrary')
        for op in protocols.decompose_once_with_qubits(self, (qubit,)):
            mat = protocols.unitary(op).dot(mat)
        return mat

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        (qubit,) = qubits
        if self == SingleQubitCliffordGate.H:
            return (common_gates.H(qubit),)
        rotations = self.decompose_rotation()
        return tuple(r.on(qubit) ** (qt / 2) for r, qt in rotations)

    def decompose_rotation(self) -> Sequence[Tuple[Pauli, int]]:
        """Returns ((first_rotation_axis, first_rotation_quarter_turns), ...)

        This is a sequence of zero, one, or two rotations."""
        x_rot = self.transform(pauli_gates.X)
        y_rot = self.transform(pauli_gates.Y)
        z_rot = self.transform(pauli_gates.Z)
        whole_arr = (
            x_rot.to == pauli_gates.X,
            y_rot.to == pauli_gates.Y,
            z_rot.to == pauli_gates.Z,
        )
        num_whole = sum(whole_arr)
        flip_arr = (x_rot.flip, y_rot.flip, z_rot.flip)
        num_flip = sum(flip_arr)
        if num_whole == 3:
            if num_flip == 0:
                # Gate is identity
                return []

            # 180 rotation about some axis
            pauli = Pauli.by_index(flip_arr.index(False))
            return [(pauli, 2)]
        if num_whole == 1:
            index = whole_arr.index(True)
            pauli = Pauli.by_index(index)
            next_pauli = Pauli.by_index(index + 1)
            flip = flip_arr[index]
            output = []
            if flip:
                # 180 degree rotation
                output.append((next_pauli, 2))
            # 90 degree rotation about some axis
            if self.transform(next_pauli).flip:
                # Negative 90 degree rotation
                output.append((pauli, -1))
            else:
                # Positive 90 degree rotation
                output.append((pauli, 1))
            return output
        elif num_whole == 0:
            # Gate is a 120 degree rotation
            if x_rot.to == pauli_gates.Y:
                return [
                    (pauli_gates.X, -1 if y_rot.flip else 1),
                    (pauli_gates.Z, -1 if x_rot.flip else 1),
                ]

            return [
                (pauli_gates.Z, 1 if y_rot.flip else -1),
                (pauli_gates.X, 1 if z_rot.flip else -1),
            ]
        # coverage: ignore
        assert (
            False
        ), 'Impossible condition where this gate only rotates one Pauli to a different Pauli.'

    def equivalent_gate_before(self, after: 'SingleQubitCliffordGate') -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate such that the circuits
            --output--self-- and --self--gate--
        are equivalent up to global phase."""
        return self.merged_with(after).merged_with(self ** -1)

    def __repr__(self) -> str:
        x = self.transform(pauli_gates.X)
        y = self.transform(pauli_gates.Y)
        z = self.transform(pauli_gates.Z)
        x_sign = '-' if x.flip else '+'
        y_sign = '-' if y.flip else '+'
        z_sign = '-' if z.flip else '+'
        return (
            f'cirq.SingleQubitCliffordGate(X:{x_sign}{x.to!s}, '
            f'Y:{y_sign}{y.to!s}, Z:{z_sign}{z.to!s})'
        )

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        well_known_map = {
            SingleQubitCliffordGate.I: 'I',
            SingleQubitCliffordGate.H: 'H',
            SingleQubitCliffordGate.X: 'X',
            SingleQubitCliffordGate.Y: 'Y',
            SingleQubitCliffordGate.Z: 'Z',
            SingleQubitCliffordGate.X_sqrt: 'X',
            SingleQubitCliffordGate.Y_sqrt: 'Y',
            SingleQubitCliffordGate.Z_sqrt: 'Z',
            SingleQubitCliffordGate.X_nsqrt: 'X',
            SingleQubitCliffordGate.Y_nsqrt: 'Y',
            SingleQubitCliffordGate.Z_nsqrt: 'Z',
        }
        if self in well_known_map:
            symbol = well_known_map[self]
        else:
            rotations = self.decompose_rotation()
            symbol = '-'.join(str(r) + ('^' + str(qt / 2)) * (qt % 4 != 2) for r, qt in rotations)
            symbol = f'({symbol})'
        return protocols.CircuitDiagramInfo(
            wire_symbols=(symbol,),
            exponent={
                SingleQubitCliffordGate.X_sqrt: 0.5,
                SingleQubitCliffordGate.Y_sqrt: 0.5,
                SingleQubitCliffordGate.Z_sqrt: 0.5,
                SingleQubitCliffordGate.X_nsqrt: -0.5,
                SingleQubitCliffordGate.Y_nsqrt: -0.5,
                SingleQubitCliffordGate.Z_nsqrt: -0.5,
            }.get(self, 1),
        )


SingleQubitCliffordGate.I = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.X, False), (pauli_gates.Z, False)
)
SingleQubitCliffordGate.H = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.Z, False), (pauli_gates.X, False)
)
SingleQubitCliffordGate.X = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.X, False), (pauli_gates.Z, True)
)
SingleQubitCliffordGate.Y = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.X, True), (pauli_gates.Z, True)
)
SingleQubitCliffordGate.Z = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.X, True), (pauli_gates.Z, False)
)
SingleQubitCliffordGate.X_sqrt = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.X, False), (pauli_gates.Y, True)
)
SingleQubitCliffordGate.X_nsqrt = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.X, False), (pauli_gates.Y, False)
)
SingleQubitCliffordGate.Y_sqrt = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.Z, True), (pauli_gates.X, False)
)
SingleQubitCliffordGate.Y_nsqrt = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.Z, False), (pauli_gates.X, True)
)
SingleQubitCliffordGate.Z_sqrt = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.Y, False), (pauli_gates.Z, False)
)
SingleQubitCliffordGate.Z_nsqrt = SingleQubitCliffordGate.from_xz_map(
    (pauli_gates.Y, True), (pauli_gates.Z, False)
)

SQRT_EXP_MAP = {
    0.5: {
        SingleQubitCliffordGate.X: SingleQubitCliffordGate.X_sqrt,
        SingleQubitCliffordGate.Y: SingleQubitCliffordGate.Y_sqrt,
        SingleQubitCliffordGate.Z: SingleQubitCliffordGate.Z_sqrt,
    },
    -0.5: {
        SingleQubitCliffordGate.X: SingleQubitCliffordGate.X_nsqrt,
        SingleQubitCliffordGate.Y: SingleQubitCliffordGate.Y_nsqrt,
        SingleQubitCliffordGate.Z: SingleQubitCliffordGate.Z_nsqrt,
    },
}
