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

from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union, cast

import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types, common_gates, op_tree, \
    named_qubit
from cirq.ops.pauli import Pauli


PauliTransform = NamedTuple('PauliTransform', [('to', Pauli), ('flip', bool)])


def _pretend_initialized() -> 'SingleQubitCliffordGate':
    # HACK: This is a workaround to fool mypy and pylint into correctly handling
    # class fields that can't be initialized until after the class is defined.
    pass


@value.value_equality
class SingleQubitCliffordGate(raw_types.Gate):
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

    def __init__(self, *,
                 _rotation_map: Dict[Pauli, PauliTransform],
                 _inverse_map: Dict[Pauli, PauliTransform]) -> None:
        self._rotation_map = _rotation_map
        self._inverse_map = _inverse_map

    @staticmethod
    def from_xz_map(x_to: Tuple[Pauli, bool],
                    z_to: Tuple[Pauli, bool]) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the specified transforms.
        The Y transform is derived from the X and Z.

        Args:
            x_to: Which Pauli to transform X to and if it should negate.
            z_to: Which Pauli to transform Z to and if it should negate.
        """
        return SingleQubitCliffordGate.from_double_map(x_to=x_to, z_to=z_to)

    @staticmethod
    def from_single_map(pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]]
                                      = None,
                        *,
                        x_to: Optional[Tuple[Pauli, bool]] = None,
                        y_to: Optional[Tuple[Pauli, bool]] = None,
                        z_to: Optional[Tuple[Pauli, bool]] = None
                        ) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        The arguments are exclusive, only one may be specified.

        Args:
            pauli_map_to: A dictionary with a single key value pair describing
                the transform.
            x_to: The transform from Pauli.X
            y_to: The transform from Pauli.Y
            z_to: The transform from Pauli.Z
        """
        rotation_map = SingleQubitCliffordGate._validate_map_input(
                                        1,
                                        pauli_map_to,
                                        x_to=x_to, y_to=y_to, z_to=z_to)
        (trans_from, (trans_to, flip)), = tuple(rotation_map.items())
        if trans_from == trans_to:
            trans_from2 = trans_to + 1  # Either +1 or +2 work
            trans_to2 = trans_from + 1
            flip2 = False
        else:
            trans_from2 = trans_to
            trans_to2 = trans_from
            flip2 = not flip
        rotation_map[trans_from2] = PauliTransform(trans_to2, flip2)
        return SingleQubitCliffordGate.from_double_map(
                        cast(Dict[Pauli, Tuple[Pauli, bool]], rotation_map))

    @staticmethod
    def from_double_map(pauli_map_to: Optional[Dict[Pauli, Tuple[Pauli, bool]]]
                                      = None,
                        *,
                        x_to: Optional[Tuple[Pauli, bool]] = None,
                        y_to: Optional[Tuple[Pauli, bool]] = None,
                        z_to: Optional[Tuple[Pauli, bool]] = None
                        ) -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate for the
        specified transform with a 90 or 180 degree rotation.

        Either pauli_map_to or two of (x_to, y_to, z_to) may be specified.

        Args:
            pauli_map_to: A dictionary with two key value pairs describing
                two transforms.
            x_to: The transform from Pauli.X
            y_to: The transform from Pauli.Y
            z_to: The transform from Pauli.Z
        """
        rotation_map = SingleQubitCliffordGate._validate_map_input(
                                        2,
                                        pauli_map_to,
                                        x_to=x_to, y_to=y_to, z_to=z_to)
        (from1, trans1), (from2, trans2) = tuple(rotation_map.items())
        from3 = from1.third(from2)
        to3 = trans1.to.third(trans2.to)
        flip3 = (trans1.flip ^ trans2.flip
                 ^ ((from1 < from2) != (trans1.to < trans2.to)))
        rotation_map[from3] = PauliTransform(to3, flip3)
        inverse_map = {to: PauliTransform(frm, flip)
                       for frm, (to, flip) in rotation_map.items()}
        return SingleQubitCliffordGate(_rotation_map=rotation_map,
                            _inverse_map=inverse_map)

    @staticmethod
    def from_pauli(pauli: Pauli,
                   sqrt: bool = False) -> 'SingleQubitCliffordGate':
        if sqrt:
            rotation_map = {pauli:   PauliTransform(pauli, False),
                            pauli+1: PauliTransform(pauli+2, False),
                            pauli+2: PauliTransform(pauli+1, True)}
        else:
            rotation_map = {pauli:   PauliTransform(pauli, False),
                            pauli+1: PauliTransform(pauli+1, True),
                            pauli+2: PauliTransform(pauli+2, True)}
        inverse_map = {to: PauliTransform(frm, flip)
                       for frm, (to, flip) in rotation_map.items()}
        return SingleQubitCliffordGate(_rotation_map=rotation_map,
                            _inverse_map=inverse_map)

    @staticmethod
    def from_quarter_turns(pauli: Pauli,
                           quarter_turns: int) -> 'SingleQubitCliffordGate':
        quarter_turns = quarter_turns % 4
        if quarter_turns == 0:
            return SingleQubitCliffordGate.I
        elif quarter_turns == 1:
            return SingleQubitCliffordGate.from_pauli(pauli, True)
        elif quarter_turns == 2:
            return SingleQubitCliffordGate.from_pauli(pauli)
        else:
            return SingleQubitCliffordGate.from_pauli(pauli, True)**-1

    @staticmethod
    def _validate_map_input(required_transform_count: int,
                            pauli_map_to: Optional[Dict[Pauli,
                                                        Tuple[Pauli, bool]]],
                            x_to: Optional[Tuple[Pauli, bool]],
                            y_to: Optional[Tuple[Pauli, bool]],
                            z_to: Optional[Tuple[Pauli, bool]]
                            ) -> Dict[Pauli, PauliTransform]:
        if pauli_map_to is None:
            pauli_map_to = {p: trans
                            for p, trans in zip(Pauli.XYZ, (x_to, y_to, z_to))
                            if trans is not None}
        elif x_to is not None or y_to is not None or z_to is not None:
            raise ValueError('{} can take either pauli_map_to or a combination'
                             ' of x_to, y_to, and z_to but both were given')
        if len(pauli_map_to) != required_transform_count:
            raise ValueError('Method takes {} transform{} but {} {} given'
                             .format(
                                required_transform_count,
                                '' if required_transform_count == 1 else 's',
                                len(pauli_map_to),
                                'was' if len(pauli_map_to) == 1 else 'were'))
        if (len(set((to for to, _ in pauli_map_to.values())))
            != len(pauli_map_to)):
            raise ValueError('A rotation cannot map two Paulis to the same')
        return {frm: PauliTransform(to, flip)
                for frm, (to, flip) in pauli_map_to.items()}

    def transform(self, pauli: Pauli) -> PauliTransform:
        return self._rotation_map[pauli]

    def _value_equality_values_(self):
        return (self.transform(Pauli.X),
                self.transform(Pauli.Y),
                self.transform(Pauli.Z))

    def __pow__(self, exponent) -> 'SingleQubitCliffordGate':
        if exponent != -1:
            return NotImplemented
        return SingleQubitCliffordGate(_rotation_map=self._inverse_map,
                                       _inverse_map=self._rotation_map)

    def commutes_with(self,
                      gate_or_pauli: Union['SingleQubitCliffordGate', Pauli]
                      ) -> bool:
        if isinstance(gate_or_pauli, SingleQubitCliffordGate):
            gate = gate_or_pauli
            return self.commutes_with_single_qubit_gate(gate)
        else:
            pauli = gate_or_pauli
            return self.commutes_with_pauli(pauli)

    def commutes_with_single_qubit_gate(self,
                                        gate: 'SingleQubitCliffordGate') \
                                        -> bool:
        """Tests if the two circuits would be equivalent up to global phase:
            --self--gate-- and --gate--self--"""
        for pauli0 in (Pauli.X, Pauli.Z):
            pauli1, flip1 = self.transform(pauli0)
            pauli2, flip2 = gate.transform(pauli1)
            pauli3, flip3 = self._inverse_map[pauli2]
            pauli4, flip4 = gate._inverse_map[pauli3]
            if pauli4 != pauli0 or (flip1 ^ flip2 ^ flip3 ^ flip4):
                return False
        return True

    def commutes_with_pauli(self, pauli: Pauli) -> bool:
        to, flip = self.transform(pauli)
        return (to == pauli and not flip)

    def merged_with(self,
                    second: 'SingleQubitCliffordGate') \
                    -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate such that the circuits
            --output-- and --self--second--
        are equivalent up to global phase."""
        x_intermediate_pauli, x_flip1 = self.transform(Pauli.X)
        x_final_pauli, x_flip2 = second.transform(x_intermediate_pauli)
        z_intermediate_pauli, z_flip1 = self.transform(Pauli.Z)
        z_final_pauli, z_flip2 = second.transform(z_intermediate_pauli)
        return SingleQubitCliffordGate.from_xz_map(
            (x_final_pauli, x_flip1 ^ x_flip2),
            (z_final_pauli, z_flip1 ^ z_flip2))

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        mat = np.eye(2)
        qubit = named_qubit.NamedQubit('arbitrary')
        for op in protocols.decompose_once_with_qubits(self, (qubit,)):
            mat = protocols.unitary(op).dot(mat)
        return mat

    def _decompose_(self, qubits: Sequence[raw_types.QubitId]
                          ) -> op_tree.OP_TREE:
        qubit, = qubits
        if self == SingleQubitCliffordGate.H:
            return common_gates.H(qubit),
        rotations = self.decompose_rotation()
        pauli_gate_map = {Pauli.X: common_gates.X,
                          Pauli.Y: common_gates.Y,
                          Pauli.Z: common_gates.Z}
        return tuple((pauli_gate_map[r](qubit) ** (qt / 2)
                      for r, qt in rotations))

    def decompose_rotation(self) -> Sequence[Tuple[Pauli, int]]:
        """Returns ((first_rotation_axis, first_rotation_quarter_turns), ...)

        This is a sequence of zero, one, or two rotations."""
        x_rot = self.transform(Pauli.X)
        y_rot = self.transform(Pauli.Y)
        z_rot = self.transform(Pauli.Z)
        whole_arr = (x_rot.to == Pauli.X,
                     y_rot.to == Pauli.Y,
                     z_rot.to == Pauli.Z)
        num_whole = sum(whole_arr)
        flip_arr = (x_rot.flip,
                    y_rot.flip,
                    z_rot.flip)
        num_flip = sum(flip_arr)
        if num_whole == 3:
            if num_flip == 0:
                # Gate is identity
                return []
            else:
                # 180 rotation about some axis
                pauli = Pauli.XYZ[flip_arr.index(False)]
                return [(pauli, 2)]
        elif num_whole == 1:
            index = whole_arr.index(True)
            pauli = Pauli.XYZ[index]
            flip = flip_arr[index]
            output = []
            if flip:
                # 180 degree rotation
                output.append((pauli + 1, 2))
            # 90 degree rotation about some axis
            if self.transform(pauli + 1).flip:
                # Negative 90 degree rotation
                output.append((pauli, -1))
            else:
                # Positive 90 degree rotation
                output.append((pauli, 1))
            return output
        elif num_whole == 0:
            # Gate is a 120 degree rotation
            if x_rot.to == Pauli.Y:
                return [(Pauli.X, -1 if y_rot.flip else 1),
                        (Pauli.Z, -1 if x_rot.flip else 1)]
            else:
                return [(Pauli.Z, 1 if y_rot.flip else -1),
                        (Pauli.X, 1 if z_rot.flip else -1)]
        # coverage: ignore
        assert False, ('Impossible condition where this gate only rotates one'
                       ' Pauli to a different Pauli.')

    def equivalent_gate_before(self, after: 'SingleQubitCliffordGate') \
        -> 'SingleQubitCliffordGate':
        """Returns a SingleQubitCliffordGate such that the circuits
            --output--self-- and --self--gate--
        are equivalent up to global phase."""
        return self.merged_with(after).merged_with(self**-1)

    def __repr__(self):
        return 'cirq.SingleQubitCliffordGate(X:{}{!s}, Y:{}{!s}, Z:{}{!s})' \
            .format(
                '+-'[self.transform(Pauli.X).flip], self.transform(Pauli.X).to,
                '+-'[self.transform(Pauli.Y).flip], self.transform(Pauli.Y).to,
                '+-'[self.transform(Pauli.Z).flip], self.transform(Pauli.Z).to)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
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
            symbol = '-'.join(
                str(r) + ('^' + str(qt / 2)) * (qt % 4 != 2)
                for r, qt in rotations)
            symbol = '({})'.format(symbol)
        return protocols.CircuitDiagramInfo(
            wire_symbols=(symbol,),
            exponent={
                SingleQubitCliffordGate.X_sqrt: 0.5,
                SingleQubitCliffordGate.Y_sqrt: 0.5,
                SingleQubitCliffordGate.Z_sqrt: 0.5,
                SingleQubitCliffordGate.X_nsqrt: -0.5,
                SingleQubitCliffordGate.Y_nsqrt: -0.5,
                SingleQubitCliffordGate.Z_nsqrt: -0.5,
            }.get(self, 1))


SingleQubitCliffordGate.I = SingleQubitCliffordGate.from_xz_map(
    (Pauli.X, False), (Pauli.Z, False))
SingleQubitCliffordGate.H = SingleQubitCliffordGate.from_xz_map(
    (Pauli.Z, False), (Pauli.X, False))
SingleQubitCliffordGate.X = SingleQubitCliffordGate.from_xz_map(
    (Pauli.X, False), (Pauli.Z, True))
SingleQubitCliffordGate.Y = SingleQubitCliffordGate.from_xz_map(
    (Pauli.X, True), (Pauli.Z, True))
SingleQubitCliffordGate.Z = SingleQubitCliffordGate.from_xz_map(
    (Pauli.X, True), (Pauli.Z, False))
SingleQubitCliffordGate.X_sqrt  = SingleQubitCliffordGate.from_xz_map(
    (Pauli.X, False), (Pauli.Y, True))
SingleQubitCliffordGate.X_nsqrt = SingleQubitCliffordGate.from_xz_map(
    (Pauli.X, False), (Pauli.Y, False))
SingleQubitCliffordGate.Y_sqrt  = SingleQubitCliffordGate.from_xz_map(
    (Pauli.Z, True), (Pauli.X, False))
SingleQubitCliffordGate.Y_nsqrt = SingleQubitCliffordGate.from_xz_map(
    (Pauli.Z, False), (Pauli.X, True))
SingleQubitCliffordGate.Z_sqrt  = SingleQubitCliffordGate.from_xz_map(
    (Pauli.Y, False), (Pauli.Z, False))
SingleQubitCliffordGate.Z_nsqrt = SingleQubitCliffordGate.from_xz_map(
    (Pauli.Y, True), (Pauli.Z, False))
