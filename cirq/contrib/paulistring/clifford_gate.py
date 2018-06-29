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

from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple

from cirq import ops

from cirq.contrib.paulistring.pauli import Pauli


PauliTransform = NamedTuple('PauliTransform', [('to', Pauli), ('flip', bool)])


class CliffordGate(ops.CompositeGate,
                   ops.ReversibleGate,
                   ops.TextDiagrammableGate):
    """Any single qubit Clifford rotation."""
    I = None  # type: CliffordGate
    H = None  # type: CliffordGate
    X = None  # type: CliffordGate
    Y = None  # type: CliffordGate
    Z = None  # type: CliffordGate
    X_sqrt  = None  # type: CliffordGate
    X_nsqrt = None  # type: CliffordGate
    Y_sqrt  = None  # type: CliffordGate
    Y_nsqrt = None  # type: CliffordGate
    Z_sqrt  = None  # type: CliffordGate
    Z_nsqrt = None  # type: CliffordGate

    def __init__(self, *,
                 _rotation_map: Dict[Pauli, PauliTransform],
                 _inverse_map: Dict[Pauli, PauliTransform]) -> None:
        self._rotation_map = _rotation_map
        self._inverse_map = _inverse_map

    @staticmethod
    def from_xz_map(transforms_x_to: Tuple[Pauli, bool],
                      transforms_z_to: Tuple[Pauli, bool]) -> 'CliffordGate':
        """Returns a CliffordGate that transforms X to the first arg, Z to the
        second."""
        rotates_x_to, flips_x = transforms_x_to
        rotates_z_to, flips_z = transforms_z_to
        if rotates_x_to == rotates_z_to:
            raise ValueError('A rotation cannot map both X and Z to {!s}.'
                             .format(rotates_x_to))
        rotates_y_to = rotates_x_to.third(rotates_z_to)
        flips_y = flips_x ^ flips_z ^ (rotates_x_to < rotates_z_to)
        rotation_map = {Pauli.X: PauliTransform(rotates_x_to, flips_x),
                        Pauli.Y: PauliTransform(rotates_y_to, flips_y),
                        Pauli.Z: PauliTransform(rotates_z_to, flips_z)}
        inverse_map = {rotates_x_to: PauliTransform(Pauli.X, flips_x),
                       rotates_y_to: PauliTransform(Pauli.Y, flips_y),
                       rotates_z_to: PauliTransform(Pauli.Z, flips_z)}
        return CliffordGate(_rotation_map=rotation_map,
                            _inverse_map=inverse_map)

    def transform(self, pauli: Pauli) -> PauliTransform:
        return self._rotation_map[pauli]

    def _eq_tuple(self) -> Tuple[Any, ...]:
        return (CliffordGate,
                self.transform(Pauli.X),
                self.transform(Pauli.Y),
                self.transform(Pauli.Z))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._eq_tuple() == other._eq_tuple()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._eq_tuple())

    def inverse(self) -> 'CliffordGate':
        return CliffordGate(_rotation_map=self._inverse_map,
                            _inverse_map=self._rotation_map)

    def commutes_with_single_qubit_gate(self, gate: 'CliffordGate') -> bool:
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

    def merged_with(self, second: 'CliffordGate') -> 'CliffordGate':
        """Returns a CliffordGate such that the circuits
            --output-- and --self--second--
        are equivalent up to global phase."""
        x_intermediate_pauli, x_flip1 = self.transform(Pauli.X)
        x_final_pauli, x_flip2 = second.transform(x_intermediate_pauli)
        z_intermediate_pauli, z_flip1 = self.transform(Pauli.Z)
        z_final_pauli, z_flip2 = second.transform(z_intermediate_pauli)
        return CliffordGate.from_xz_map((x_final_pauli, x_flip1 ^ x_flip2),
                                          (z_final_pauli, z_flip1 ^ z_flip2))

    def default_decompose(self, qubits: Sequence[ops.QubitId]) -> ops.OP_TREE:
        qubit, = qubits
        if self == CliffordGate.H:
            return ops.H(qubit),
        rotations = self.decompose_rotation()
        pauli_gate_map = {Pauli.X: ops.X,
                          Pauli.Y: ops.Y,
                          Pauli.Z: ops.Z}
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

    def equivalent_gate_before(self, after: 'CliffordGate') -> 'CliffordGate':
        """Returns a CliffordGate such that the circuits
            --output--self-- and --self--gate--
        are equivalent up to global phase."""
        return self.merged_with(after).merged_with(self.inverse())

    def __repr__(self):
        return 'CliffordGate(X:{}{!s}, Y:{}{!s}, Z:{}{!s})'.format(
                '+-'[self.transform(Pauli.X).flip], self.transform(Pauli.X).to,
                '+-'[self.transform(Pauli.Y).flip], self.transform(Pauli.Y).to,
                '+-'[self.transform(Pauli.Z).flip], self.transform(Pauli.Z).to)

    def text_diagram_wire_symbols(self,
                                  qubit_count: Optional[int] = None,
                                  use_unicode_characters: bool = True,
                                  precision: Optional[int] = 3
                                  ) -> Tuple[str, ...]:
        well_known_map = {
            CliffordGate.I: 'I',
            CliffordGate.H: 'H',
            CliffordGate.X: 'X',
            CliffordGate.Y: 'Y',
            CliffordGate.Z: 'Z',
            CliffordGate.X_sqrt: 'X',
            CliffordGate.Y_sqrt: 'Y',
            CliffordGate.Z_sqrt: 'Z',
            CliffordGate.X_nsqrt: 'X',
            CliffordGate.Y_nsqrt: 'Y',
            CliffordGate.Z_nsqrt: 'Z',
        }
        if self in well_known_map:
            return (well_known_map[self],)
        else:
            rotations = self.decompose_rotation()
            return ('-'.join((
                        str(r) + ('^' + str(qt / 2)) * (qt % 4 != 2)
                        for r, qt in rotations)),)

    def text_diagram_exponent(self) -> float:
        return {CliffordGate.X_sqrt: 0.5,
                CliffordGate.Y_sqrt: 0.5,
                CliffordGate.Z_sqrt: 0.5,
                CliffordGate.X_nsqrt: -0.5,
                CliffordGate.Y_nsqrt: -0.5,
                CliffordGate.Z_nsqrt: -0.5,
               }.get(self, 1)


CliffordGate.I = CliffordGate.from_xz_map((Pauli.X, False), (Pauli.Z, False))
CliffordGate.H = CliffordGate.from_xz_map((Pauli.Z, False), (Pauli.X, False))
CliffordGate.X = CliffordGate.from_xz_map((Pauli.X, False), (Pauli.Z, True))
CliffordGate.Y = CliffordGate.from_xz_map((Pauli.X, True),  (Pauli.Z, True))
CliffordGate.Z = CliffordGate.from_xz_map((Pauli.X, True),  (Pauli.Z, False))
CliffordGate.X_sqrt  = CliffordGate.from_xz_map((Pauli.X, False),
                                                  (Pauli.Y, True))
CliffordGate.X_nsqrt = CliffordGate.from_xz_map((Pauli.X, False),
                                                  (Pauli.Y, False))
CliffordGate.Y_sqrt  = CliffordGate.from_xz_map((Pauli.Z, True),
                                                  (Pauli.X, False))
CliffordGate.Y_nsqrt = CliffordGate.from_xz_map((Pauli.Z, False),
                                                  (Pauli.X, True))
CliffordGate.Z_sqrt  = CliffordGate.from_xz_map((Pauli.Y, False),
                                                  (Pauli.Z, False))
CliffordGate.Z_nsqrt = CliffordGate.from_xz_map((Pauli.Y, True),
                                                  (Pauli.Z, False))
