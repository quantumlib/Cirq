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

from cirq.contrib.paulistring.pauli import Pauli, PAULI_X, PAULI_Y, PAULI_Z


PauliTransform = NamedTuple('PauliTransform', [('to', Pauli), ('flip', bool)])


class CliffordGate(ops.CompositeGate,
                   ops.ReversibleGate,
                   ops.TextDiagrammableGate):
    '''Any single qubit Clifford rotation.'''
    def __init__(self,
                 transforms_x_to: Tuple[Pauli, bool],
                 transforms_z_to: Tuple[Pauli, bool],
                 _maps: Tuple[Dict[Pauli, PauliTransform],
                              Dict[Pauli, PauliTransform]] = None) -> None:
        if _maps is not None:
            self._rotation_map, self._inverse_map = _maps
            return
        rotates_x_to, flips_x = transforms_x_to
        rotates_z_to, flips_z = transforms_z_to
        if rotates_x_to == rotates_z_to:
            raise ValueError('A rotation cannot map both X and Z to {!s}.'
                             .format(rotates_x_to))
        rotates_y_to = rotates_x_to.third(rotates_z_to)
        flips_y = flips_x ^ flips_z ^ (rotates_x_to - rotates_z_to != 1)
        self._rotation_map = {PAULI_X: PauliTransform(rotates_x_to, flips_x),
                              PAULI_Y: PauliTransform(rotates_y_to, flips_y),
                              PAULI_Z: PauliTransform(rotates_z_to, flips_z)}
        self._inverse_map = {rotates_x_to: PauliTransform(PAULI_X, flips_x),
                             rotates_y_to: PauliTransform(PAULI_Y, flips_y),
                             rotates_z_to: PauliTransform(PAULI_Z, flips_z)}

    def transform(self, pauli: Pauli) -> PauliTransform:
        return self._rotation_map[pauli]

    def rotates_pauli_to(self, pauli: Pauli) -> Pauli:
        return self.transform(pauli).to

    def flips_pauli(self, pauli: Pauli) -> bool:
        return self.transform(pauli).flip

    def _tuple(self) -> Tuple[Any, ...]:
        return (CliffordGate,
                self.transform(PAULI_X),
                self.transform(PAULI_Y),
                self.transform(PAULI_Z))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._tuple() == other._tuple()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self._tuple())

    def inverse(self) -> 'CliffordGate':
        inverse_map = self._inverse_map
        return CliffordGate(inverse_map[PAULI_X], inverse_map[PAULI_Z],
                            _maps = (inverse_map, self._rotation_map))

    def commutes_with_single_qubit_gate(self, gate: 'CliffordGate') -> bool:
        '''Tests if the two circuits would be equivalent:
            --self--gate-- and --gate--self--'''
        for pauli0 in (PAULI_X, PAULI_Z):
            pauli1, flip1 = self.transform(pauli0)
            pauli2, flip2 = gate.transform(pauli1)
            pauli3, flip3 = self._inverse_map[pauli2]
            pauli4, flip4 = gate._inverse_map[pauli3]
            if pauli4 != pauli0 or (flip1 ^ flip2 ^ flip3 ^ flip4):
                return False
        return True

    def commutes_with_pauli(self, pauli: Pauli, whole=False) -> bool:
        return (self.rotates_pauli_to(pauli) == pauli
                and (not self.flips_pauli(pauli) or whole))

    def merged_with(self, second: 'CliffordGate') -> 'CliffordGate':
        '''Returns a CliffordGate such that the circuits
            --output-- and --self--second--
        are equivalent.'''
        x_intermediate_pauli, x_flip1 = self.transform(PAULI_X)
        x_final_pauli, x_flip2 = second.transform(x_intermediate_pauli)
        z_intermediate_pauli, z_flip1 = self.transform(PAULI_Z)
        z_final_pauli, z_flip2 = second.transform(z_intermediate_pauli)
        return CliffordGate((x_final_pauli, x_flip1 ^ x_flip2),
                            (z_final_pauli, z_flip1 ^ z_flip2))

    def default_decompose(self, qubits: Sequence[ops.QubitId]) -> ops.OP_TREE:
        qubit, = qubits
        if self == CLIFFORD_H:
            return (ops.H(qubit),)
        rotations = self.decompose_rotation()
        return tuple((
                {PAULI_X: ops.X,
                 PAULI_Y: ops.Y,
                 PAULI_Z: ops.Z}[r](qubit) ** (qt / 2)
                for r, qt in rotations))

    def decompose_rotation(self) -> Sequence[Tuple[Pauli, int]]:
        '''Returns ((first_rotation_axis, first_rotation_quarter_turns), ...)

        This is a sequence of zero, one, or two rotations.'''
        x_rot = self.transform(PAULI_X)
        y_rot = self.transform(PAULI_Y)
        z_rot = self.transform(PAULI_Z)
        whole_arr = (x_rot.to == PAULI_X,
                     y_rot.to == PAULI_Y,
                     z_rot.to == PAULI_Z)
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
                pauli = Pauli(flip_arr.index(False))
                return [(pauli, 2)]
        elif num_whole == 1:
            index = whole_arr.index(True)
            pauli = Pauli(index)
            flip = flip_arr[index]
            output = []
            if flip:
                # 180 degree rotation
                output.append((pauli + 1, 2))
            # 90 degree rotation about some axis
            if not self.transform(pauli + 1).flip:
                # Positive 90 degree rotation
                output.append((pauli, 1))
            else:
                # Negative 90 degree rotation
                output.append((pauli, -1))
            return output
        elif num_whole == 0:
            # Gate is a 120 degree rotation
            if x_rot.to == PAULI_Y:
                return [(PAULI_X, -1 if y_rot.flip else 1),
                        (PAULI_Z, -1 if x_rot.flip else 1)]
            else:
                return [(PAULI_Z, 1 if y_rot.flip else -1),
                        (PAULI_X, 1 if z_rot.flip else -1)]
        assert False  # coverage: ignore

    def single_qubit_gate_after_switching_order(self, gate: 'CliffordGate'
                                                ) -> 'CliffordGate':
        '''Returns a CliffordGate such that the circuits
            --output--self-- and --self--gate--
        are equivalent.'''
        return self.merged_with(gate).merged_with(self.inverse())

    def __repr__(self):
        return 'CliffordGate(X:{}{!s}, Y:{}{!s}, Z:{}{!s})'.format(
                '+-'[self.flips_pauli(PAULI_X)], self.rotates_pauli_to(PAULI_X),
                '+-'[self.flips_pauli(PAULI_Y)], self.rotates_pauli_to(PAULI_Y),
                '+-'[self.flips_pauli(PAULI_Z)], self.rotates_pauli_to(PAULI_Z))

    def text_diagram_wire_symbols(self,
                                  qubit_count: Optional[int] = None,
                                  use_unicode_characters: bool = True,
                                  precision: Optional[int] = 3
                                  ) -> Tuple[str, ...]:
        well_known_map = {
            CLIFFORD_I: 'I',
            CLIFFORD_H: 'H',
            CLIFFORD_X: 'X',
            CLIFFORD_Y: 'Y',
            CLIFFORD_Z: 'Z',
            CLIFFORD_X_sqrt: 'X',
            CLIFFORD_Y_sqrt: 'Y',
            CLIFFORD_Z_sqrt: 'Z',
            CLIFFORD_X_nsqrt: 'X',
            CLIFFORD_Y_nsqrt: 'Y',
            CLIFFORD_Z_nsqrt: 'Z',
        }
        if self in well_known_map:
            return (well_known_map[self],)
        else:
            rotations = self.decompose_rotation()
            return ('-'.join((
                        str(r) + ('^' + str(qt / 2)) * (qt % 4 != 2)
                        for r, qt in rotations)),)

    def text_diagram_exponent(self) -> float:
        return {CLIFFORD_X_sqrt: 0.5,
                CLIFFORD_Y_sqrt: 0.5,
                CLIFFORD_Z_sqrt: 0.5,
                CLIFFORD_X_nsqrt: -0.5,
                CLIFFORD_Y_nsqrt: -0.5,
                CLIFFORD_Z_nsqrt: -0.5,
               }.get(self, 1)


CLIFFORD_I = CliffordGate((PAULI_X, False), (PAULI_Z, False))
CLIFFORD_H = CliffordGate((PAULI_Z, False), (PAULI_X, False))
CLIFFORD_X = CliffordGate((PAULI_X, False), (PAULI_Z, True))
CLIFFORD_Y = CliffordGate((PAULI_X, True),  (PAULI_Z, True))
CLIFFORD_Z = CliffordGate((PAULI_X, True),  (PAULI_Z, False))
CLIFFORD_X_sqrt  = CliffordGate((PAULI_X, False), (PAULI_Y, True))
CLIFFORD_X_nsqrt = CliffordGate((PAULI_X, False), (PAULI_Y, False))
CLIFFORD_Y_sqrt  = CliffordGate((PAULI_Z, True),  (PAULI_X, False))
CLIFFORD_Y_nsqrt = CliffordGate((PAULI_Z, False), (PAULI_X, True))
CLIFFORD_Z_sqrt  = CliffordGate((PAULI_Y, False), (PAULI_Z, False))
CLIFFORD_Z_nsqrt = CliffordGate((PAULI_Y, True),  (PAULI_Z, False))
