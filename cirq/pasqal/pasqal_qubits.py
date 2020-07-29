# Copyright 2020 The Cirq Developers
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
from typing import List
from numpy import sqrt
import numpy as np

import cirq


class ThreeDQubit(cirq.ops.Qid):
    """A qubit in 3d.

    ThreeDQubits use z-y-x ordering:

        ThreeDQubit(0, 0, 0) < ThreeDQubit(1, 0, 0)
        < ThreeDQubit(0, 1, 0) < ThreeDQubit(1, 1, 0)
        < ThreeDQubit(0, 0, 1) < ThreeDQubit(1, 0, 1)
        < ThreeDQubit(0, 1, 1) < ThreeDQubit(1, 1, 1)
    """

    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def _comparison_key(self):
        return round(self.z, 15), round(self.y, 15), round(self.x, 15)

    @property
    def dimension(self) -> int:
        return 2

    def distance(self, other: cirq.ops.Qid) -> float:
        """Returns the distance between two qubits in 3d."""
        if not isinstance(other, ThreeDQubit):
            raise TypeError(
                "Can compute distance to another ThreeDQubit, but {}".format(
                    other))
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2 +
                    (self.z - other.z)**2)

    @staticmethod
    def cube(diameter: int, x0: float = 0, y0: float = 0,
             z0: float = 0) -> List['ThreeDQubit']:
        """Returns a cube of ThreeDQubits.

        Args:
            diameter: Length of a side of the square
            x0: x-coordinate of the first qubit
            y0: y-coordinate of the first qubit
            z0: z-coordinate of the first qubit

        Returns:
            A list of ThreeDQubits filling in a square grid
        """
        return ThreeDQubit.parallelep(diameter,
                                      diameter,
                                      diameter,
                                      x0=x0,
                                      y0=y0,
                                      z0=z0)

    @staticmethod
    def parallelep(rows: int,
                   cols: int,
                   lays: int,
                   x0: float = 0,
                   y0: float = 0,
                   z0: float = 0) -> List['ThreeDQubit']:
        """Returns a parallelepiped of ThreeDQubits.

        Args:
            rows: Number of rows in the rectangle
            cols: Number of columns in the rectangle
            x0: x-coordinate of the first qubit
            y0: y-coordinate of the first qubit
            z0: z-coordinate of the first qubit

        Returns:
            A list of ThreeDQubits filling in a 3d grid
        """
        return [
            ThreeDQubit(x0 + x, y0 + y, z0 + z) for z in range(lays)
            for y in range(cols) for x in range(rows)
        ]

    def __repr__(self):
        return 'pasqal.ThreeDQubit({}, {}, {})'.format(self.x, self.y, self.z)

    def __str__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['x', 'y', 'z'])


class TwoDQubit(ThreeDQubit):
    """A qubit in 2d."""

    def __init__(self, x: float, y: float):
        super().__init__(x, y, z=0)

    @staticmethod
    def square(diameter: int, x0: float = 0,
               y0: float = 0) -> List['TwoDQubit']:
        """Returns a square of TwoDQubit.

        Args:
            diameter: Length of a side of the square
            x0: x-coordinate of the first qubit
            y0: y-coordinate of the first qubit

        Returns:
            A list of TwoDQubits filling in a square grid
        """
        return TwoDQubit.rect(diameter, diameter, x0=x0, y0=y0)

    @staticmethod
    def rect(rows: int, cols: int, x0: float = 0,
             y0: float = 0) -> List['TwoDQubit']:
        """Returns a rectangle of TwoDQubit.

        Args:
            rows: Number of rows in the rectangle
            cols: Number of columns in the rectangle
            x0: x-coordinate of the first qubit
            y0: y-coordinate of the first qubit

        Returns:
            A list of TwoDQubits filling in a rectangular grid
        """
        return [
            TwoDQubit(x0 + x, y0 + y) for y in range(cols) for x in range(rows)
        ]

    @staticmethod
    def triangular_lattice(l: int, x0: float = 0, y0: float = 0):
        """Returns a triangular lattice of TwoDQubits.

        Args:
            l: Number of qubits along one direction
            x0: x-coordinate of the first qubit
            y0: y-coordinate of the first qubit

        Returns:
            A list of TwoDQubits filling in a triangular lattice.
        """
        coords = np.array([[x, y] for x in range(l + 1) for y in range(l + 1)],
                          dtype=float)
        coords[:, 0] += 0.5 * np.mod(coords[:, 1], 2)
        coords[:, 1] *= np.sqrt(3) / 2
        coords += [x0, y0]

        return [TwoDQubit(coord[0], coord[1]) for coord in coords]

    def __repr__(self):
        return 'pasqal.TwoDQubit({}, {})'.format(self.x, self.y)

    def __str__(self):
        return '({}, {})'.format(self.x, self.y)

    def _json_dict_(self):
        return cirq.protocols.obj_to_dict_helper(self, ['x', 'y'])
