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

from typing import Tuple

def line_macro(end: Tuple[int, int],
               start: Tuple[int, int] = (0,0),
               thickness: int = 1,
               style: str = '-'
               ) -> str:
    """Produces Xy-pic (tex) code for drawing a line.

    Args:
        end: The end point of the line, relative to the xymatrix element in
            which the code appears.
        start: The starting point of the line, relative to the xymatrix element
            in which the code appears. Default to (0, 0).
        thickness: The thickness of the line, as an integer multiple of the
            standard thickness.
        style: The style of the line, e.g. '.' draws a dashed line. Defaults
            to '-', specifying a solid line.
    Returns:
        The tex code.
    """
    tex = '\\ar '
    tex += '@{' + str(style) + '} '
    if thickness != 1:
        tex += '@*{[|(' + str(thickness) + ')]} '
    if tuple(start) != (0, 0):
        tex += '[{1}, {0}];'.format(*start)
    tex += '[{1}, {0}]'.format(*end)
    return tex


def gate_macro(label: str = '') -> str:
    r"""Same as qcircuit's '\gate' but without final '\qw'."""
    return ('*+<.6em>{' + str(label) + '} \POS ="i","i"+UR;"i"+UL **\dir{-};'
            '"i"+DL **\dir{-};"i"+DR **\dir{-};"i"+UR **\dir{-},"i"')

def ghost_macro(label: str='') -> str:
    r"""Same as qcircuit's '\nghost'."""
    return '*+<1em,.9em>{\hphantom{' + str(label) + '}}'


def multigate_macro(n_qubits: int, label: str='') -> str:
    r"""Same as qcircuit's '\multigate' but without final '\qw'."""
    return (
        '*+<1em,.9em>{\hphantom{' + str(label) + '}} \POS [0,0]="i",[0,0].[' +
        str(n_qubits - 1) + ',0]="e",!C *{' + str(label) +
        '},"e"+UR;"e"+UL **\dir{-};"e"+DL **\dir{-};' +
        '"e"+DR **\dir{-};"e"+UR **\dir{-},"i"')
