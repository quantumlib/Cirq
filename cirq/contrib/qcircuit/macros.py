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
