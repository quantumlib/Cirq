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
from typing import List, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Tuple, Dict

_HorizontalLine = NamedTuple('HorizontalLine', [
    ('y', int),
    ('x1', int),
    ('x2', int),
    ('emphasize', bool),
])
_VerticalLine = NamedTuple('VerticalLine', [
    ('x', int),
    ('y1', int),
    ('y2', int),
    ('emphasize', bool),
])
_MomentGroup = NamedTuple('MomentGroup', [
    ('start', int),
    ('end', int),
])
_MomentGroupParts = NamedTuple('MomentGroupParts', [
    ('start_char', str),
    ('mid_char', str),
    ('end_char', str),
])


class TextDiagramDrawer:
    """A utility class for creating simple text diagrams.
    """

    def __init__(self):
        self.entries = dict()  # type: Dict[Tuple[int, int], str]
        self.vertical_lines = []  # type: List[_VerticalLine]
        self.horizontal_lines = []  # type: List[_HorizontalLine]
        self.moment_groups = []  # type: List[_MomentGroup]
        self.groups_are_vertical = False

    def write(self, x: int, y: int, text: str):
        """Adds text to the given location."""
        if (x, y) in self.entries:
            self.entries[(x, y)] += text
        else:
            self.entries[(x, y)] = text

    def content_present(self, x: int, y: int) -> bool:
        """Determines if a line or printed text is at the given location."""

        # Text?
        if (x, y) in self.entries:
            return True

        # Vertical line?
        if any(v.x == x and v.y1 < y < v.y2 for v in self.vertical_lines):
            return True

        # Horizontal line?
        if any(line_y == y and x1 < x < x2
               for line_y, x1, x2, _ in self.horizontal_lines):
            return True

        return False

    def grid_line(self, x1: int, y1: int, x2: int, y2: int,
                  emphasize: bool = False):
        """Adds a vertical or horizontal line from (x1, y1) to (x2, y2).

        Horizontal line is selected on equality in the second coordinate and
        vertical line is selected on equality in the first coordinate.

        Raises:
            ValueError: If line is neither horizontal nor vertical.
        """
        if x1 == x2:
            self.vertical_line(x1, y1, y2, emphasize)
        elif y1 == y2:
            self.horizontal_line(y1, x1, x2, emphasize)
        else:
            raise ValueError("Line is neither horizontal nor vertical")

    def vertical_line(self, x: int, y1: int, y2: int, emphasize: bool = False
                      ) -> None:
        """Adds a line from (x, y1) to (x, y2)."""
        y1, y2 = sorted([y1, y2])
        self.vertical_lines.append(_VerticalLine(x, y1, y2, emphasize))

    def horizontal_line(self, y, x1, x2, emphasize: bool = False
                        ) -> None:
        """Adds a line from (x1, y) to (x2, y)."""
        x1, x2 = sorted([x1, x2])
        self.horizontal_lines.append(_HorizontalLine(y, x1, x2, emphasize))

    def moment_group(self, start, end) -> None:
        """Groups columns start to end as belonging to the same Moment."""
        start, end = sorted([start, end])
        self.moment_groups.append(_MomentGroup(start, end))

    def transpose(self) -> 'TextDiagramDrawer':
        """Returns the same diagram, but mirrored across its diagonal."""
        out = TextDiagramDrawer()
        out.entries = {(y, x): v for (x, y), v in self.entries.items()}
        out.vertical_lines = [_VerticalLine(*e)
                              for e in self.horizontal_lines]
        out.horizontal_lines = [_HorizontalLine(*e)
                                for e in self.vertical_lines]
        out.moment_groups = [_MomentGroup(*e)
                             for e in self.moment_groups]
        out.groups_are_vertical = not self.groups_are_vertical
        return out

    def width(self) -> int:
        """Determines how many entry columns are in the diagram."""
        max_x = -1
        for x, _ in self.entries.keys():
            max_x = max(max_x, x)
        for v in self.vertical_lines:
            max_x = max(max_x, v.x)
        for h in self.horizontal_lines:
            max_x = max(max_x, h.x1, h.x2)
        return 1 + max_x

    def height(self) -> int:
        """Determines how many entry rows are in the diagram."""
        max_y = -1
        for _, y in self.entries.keys():
            max_y = max(max_y, y)
        for h in self.horizontal_lines:
            max_y = max(max_y, h.y)
        for v in self.vertical_lines:
            max_y = max(max_y, v.y1, v.y2)
        return 1 + max_y

    def render(self,
               horizontal_spacing: int = 1,
               vertical_spacing: int = 1,
               crossing_char: str = None,
               use_unicode_characters: bool = True) -> str:
        """Outputs text containing the diagram."""

        char = _normal_char if use_unicode_characters else _ascii_char

        w = self.width()
        h = self.height()

        grid = [[''] * w for _ in range(h)]
        horizontal_separator = [[' '] * w for _ in range(h)]
        vertical_separator = [[' '] * w for _ in range(h)]

        # Place lines.
        verticals = {
            (v.x, y): v.emphasize
            for v in self.vertical_lines
            for y in range(v.y1, v.y2)
        }
        horizontals = {
            (x, h.y): h.emphasize
            for h in self.horizontal_lines
            for x in range(h.x1, h.x2)
        }
        for (x, y), emph in verticals.items():
            c = char('│', emph)
            grid[y][x] = c
            vertical_separator[y][x] = c
        for (x, y), emph in horizontals.items():
            c = char('─', emph)
            grid[y][x] = c
            horizontal_separator[y][x] = c
        for x, y in set(horizontals.keys()) & set(verticals.keys()):
            grid[y][x] = crossing_char or _cross_char(
                not use_unicode_characters,
                horizontals[(x, y)],
                verticals[(x, y)])

        # Place entries.
        for (x, y), v in self.entries.items():
            grid[y][x] = v

        # Prepare Moment groups.
        if self.groups_are_vertical:
            moment_group_parts = (char('┬', False),
                                  char('│', False),
                                  char('┴', False))
        else:
            moment_group_parts = (char('├', False),
                                  char('─', False),
                                  char('┤', False))

        # Pad rows and columns to fit contents with desired spacing.
        multiline_grid = _pad_into_multiline(w,
                                             grid,
                                             horizontal_separator,
                                             vertical_separator,
                                             horizontal_spacing,
                                             vertical_spacing,
                                             self.moment_groups,
                                             moment_group_parts,
                                             self.groups_are_vertical)

        # Concatenate it all together.
        return '\n'.join(''.join(sub_row).rstrip()
                         for row in multiline_grid
                         for sub_row in row).rstrip()


_BoxChars = [
    ('─', '━', '-'),
    ('│', '┃', '|'),
    ('┌', '┏', '/'),
    ('└', '┗', '\\'),
    ('┐', '┓', '\\'),
    ('┘', '┛', '/'),
    ('├', '┣', '|'),
    ('┼', '╋', '+'),
    ('┤', '┫', '|'),
    ('┬', '┳', '-'),
    ('┴', '┻', '-'),
]  # type: List[Tuple[str, ...]]

_EmphasisMap = {k: v for k, v, _ in _BoxChars}
_AsciiMap = {k: v for k, _, v in _BoxChars}


def _normal_char(k: str, emphasize: bool = False) -> str:
    return _EmphasisMap.get(k, k) if emphasize else k


def _ascii_char(k: str, emphasize: bool = False) -> str:
    del emphasize
    return _AsciiMap.get(k, k)


def _cross_char(use_ascii: bool, horizontal_emph: bool, vertical_emph: bool
                ) -> str:
    if use_ascii:
        return '+'
    if horizontal_emph != vertical_emph:
        return '┿' if horizontal_emph else '╂'
    return _normal_char('┼', horizontal_emph)


def _pad_border(border: List[str],
                index: int,
                length: int,
                spacing: int,
                moment_groups: List[_MomentGroup],
                moment_group_parts: _MomentGroupParts) -> int:
    """Pad a border with indicators for Moment groups.

    Args:
        border: A list of cells corresponding to the border of a circuit grid.
        index: The index of the border cell to be padded.
        length: The width (if horizontal) or height (if vertical) of the
            longest contents of a cell.
        spacing: The number of spaces (if horizontal) or rows (if vertical)
            between this border cell and the next. If spacing is 2 or greater,
            the start and end of each interval is drawn in the adjacent cell.
        moment_groups: A set of intervals each of which groups together
            columns (if horizontal) or rows (if vertical) belonging to
            the same Moment.
        moment_group_parts: The characters to use for drawing the start,
            middle, and end of an interval indicating a Moment group.

    Returns:
        The new length for that border cell.
    """
    (start_char, middle_char, end_char) = moment_group_parts
    overshoot = spacing >= 2
    if any(index in range(start, end) for (start, end) in moment_groups):
        border.append(middle_char * length)
    else:
        new_length = length + spacing
        if any(index == end for (_, end) in moment_groups):
            middle_char_length = length if overshoot else length - 1
            border.append((middle_char * middle_char_length +
                           end_char).ljust(new_length, ' '))
        else:
            border.append(' ' * new_length)
        length = new_length
    if any(index == start for (start, _) in moment_groups):
        if overshoot:
            border[index - 1] = border[index - 1][:-1] + start_char
        else:
            border[index] = start_char + border[index][1:]
    return length


def _pad_into_multiline(width: int,
                        grid: List[List[str]],
                        horizontal_separator: List[List[str]],
                        vertical_separator: List[List[str]],
                        horizontal_spacing: int,
                        vertical_spacing: int,
                        moment_groups: List[_MomentGroup],
                        moment_group_parts: _MomentGroupParts,
                        groups_are_vertical: bool
                        ) -> List[List[List[str]]]:
    multiline_grid = []  # type: List[List[List[str]]]
    border = []

    # Vertical padding.
    for row in range(len(grid)):
        multiline_cells = [cell.split('\n') for cell in grid[row]]
        row_height = max(1, max(len(cell) for cell in multiline_cells))
        if groups_are_vertical:
            row_height = _pad_border(border, row, row_height, vertical_spacing,
                                     moment_groups, moment_group_parts)
        elif row < len(grid) - 1:
            row_height += vertical_spacing

        multiline_row = []
        for sub_row in range(row_height):
            sub_row_cells = []
            for col in range(width):
                cell_lines = multiline_cells[col]
                sub_row_cells.append(cell_lines[sub_row]
                                     if sub_row < len(cell_lines)
                                     else vertical_separator[row][col])
            multiline_row.append(sub_row_cells)

        multiline_grid.append(multiline_row)

    # Horizontal padding.
    for col in range(width):
        col_width = max(1, max(len(sub_row[col])
                               for row in multiline_grid
                               for sub_row in row))
        if groups_are_vertical:
            col_width += horizontal_spacing
        else:
            col_width = _pad_border(border, col, col_width, horizontal_spacing,
                                    moment_groups, moment_group_parts)
        for row in range(len(multiline_grid)):
            for sub_row in range(len(multiline_grid[row])):
                sub_row_contents = multiline_grid[row][sub_row]
                pad_char = (horizontal_separator[row][col]
                            if sub_row == 0
                            else ' ')
                sub_row_contents[col] = sub_row_contents[col].ljust(
                    col_width, pad_char)

    if moment_groups:
        if groups_are_vertical:
            for row in range(len(multiline_grid)):
                for sub_row in range(len(multiline_grid[row])):
                    sub_row_contents = multiline_grid[row][sub_row]
                    sub_row_contents.insert(0, border[row][sub_row] + ' ')
                    sub_row_contents.append(border[row][sub_row])
        else:
            multiline_grid.insert(0, [border])
            multiline_grid.append([border])
    return multiline_grid
