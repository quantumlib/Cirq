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


class TextDiagramDrawer:
    """A utility class for creating simple text diagrams.
    """

    def __init__(self):
        self.entries = dict()
        self.vertical_lines = []
        self.horizontal_lines = []

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
        if any(line_x == x and y1 < y < y2
               for line_x, y1, y2 in self.vertical_lines):
            return True

        # Horizontal line?
        if any(line_y == y and x1 < x < x2
               for line_y, x1, x2 in self.horizontal_lines):
            return True

        return False

    def line(self, x1: int, y1: int, x2: int, y2: int):
        """Adds a line from (x1, y1) to (x2, y2).

        Horizontal line is selected on equality in the second coordinate and
        vertical line is selected on equality in the first coordinate.

        Raises:
            ValueError: If line is neither horizontal nor vertical.
        """
        if x1 == x2:
            self.vertical_line(x1, y1, y2)
        elif y1 == y2:
            self.horizontal_line(y1, x1, x2)
        else:
            raise ValueError("Line is neither horizontal nor vertical")

    def vertical_line(self, x: int, y1: int, y2: int):
        """Adds a line from (x, y1) to (x, y2)."""
        y1, y2 = sorted([y1, y2])
        self.vertical_lines.append((x, y1, y2))

    def horizontal_line(self, y, x1, x2):
        """Adds a line from (x1, y) to (x2, y)."""
        x1, x2 = sorted([x1, x2])
        self.horizontal_lines.append((y, x1, x2))

    def transpose(self):
        """Returns the same diagram, but mirrored across its diagonal."""
        out = TextDiagramDrawer()
        out.entries = {(y, x): v for (x, y), v in self.entries.items()}
        out.vertical_lines = list(self.horizontal_lines)
        out.horizontal_lines = list(self.vertical_lines)
        return out

    def width(self):
        """Determines how many entry columns are in the diagram."""
        max_x = -1
        for x, _ in self.entries.keys():
            max_x = max(max_x, x)
        for x, _, _ in self.vertical_lines:
            max_x = max(max_x, x)
        for _, x1, x2 in self.horizontal_lines:
            max_x = max(max_x, x1, x2)
        return 1 + max_x

    def height(self):
        """Determines how many entry rows are in the diagram."""
        max_y = -1
        for _, y in self.entries.keys():
            max_y = max(max_y, y)
        for y, _, _ in self.horizontal_lines:
            max_y = max(max_y, y)
        for _, y1, y2 in self.vertical_lines:
            max_y = max(max_y, y1, y2)
        return 1 + max_y

    def render(self,
               horizontal_spacing: int = 1,
               vertical_spacing: int = 1,
               crossing_char: str = None,
               use_unicode_characters: bool = True) -> str:
        """Outputs text containing the diagram."""

        pipe = '│' if use_unicode_characters else '|'
        dash = '─' if use_unicode_characters else '-'
        if crossing_char is None:
            crossing_char = '┼' if use_unicode_characters else '+'

        dx = 1 + horizontal_spacing
        dy = 1 + vertical_spacing
        w = self.width() * dx - horizontal_spacing
        h = self.height() * dy - vertical_spacing

        grid = [[''] * w for _ in range(h)]
        extend_char = [[' '] * w for _ in range(h)]

        for x, y1, y2 in self.vertical_lines:
            x *= dx
            y1 *= dy
            y2 *= dy
            for y in range(y1, y2):
                grid[y][x] = pipe

        for y, x1, x2 in self.horizontal_lines:
            y *= dy
            x1 *= dx
            x2 *= dx
            for x in range(x1, x2):
                if grid[y][x] == pipe:
                    grid[y][x] = crossing_char
                else:
                    grid[y][x] = dash
                extend_char[y][x] = dash

        for (x, y), v in self.entries.items():
            x *= dx
            y *= dy
            grid[y][x] = v

        for col in range(w):
            col_width = max(1, max(len(grid[y][col]) for y in range(h)))
            for row in range(h):
                missing = col_width - len(grid[row][col])
                grid[row][col] += extend_char[row][col] * missing

        return '\n'.join(''.join(row).rstrip() for row in grid)
