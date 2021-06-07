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

from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
    BoxDrawCharacterSet,
    NORMAL_BOX_CHARS,
    BOLD_BOX_CHARS,
    ASCII_BOX_CHARS,
)

_HorizontalLine = NamedTuple(
    'HorizontalLine',
    [
        ('y', Union[int, float]),
        ('x1', Union[int, float]),
        ('x2', Union[int, float]),
        ('emphasize', bool),
    ],
)
_VerticalLine = NamedTuple(
    'VerticalLine',
    [
        ('x', Union[int, float]),
        ('y1', Union[int, float]),
        ('y2', Union[int, float]),
        ('emphasize', bool),
    ],
)
_DiagramText = NamedTuple(
    'DiagramText',
    [
        ('text', str),
        ('transposed_text', str),
    ],
)


def pick_charset(use_unicode: bool, emphasize: bool) -> BoxDrawCharacterSet:
    if not use_unicode:
        return ASCII_BOX_CHARS
    if emphasize:
        return BOLD_BOX_CHARS
    return NORMAL_BOX_CHARS


@value.value_equality(unhashable=True)
class TextDiagramDrawer:
    """A utility class for creating simple text diagrams."""

    def __init__(
        self,
        entries: Optional[Mapping[Tuple[int, int], _DiagramText]] = None,
        horizontal_lines: Optional[Iterable[_HorizontalLine]] = None,
        vertical_lines: Optional[Iterable[_VerticalLine]] = None,
        horizontal_padding: Optional[Mapping[int, int]] = None,
        vertical_padding: Optional[Mapping[int, int]] = None,
    ) -> None:
        self.entries: Dict[Tuple[int, int], _DiagramText] = (
            dict() if entries is None else dict(entries)
        )
        self.horizontal_lines: List[_HorizontalLine] = (
            [] if horizontal_lines is None else list(horizontal_lines)
        )
        self.vertical_lines: List[_VerticalLine] = (
            [] if vertical_lines is None else list(vertical_lines)
        )
        self.horizontal_padding: Dict[int, Union[int, float]] = (
            dict() if horizontal_padding is None else dict(horizontal_padding)
        )
        self.vertical_padding: Dict[int, Union[int, float]] = (
            dict() if vertical_padding is None else dict(vertical_padding)
        )

    def _value_equality_values_(self):
        attrs = (
            'entries',
            'horizontal_lines',
            'vertical_lines',
            'horizontal_padding',
            'vertical_padding',
        )
        return tuple(getattr(self, attr) for attr in attrs)

    def __bool__(self):
        return any(self._value_equality_values_())

    def write(self, x: int, y: int, text: str, transposed_text: Optional[str] = None):
        """Adds text to the given location.

        Args:
            x: The column in which to write the text.
            y: The row in which to write the text.
            text: The text to write at location (x, y).
            transposed_text: Optional text to write instead, if the text
                diagram is transposed.
        """
        entry = self.entries.get((x, y), _DiagramText('', ''))
        self.entries[(x, y)] = _DiagramText(
            entry.text + text,
            entry.transposed_text + (transposed_text if transposed_text else text),
        )

    def content_present(self, x: int, y: int) -> bool:
        """Determines if a line or printed text is at the given location."""

        # Text?
        if (x, y) in self.entries:
            return True

        # Vertical line?
        if any(v.x == x and v.y1 < y < v.y2 for v in self.vertical_lines):
            return True

        # Horizontal line?
        if any(line_y == y and x1 < x < x2 for line_y, x1, x2, _ in self.horizontal_lines):
            return True

        return False

    def grid_line(self, x1: int, y1: int, x2: int, y2: int, emphasize: bool = False):
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

    def vertical_line(
        self,
        x: Union[int, float],
        y1: Union[int, float],
        y2: Union[int, float],
        emphasize: bool = False,
    ) -> None:
        """Adds a line from (x, y1) to (x, y2)."""
        y1, y2 = sorted([y1, y2])
        self.vertical_lines.append(_VerticalLine(x, y1, y2, emphasize))

    def horizontal_line(
        self,
        y: Union[int, float],
        x1: Union[int, float],
        x2: Union[int, float],
        emphasize: bool = False,
    ) -> None:
        """Adds a line from (x1, y) to (x2, y)."""
        x1, x2 = sorted([x1, x2])
        self.horizontal_lines.append(_HorizontalLine(y, x1, x2, emphasize))

    def transpose(self) -> 'TextDiagramDrawer':
        """Returns the same diagram, but mirrored across its diagonal."""
        out = TextDiagramDrawer()
        out.entries = {
            (y, x): _DiagramText(v.transposed_text, v.text) for (x, y), v in self.entries.items()
        }
        out.vertical_lines = [_VerticalLine(*e) for e in self.horizontal_lines]
        out.horizontal_lines = [_HorizontalLine(*e) for e in self.vertical_lines]
        out.vertical_padding = self.horizontal_padding.copy()
        out.horizontal_padding = self.vertical_padding.copy()
        return out

    def width(self) -> int:
        """Determines how many entry columns are in the diagram."""
        max_x = -1.0
        for x, _ in self.entries.keys():
            max_x = max(max_x, x)
        for v in self.vertical_lines:
            max_x = max(max_x, v.x)
        for h in self.horizontal_lines:
            max_x = max(max_x, h.x1, h.x2)
        return 1 + int(max_x)

    def height(self) -> int:
        """Determines how many entry rows are in the diagram."""
        max_y = -1.0
        for _, y in self.entries.keys():
            max_y = max(max_y, y)
        for h in self.horizontal_lines:
            max_y = max(max_y, h.y)
        for v in self.vertical_lines:
            max_y = max(max_y, v.y1, v.y2)
        return 1 + int(max_y)

    def force_horizontal_padding_after(self, index: int, padding: Union[int, float]) -> None:
        """Change the padding after the given column."""
        self.horizontal_padding[index] = padding

    def force_vertical_padding_after(self, index: int, padding: Union[int, float]) -> None:
        """Change the padding after the given row."""
        self.vertical_padding[index] = padding

    def _transform_coordinates(
        self,
        func: Callable[
            [Union[int, float], Union[int, float]], Tuple[Union[int, float], Union[int, float]]
        ],
    ) -> None:
        """Helper method to transformer either row or column coordinates."""

        def func_x(x: Union[int, float]) -> Union[int, float]:
            return func(x, 0)[0]

        def func_y(y: Union[int, float]) -> Union[int, float]:
            return func(0, y)[1]

        self.entries = {
            cast(Tuple[int, int], func(int(x), int(y))): v for (x, y), v in self.entries.items()
        }
        self.vertical_lines = [
            _VerticalLine(func_x(x), func_y(y1), func_y(y2), emph)
            for x, y1, y2, emph in self.vertical_lines
        ]
        self.horizontal_lines = [
            _HorizontalLine(func_y(y), func_x(x1), func_x(x2), emph)
            for y, x1, x2, emph in self.horizontal_lines
        ]
        self.horizontal_padding = {
            int(func_x(int(x))): padding for x, padding in self.horizontal_padding.items()
        }
        self.vertical_padding = {
            int(func_y(int(y))): padding for y, padding in self.vertical_padding.items()
        }

    def insert_empty_columns(self, x: int, amount: int = 1) -> None:
        """Insert a number of columns after the given column."""

        def transform_columns(
            column: Union[int, float], row: Union[int, float]
        ) -> Tuple[Union[int, float], Union[int, float]]:
            return column + (amount if column >= x else 0), row

        self._transform_coordinates(transform_columns)

    def insert_empty_rows(self, y: int, amount: int = 1) -> None:
        """Insert a number of rows after the given row."""

        def transform_rows(
            column: Union[int, float], row: Union[int, float]
        ) -> Tuple[Union[int, float], Union[int, float]]:
            return column, row + (amount if row >= y else 0)

        self._transform_coordinates(transform_rows)

    def render(
        self,
        horizontal_spacing: int = 1,
        vertical_spacing: int = 1,
        crossing_char: str = None,
        use_unicode_characters: bool = True,
    ) -> str:
        """Outputs text containing the diagram."""

        block_diagram = BlockDiagramDrawer()

        w = self.width()
        h = self.height()

        # Communicate padding into block diagram.
        for x in range(0, w - 1):
            block_diagram.set_col_min_width(
                x * 2 + 1,
                # Horizontal separation looks narrow, so partials round up.
                int(np.ceil(self.horizontal_padding.get(x, horizontal_spacing))),
            )
            block_diagram.set_col_min_width(x * 2, 1)
        for y in range(0, h - 1):
            block_diagram.set_row_min_height(
                y * 2 + 1,
                # Vertical separation looks wide, so partials round down.
                int(np.floor(self.vertical_padding.get(y, vertical_spacing))),
            )
            block_diagram.set_row_min_height(y * 2, 1)

        # Draw vertical lines.
        for x_b, y1_b, y2_b, emphasize in self.vertical_lines:
            x = int(x_b * 2)
            y1, y2 = int(min(y1_b, y2_b) * 2), int(max(y1_b, y2_b) * 2)
            charset = pick_charset(use_unicode_characters, emphasize)

            # Caps.
            block_diagram.mutable_block(x, y1).draw_curve(charset, bottom=True)
            block_diagram.mutable_block(x, y2).draw_curve(charset, top=True)

            # Span.
            for y in range(y1 + 1, y2):
                block_diagram.mutable_block(x, y).draw_curve(charset, top=True, bottom=True)

        # Draw horizontal lines.
        for y_b, x1_b, x2_b, emphasize in self.horizontal_lines:
            y = int(y_b * 2)
            x1, x2 = int(min(x1_b, x2_b) * 2), int(max(x1_b, x2_b) * 2)
            charset = pick_charset(use_unicode_characters, emphasize)

            # Caps.
            block_diagram.mutable_block(x1, y).draw_curve(charset, right=True)
            block_diagram.mutable_block(x2, y).draw_curve(charset, left=True)

            # Span.
            for x in range(x1 + 1, x2):
                block_diagram.mutable_block(x, y).draw_curve(
                    charset, left=True, right=True, crossing_char=crossing_char
                )

        # Place entries.
        for (x, y), v in self.entries.items():
            x *= 2
            y *= 2
            block_diagram.mutable_block(x, y).content = v.text

        return block_diagram.render()

    def copy(self):
        return self.__class__(
            entries=self.entries,
            vertical_lines=self.vertical_lines,
            horizontal_lines=self.horizontal_lines,
            vertical_padding=self.vertical_padding,
            horizontal_padding=self.horizontal_padding,
        )

    def shift(self, dx: int = 0, dy: int = 0) -> 'TextDiagramDrawer':
        self._transform_coordinates(lambda x, y: (x + dx, y + dy))
        return self

    def shifted(self, dx: int = 0, dy: int = 0) -> 'TextDiagramDrawer':
        return self.copy().shift(dx, dy)

    def superimpose(self, other: 'TextDiagramDrawer') -> 'TextDiagramDrawer':
        self.entries.update(other.entries)
        self.horizontal_lines += other.horizontal_lines
        self.vertical_lines += other.vertical_lines
        self.horizontal_padding.update(other.horizontal_padding)
        self.vertical_padding.update(other.vertical_padding)
        return self

    def superimposed(self, other: 'TextDiagramDrawer') -> 'TextDiagramDrawer':
        return self.copy().superimpose(other)

    @classmethod
    def vstack(
        cls,
        diagrams: Sequence['TextDiagramDrawer'],
        padding_resolver: Optional[Callable[[Sequence[Optional[int]]], int]] = None,
    ):
        """Vertically stack text diagrams.

        Args:
            diagrams: The diagrams to stack, ordered from bottom to top.
            padding_resolver: A function that takes a list of paddings
                specified for a column and returns the padding to use in the
                stacked diagram. If None, defaults to raising ValueError if the
                diagrams to stack contain inconsistent padding in any column,
                including if some specify a padding and others don't.

        Raises:
            ValueError: Inconsistent padding cannot be resolved.

        Returns:
            The vertically stacked diagram.
        """

        if padding_resolver is None:
            padding_resolver = _same_element_or_throw_error

        stacked = cls()
        dy = 0
        for diagram in diagrams:
            stacked.superimpose(diagram.shifted(dy=dy))
            dy += diagram.height()
        for x in stacked.horizontal_padding:
            resolved_padding = padding_resolver(
                tuple(
                    cast(Optional[int], diagram.horizontal_padding.get(x)) for diagram in diagrams
                )
            )
            if resolved_padding is not None:
                stacked.horizontal_padding[x] = resolved_padding
        return stacked

    @classmethod
    def hstack(
        cls,
        diagrams: Sequence['TextDiagramDrawer'],
        padding_resolver: Optional[Callable[[Sequence[Optional[int]]], int]] = None,
    ):
        """Horizontally stack text diagrams.

        Args:
            diagrams: The diagrams to stack, ordered from left to right.
            padding_resolver: A function that takes a list of paddings
                specified for a row and returns the padding to use in the
                stacked diagram. Defaults to raising ValueError if the diagrams
                to stack contain inconsistent padding in any row, including
                if some specify a padding and others don't.

        Raises:
            ValueError: Inconsistent padding cannot be resolved.

        Returns:
            The horizontally stacked diagram.
        """

        if padding_resolver is None:
            padding_resolver = _same_element_or_throw_error

        stacked = cls()
        dx = 0
        for diagram in diagrams:
            stacked.superimpose(diagram.shifted(dx=dx))
            dx += diagram.width()
        for y in stacked.vertical_padding:
            resolved_padding = padding_resolver(
                tuple(cast(Optional[int], diagram.vertical_padding.get(y)) for diagram in diagrams)
            )
            if resolved_padding is not None:
                stacked.vertical_padding[y] = resolved_padding
        return stacked


def _same_element_or_throw_error(elements: Sequence[Any]):
    """Extract an element or throw an error.

    Args:
        elements: A sequence of something.

    copies of it. Returns None on an empty sequence.

    Raises:
        ValueError: The sequence contains more than one unique element.

    Returns:
        The element when given a sequence containing only multiple copies of a
        single element. None if elements is empty.
    """
    unique_elements = set(elements)
    if len(unique_elements) > 1:
        raise ValueError(f'len(set({elements})) > 1')
    return unique_elements.pop() if elements else None
