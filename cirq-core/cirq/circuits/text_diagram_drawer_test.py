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

from unittest import mock
import pytest

from cirq.circuits import TextDiagramDrawer
from cirq.circuits._block_diagram_drawer_test import _assert_same_diagram
from cirq.circuits._box_drawing_character_data import (
    ASCII_BOX_CHARS,
    NORMAL_BOX_CHARS,
    DOUBLED_BOX_CHARS,
    BOLD_BOX_CHARS,
)
from cirq.circuits.text_diagram_drawer import (
    _HorizontalLine,
    _VerticalLine,
    _DiagramText,
    pick_charset,
)
import cirq.testing as ct


def assert_has_rendering(actual: TextDiagramDrawer, desired: str, **kwargs) -> None:
    """Determines if a given diagram has the desired rendering.

    Args:
        actual: The text diagram.
        desired: The desired rendering as a string.
        **kwargs: Keyword arguments to be passed to actual.render.
    """
    actual_diagram = actual.render(**kwargs)
    desired_diagram = desired
    assert actual_diagram == desired_diagram, (
        "Diagram's rendering differs from the desired rendering.\n"
        '\n'
        'Actual rendering:\n'
        '{}\n'
        '\n'
        'Desired rendering:\n'
        '{}\n'
        '\n'
        'Highlighted differences:\n'
        '{}\n'.format(
            actual_diagram,
            desired_diagram,
            ct.highlight_text_differences(actual_diagram, desired_diagram),
        )
    )


def test_draw_entries_and_lines_with_options():
    d = TextDiagramDrawer()
    d.write(0, 0, '!')
    d.write(6, 2, 'span')
    d.horizontal_line(y=3, x1=2, x2=8)
    d.vertical_line(x=7, y1=1, y2=4)
    _assert_same_diagram(
        d.render().strip(),
        """
!

                 ╷
                 │
            span │
                 │
    ╶────────────┼─
                 │
    """.strip(),
    )

    _assert_same_diagram(
        d.render(use_unicode_characters=False).strip(),
        """
!


                 |
            span |
                 |
     ------------+-
                 |
    """.strip(),
    )

    _assert_same_diagram(
        d.render(crossing_char='@').strip(),
        """
!

                 ╷
                 │
            span │
                 │
    ╶────────────@─
                 │
    """.strip(),
    )

    _assert_same_diagram(
        d.render(horizontal_spacing=0).strip(),
        """
!

          ╷
          │
      span│
          │
  ╶───────┼
          │
    """.strip(),
    )

    _assert_same_diagram(
        d.render(vertical_spacing=0).strip(),
        """
!
                 ╷
            span │
    ╶────────────┼─
    """.strip(),
    )


def test_draw_entries_and_lines_with_emphasize():
    d = TextDiagramDrawer()
    d.write(0, 0, '!')
    d.write(6, 2, 'span')
    d.horizontal_line(y=3, x1=2, x2=8, emphasize=True)
    d.horizontal_line(y=5, x1=2, x2=9, emphasize=False)
    d.vertical_line(x=7, y1=1, y2=6, emphasize=True)
    d.vertical_line(x=5, y1=1, y2=7, emphasize=False)
    _assert_same_diagram(
        d.render().strip(),
        """
!

          ╷      ╻
          │      ┃
          │ span ┃
          │      ┃
    ╺━━━━━┿━━━━━━╋━╸
          │      ┃
          │      ┃
          │      ┃
    ╶─────┼──────╂───
          │      ┃
          │      ╹
          │
    """.strip(),
    )


def test_line_detects_horizontal():
    d = TextDiagramDrawer()
    with mock.patch.object(d, 'vertical_line') as vertical_line:
        d.grid_line(1, 2, 1, 5, True)
        vertical_line.assert_called_once_with(1, 2, 5, True, False)


def test_line_detects_vertical():
    d = TextDiagramDrawer()
    with mock.patch.object(d, 'horizontal_line') as horizontal_line:
        d.grid_line(2, 1, 5, 1, True)
        horizontal_line.assert_called_once_with(1, 2, 5, True, False)


def test_line_fails_when_not_aligned():
    d = TextDiagramDrawer()
    with pytest.raises(ValueError):
        d.grid_line(1, 2, 3, 4)


def test_multiline_entries():
    d = TextDiagramDrawer()
    d.write(0, 0, 'hello\nthere')
    d.write(0, 1, 'next')
    d.write(5, 1, '1\n2\n3')
    d.write(5, 2, '4n')
    d.vertical_line(x=5, y1=1, y2=2)
    d.horizontal_line(y=1, x1=0, x2=8)
    _assert_same_diagram(
        d.render().strip(),
        """
hello
there

              1
next──────────2──────
              3
              │
              4n
    """.strip(),
    )

    d = TextDiagramDrawer()
    d.vertical_line(x=0, y1=0, y2=3)
    d.vertical_line(x=1, y1=0, y2=3)
    d.vertical_line(x=2, y1=0, y2=3)
    d.vertical_line(x=3, y1=0, y2=3)
    d.write(0, 0, 'long line\nshort')
    d.write(2, 2, 'short\nlong line')
    _assert_same_diagram(
        d.render().strip(),
        """
long line ╷ ╷         ╷
short     │ │         │
│         │ │         │
│         │ │         │
│         │ │         │
│         │ short     │
│         │ long line │
│         │ │         │
    """.strip(),
    )


def test_drawer_copy():
    orig_entries = {(0, 0): _DiagramText('entry', '')}
    orig_vertical_lines = [_VerticalLine(1, 1, 3, True, False)]
    orig_horizontal_lines = [_HorizontalLine(0, 0, 3, False, False)]
    orig_vertical_padding = {0: 2}
    orig_horizontal_padding = {1: 3}
    kwargs = {
        'entries': orig_entries,
        'vertical_lines': orig_vertical_lines,
        'horizontal_lines': orig_horizontal_lines,
        'vertical_padding': orig_vertical_padding,
        'horizontal_padding': orig_horizontal_padding,
    }
    orig_drawer = TextDiagramDrawer(**kwargs)

    same_drawer = TextDiagramDrawer(**kwargs)
    assert orig_drawer == same_drawer

    copy_drawer = orig_drawer.copy()
    assert orig_drawer == copy_drawer

    copy_drawer.write(0, 1, 'new_entry')
    assert copy_drawer != orig_drawer

    copy_drawer = orig_drawer.copy()
    copy_drawer.vertical_line(2, 1, 3)
    assert copy_drawer != orig_drawer

    copy_drawer = orig_drawer.copy()
    copy_drawer.horizontal_line(2, 1, 3)
    assert copy_drawer != orig_drawer

    copy_drawer = orig_drawer.copy()
    copy_drawer.force_horizontal_padding_after(1, 4)
    assert copy_drawer != orig_drawer

    copy_drawer = orig_drawer.copy()
    copy_drawer.force_vertical_padding_after(1, 4)
    assert copy_drawer != orig_drawer


def test_drawer_stack():
    d = TextDiagramDrawer()
    d.write(0, 0, 'A')
    d.write(1, 0, 'B')
    d.write(1, 1, 'C')
    dd = TextDiagramDrawer()
    dd.write(0, 0, 'D')
    dd.write(0, 1, 'E')
    dd.write(1, 1, 'F')

    vstacked = TextDiagramDrawer.vstack((dd, d))
    expected = """
D

E F

A B

  C
    """.strip()
    assert_has_rendering(vstacked, expected)

    hstacked = TextDiagramDrawer.hstack((d, dd))
    expected = """
A B D

  C E F
    """.strip()
    assert_has_rendering(hstacked, expected)

    d.force_horizontal_padding_after(0, 0)

    with pytest.raises(ValueError):
        TextDiagramDrawer.vstack((dd, d))

    dd.force_horizontal_padding_after(0, 0)
    expected = """
D

EF

AB

 C
    """.strip()
    vstacked = TextDiagramDrawer.vstack((dd, d))
    assert_has_rendering(vstacked, expected)

    d.force_vertical_padding_after(0, 0)
    with pytest.raises(ValueError):
        print(d.vertical_padding)
        print(dd.vertical_padding)
        TextDiagramDrawer.hstack((d, dd))

    dd.force_vertical_padding_after(0, 0)
    expected = """
AB D
 C EF
    """.strip()
    hstacked = TextDiagramDrawer.hstack((d, dd))
    assert_has_rendering(hstacked, expected)

    d.force_horizontal_padding_after(0, 0)
    dd.force_horizontal_padding_after(0, 2)
    d.force_vertical_padding_after(0, 1)
    dd.force_vertical_padding_after(0, 3)

    with pytest.raises(ValueError):
        TextDiagramDrawer.vstack((d, dd))

    vstacked = TextDiagramDrawer.vstack((dd, d), padding_resolver=max)
    expected = """
D



E  F

A  B

   C
    """.strip()
    assert_has_rendering(vstacked, expected)

    hstacked = TextDiagramDrawer.hstack((d, dd), padding_resolver=max)
    expected = """
AB D



 C E  F
    """.strip()
    assert_has_rendering(hstacked, expected)

    vstacked_min = TextDiagramDrawer.vstack((dd, d), padding_resolver=min)
    expected = """
D



EF

AB

 C
    """.strip()
    assert_has_rendering(vstacked_min, expected)

    hstacked_min = TextDiagramDrawer.hstack((d, dd), padding_resolver=min)
    expected = """
AB D

 C E  F
    """.strip()
    assert_has_rendering(hstacked_min, expected)


def test_drawer_eq():
    assert TextDiagramDrawer().__eq__(23) == NotImplemented

    eq = ct.EqualsTester()

    d = TextDiagramDrawer()
    d.write(0, 0, 'A')
    d.write(1, 0, 'B')
    d.write(1, 1, 'C')

    alt_d = TextDiagramDrawer()
    alt_d.write(0, 0, 'A')
    alt_d.write(1, 0, 'B')
    alt_d.write(1, 1, 'C')
    eq.add_equality_group(d, alt_d)

    dd = TextDiagramDrawer()
    dd.write(0, 0, 'D')
    dd.write(0, 1, 'E')
    dd.write(1, 1, 'F')

    eq.add_equality_group(dd)


def test_drawer_superimposed():
    empty_drawer = TextDiagramDrawer()
    assert not empty_drawer
    drawer_with_something = TextDiagramDrawer()
    drawer_with_something.write(0, 0, 'A')
    assert drawer_with_something
    superimposed_drawer = empty_drawer.superimposed(drawer_with_something)
    assert superimposed_drawer == drawer_with_something
    assert not empty_drawer


def test_pick_charset():
    assert pick_charset(use_unicode=False, emphasize=False, doubled=False) == ASCII_BOX_CHARS
    assert pick_charset(use_unicode=False, emphasize=False, doubled=True) == ASCII_BOX_CHARS
    assert pick_charset(use_unicode=False, emphasize=True, doubled=False) == ASCII_BOX_CHARS
    assert pick_charset(use_unicode=False, emphasize=True, doubled=True) == ASCII_BOX_CHARS
    assert pick_charset(use_unicode=True, emphasize=False, doubled=False) == NORMAL_BOX_CHARS
    assert pick_charset(use_unicode=True, emphasize=False, doubled=True) == DOUBLED_BOX_CHARS
    assert pick_charset(use_unicode=True, emphasize=True, doubled=False) == BOLD_BOX_CHARS
    with pytest.raises(ValueError):
        pick_charset(use_unicode=True, emphasize=True, doubled=True)
