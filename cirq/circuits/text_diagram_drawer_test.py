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

import pytest

from cirq.circuits import TextDiagramDrawer
from cirq.circuits.text_diagram_drawer import (
        _HorizontalLine, _VerticalLine, _DiagramText)
from cirq.testing.mock import mock
from cirq.testing import assert_has_diagram


def test_draw_entries_and_lines_with_options():
    d = TextDiagramDrawer()
    d.write(0, 0, '!')
    d.write(6, 2, 'span')
    d.horizontal_line(y=3, x1=2, x2=8)
    d.vertical_line(x=7, y1=1, y2=4)
    assert d.render().strip() == """
!

                 │
                 │
            span │
                 │
    ─────────────┼─
                 │
    """.strip()

    assert d.render(use_unicode_characters=False).strip() == """
!

                 |
                 |
            span |
                 |
    -------------+-
                 |
    """.strip()

    assert d.render(crossing_char='@').strip() == """
!

                 │
                 │
            span │
                 │
    ─────────────@─
                 │
    """.strip()

    assert d.render(horizontal_spacing=0).strip() == """
!

          │
          │
      span│
          │
  ────────┼
          │
    """.strip()

    assert d.render(vertical_spacing=0).strip() == """
!
                 │
            span │
    ─────────────┼─
    """.strip()


def test_draw_entries_and_lines_with_emphasize():
    d = TextDiagramDrawer()
    d.write(0, 0, '!')
    d.write(6, 2, 'span')
    d.horizontal_line(y=3, x1=2, x2=8, emphasize=True)
    d.horizontal_line(y=5, x1=2, x2=9, emphasize=False)
    d.vertical_line(x=7, y1=1, y2=6, emphasize=True)
    d.vertical_line(x=5, y1=1, y2=7, emphasize=False)
    assert d.render().strip() == """
!

          │      ┃
          │      ┃
          │ span ┃
          │      ┃
    ━━━━━━┿━━━━━━╋━
          │      ┃
          │      ┃
          │      ┃
    ──────┼──────╂───
          │      ┃
          │
          │
    """.strip()


def test_line_detects_horizontal():
    d = TextDiagramDrawer()
    with mock.patch.object(d, 'vertical_line') as vertical_line:
        d.grid_line(1, 2, 1, 5, True)
        vertical_line.assert_called_once_with(1, 2, 5, True)


def test_line_detects_vertical():
    d = TextDiagramDrawer()
    with mock.patch.object(d, 'horizontal_line') as horizontal_line:
        d.grid_line(2, 1, 5, 1, True)
        horizontal_line.assert_called_once_with(1, 2, 5, True)


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
    assert d.render().strip() == """
hello
there

next──────────1──────
              2
              3
              │
              4n
    """.strip()

    d = TextDiagramDrawer()
    d.vertical_line(x=0, y1=0, y2=3)
    d.vertical_line(x=1, y1=0, y2=3)
    d.vertical_line(x=2, y1=0, y2=3)
    d.vertical_line(x=3, y1=0, y2=3)
    d.write(0, 0, 'long line\nshort')
    d.write(2, 2, 'short\nlong line')
    assert d.render().strip() == """
long line │ │         │
short     │ │         │
│         │ │         │
│         │ │         │
│         │ │         │
│         │ short     │
│         │ long line │
│         │ │         │
    """.strip()

def test_drawer_copy():
    orig_entries = {(0, 0): _DiagramText('entry', '')}
    orig_vertical_lines = [_VerticalLine(1, 1, 3, True)]
    orig_horizontal_lines = [_HorizontalLine(0, 0, 3, False)]
    orig_vertical_padding = {0: 2}
    orig_horizontal_padding = {1: 3}
    kwargs = {
            'entries': orig_entries,
            'vertical_lines': orig_vertical_lines,
            'horizontal_lines': orig_horizontal_lines,
            'vertical_padding': orig_vertical_padding,
            'horizontal_padding': orig_horizontal_padding}
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

    vstacked = TextDiagramDrawer.vstack(d, dd)
    expected = """
D

E F

A B

  C
    """.strip()
    assert_has_diagram(vstacked, expected, render_method_name='render')

    hstacked = TextDiagramDrawer.hstack(d, dd)
    expected = """
A B D

  C E F
    """.strip()
    assert_has_diagram(hstacked, expected, render_method_name='render')
