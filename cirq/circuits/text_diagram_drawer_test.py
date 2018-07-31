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
from cirq.testing.mock import mock


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
