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

import cirq.testing as ct
from cirq.circuits import TextDiagramDrawer
from cirq.testing.mock import mock


def assert_has_rendering(
        actual: TextDiagramDrawer,
        desired: str,
        **kwargs) -> None:
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
        '{}\n'.format(actual_diagram, desired_diagram,
                      ct.highlight_text_differences(actual_diagram,
                                                 desired_diagram))
    )


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

def test_horizontal_cut():
    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=0, x2=6)
    d.horizontal_line(y=1, x1=1, x2=5)
    d.horizontal_cut(y=0, x1=2, x2=4)
    expected_rendering = """
────┤   ├───

  ────────
"""[1:-1]
    assert_has_rendering(d, expected_rendering)
    expected_transpose_rendering = """
│
│
│ │
│ │
┴ │
  │
  │
  │
┬ │
│ │
│
│
"""[1:-1]
    assert_has_rendering(d.transpose(), expected_transpose_rendering)

    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=0, x2=2)
    d.horizontal_line(y=0, x1=5, x2=6)
    d.horizontal_cut(y=0, x1=2, x2=4)
    expected_rendering = """
────┤     ──
"""[1:-1]
    assert_has_rendering(d, expected_rendering)

    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=0, x2=1)
    d.horizontal_line(y=0, x1=5, x2=6)
    d.horizontal_cut(y=0, x1=2, x2=4)
    expected_rendering = """
──        ──
"""[1:-1]
    assert_has_rendering(d, expected_rendering)

    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=0, x2=6)
    d.horizontal_cut(y=0, x1=2, x2=4, emphasize=True)
    expected_rendering = """
────┨   ┠───
"""[1:-1]
    assert_has_rendering(d, expected_rendering)
    expected_transpose_rendering = """
│
│
│
│
┷



┯
│
│
│
"""[1:-1]
    assert_has_rendering(d.transpose(), expected_transpose_rendering)

    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=0, x2=6)
    d.horizontal_cut(y=0, x1=2, x2=4, use_unicode_characters=False)
    expected_rendering = """
----|   |---
"""[1:-1]
    assert_has_rendering(d, expected_rendering, use_unicode_characters=False)

    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=0, x2=6, emphasize=True)
    d.horizontal_cut(y=0, x1=2, x2=4)
    expected_rendering = """
━━━━┥   ┝━━━
"""[1:-1]
    assert_has_rendering(d, expected_rendering)
    expected_transpose_rendering = """
┃
┃
┃
┃
┸



┰
┃
┃
┃
"""[1:-1]
    assert_has_rendering(d.transpose(), expected_transpose_rendering)

    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=0, x2=6, emphasize=True)
    d.horizontal_cut(y=0, x1=2, x2=4, use_unicode_characters=False)
    expected_rendering = """
|
|
|
|
-



-
|
|
|
"""[1:-1]
    assert_has_rendering(d.transpose(), expected_rendering,
            use_unicode_characters=False)

    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=0, x2=6, emphasize=True)
    d.horizontal_cut(y=0, x1=2, x2=4, emphasize=True)
    expected_rendering = """
━━━━┫   ┣━━━
"""[1:-1]
    assert_has_rendering(d, expected_rendering)
    expected_transpose_rendering = """
┃
┃
┃
┃
┻



┳
┃
┃
┃
"""[1:-1]
    assert_has_rendering(d.transpose(), expected_transpose_rendering)

    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=2, x2=6)
    d.horizontal_cut(y=0, x1=2, x2=4)
    expected_rendering = """
        ├───
"""[1:-1]
    assert_has_rendering(d, expected_rendering)

    d = TextDiagramDrawer()
    d.horizontal_line(y=0, x1=0, x2=4)
    d.horizontal_cut(y=0, x1=2, x2=4)
    expected_rendering = """
────┤
"""[1:-1]
    assert_has_rendering(d, expected_rendering)
