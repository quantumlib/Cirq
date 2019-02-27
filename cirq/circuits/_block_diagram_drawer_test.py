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
import itertools

import pytest

import cirq
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
    NORMAL_BOX_CHARS,
    BOLD_BOX_CHARS,
    DOUBLED_BOX_CHARS,
    ASCII_BOX_CHARS,
    BoxDrawCharacterSet,
)


def _assert_same_diagram(actual: str, expected: str):
    assert actual == expected, (
        "Diagram differs from the desired diagram.\n"
        '\n'
        'Actual diagram:\n'
        '{}\n'
        '\n'
        'Desired diagram:\n'
        '{}\n'
        '\n'
        'Highlighted differences:\n'
        '{}\n'.format(actual,
                      expected,
                      cirq.testing.highlight_text_differences(actual,
                                                              expected))
    )


def _curve_pieces_diagram(chars: BoxDrawCharacterSet) -> BlockDiagramDrawer:
    d = BlockDiagramDrawer()
    for x in range(4):
        for y in range(4):
            block = d.mutable_block(x*2, y*2)
            block.horizontal_alignment = 0.5
            block.draw_curve(
                chars,
                top=bool(y & 1),
                bottom=bool(y & 2),
                left=bool(x & 2),
                right=bool(x & 1))
    return d


def test_block_curve():
    d = _curve_pieces_diagram(NORMAL_BOX_CHARS)
    actual = d.render(min_block_width=5, min_block_height=5)
    expected = """

            ╶──     ──╴       ─────







  │         │         │         │
  │         │         │         │
  ╵         └──     ──┘       ──┴──









  ╷         ┌──     ──┐       ──┬──
  │         │         │         │
  │         │         │         │





  │         │         │         │
  │         │         │         │
  │         ├──     ──┤       ──┼──
  │         │         │         │
  │         │         │         │"""
    _assert_same_diagram(actual, expected)

    d = _curve_pieces_diagram(DOUBLED_BOX_CHARS)
    actual = d.render(min_block_width=3, min_block_height=3)
    expected = """
       ══   ══    ═══




 ║     ║     ║     ║
 ║     ╚═   ═╝    ═╩═





 ║     ╔═   ═╗    ═╦═
 ║     ║     ║     ║



 ║     ║     ║     ║
 ║     ╠═   ═╣    ═╬═
 ║     ║     ║     ║"""
    _assert_same_diagram(actual, expected)

    d = _curve_pieces_diagram(BOLD_BOX_CHARS)
    actual = d.render(min_block_width=4, min_block_height=4)
    expected = """
         ╺━━    ━╸      ━━━━






 ┃       ┃       ┃       ┃
 ╹       ┗━━    ━┛      ━┻━━







 ╻       ┏━━    ━┓      ━┳━━
 ┃       ┃       ┃       ┃
 ┃       ┃       ┃       ┃




 ┃       ┃       ┃       ┃
 ┃       ┣━━    ━┫      ━╋━━
 ┃       ┃       ┃       ┃
 ┃       ┃       ┃       ┃"""
    _assert_same_diagram(actual, expected)

    d = _curve_pieces_diagram(ASCII_BOX_CHARS)
    actual = d.render(min_block_width=3, min_block_height=3)
    expected = r"""
        -   -     ---




 |     |     |     |
       \-   -/    -+-





       /-   -\    -+-
 |     |     |     |



 |     |     |     |
 |     +-   -+    -+-
 |     |     |     |"""
    _assert_same_diagram(actual, expected)


def test_mixed_block_curve():
    diagram = BlockDiagramDrawer()
    for a, b, c, d in itertools.product(range(3), repeat=4):
        x = (a * 3 + b) * 2
        y = (c * 3 + d) * 2
        block = diagram.mutable_block(x, y)
        block.horizontal_alignment = 0.5
        block.draw_curve(
            NORMAL_BOX_CHARS,
            top=a == 2,
            bottom=b == 2,
            left=c == 2,
            right=d == 2)
        block.draw_curve(
            BOLD_BOX_CHARS,
            top=a == 1,
            bottom=b == 1,
            left=c == 1,
            right=d == 1)
    actual = diagram.render(min_block_width=3, min_block_height=3)
    expected = """
                   ┃     ┃     ┃     │     │     │
       ╻     ╷     ╹     ┃     ╿     ╵     ╽     │
       ┃     │           ┃     │           ┃     │



                   ┃     ┃     ┃     │     │     │
 ╺━    ┏━    ┍━    ┗━    ┣━    ┡━    ┕━    ┢━    ┝━
       ┃     │           ┃     │           ┃     │



                   ┃     ┃     ┃     │     │     │
 ╶─    ┎─    ┌─    ┖─    ┠─    ┞─    └─    ┟─    ├─
       ┃     │           ┃     │           ┃     │



                   ┃     ┃     ┃     │     │     │
━╸    ━┓    ━┑    ━┛    ━┫    ━┩    ━┙    ━┪    ━┥
       ┃     │           ┃     │           ┃     │



                   ┃     ┃     ┃     │     │     │
━━━   ━┳━   ━┯━   ━┻━   ━╋━   ━╇━   ━┷━   ━╈━   ━┿━
       ┃     │           ┃     │           ┃     │



                   ┃     ┃     ┃     │     │     │
━╾─   ━┱─   ━┭─   ━┹─   ━╉─   ━╃─   ━┵─   ━╅─   ━┽─
       ┃     │           ┃     │           ┃     │



                   ┃     ┃     ┃     │     │     │
─╴    ─┒    ─┐    ─┚    ─┨    ─┦    ─┘    ─┧    ─┤
       ┃     │           ┃     │           ┃     │



                   ┃     ┃     ┃     │     │     │
─╼━   ─┲━   ─┮━   ─┺━   ─╊━   ─╄━   ─┶━   ─╆━   ─┾━
       ┃     │           ┃     │           ┃     │



                   ┃     ┃     ┃     │     │     │
───   ─┰─   ─┬─   ─┸─   ─╂─   ─╀─   ─┴─   ─╁─   ─┼─
       ┃     │           ┃     │           ┃     │"""[1:]
    _assert_same_diagram(actual, expected)


def test_lines_meet_content():
    d = BlockDiagramDrawer()
    b = d.mutable_block(0, 0)
    b.content = 'long text\nwith multiple lines'
    b.left = '>'
    b.right = '<'
    b.top = 'v'
    b.bottom = '^'

    _assert_same_diagram(d.render(), """
long text<<<<<<<<<<
with multiple lines"""[1:])

    b.horizontal_alignment = 0.5
    _assert_same_diagram(d.render(), """
>>>>>long text<<<<<
with multiple lines"""[1:])

    _assert_same_diagram(d.render(min_block_height=5), """
         v
         v
>>>>>long text<<<<<
with multiple lines
         ^"""[1:])

    _assert_same_diagram(d.render(min_block_height=4), """
         v
>>>>>long text<<<<<
with multiple lines
         ^"""[1:])

    _assert_same_diagram(d.render(min_block_height=20, min_block_width=40), """
                   v
                   v
                   v
                   v
                   v
                   v
                   v
                   v
                   v
>>>>>>>>>>>>>>>long text<<<<<<<<<<<<<<<<
          with multiple lines
                   ^
                   ^
                   ^
                   ^
                   ^
                   ^
                   ^
                   ^
                   ^"""[1:])

    _assert_same_diagram(d.render(min_block_height=21, min_block_width=41), """
                    v
                    v
                    v
                    v
                    v
                    v
                    v
                    v
                    v
                    v
>>>>>>>>>>>>>>>>long text<<<<<<<<<<<<<<<<
           with multiple lines
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^"""[1:])

    b.content = 'short text'
    _assert_same_diagram(d.render(min_block_height=21, min_block_width=41), """
                    v
                    v
                    v
                    v
                    v
                    v
                    v
                    v
                    v
                    v
>>>>>>>>>>>>>>>>short text<<<<<<<<<<<<<<<
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^"""[1:])

    b.content = 'abc\ndef\nghi'
    _assert_same_diagram(d.render(min_block_height=21, min_block_width=41), """
                    v
                    v
                    v
                    v
                    v
                    v
                    v
                    v
                    v
                   abc
>>>>>>>>>>>>>>>>>>>def<<<<<<<<<<<<<<<<<<<
                   ghi
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^
                    ^"""[1:])


def test_content_stretches_other_blocks():
    d = BlockDiagramDrawer()
    d.mutable_block(0, 0).horizontal_alignment = 0.5
    d.mutable_block(1, 0).horizontal_alignment = 0.5
    d.mutable_block(0, 1).horizontal_alignment = 0.5
    d.mutable_block(1, 1).horizontal_alignment = 0.5
    d.mutable_block(0, 0).content = 'long text\nwith multiple lines'
    d.mutable_block(1, 0).draw_curve(
        NORMAL_BOX_CHARS, top=True, bottom=True, left=True, right=True)
    d.mutable_block(1, 1).draw_curve(
        NORMAL_BOX_CHARS, top=True, bottom=True, left=True, right=True)
    d.mutable_block(0, 1).draw_curve(
        NORMAL_BOX_CHARS, top=True, bottom=True, left=True, right=True)
    _assert_same_diagram(d.render(), """
     long text     ┼
with multiple lines│
─────────┼─────────┼"""[1:])


def test_lines_stretch_content():
    d = BlockDiagramDrawer()
    d.mutable_block(0, 0).left = 'l'
    d.mutable_block(2, 4).right = 'r'
    d.mutable_block(11, 15).bottom = 'b'
    d.mutable_block(16, 17).top = 't'
    d.mutable_block(19, 20).center = 'c'
    d.mutable_block(21, 23).content = 'C'
    _assert_same_diagram(d.render(), """


  C"""[1:])


def test_indices():
    d = BlockDiagramDrawer()
    with pytest.raises(IndexError):
        d.mutable_block(-1, -1)
    with pytest.raises(IndexError):
        d.set_row_min_height(-1, 500)
    with pytest.raises(IndexError):
        d.set_col_min_width(-1, 500)
