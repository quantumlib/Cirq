# Copyright 2018 Google LLC
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

from cirq._circuits import TextDiagramDrawer


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
