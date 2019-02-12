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

from cirq.circuits._box_drawing_character_data import (
    box_draw_character,
    NORMAL_BOX_CHARS,
    NORMAL_THEN_BOLD_MIXED_BOX_CHARS,
    BOLD_BOX_CHARS,
)


def test_chars():
    assert NORMAL_BOX_CHARS.char() is None
    assert NORMAL_BOX_CHARS.char(top=True, bottom=True) == '│'
    assert NORMAL_THEN_BOLD_MIXED_BOX_CHARS.char() is None
    assert NORMAL_THEN_BOLD_MIXED_BOX_CHARS.char(top=1, bottom=-1) == '╿'
    assert NORMAL_THEN_BOLD_MIXED_BOX_CHARS.char(top=1, bottom=1) == '┃'
    assert NORMAL_THEN_BOLD_MIXED_BOX_CHARS.char(top=-1, bottom=-1) == '│'

    assert box_draw_character(None, NORMAL_BOX_CHARS) is None
    assert box_draw_character(NORMAL_BOX_CHARS,
                              BOLD_BOX_CHARS,
                              top=-1,
                              bottom=+1) == '╽'
    assert box_draw_character(BOLD_BOX_CHARS,
                              NORMAL_BOX_CHARS,
                              top=-1,
                              bottom=+1) == '╿'
