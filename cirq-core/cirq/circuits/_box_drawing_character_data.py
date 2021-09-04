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

"""Exposes structured data about unicode/ascii box drawing characters."""

from typing import List, NamedTuple, Optional


_BoxDrawCharacterSet = NamedTuple(
    '_BoxDrawCharacterSet',
    [
        ('top', str),
        ('bottom', str),
        ('left', str),
        ('right', str),
        ('top_bottom', str),
        ('top_left', str),
        ('top_right', str),
        ('bottom_left', str),
        ('bottom_right', str),
        ('left_right', str),
        ('top_bottom_left', str),
        ('top_bottom_right', str),
        ('top_left_right', str),
        ('bottom_left_right', str),
        ('top_bottom_left_right', str),
    ],
)


class BoxDrawCharacterSet(_BoxDrawCharacterSet):
    def char(
        self, top: bool = False, bottom: bool = False, left: bool = False, right: bool = False
    ) -> Optional[str]:
        parts = []
        if top:
            parts.append('top')
        if bottom:
            parts.append('bottom')
        if left:
            parts.append('left')
        if right:
            parts.append('right')
        if not parts:
            return None
        return getattr(self, '_'.join(parts))


_MixedBoxDrawCharacterSet = NamedTuple(
    '_MixedBoxDrawCharacterSet',
    [
        ('first_char_set', BoxDrawCharacterSet),
        ('second_char_set', BoxDrawCharacterSet),
        ('top_then_bottom', str),
        ('top_then_left', str),
        ('top_then_right', str),
        ('top_then_bottom_left', str),
        ('top_then_bottom_right', str),
        ('top_then_left_right', str),
        ('top_then_bottom_left_right', str),
        ('bottom_then_top', str),
        ('bottom_then_left', str),
        ('bottom_then_right', str),
        ('bottom_then_top_left', str),
        ('bottom_then_top_right', str),
        ('bottom_then_left_right', str),
        ('bottom_then_top_left_right', str),
        ('left_then_top', str),
        ('left_then_bottom', str),
        ('left_then_right', str),
        ('left_then_top_bottom', str),
        ('left_then_bottom_right', str),
        ('left_then_top_right', str),
        ('left_then_top_bottom_right', str),
        ('right_then_top', str),
        ('right_then_bottom', str),
        ('right_then_left', str),
        ('right_then_top_bottom', str),
        ('right_then_top_left', str),
        ('right_then_bottom_left', str),
        ('right_then_top_bottom_left', str),
        ('top_bottom_then_left', str),
        ('top_bottom_then_right', str),
        ('top_bottom_then_left_right', str),
        ('top_left_then_bottom', str),
        ('top_left_then_right', str),
        ('top_left_then_bottom_right', str),
        ('top_right_then_bottom', str),
        ('top_right_then_left', str),
        ('top_right_then_bottom_left', str),
        ('bottom_left_then_top', str),
        ('bottom_left_then_right', str),
        ('bottom_left_then_top_right', str),
        ('bottom_right_then_top', str),
        ('bottom_right_then_left', str),
        ('bottom_right_then_top_left', str),
        ('left_right_then_top', str),
        ('left_right_then_bottom', str),
        ('left_right_then_top_bottom', str),
        ('top_bottom_left_then_right', str),
        ('top_bottom_right_then_left', str),
        ('top_left_right_then_bottom', str),
        ('bottom_left_right_then_top', str),
    ],
)


class MixedBoxDrawCharacterSet(_MixedBoxDrawCharacterSet):
    def char(
        self, *, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0
    ) -> Optional[str]:
        def parts_with(val: int) -> List[str]:
            parts = []
            if top == val:
                parts.append('top')
            if bottom == val:
                parts.append('bottom')
            if left == val:
                parts.append('left')
            if right == val:
                parts.append('right')
            return parts

        first_key = '_'.join(parts_with(-1))
        second_key = '_'.join(parts_with(+1))

        if not first_key and not second_key:
            return None
        if not first_key:
            return getattr(self.second_char_set, second_key)
        if not second_key:
            return getattr(self.first_char_set, first_key)
        return getattr(self, f'{first_key}_then_{second_key}')


NORMAL_BOX_CHARS = BoxDrawCharacterSet(
    top='╵',
    bottom='╷',
    left='╴',
    right='╶',
    top_bottom='│',
    top_left='┘',
    top_right='└',
    bottom_left='┐',
    bottom_right='┌',
    left_right='─',
    top_bottom_left='┤',
    top_bottom_right='├',
    top_left_right='┴',
    bottom_left_right='┬',
    top_bottom_left_right='┼',
)


BOLD_BOX_CHARS = BoxDrawCharacterSet(
    top='╹',
    bottom='╻',
    left='╸',
    right='╺',
    top_bottom='┃',
    top_left='┛',
    top_right='┗',
    bottom_left='┓',
    bottom_right='┏',
    left_right='━',
    top_bottom_left='┫',
    top_bottom_right='┣',
    top_left_right='┻',
    bottom_left_right='┳',
    top_bottom_left_right='╋',
)


DOUBLED_BOX_CHARS = BoxDrawCharacterSet(
    # No special end caps for these ones :(.
    top='║',
    bottom='║',
    left='═',
    right='═',
    top_bottom='║',
    top_left='╝',
    top_right='╚',
    bottom_left='╗',
    bottom_right='╔',
    left_right='═',
    top_bottom_left='╣',
    top_bottom_right='╠',
    top_left_right='╩',
    bottom_left_right='╦',
    top_bottom_left_right='╬',
)


ASCII_BOX_CHARS = BoxDrawCharacterSet(
    # We can round the half-caps up to full or down to nothing.
    top=' ',
    bottom=' ',
    left=' ',
    right=' ',
    top_bottom='|',
    top_left='/',
    top_right='\\',
    bottom_left='\\',
    bottom_right='/',
    left_right='-',
    top_bottom_left='+',
    top_bottom_right='+',
    top_left_right='+',
    bottom_left_right='+',
    top_bottom_left_right='+',
)


NORMAL_THEN_BOLD_MIXED_BOX_CHARS = MixedBoxDrawCharacterSet(
    first_char_set=NORMAL_BOX_CHARS,
    second_char_set=BOLD_BOX_CHARS,
    top_then_bottom='╽',
    top_then_left='┙',
    top_then_right='┕',
    top_then_bottom_left='┪',
    top_then_bottom_right='┢',
    top_then_left_right='┷',
    top_then_bottom_left_right='╈',
    bottom_then_top='╿',
    bottom_then_left='┑',
    bottom_then_right='┍',
    bottom_then_top_left='┩',
    bottom_then_top_right='┡',
    bottom_then_left_right='┯',
    bottom_then_top_left_right='╇',
    left_then_top='┚',
    left_then_bottom='┒',
    left_then_right='╼',
    left_then_top_bottom='┨',
    left_then_bottom_right='┲',
    left_then_top_right='┺',
    left_then_top_bottom_right='╊',
    right_then_top='┖',
    right_then_bottom='┎',
    right_then_left='╾',
    right_then_top_bottom='┠',
    right_then_top_left='┹',
    right_then_bottom_left='┱',
    right_then_top_bottom_left='╉',
    top_bottom_then_left='┥',
    top_bottom_then_right='┝',
    top_bottom_then_left_right='┿',
    top_left_then_bottom='┧',
    top_left_then_right='┶',
    top_left_then_bottom_right='╆',
    top_right_then_bottom='┟',
    top_right_then_left='┵',
    top_right_then_bottom_left='╅',
    bottom_left_then_top='┦',
    bottom_left_then_right='┮',
    bottom_left_then_top_right='╄',
    bottom_right_then_top='┞',
    bottom_right_then_left='┭',
    bottom_right_then_top_left='╃',
    left_right_then_top='┸',
    left_right_then_bottom='┰',
    left_right_then_top_bottom='╂',
    top_bottom_left_then_right='┾',
    top_bottom_right_then_left='┽',
    top_left_right_then_bottom='╁',
    bottom_left_right_then_top='╀',
    # You're right, it *was* tedious.
    # If the box drawing character set was laid out so that certain bits
    # corresponded to certain legs in a reasonable way, this wouldn't have been
    # needed...
)


NORMAL_THEN_DOUBLED_MIXED_BOX_CHARS = MixedBoxDrawCharacterSet(
    first_char_set=NORMAL_BOX_CHARS,
    second_char_set=DOUBLED_BOX_CHARS,
    top_then_bottom=' ',
    top_then_left='╛',
    top_then_right='╘',
    top_then_bottom_left=' ',
    top_then_bottom_right=' ',
    top_then_left_right='╧',
    top_then_bottom_left_right=' ',
    bottom_then_top=' ',
    bottom_then_left='╕',
    bottom_then_right='╒',
    bottom_then_top_left=' ',
    bottom_then_top_right=' ',
    bottom_then_left_right='╤',
    bottom_then_top_left_right=' ',
    left_then_top='╜',
    left_then_bottom='╖',
    left_then_right=' ',
    left_then_top_bottom='╢',
    left_then_bottom_right=' ',
    left_then_top_right=' ',
    left_then_top_bottom_right=' ',
    right_then_top='╙',
    right_then_bottom='╓',
    right_then_left=' ',
    right_then_top_bottom='╟',
    right_then_top_left=' ',
    right_then_bottom_left=' ',
    right_then_top_bottom_left=' ',
    top_bottom_then_left='╡',
    top_bottom_then_right='╞',
    top_bottom_then_left_right='╪',
    top_left_then_bottom=' ',
    top_left_then_right=' ',
    top_left_then_bottom_right=' ',
    top_right_then_bottom=' ',
    top_right_then_left=' ',
    top_right_then_bottom_left=' ',
    bottom_left_then_top=' ',
    bottom_left_then_right=' ',
    bottom_left_then_top_right=' ',
    bottom_right_then_top=' ',
    bottom_right_then_left=' ',
    bottom_right_then_top_left=' ',
    left_right_then_top='╨',
    left_right_then_bottom='╥',
    left_right_then_top_bottom='╫',
    top_bottom_left_then_right=' ',
    top_bottom_right_then_left=' ',
    top_left_right_then_bottom=' ',
    bottom_left_right_then_top=' ',
)


def box_draw_character(
    first: Optional[BoxDrawCharacterSet],
    second: BoxDrawCharacterSet,
    *,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
) -> Optional[str]:
    """Finds a box drawing character based on its connectivity.

    For example:

        box_draw_character(
            NORMAL_BOX_CHARS,
            BOLD_BOX_CHARS,
            top=-1,
            right=+1)

    evaluates to '┕', which has a normal upward leg and bold rightward leg.

    Args:
        first: The character set to use for legs set to -1. If set to None,
            defaults to the same thing as the second character set.
        second: The character set to use for legs set to +1.
        top: Whether the upward leg should be present.
        bottom: Whether the bottom leg should be present.
        left: Whether the left leg should be present.
        right: Whether the right leg should be present.

    Returns:
        A box drawing character approximating the desired properties, or None
        if all legs are set to 0.
    """
    if first is None:
        first = second
    sign = +1
    combo = None

    # Known combinations.
    if first is NORMAL_BOX_CHARS and second is BOLD_BOX_CHARS:
        combo = NORMAL_THEN_BOLD_MIXED_BOX_CHARS
    if first is BOLD_BOX_CHARS and second is NORMAL_BOX_CHARS:
        combo = NORMAL_THEN_BOLD_MIXED_BOX_CHARS
        sign = -1
    if first is NORMAL_BOX_CHARS and second is DOUBLED_BOX_CHARS:
        combo = NORMAL_THEN_DOUBLED_MIXED_BOX_CHARS
    if first is DOUBLED_BOX_CHARS and second is NORMAL_BOX_CHARS:
        combo = NORMAL_THEN_DOUBLED_MIXED_BOX_CHARS
        sign = -1

    if combo is None:
        choice = second if +1 in [top, bottom, left, right] else first
        return choice.char(top=bool(top), bottom=bool(bottom), left=bool(left), right=bool(right))

    return combo.char(top=top * sign, bottom=bottom * sign, left=left * sign, right=right * sign)
