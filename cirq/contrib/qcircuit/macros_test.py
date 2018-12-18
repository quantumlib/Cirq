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

import cirq.contrib.qcircuit as ccq


def test_line_macro():
    assert ccq.line_macro((1, 2)) == r'\ar @{-} [2, 1]'
    assert ccq.line_macro((1, 0), thickness=2) == r'\ar @{-} @*{[|(2)]} [0, 1]'
    assert ccq.line_macro((1, 1), start=(1, 0)) == r'\ar @{-} [0, 1];[1, 1]'
    assert (ccq.line_macro((1, 1), start=(1, 0), thickness=3) ==
            r'\ar @{-} @*{[|(3)]} [0, 1];[1, 1]')


def test_ghost_macro():
    assert ccq.ghost_macro('foo') == '*+<1em,.9em>{\hphantom{foo}}'
    assert ccq.ghost_macro() == '*+<1em,.9em>{\hphantom{}}'


def test_multigate_macro():
    assert ccq.multigate_macro(3) == (
        '*+<1em,.9em>{\hphantom{}} \POS [0,0]="i",[0,0].[2,0]="e",'
        '!C *{},"e"+UR;"e"+UL **\dir{-};"e"+DL **\dir{-};'
        '"e"+DR **\dir{-};"e"+UR **\dir{-},"i"')
    assert ccq.multigate_macro(5, 'foo') == (
        '*+<1em,.9em>{\hphantom{foo}} \POS [0,0]="i",[0,0].[4,0]="e",'
        '!C *{foo},"e"+UR;"e"+UL **\dir{-};"e"+DL **\dir{-};'
        '"e"+DR **\dir{-};"e"+UR **\dir{-},"i"')
