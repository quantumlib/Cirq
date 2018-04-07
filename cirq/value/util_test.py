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

from cirq.value import sorting_str


def test_sorting_str():
    assert sorting_str('') == ''
    assert sorting_str('a') == 'a'
    assert sorting_str('a0') == 'a00000000:1'
    assert sorting_str('a00') == 'a00000000:2'
    assert sorting_str('a1bc23') == 'a00000001:1bc00000023:2'
    assert sorting_str('a9') == 'a00000009:1'
    assert sorting_str('a09') == 'a00000009:2'
    assert sorting_str('a00000000:8') == 'a00000000:8:00000008:1'


def test_sorted_by_sorting_str():
    actual = [
        '',
        '1',
        'a',
        'a00000000',
        'a00000000:8',
        'a9',
        'a09',
        'a10',
        'a11',
    ]
    assert sorted(actual, key=sorting_str) == actual
    assert sorted(reversed(actual), key=sorting_str) == actual
