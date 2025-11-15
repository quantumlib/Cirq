# Copyright 2025 The Cirq Developers
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

import cirq
import cirq_google


def test_convert_to_zip_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        cirq_google.study.convert_to_zip([])


def test_convert_to_zip_mismatched_keys() -> None:
    with pytest.raises(ValueError, match="Keys must be the same"):
        cirq_google.study.convert_to_zip([{'a': 4.0}, {'a': 2.0, 'b': 1.0}])


def test_convert_to_zip() -> None:
    param_dict = [
        {'a': 1.0, 'b': 2.0, 'c': 10.0},
        {'a': 2.0, 'b': 4.0, 'c': 9.0},
        {'a': 3.0, 'b': 8.0, 'c': 8.0},
    ]
    param_zip = cirq.Zip(
        cirq.Points('a', [1.0, 2.0, 3.0]),
        cirq.Points('b', [2.0, 4.0, 8.0]),
        cirq.Points('c', [10.0, 9.0, 8.0]),
    )
    assert cirq_google.study.convert_to_zip(param_dict) == param_zip
