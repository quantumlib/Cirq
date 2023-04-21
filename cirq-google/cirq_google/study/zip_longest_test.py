# Copyright 2023 The Cirq Developers
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
import cirq
import cirq_google as cg


def test_zip_longest():
    sweep = cg.ZipLongest(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6, 7]))
    assert len(sweep) == 4
    assert tuple(sweep.param_tuples()) == (
        (('a', 1), ('b', 4)),
        (('a', 2), ('b', 5)),
        (('a', 3), ('b', 6)),
        (('a', 3), ('b', 7)),
    )
    assert sweep.keys == ['a', 'b']
    assert (
        str(sweep) == 'ZipLongest(cirq.Points(\'a\', [1, 2, 3]), cirq.Points(\'b\', [4, 5, 6, 7]))'
    )
    assert (
        repr(sweep)
        == 'cirq_google.ZipLongest(cirq.Points(\'a\', [1, 2, 3]), cirq.Points(\'b\', [4, 5, 6, 7]))'
    )


def test_empty_zip():
    assert len(cg.ZipLongest()) == 0


def test_zip_eq():
    sweep1 = cg.ZipLongest(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6, 7]))
    sweep2 = cg.ZipLongest(cirq.Points('a', [1, 2, 3]), cirq.Points('b', [4, 5, 6, 7]))
    sweep3 = cg.ZipLongest(cirq.Points('a', [1, 2]), cirq.Points('b', [4, 5, 6, 7]))
    sweep4 = cirq.Zip(cirq.Points('a', [1, 2]), cirq.Points('b', [4, 5, 6, 7]))

    assert sweep1 == sweep2
    assert hash(sweep1) == hash(sweep2)
    assert sweep2 != sweep3
    assert hash(sweep2) != hash(sweep3)
    assert sweep1 != sweep4
    assert hash(sweep1) != hash(sweep4)
