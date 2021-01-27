# Copyright 2020 The Cirq Developers
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


def test_equality():
    assert cirq.google.PhysicalZTag() == cirq.google.PhysicalZTag()
    assert hash(cirq.google.PhysicalZTag()) == hash(cirq.google.PhysicalZTag())


def test_syc_str_repr():
    assert str(cirq.google.PhysicalZTag()) == 'PhysicalZTag()'
    assert repr(cirq.google.PhysicalZTag()) == 'cirq.google.PhysicalZTag()'
    cirq.testing.assert_equivalent_repr(cirq.google.PhysicalZTag(), setup_code=('import cirq\n'))
