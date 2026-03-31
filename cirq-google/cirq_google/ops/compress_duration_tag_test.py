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

from __future__ import annotations

import cirq
import cirq_google


def test_equality():
    assert cirq_google.CompressDurationTag() == cirq_google.CompressDurationTag()
    assert hash(cirq_google.CompressDurationTag()) == hash(cirq_google.CompressDurationTag())


def test_syc_str_repr():
    assert str(cirq_google.CompressDurationTag()) == 'CompressDurationTag()'
    assert repr(cirq_google.CompressDurationTag()) == 'cirq_google.CompressDurationTag()'
    cirq.testing.assert_equivalent_repr(
        cirq_google.CompressDurationTag(), setup_code=('import cirq\nimport cirq_google\n')
    )


def test_proto():
    tag = cirq_google.CompressDurationTag()
    msg = tag.to_proto()
    assert tag == cirq_google.CompressDurationTag.from_proto(msg)
