# Copyright 2024 The Cirq Developers
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

from cirq_google.api.v2 import program_pb2


def test_equality():
    assert cirq_google.FSimViaModelTag() == cirq_google.FSimViaModelTag()
    assert hash(cirq_google.FSimViaModelTag()) == hash(cirq_google.FSimViaModelTag())


def test_str_repr():
    assert str(cirq_google.FSimViaModelTag()) == 'FSimViaModelTag()'
    assert repr(cirq_google.FSimViaModelTag()) == 'cirq_google.FSimViaModelTag()'
    cirq.testing.assert_equivalent_repr(
        cirq_google.FSimViaModelTag(), setup_code=('import cirq\nimport cirq_google\n')
    )


def test_proto():
    tag = cirq_google.FSimViaModelTag()
    msg = tag.to_proto()
    assert tag == cirq_google.FSimViaModelTag.from_proto(msg)

    with pytest.raises(ValueError, match="Message is not a FSimViaModelTag"):
        msg = program_pb2.Tag()
        msg.internal_tag.SetInParent()
        cirq_google.FSimViaModelTag.from_proto(msg)
