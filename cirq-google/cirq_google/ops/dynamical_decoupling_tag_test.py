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

from cirq_google.api.v2 import program_pb2
from cirq_google.ops.dynamical_decoupling_tag import DynamicalDecouplingTag


def test_invalid_value():
    with pytest.raises(AssertionError):
        _ = DynamicalDecouplingTag('_Z')


def test_proto_serialization():
    tag = DynamicalDecouplingTag('X')
    msg = tag.to_proto()
    assert tag == DynamicalDecouplingTag.from_proto(msg)

    with pytest.raises(ValueError, match="Message is not a DynamicalDecouplingTag"):
        msg = program_pb2.Tag()
        msg.fsim_via_model.SetInParent()
        DynamicalDecouplingTag.from_proto(msg)
