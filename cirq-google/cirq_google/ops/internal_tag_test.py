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

import cirq_google
import pytest

from cirq_google.api.v2 import program_pb2


def test_internal_tag():
    g = cirq_google.InternalTag(
        name="IdleWaitTag",
        package='internal_module',
        phase_match=True,
        pad_ns=10,
        pad_start=True,
        pad_end=False,
    )
    assert (
        str(g) == "internal_module.IdleWaitTag("
        "phase_match=True, pad_ns=10, pad_start=True, pad_end=False)"
    )
    assert (
        repr(g) == "cirq_google.InternalTag(name='IdleWaitTag', "
        "package='internal_module', phase_match=True, "
        "pad_ns=10, pad_start=True, pad_end=False)"
    )


def test_internal_tag_with_no_args():
    g = cirq_google.InternalTag(name="TagWithNoArgs", package='test')
    assert str(g) == 'test.TagWithNoArgs()'
    assert repr(g) == "cirq_google.InternalTag(name='TagWithNoArgs', package='test')"


def test_internal_tag_with_hashable_args_is_hashable():
    hashable = cirq_google.InternalTag(
        name="TagWithHashableArgs", package='test', foo=1, bar="2", baz=(("a", 1),)
    )
    _ = hash(hashable)

    unhashable = cirq_google.InternalTag(
        name="TagWithUnHashableArgs", package='test', foo=1, bar="2", baz={"a": 1}
    )
    with pytest.raises(TypeError, match="unhashable"):
        _ = hash(unhashable)


def test_proto():
    tag = cirq_google.InternalTag(name="TagWithNoParams", package='test', param1=2.5)
    msg = tag.to_proto()
    assert tag == cirq_google.InternalTag.from_proto(msg)

    with pytest.raises(ValueError, match="Message is not a InternalTag"):
        msg = program_pb2.Tag()
        msg.fsim_via_model.SetInParent()
        cirq_google.InternalTag.from_proto(msg)
