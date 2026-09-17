# Copyright 2026 The Cirq Developers
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

import pytest

import cirq
import cirq_google
from cirq_google.api.v2 import program_pb2


def test_equality() -> None:
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq_google.NoSyncTag(), cirq_google.NoSyncTag())
    eq.add_equality_group(
        cirq_google.NoSyncTag(reverse=0, forward=1), cirq_google.NoSyncTag(forward=1)
    )
    eq.add_equality_group(cirq_google.NoSyncTag(reverse=1, forward=1))
    eq.add_equality_group(cirq_google.NoSyncTag(reverse=0, forward=2))
    eq.add_equality_group(cirq_google.NoSyncTag(remove_all_syncs_before=True))
    eq.add_equality_group(cirq_google.NoSyncTag(remove_all_syncs_after=True))
    eq.add_equality_group(
        cirq_google.NoSyncTag(remove_all_syncs_before=True, remove_all_syncs_after=True)
    )


def test_invalid_args() -> None:
    with pytest.raises(ValueError, match="Cannot specify both reverse and remove_all_syncs_before"):
        _ = cirq_google.NoSyncTag(reverse=1, remove_all_syncs_before=True)

    with pytest.raises(ValueError, match="Cannot specify both forward and remove_all_syncs_after"):
        _ = cirq_google.NoSyncTag(forward=1, remove_all_syncs_after=True)

    with pytest.raises(ValueError, match="reverse must be non-negative"):
        _ = cirq_google.NoSyncTag(reverse=-1)

    with pytest.raises(ValueError, match="forward must be non-negative"):
        _ = cirq_google.NoSyncTag(forward=-1)

    with pytest.raises(TypeError):
        _ = cirq_google.NoSyncTag(1)  # type: ignore[misc]


def test_str_repr() -> None:
    assert str(cirq_google.NoSyncTag()) == 'NoSyncTag()'
    assert repr(cirq_google.NoSyncTag()) == 'cirq_google.NoSyncTag()'
    cirq.testing.assert_equivalent_repr(cirq_google.NoSyncTag(), setup_code='import cirq_google')

    tag = cirq_google.NoSyncTag(reverse=2, forward=1)
    assert str(tag) == 'NoSyncTag(reverse=2, forward=1)'
    assert repr(tag) == 'cirq_google.NoSyncTag(reverse=2, forward=1)'
    cirq.testing.assert_equivalent_repr(tag, setup_code='import cirq_google')

    tag_bool = cirq_google.NoSyncTag(remove_all_syncs_before=True, remove_all_syncs_after=True)
    assert str(tag_bool) == 'NoSyncTag(remove_all_syncs_before=True, remove_all_syncs_after=True)'
    assert (
        repr(tag_bool)
        == 'cirq_google.NoSyncTag(remove_all_syncs_before=True, remove_all_syncs_after=True)'
    )
    cirq.testing.assert_equivalent_repr(tag_bool, setup_code='import cirq_google')


def test_proto() -> None:
    # Empty tag
    tag = cirq_google.NoSyncTag()
    msg = tag.to_proto()
    assert msg.WhichOneof('tag') == 'no_sync'
    assert tag == cirq_google.NoSyncTag.from_proto(msg)

    # Existing message passed to to_proto
    existing_msg = program_pb2.Tag()
    assert tag.to_proto(existing_msg) is existing_msg
    assert tag == cirq_google.NoSyncTag.from_proto(existing_msg)

    # Tag with reverse=2, forward=1
    tag_with_fields = cirq_google.NoSyncTag(reverse=2, forward=1)
    msg_fields = tag_with_fields.to_proto()
    assert msg_fields.no_sync.WhichOneof('rev') == 'reverse'
    assert msg_fields.no_sync.reverse == 2
    assert msg_fields.no_sync.WhichOneof('fwd') == 'forward'
    assert msg_fields.no_sync.forward == 1
    assert tag_with_fields == cirq_google.NoSyncTag.from_proto(msg_fields)

    # from_proto directly with manually created proto matching "reverse: 2, forward: 1"
    raw_tag_msg = program_pb2.Tag()
    raw_tag_msg.no_sync.reverse = 2
    raw_tag_msg.no_sync.forward = 1
    deserialized = cirq_google.NoSyncTag.from_proto(raw_tag_msg)
    assert deserialized.reverse == 2
    assert deserialized.forward == 1
    assert deserialized == tag_with_fields

    # Tag with remove_all_syncs
    tag_all_syncs = cirq_google.NoSyncTag(remove_all_syncs_before=True, remove_all_syncs_after=True)
    msg_all_syncs = tag_all_syncs.to_proto()
    assert msg_all_syncs.no_sync.WhichOneof('rev') == 'remove_all_syncs_before'
    assert msg_all_syncs.no_sync.remove_all_syncs_before is True
    assert msg_all_syncs.no_sync.WhichOneof('fwd') == 'remove_all_syncs_after'
    assert msg_all_syncs.no_sync.remove_all_syncs_after is True
    assert tag_all_syncs == cirq_google.NoSyncTag.from_proto(msg_all_syncs)

    with pytest.raises(ValueError, match="Message is not a NoSyncTag"):
        msg_wrong = program_pb2.Tag()
        msg_wrong.internal_tag.SetInParent()
        cirq_google.NoSyncTag.from_proto(msg_wrong)
