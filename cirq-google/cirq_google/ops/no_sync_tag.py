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

"""A class that can be used to denote an operation that ignores moment-based synchronization."""

from __future__ import annotations

from typing import Any

import cirq
from cirq_google.api.v2 import program_pb2


class NoSyncTag:
    """A tag class to direct hardware to ignore moment-based synchronization.

    Directs the hardware to ignore moment-based synchronization and to instead
    schedule operations as soon as possible for these qubits.

    Args:
        reverse: Number of synchronizations before the operation to remove.
        forward: Number of synchronizations after the operation to remove.
        remove_all_syncs_before: Remove all possible synchronizations before the operation.
        remove_all_syncs_after: Remove all possible synchronizations after the operation.
    """

    def __init__(
        self,
        reverse: int | None = None,
        forward: int | None = None,
        *,
        remove_all_syncs_before: bool = False,
        remove_all_syncs_after: bool = False,
    ):
        if reverse is not None and remove_all_syncs_before:
            raise ValueError("Cannot specify both reverse and remove_all_syncs_before")
        if forward is not None and remove_all_syncs_after:
            raise ValueError("Cannot specify both forward and remove_all_syncs_after")
        if reverse is not None and reverse < 0:
            raise ValueError(f"reverse must be non-negative, got {reverse}")
        if forward is not None and forward < 0:
            raise ValueError(f"forward must be non-negative, got {forward}")
        self.reverse = reverse
        self.forward = forward
        self.remove_all_syncs_before = remove_all_syncs_before
        self.remove_all_syncs_after = remove_all_syncs_after

    def __str__(self) -> str:
        args = []
        if self.reverse is not None:
            args.append(f'reverse={self.reverse}')
        if self.forward is not None:
            args.append(f'forward={self.forward}')
        if self.remove_all_syncs_before:
            args.append(f'remove_all_syncs_before={self.remove_all_syncs_before}')
        if self.remove_all_syncs_after:
            args.append(f'remove_all_syncs_after={self.remove_all_syncs_after}')
        return f"NoSyncTag({', '.join(args)})"

    def __repr__(self) -> str:
        args = []
        if self.reverse is not None:
            args.append(f'reverse={self.reverse!r}')
        if self.forward is not None:
            args.append(f'forward={self.forward!r}')
        if self.remove_all_syncs_before:
            args.append(f'remove_all_syncs_before={self.remove_all_syncs_before!r}')
        if self.remove_all_syncs_after:
            args.append(f'remove_all_syncs_after={self.remove_all_syncs_after!r}')
        return f"cirq_google.NoSyncTag({', '.join(args)})"

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(
            self,
            [
                'reverse',
                'forward',
                'remove_all_syncs_before',
                'remove_all_syncs_after',
            ],
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NoSyncTag):
            return NotImplemented
        return (
            self.reverse == other.reverse
            and self.forward == other.forward
            and self.remove_all_syncs_before == other.remove_all_syncs_before
            and self.remove_all_syncs_after == other.remove_all_syncs_after
        )

    def __hash__(self) -> int:
        return hash((
            self.reverse,
            self.forward,
            self.remove_all_syncs_before,
            self.remove_all_syncs_after,
        ))

    def to_proto(
        self, msg: program_pb2.Tag | program_pb2.NoSyncTag | None = None
    ) -> program_pb2.Tag | program_pb2.NoSyncTag:
        if msg is None:
            msg = program_pb2.Tag()
        if isinstance(msg, program_pb2.NoSyncTag):
            no_sync_msg = msg
        else:
            no_sync_msg = msg.no_sync

        if self.remove_all_syncs_before:
            no_sync_msg.remove_all_syncs_before = True
        elif self.reverse is not None:
            no_sync_msg.reverse = self.reverse

        if self.remove_all_syncs_after:
            no_sync_msg.remove_all_syncs_after = True
        elif self.forward is not None:
            no_sync_msg.forward = self.forward

        if (
            self.reverse is None
            and not self.remove_all_syncs_before
            and self.forward is None
            and not self.remove_all_syncs_after
            and isinstance(msg, program_pb2.Tag)
        ):
            msg.no_sync.SetInParent()
        return msg

    @staticmethod
    def from_proto(msg: program_pb2.Tag | program_pb2.NoSyncTag) -> NoSyncTag:
        if isinstance(msg, program_pb2.Tag):
            if msg.WhichOneof("tag") != "no_sync":
                raise ValueError(f"Message is not a NoSyncTag, {msg}")
            no_sync = msg.no_sync
        elif isinstance(msg, program_pb2.NoSyncTag):
            no_sync = msg
        else:
            raise ValueError(f"Expected Tag or NoSyncTag, got {type(msg)}")

        reverse = None
        remove_all_syncs_before = False
        rev_which = no_sync.WhichOneof("rev")
        if rev_which == "reverse":
            reverse = no_sync.reverse
        elif rev_which == "remove_all_syncs_before":
            remove_all_syncs_before = no_sync.remove_all_syncs_before

        forward = None
        remove_all_syncs_after = False
        fwd_which = no_sync.WhichOneof("fwd")
        if fwd_which == "forward":
            forward = no_sync.forward
        elif fwd_which == "remove_all_syncs_after":
            remove_all_syncs_after = no_sync.remove_all_syncs_after

        return NoSyncTag(
            reverse=reverse,
            forward=forward,
            remove_all_syncs_before=remove_all_syncs_before,
            remove_all_syncs_after=remove_all_syncs_after,
        )
