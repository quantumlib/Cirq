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

import attrs

import cirq
from cirq_google.api.v2 import program_pb2


@attrs.frozen(kw_only=True)
class NoSyncTag:
    """A tag class to direct hardware to ignore moment-based synchronization.

    Directs the hardware to ignore moment-based synchronization and to instead
    schedule operations as soon as possible for these qubits.

    Args:
        reverse: Number of synchronizations before the operation to remove.
            Mutually exclusive with remove_all_syncs_before.
        remove_all_syncs_before: Remove all possible synchronizations before the operation.
            Mutually exclusive with reverse.
        forward: Number of synchronizations after the operation to remove.
            Mutually exclusive with remove_all_syncs_after.
        remove_all_syncs_after: Remove all possible synchronizations after the operation.
            Mutually exclusive with forward.
    """

    reverse: int = 0
    remove_all_syncs_before: bool = False
    forward: int = 0
    remove_all_syncs_after: bool = False

    def __attrs_post_init__(self):
        if self.reverse and self.remove_all_syncs_before:
            raise ValueError("Cannot specify both reverse and remove_all_syncs_before")
        if self.forward and self.remove_all_syncs_after:
            raise ValueError("Cannot specify both forward and remove_all_syncs_after")
        if self.reverse < 0:
            raise ValueError(f"reverse must be non-negative, got {self.reverse}")
        if self.forward < 0:
            raise ValueError(f"forward must be non-negative, got {self.forward}")

    def __str__(self) -> str:
        args = []
        if self.reverse:
            args.append(f'reverse={self.reverse}')
        if self.remove_all_syncs_before:
            args.append(f'remove_all_syncs_before={self.remove_all_syncs_before}')
        if self.forward:
            args.append(f'forward={self.forward}')
        if self.remove_all_syncs_after:
            args.append(f'remove_all_syncs_after={self.remove_all_syncs_after}')
        return f"NoSyncTag({', '.join(args)})"

    def __repr__(self) -> str:
        args = []
        if self.reverse:
            args.append(f'reverse={self.reverse!r}')
        if self.remove_all_syncs_before:
            args.append(f'remove_all_syncs_before={self.remove_all_syncs_before!r}')
        if self.forward:
            args.append(f'forward={self.forward!r}')
        if self.remove_all_syncs_after:
            args.append(f'remove_all_syncs_after={self.remove_all_syncs_after!r}')
        return f"cirq_google.NoSyncTag({', '.join(args)})"

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(
            self, ['reverse', 'remove_all_syncs_before', 'forward', 'remove_all_syncs_after']
        )

    def to_proto(self, msg: program_pb2.Tag | None = None) -> program_pb2.Tag:
        if msg is None:
            msg = program_pb2.Tag()
        msg.no_sync.SetInParent()
        msg.no_sync.Clear()
        if self.reverse:
            msg.no_sync.reverse = self.reverse
        if self.remove_all_syncs_before:
            msg.no_sync.remove_all_syncs_before = True
        if self.forward:
            msg.no_sync.forward = self.forward
        if self.remove_all_syncs_after:
            msg.no_sync.remove_all_syncs_after = True
        return msg

    @staticmethod
    def from_proto(msg: program_pb2.Tag) -> NoSyncTag:
        if msg.WhichOneof("tag") != "no_sync":
            raise ValueError(f"Message is not a NoSyncTag, {msg}")
        return NoSyncTag(
            reverse=msg.no_sync.reverse,
            forward=msg.no_sync.forward,
            remove_all_syncs_before=msg.no_sync.remove_all_syncs_before,
            remove_all_syncs_after=msg.no_sync.remove_all_syncs_after,
        )
