# Copyright 2019 The Cirq Developers
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
from typing import Iterator

from cirq.interop.quirk.cells.cell import (
    CellMaker,
    CELL_SIZES,
)


def generate_all_unsupported_cell_makers() -> Iterator[CellMaker]:
    # Post selection.
    yield from _unsupported_gates(
        "|0⟩⟨0|",
        "|1⟩⟨1|",
        "|+⟩⟨+|",
        "|-⟩⟨-|",
        "|X⟩⟨X|",
        "|/⟩⟨/|",
        "0",
        reason='postselection is not implemented in Cirq',
    )

    # Non-physical operations.
    yield from _unsupported_gates(
        "__error__", "__unstable__UniversalNot", reason="unphysical operation."
    )

    # Measurement.
    yield from _unsupported_gates(
        "XDetectControlReset",
        "YDetectControlReset",
        "ZDetectControlReset",
        reason="classical feedback is not implemented in Cirq.",
    )

    # Dynamic gates with discretized actions.
    yield from _unsupported_gates("X^⌈t⌉", "X^⌈t-¼⌉", reason="discrete parameter")
    yield from _unsupported_family("Counting", reason="discrete parameter")
    yield from _unsupported_family("Uncounting", reason="discrete parameter")
    yield from _unsupported_family(">>t", reason="discrete parameter")
    yield from _unsupported_family("<<t", reason="discrete parameter")

    # Gates that are no longer in the toolbox and have dominant replacements.
    yield from _unsupported_family("add", reason="deprecated; use +=A instead")
    yield from _unsupported_family("sub", reason="deprecated; use -=A instead")
    yield from _unsupported_family("c+=ab", reason="deprecated; use +=AB instead")
    yield from _unsupported_family("c-=ab", reason="deprecated; use -=AB instead")


def _unsupported_gate(identifier: str, reason: str) -> CellMaker:
    def fail(_):
        raise NotImplementedError(
            f'Converting the Quirk gate {identifier} is not implemented yet. Reason: {reason}'
        )

    return CellMaker(identifier, 0, fail)


def _unsupported_gates(*identifiers: str, reason: str) -> Iterator[CellMaker]:
    for identifier in identifiers:
        yield _unsupported_gate(identifier, reason)


def _unsupported_family(identifier_prefix: str, reason: str) -> Iterator[CellMaker]:
    for i in CELL_SIZES:
        yield _unsupported_gate(identifier_prefix + str(i), reason)
