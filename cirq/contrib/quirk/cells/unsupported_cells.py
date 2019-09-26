# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Iterator

from cirq.contrib.quirk.cells.cell import (
    CellMaker,
    CELL_SIZES,)


def generate_all_unsupported_cells() -> Iterator[CellMaker]:
    # Post selection.
    yield from reg_unsupported_gates(
        "|0⟩⟨0|",
        "|1⟩⟨1|",
        "|+⟩⟨+|",
        "|-⟩⟨-|",
        "|X⟩⟨X|",
        "|/⟩⟨/|",
        "0",
        reason='postselection is not implemented in Cirq')

    # Non-physical operations.
    yield from reg_unsupported_gates("__error__",
                                     "__unstable__UniversalNot",
                                     reason="Unphysical operation.")

    # Measurement.
    yield from reg_unsupported_gates(
        "XDetectControlReset",
        "YDetectControlReset",
        "ZDetectControlReset",
        reason="Classical feedback is not implemented in Cirq")

    # Dynamic gates with discretized actions.
    yield from reg_unsupported_gates("X^⌈t⌉",
                                     "X^⌈t-¼⌉",
                                     reason="discrete parameter")
    yield from reg_unsupported_family("Counting", reason="discrete parameter")
    yield from reg_unsupported_family("Uncounting", reason="discrete parameter")
    yield from reg_unsupported_family(">>t", reason="discrete parameter")
    yield from reg_unsupported_family("<<t", reason="discrete parameter")

    # Gates that are no longer in the toolbox and have dominant replacements.
    yield from reg_unsupported_family("add",
                                      reason="deprecated; use +=A instead")
    yield from reg_unsupported_family("sub",
                                      reason="deprecated; use -=A instead")
    yield from reg_unsupported_family("c+=ab",
                                      reason="deprecated; use +=AB instead")
    yield from reg_unsupported_family("c-=ab",
                                      reason="deprecated; use -=AB instead")


def reg_unsupported_gate(identifier: str, reason: str) -> Iterator[CellMaker]:

    def fail(_):
        raise NotImplementedError(
            f'Converting the Quirk gate {identifier} is not implemented yet. '
            f'Reason: {reason}')

    yield CellMaker(identifier, 0, fail)


def reg_unsupported_gates(*identifiers: str, reason: str) -> Iterator[CellMaker]:
    for identifier in identifiers:
        yield from reg_unsupported_gate(identifier, reason)


def reg_unsupported_family(identifier_prefix: str,
                           reason: str) -> Iterator[CellMaker]:
    for i in CELL_SIZES:
        yield from reg_unsupported_gate(identifier_prefix + str(i), reason)
