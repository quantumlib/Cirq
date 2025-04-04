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

from cirq.interop.quirk.cells.cell import CELL_SIZES, CellMaker


def generate_all_ignored_cell_makers() -> Iterator[CellMaker]:
    # Spacer.
    yield _ignored_gate("…")

    # Displays.
    yield _ignored_gate("Bloch")
    yield from _ignored_family("Amps")
    yield from _ignored_family("Chance")
    yield from _ignored_family("Sample")
    yield from _ignored_family("Density")


def _ignored_family(identifier_prefix: str) -> Iterator[CellMaker]:
    yield _ignored_gate(identifier_prefix)
    for i in CELL_SIZES:
        yield _ignored_gate(identifier_prefix + str(i))


def _ignored_gate(identifier: str) -> CellMaker:
    # No matter the arguments (qubit, position, etc), map to nothing.
    return CellMaker(identifier, size=0, maker=lambda _: None)
