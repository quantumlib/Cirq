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
    CELL_SIZES)


def generate_all_ignored_cells() -> Iterator[CellMaker]:
    # Spacer.
    yield reg_ignored_gate("…")

    # Displays.
    yield reg_ignored_gate("Bloch")
    yield from reg_ignored_family("Amps")
    yield from reg_ignored_family("Chance")
    yield from reg_ignored_family("Sample")
    yield from reg_ignored_family("Density")


def reg_ignored_family(identifier_prefix: str) -> Iterator[CellMaker]:
    yield reg_ignored_gate(identifier_prefix)
    for i in CELL_SIZES:
        yield reg_ignored_gate(identifier_prefix + str(i))


def reg_ignored_gate(identifier: str) -> CellMaker:
    return CellMaker(identifier, 0, lambda _: None)
