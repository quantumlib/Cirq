# Copyright 2018 The Cirq Developers
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
import abc
from typing import TYPE_CHECKING

from cirq_google.line.placement.sequence import GridQubitLineTuple

if TYPE_CHECKING:
    import cirq_google


class LinePlacementStrategy(metaclass=abc.ABCMeta):
    """Choice and options for the line placement calculation method.

    Currently two methods are available: cirq.line.GreedySequenceSearchMethod
    and cirq.line.AnnealSequenceSearchMethod.
    """

    @abc.abstractmethod
    def place_line(self, device: 'cirq_google.GridDevice', length: int) -> GridQubitLineTuple:
        """Runs line sequence search.

        Args:
            device: Chip description.
            length: Required line length.

        Returns:
            Linear sequences found on the chip.
        """
