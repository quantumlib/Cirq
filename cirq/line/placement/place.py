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

from typing import List

from cirq.google import XmonDevice, XmonQubit
from cirq.line.placement import anneal
from cirq.line.placement import greedy
from cirq.line.placement.search_method import SequenceSearchMethod


class LinePlacementOptions:
    """Options to control line placement calculation.
    """

    def __init__(self,
                 search_method: SequenceSearchMethod =
                 greedy.GreedySequenceSearchMethod(),
                 seed=None):
        """Creates options to control line placement calculation.

        Args:
            search_method: Search method to use. Currently two methods are
                           available: cirq.line.GreedySequenceSearchMethod and
                           cirq.line.AnnealSequenceSearchMethod.
            seed: Seed for the random number generator used during the search.
        """
        self.search_method = search_method
        self.seed = seed


def place_on_device(device: XmonDevice,
                    options: LinePlacementOptions = LinePlacementOptions()) \
        -> List[List[XmonQubit]]:
    """Searches for linear sequence of qubits on device.

    Args:
        device: Google Xmon device instance.
        options: Line placement search options.

    Returns:
        List of sequences found.

    Raises:
        ValueError: When unknown search method is requested.
    """
    if isinstance(options.search_method, greedy.GreedySequenceSearchMethod):
        return greedy.greedy_sequence(device, options.search_method)
    elif isinstance(options.search_method, anneal.AnnealSequenceSearchMethod):
        return anneal.anneal_sequence(device, options.search_method,
                                      seed=options.seed)
    else:
        raise ValueError(
            "Unknown linear sequence search method '%s'" %
            options.search_method)
