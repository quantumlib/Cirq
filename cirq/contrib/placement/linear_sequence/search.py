# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

from cirq.google import XmonDevice, XmonQubit
from cirq.contrib.placement.linear_sequence import anneal
from cirq.contrib.placement.linear_sequence import greedy


def search_sequence(device: XmonDevice,
                    method: str,
                    method_opts: dict = None,
                    seed: int = None) -> List[List[XmonQubit]]:
    """Searches for linear sequence of nodes on the grid.

    Args:
      device: Google Xmon device instance.
      method: Search method to use. Currently only 'greedy' method is supported.
      method_opts: Search method specific options.
      seed: Seed for the random number generator used during the search. Not yet
            implemented.

    Returns:
      Future that gives a list of sequences found upon completion.

    Raises:
      ValueError: When unknown search method is requested.
    """
    if method == 'greedy':
        return greedy.greedy_sequence(device, method_opts)
    elif method == 'anneal':
        return anneal.anneal_sequence(device, method_opts, seed=seed)
    else:
        raise ValueError("Unknown linear sequence search method '%s'" % method)
