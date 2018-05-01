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

from cirq.google import XmonQubit


def longest_sequence_index(sequences: List[List[XmonQubit]]) -> int:
  """Gives the position of a longest sequence.

  Args:
    sequences: List of node sequences.

  Returns:
    Index of the longest sequence from the sequences list. If more than one
    longest sequence exist, the first one is returned. None is returned for
    empty list.
  """
  if sequences:
    return max(range(len(sequences)), key=lambda i: len(sequences[i]))
