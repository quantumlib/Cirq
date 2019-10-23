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

from typing import cast, Optional, Union

import numpy as np

RANDOM_STATE_LIKE = Optional[Union[np.random.RandomState, int]]


def parse_random_state(random_state: RANDOM_STATE_LIKE
                      ) -> np.random.RandomState:
    if random_state is None:
        return cast(np.random.RandomState, np.random)
    elif (isinstance(random_state, np.random.RandomState) or
          random_state == np.random):
        return cast(np.random.RandomState, random_state)
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state)
    raise TypeError(f'Argument must be of type cirq.value.RANDOM_STATE_LIKE.')
