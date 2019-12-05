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

from typing import cast, Any, Optional, Union

import numpy as np

RANDOM_STATE_LIKE = Optional[Union[np.random.RandomState, int, Any]]


def parse_random_state(random_state: RANDOM_STATE_LIKE
                      ) -> np.random.RandomState:
    """Interpret an object as the state of a pseudorandom number generator.

    If `random_state` is None, returns the module `np.random`.
    If `random_state` is an integer, returns
    `np.random.RandomState(random_state)`.
    Otherwise, returns `random_state` unmodified.
    """
    if random_state is None:
        return cast(np.random.RandomState, np.random)
    elif isinstance(random_state, int):
        return np.random.RandomState(random_state)
    else:
        return cast(np.random.RandomState, random_state)
