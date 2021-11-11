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

from typing import (
    Union,
    Sequence,
    TYPE_CHECKING,
)

import numpy as np
from typing_extensions import Protocol

from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


class SupportsApplyToTableau(Protocol):
    @doc_private
    def _apply_to_tableau_(
        self, tableau: 'cirq.CliffordTableau', axes: Sequence[int], prng: np.random.RandomState
    ) -> Union[bool, NotImplementedType]:
        """Write me!"""


def apply_to_tableau(
    val: 'cirq.Gate',
    tableau: 'cirq.CliffordTableau',
    axes: Sequence[int],
    prng: np.random.RandomState,
) -> Union[bool, NotImplementedType]:
    getter = getattr(val, '_apply_to_tableau_', None)
    return NotImplemented if getter is None else getter(tableau, axes, prng)


class SupportsApplyToChForm(Protocol):
    @doc_private
    def _apply_to_ch_form_(
        self, state: 'cirq.StabilizerStateChForm', axes: Sequence[int], prng: np.random.RandomState
    ) -> Union[bool, NotImplementedType]:
        """Write me!"""


def apply_to_ch_form(
    val: 'cirq.Gate',
    state: 'cirq.StabilizerStateChForm',
    axes: Sequence[int],
    prng: np.random.RandomState,
) -> Union[bool, NotImplementedType]:
    getter = getattr(val, '_apply_to_ch_form_', None)
    return NotImplemented if getter is None else getter(state, axes, prng)
