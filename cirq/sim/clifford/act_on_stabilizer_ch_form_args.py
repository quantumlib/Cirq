# Copyright 2020 The Cirq Developers
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

from typing import Iterable

from cirq.sim.clifford.stabilizer_state_ch_form import StabilizerStateChForm


class ActOnStabilizerCHFormArgs:
    """Wrapper around a stabilizer state in CH form for the act_on protocol.

    To act on this object, directly edit the `state` property, which is
    storing the stabilizer state of the quantum system with one axis per qubit.
    Measurements are currently not supported on this object.
    """

    def __init__(self, state: StabilizerStateChForm, axes: Iterable[int]):
        """Initializes with the given state and the axes for the operation.

        Args:
            state: The StabilizerStateChForm to act on. Operations are expected
                to perform inplace edits of this object.
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
        """
        self.state = state
        self.axes = tuple(axes)
