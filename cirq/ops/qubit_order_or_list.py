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

"""Any method taking a QubitOrder instance should also take raw qubit lists.

This type is defined in its own file to work around an "invalid type" bug in
mypy.
"""

from typing import Iterable, Union

from cirq._doc import document
from cirq.ops import qubit_order, raw_types

QubitOrderOrList = Union[qubit_order.QubitOrder, Iterable[raw_types.Qid]]
document(
    QubitOrderOrList,  # type: ignore
    """Specifies a qubit ordering.

    The ordering can either be specified by an iterable (such as a list) with
    the qubits in the desired order, or by a `cirq.QubitOrder` object.
    """,
)
