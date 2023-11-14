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

from typing import Any

from cirq import protocols


def assert_specifies_has_unitary_if_unitary(val: Any) -> None:
    """Checks that unitary values can be cheaply identifies as unitary."""

    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    assert not protocols.has_unitary(val) or hasattr(val, '_has_unitary_'), (
        "Value is unitary but doesn't specify a _has_unitary_ method that "
        "can be used to cheaply verify this fact.\n"
        "\n"
        f"val: {val!r}"
    )
