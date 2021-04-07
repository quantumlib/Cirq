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

from cirq import ops


def assert_equivalent_op_tree(x: ops.OP_TREE, y: ops.OP_TREE):
    """Ensures that the two OP_TREEs are equivalent.

    Args:
        x: OP_TREE one
        y: OP_TREE two
    Returns:
        None
    Raises:
         AssertionError if x != y
    """

    a = list(ops.flatten_op_tree(x))
    b = list(ops.flatten_op_tree(y))
    assert a == b
