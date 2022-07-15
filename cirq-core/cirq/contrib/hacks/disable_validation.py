# Copyright 2022 The Cirq Developers
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

"""Tools for disabling validation in circuit construction."""

import contextlib


@contextlib.contextmanager
def disable_op_validation(*, accept_debug_responsibility: bool = False):
    if not accept_debug_responsibility:
        raise ValueError(
            "WARNING! Using disable_op_validation with invalid ops can cause "
            "mysterious and terrible things to happen. cirq-maintainers will "
            "not help you debug beyond this point!\n"
            "If you still wish to continue, call this method with "
            "accept_debug_responsibility=True."
        )

    from cirq.ops import raw_types

    temp = raw_types._validate_qid_shape
    raw_types._validate_qid_shape = lambda *args: None
    try:
        yield None
        # ...run context...
    finally:
        raw_types._validate_qid_shape = temp
