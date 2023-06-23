# Copyright 2023 The Cirq Developers
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

import abc
from cirq_ft.algos import select_and_prepare


class RandomVariableEncoder(select_and_prepare.SelectOracle):
    r"""Abstract base class that defines the API for a Random Variable Encoder.

    This class extends the SELECT Oracle and adds two additional properties:
    target_bitsize_before_decimal and target_bitsize_after_decimal. These variables specify
    the number of bits of precision before and after the decimal when encoding random variables
    in registers.
    """

    @property
    @abc.abstractmethod
    def target_bitsize_before_decimal(self) -> int:
        """Returns the number of bits before the decimal point in the target register."""
        ...

    @property
    @abc.abstractmethod
    def target_bitsize_after_decimal(self) -> int:
        """Returns the number of bits after the decimal point in the target register."""
        ...
