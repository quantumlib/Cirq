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

"""A type for specifying thresholds for doing approximate equality."""

import numpy as np


class Tolerance:
    """Specifies thresholds for doing approximate equality."""

    ZERO = None  # type: Tolerance
    DEFAULT = None  # type: Tolerance

    def __init__(self,
                 rtol: float = 1e-5,
                 atol: float = 1e-8,
                 equal_nan: bool = False) -> None:
        """Initializes a Tolerance instance with the specified parameters.

        Notes:
          Matrix Comparisons (methods beginning with "all_") are done by
          numpy.allclose, which considers x and y
          to be close when abs(x - y) <= atol + rtol * abs(y). See
          numpy.allclose's documentation for more details.   The scalar
          methods perform the same calculations without the numpy
          matrix construction.

        Args:
          rtol: Relative tolerance.
          atol: Absolute tolerance.
          equal_nan: Whether NaNs are equal to each other.
        """
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    # Matrix methods
    def all_close(self, a, b):
        return np.allclose(
            a, b, rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan)

    def all_near_zero(self, a):
        return self.all_close(a, np.zeros(np.shape(a)))

    def all_near_zero_mod(self, a, period):
        return self.all_close((np.array(a) + (period/2)) % period - period/2,
                              np.zeros(np.shape(a)))

    # Scalar methods
    def close(self, a, b):
        return abs(a - b) <= self.atol + self.rtol * abs(b)

    def near_zero(self, a):
        return abs(a) <= self.atol

    def near_zero_mod(self, a, period):
        half_period = period / 2
        return self.near_zero((a + half_period) % period - half_period)

    def __repr__(self):
        return "Tolerance(rtol={}, atol={}, equal_nan={})".format(
            repr(self.rtol), repr(self.atol), repr(self.equal_nan))


Tolerance.ZERO = Tolerance(rtol=0, atol=0)
Tolerance.DEFAULT = Tolerance()
