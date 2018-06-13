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


class AxisAbc:
    # Abstract
    def _commutes_with(self, other):
        return NotImplemented
    def commutes_with(self, other):
        ret = self._commutes_with(other)
        if ret == NotImplemented:
            ret = other._commutes_with(self)
        if ret == NotImplemented:
            raise NotImplementedError('...')
        return ret
    # Abstract
    def negative(self):
        raise NotImplementedError
    # Abstract
    def is_negative(self):
        raise NotImplementedError
    # Abstract
    def abs(self):
        raise NotImplementedError
    # Abstract
    def _merge_rotations(self, other):
        return NotImplemented
    def merge_rotations(self, other):
        ret = self._merge_rotations(other)
        if ret == NotImplemented:
            ret = other._merge_rotations(self)
        if ret == NotImplemented:
            raise NotImplementedError('...')
        return ret


class IdentAxis(AxisAbc):
    def _commutes_with(self, other):
        return True
    def negative(self):
        return self
    def abs(self):
        return self
    def is_negative(self):
        return False
    def _merge_rotations(self, other):
        return other
    def __eq__(self, other):
        return isinstance(other, type(self))
    def __ne__(self, other):
        return not self == other
    def __str__(self):
        return 'I'
    def __repr__(self):
        return 'IdentAxis()'


class Axis(AxisAbc):
    def __init__(self, axis_i):
        assert 0 <= axis_i < 6
        self.axis_i = axis_i
    def _commutes_with(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.axis_i % 3 == other.axis_i % 3
    def negative(self, negate=True):
        if not negate:
            return self
        return Axis((self.axis_i + 3) % 6)
    def abs(self):
        return Axis(self.axis_i % 3)
    def is_negative(self):
        return self.axis_i >= 3
    def next(self, skip=1):
        return Axis((self.axis_i + skip) % 6)
    def third(self, second):
        return Axis((-self.axis_i - second.axis_i) % 3)
    def complement(self, second):
        neg = (self.axis_i - second.axis_i) % 3 == 2
        neg ^= self.is_negative()
        neg ^= second.is_negative()
        axis_i = (-self.axis_i - second.axis_i) % 3
        if neg:
            axis_i += 3
        return Axis(axis_i)
    def _merge_rotations(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.axis_i % 3 == other.axis_i % 3:
            return I_AXIS  # Two of the same cancel each other
        else:
            return self.third(other)  # Two different become the third axis
    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.axis_i == other.axis_i
    def __ne__(self, other):
        return not self == other
    def __str__(self):
        return ('X', 'Y', 'Z', 'nX', 'nY', 'nZ')[self.axis_i]
    def __repr__(self):
        return 'Axis({})'.format(self.axis_i)


I_AXIS = IdentAxis()
X_AXIS = Axis(0)
Y_AXIS = Axis(1)
Z_AXIS = Axis(2)
nX_AXIS = X_AXIS.negative()
nY_AXIS = Y_AXIS.negative()
nZ_AXIS = Z_AXIS.negative()
