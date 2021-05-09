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

from cirq.linalg.tolerance import all_near_zero, all_near_zero_mod, near_zero, near_zero_mod


def test_all_zero():
    atol = 5
    assert all_near_zero(0, atol=atol)
    assert all_near_zero(4.5, atol=atol)
    assert not all_near_zero(5.5, atol=atol)

    assert all_near_zero([-4.5, 0, 1, 4.5, 3], atol=atol)
    assert not all_near_zero([-4.5, 0, 1, 4.5, 30], atol=atol)


def test_all_zero_mod():
    atol = 5
    assert all_near_zero_mod(0, 100, atol=atol)
    assert all_near_zero_mod(4.5, 100, atol=atol)
    assert not all_near_zero_mod(5.5, 100, atol=atol)

    assert all_near_zero_mod(100, 100, atol=atol)
    assert all_near_zero_mod(95.5, 100, atol=atol)
    assert not all_near_zero_mod(94.5, 100, atol=atol)

    assert all_near_zero_mod(-4.5, 100, atol=atol)
    assert not all_near_zero_mod(-5.5, 100, atol=atol)

    assert all_near_zero_mod(104.5, 100, atol=atol)
    assert not all_near_zero_mod(105.5, 100, atol=atol)

    assert all_near_zero_mod([-4.5, 0, 1, 4.5, 3, 95.5, 104.5], 100, atol=atol)
    assert not all_near_zero_mod([-4.5, 0, 1, 4.5, 30], 100, atol=atol)


def test_near_zero():
    atol = 5
    assert near_zero(0, atol=atol)
    assert near_zero(4.5, atol=atol)
    assert not near_zero(5.5, atol=atol)


def test_near_zero_mod():
    atol = 5
    assert near_zero_mod(0, 100, atol=atol)
    assert near_zero_mod(4.5, 100, atol=atol)
    assert not near_zero_mod(5.5, 100, atol=atol)

    assert near_zero_mod(100, 100, atol=atol)
    assert near_zero_mod(95.5, 100, atol=atol)
    assert not near_zero_mod(94.5, 100, atol=atol)

    assert near_zero_mod(-4.5, 100, atol=atol)
    assert not near_zero_mod(-5.5, 100, atol=atol)

    assert near_zero_mod(104.5, 100, atol=atol)
    assert not near_zero_mod(105.5, 100, atol=atol)
