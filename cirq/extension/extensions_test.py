# Copyright 2018 Google LLC
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

import pytest

from cirq import extension
from cirq.extension import PotentialImplementation


class DesiredType:
    def make(self):
        raise NotImplementedError()


class ChildType(DesiredType):
    def make(self):
        return 'child'


class OtherType:
    pass


class WrapType(DesiredType):
    def __init__(self, other):
        self.other = other

    def make(self):
        return 'wrap'


class UnrelatedType:
    pass


def test_wrap():
    e = extension.Extensions({DesiredType: {OtherType: WrapType}})
    o = OtherType()
    w = e.cast(o, DesiredType)
    assert w.make() == 'wrap'
    assert isinstance(w, WrapType)
    assert w.other is o


def test_empty():
    e = extension.Extensions()
    o = OtherType()
    c = ChildType()
    u = UnrelatedType()

    with pytest.raises(TypeError):
        _ = e.cast(u, DesiredType)
    assert e.cast(c, DesiredType) is c
    with pytest.raises(TypeError):
        _ = e.cast(o, DesiredType)

    assert e.cast(u, UnrelatedType) is u
    with pytest.raises(TypeError):
        _ = e.cast(c, UnrelatedType)
    with pytest.raises(TypeError):
        _ = e.cast(o, UnrelatedType)


def test_cast_hit_vs_miss():
    e = extension.Extensions({DesiredType: {OtherType: WrapType}})
    o = OtherType()
    c = ChildType()
    u = UnrelatedType()

    with pytest.raises(TypeError):
        _ = e.cast(None, DesiredType)
    with pytest.raises(TypeError):
        _ = e.cast(u, DesiredType)
    assert e.cast(c, DesiredType) is c
    assert e.cast(o, DesiredType) is not o

    with pytest.raises(TypeError):
        _ = e.cast(None, UnrelatedType)
    assert e.cast(u, UnrelatedType) is u
    with pytest.raises(TypeError):
        _ = e.cast(c, UnrelatedType)
    with pytest.raises(TypeError):
        _ = e.cast(o, UnrelatedType)


def test_try_cast_hit_vs_miss():
    e = extension.Extensions({DesiredType: {OtherType: WrapType}})
    o = OtherType()
    c = ChildType()
    u = UnrelatedType()

    assert e.try_cast(None, DesiredType) is None
    assert e.try_cast(u, DesiredType) is None
    assert e.try_cast(c, DesiredType) is c
    assert e.try_cast(o, DesiredType) is not o

    assert e.try_cast(None, UnrelatedType) is None
    assert e.try_cast(u, UnrelatedType) is u
    assert e.try_cast(c, UnrelatedType) is None
    assert e.try_cast(o, UnrelatedType) is None


def test_can_cast():
    e = extension.Extensions({DesiredType: {OtherType: WrapType}})
    c = ChildType()
    u = UnrelatedType()
    assert not e.can_cast(u, DesiredType)
    assert e.can_cast(c, DesiredType)


def test_override_order():
    e = extension.Extensions({
        DesiredType: {
            object: lambda _: 'obj',
            UnrelatedType: lambda _: 'unrelated',
        }
    })
    assert e.cast('', DesiredType) == 'obj'
    assert e.cast(None, DesiredType) == 'obj'
    assert e.cast(UnrelatedType(), DesiredType) == 'unrelated'
    assert e.cast(ChildType(), DesiredType) == 'obj'


def test_try_cast_potential_implementation():

    class PotentialOther(PotentialImplementation):
        def __init__(self, is_other):
            self.is_other = is_other

        def try_cast_to(self, desired_type):
            if desired_type is OtherType and self.is_other:
                return OtherType()
            return None

    e = extension.Extensions()
    o = PotentialOther(is_other=False)
    u = PotentialOther(is_other=False)

    assert e.try_cast(o, OtherType) is None
    assert e.try_cast(u, OtherType) is None


def test_add_cast():
    e = extension.Extensions()
    d = DesiredType()

    assert e.try_cast(UnrelatedType(), DesiredType) is None
    e.add_cast(desired_type=DesiredType,
               actual_type=UnrelatedType,
               conversion=lambda _: d)
    assert e.try_cast(UnrelatedType(), DesiredType) is d


class Grandparent:
    pass


class Parent(Grandparent):
    pass


class Aunt(Grandparent):
    pass


class Child(Parent):
    pass


class Cousin(Aunt):
    pass


def test_add_cast_include_subtypes():

    e = extension.Extensions()
    e.add_cast(desired_type=Cousin,
               actual_type=Child,
               conversion=lambda _: 'boo',
               also_add_inherited_conversions=True)
    e2 = extension.Extensions()
    e2.add_cast(desired_type=Cousin,
                actual_type=Child,
                conversion=lambda _: 'boo',
                also_add_inherited_conversions=False)

    c = Child()
    assert e.try_cast(c, Child) is c
    assert e.try_cast(c, Parent) is c
    assert e.try_cast(c, Grandparent) is c
    assert e.try_cast(c, object) is c
    assert e.try_cast(c, Aunt) == 'boo'
    assert e.try_cast(c, Cousin) == 'boo'

    assert e2.try_cast(c, Child) is c
    assert e2.try_cast(c, Parent) is c
    assert e2.try_cast(c, Grandparent) is c
    assert e2.try_cast(c, object) is c
    assert e2.try_cast(c, Aunt) is None
    assert e2.try_cast(c, Cousin) == 'boo'


def test_add_cast_redundant():
    o1 = Cousin()
    o2 = Cousin()
    o3 = Cousin()

    e = extension.Extensions()
    e.add_cast(desired_type=Cousin,
               actual_type=Child,
               conversion=lambda _: o1,
               also_add_inherited_conversions=False,
               overwrite_existing=False)
    with pytest.raises(ValueError):
        e.add_cast(desired_type=Cousin,
                   actual_type=Child,
                   conversion=lambda _: o2,
                   also_add_inherited_conversions=False,
                   overwrite_existing=False)
    assert e.try_cast(Child(), Cousin) is o1
    e.add_cast(desired_type=Cousin,
               actual_type=Child,
               conversion=lambda _: o3,
               also_add_inherited_conversions=False,
               overwrite_existing=True)
    assert e.try_cast(Child(), Cousin) is o3


def test_add_cast_redundant_including_subtypes():
    o1 = Cousin()
    o2 = Cousin()
    o3 = Cousin()

    e = extension.Extensions()
    e.add_cast(desired_type=Aunt,
               actual_type=Child,
               conversion=lambda _: o1,
               also_add_inherited_conversions=True,
               overwrite_existing=False)
    with pytest.raises(ValueError):
        e.add_cast(desired_type=Cousin,
                   actual_type=Child,
                   conversion=lambda _: o2,
                   also_add_inherited_conversions=True,
                   overwrite_existing=False)
    assert e.try_cast(Child(), Aunt) is o1
    assert e.try_cast(Child(), Cousin) is None
    e.add_cast(desired_type=Cousin,
               actual_type=Child,
               conversion=lambda _: o3,
               also_add_inherited_conversions=True,
               overwrite_existing=True)
    assert e.try_cast(Child(), Aunt) is o3
    assert e.try_cast(Child(), Cousin) is o3


def test_add_potential_cast():
    a = Aunt()
    c1 = Child()
    c2 = Child()

    e = extension.Extensions()
    e.add_cast(desired_type=Aunt,
               actual_type=Child,
               conversion=lambda e: a if e is c1 else None)

    assert e.try_cast(c1, Aunt) is a
    assert e.try_cast(c2, Aunt) is None
