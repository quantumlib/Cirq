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
    def __init__(self, extensions, other):
        del extensions
        self.other = other

    def make(self):
        return 'wrap'


class UnrelatedType:
    pass


def test_wrap():
    e = extension.Extensions({DesiredType: {OtherType: WrapType}})
    o = OtherType()
    w = e.cast(DesiredType, o)
    assert w.make() == 'wrap'
    assert isinstance(w, WrapType)
    assert w.other is o


def test_empty():
    e = extension.Extensions()
    o = OtherType()
    c = ChildType()
    u = UnrelatedType()

    with pytest.raises(TypeError):
        _ = e.cast(DesiredType, u)
    assert e.cast(DesiredType, c) is c
    with pytest.raises(TypeError):
        _ = e.cast(DesiredType, o)

    assert e.cast(UnrelatedType, u) is u
    with pytest.raises(TypeError):
        _ = e.cast(UnrelatedType, c)
    with pytest.raises(TypeError):
        _ = e.cast(UnrelatedType, o)


def test_cast_hit_vs_miss():
    e = extension.Extensions({DesiredType: {OtherType: WrapType}})
    o = OtherType()
    c = ChildType()
    u = UnrelatedType()

    with pytest.raises(TypeError):
        _ = e.cast(DesiredType, None)
    with pytest.raises(TypeError):
        _ = e.cast(DesiredType, u)
    assert e.cast(DesiredType, c) is c
    assert e.cast(DesiredType, o) is not o

    with pytest.raises(TypeError):
        _ = e.cast(UnrelatedType, None)
    assert e.cast(UnrelatedType, u) is u
    with pytest.raises(TypeError):
        _ = e.cast(UnrelatedType, c)
    with pytest.raises(TypeError):
        _ = e.cast(UnrelatedType, o)


def test_try_cast_hit_vs_miss():
    e = extension.Extensions({DesiredType: {OtherType: WrapType}})
    o = OtherType()
    c = ChildType()
    u = UnrelatedType()

    assert e.try_cast(DesiredType, None) is None
    assert e.try_cast(DesiredType, u) is None
    assert e.try_cast(DesiredType, c) is c
    assert e.try_cast(DesiredType, o) is not o

    assert e.try_cast(UnrelatedType, None) is None
    assert e.try_cast(UnrelatedType, u) is u
    assert e.try_cast(UnrelatedType, c) is None
    assert e.try_cast(UnrelatedType, o) is None


def test_can_cast():
    e = extension.Extensions({DesiredType: {OtherType: WrapType}})
    c = ChildType()
    u = UnrelatedType()
    assert not e.can_cast(DesiredType, u)
    assert e.can_cast(DesiredType, c)


def test_override_order():
    e = extension.Extensions({
        DesiredType: {
            object: lambda _, _2: 'obj',
            UnrelatedType: lambda _, _2: 'unrelated',
        }
    })
    assert e.cast(DesiredType, '') == 'obj'
    assert e.cast(DesiredType, None) == 'obj'
    assert e.cast(DesiredType, UnrelatedType()) == 'unrelated'
    assert e.cast(DesiredType, ChildType()) == 'obj'


def test_try_cast_potential_implementation():

    class PotentialOther(PotentialImplementation):
        def __init__(self, is_other):
            self.is_other = is_other

        def try_cast_to(self, desired_type, extensions):
            if desired_type is OtherType and self.is_other:
                return OtherType()
            return None

    e = extension.Extensions()
    o = PotentialOther(is_other=False)
    u = PotentialOther(is_other=False)

    assert e.try_cast(OtherType, o) is None
    assert e.try_cast(OtherType, u) is None


def test_add_cast():
    e = extension.Extensions()
    d = DesiredType()

    assert e.try_cast(DesiredType, UnrelatedType()) is None
    e.add_cast(desired_type=DesiredType,
               actual_type=UnrelatedType,
               conversion=lambda _: d)
    assert e.try_cast(DesiredType, UnrelatedType()) is d


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
    assert e.try_cast(Child, c) is c
    assert e.try_cast(Parent, c) is c
    assert e.try_cast(Grandparent, c) is c
    assert e.try_cast(object, c) is c
    assert e.try_cast(Aunt, c) == 'boo'
    assert e.try_cast(Cousin, c) == 'boo'

    assert e2.try_cast(Child, c) is c
    assert e2.try_cast(Parent, c) is c
    assert e2.try_cast(Grandparent, c) is c
    assert e2.try_cast(object, c) is c
    assert e2.try_cast(Aunt, c) is None
    assert e2.try_cast(Cousin, c) == 'boo'


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
    assert e.try_cast(Cousin, Child()) is o1
    e.add_cast(desired_type=Cousin,
               actual_type=Child,
               conversion=lambda _: o3,
               also_add_inherited_conversions=False,
               overwrite_existing=True)
    assert e.try_cast(Cousin, Child()) is o3


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
    assert e.try_cast(Aunt, Child()) is o1
    assert e.try_cast(Cousin, Child()) is None
    e.add_cast(desired_type=Cousin,
               actual_type=Child,
               conversion=lambda _: o3,
               also_add_inherited_conversions=True,
               overwrite_existing=True)
    assert e.try_cast(Aunt, Child()) is o3
    assert e.try_cast(Cousin, Child()) is o3


def test_add_potential_cast():
    a = Aunt()
    c1 = Child()
    c2 = Child()

    e = extension.Extensions()
    e.add_cast(desired_type=Aunt,
               actual_type=Child,
               conversion=lambda e: a if e is c1 else None)

    assert e.try_cast(Aunt, c1) is a
    assert e.try_cast(Aunt, c2) is None


def test_add_recursive_cast():
    cousin = Cousin()
    child = Child()

    e = extension.Extensions()
    e.add_cast(desired_type=Cousin,
               actual_type=Child,
               conversion=lambda _: cousin)
    e.add_recursive_cast(
        desired_type=Cousin,
        actual_type=Grandparent,
        conversion=lambda ext, _: ext.try_cast(Cousin, child))

    assert e.try_cast(Cousin, Grandparent()) is cousin
