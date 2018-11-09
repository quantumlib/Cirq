from lib3to2.tests.support import lib3to2FixerTestCase

class Test_unittest(lib3to2FixerTestCase):
    fixer = 'unittest'

    def test_imported(self):
        b = "import unittest"
        a = "import unittest2"
        self.check(b, a)

    def test_used(self):
        b = "unittest.AssertStuff(True)"
        a = "unittest2.AssertStuff(True)"
        self.check(b, a)

    def test_from_import(self):
        b = "from unittest import *"
        a = "from unittest2 import *"
        self.check(b, a)

    def test_imported_from(self):
        s = "from whatever import unittest"
        self.unchanged(s)

    def test_not_base(self):
        s = "not_unittest.unittest.stuff()"
        self.unchanged(s)

