from lib3to2.tests.support import lib3to2FixerTestCase

class Test_newstyle(lib3to2FixerTestCase):
    fixer = "newstyle"

    def test_oneline(self):

        b = """class Foo: pass"""
        a = """class Foo(object): pass"""
        self.check(b, a)

    def test_suite(self):

        b = """
        class Foo():
            do_stuff()"""
        a = """
        class Foo(object):
            do_stuff()"""
        self.check(b, a)

