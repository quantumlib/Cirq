from lib3to2.tests.support import lib3to2FixerTestCase

class Test_except(lib3to2FixerTestCase):
    fixer = "except"

    def test_prefix_preservation(self):
        a = """
            try:
                pass
            except (RuntimeError, ImportError),    e:
                pass"""
        b = """
            try:
                pass
            except (RuntimeError, ImportError) as    e:
                pass"""
        self.check(b, a)

    def test_simple(self):
        a = """
            try:
                pass
            except Foo, e:
                pass"""
        b = """
            try:
                pass
            except Foo as e:
                pass"""
        self.check(b, a)
