from lib3to2.tests.support import lib3to2FixerTestCase

class Test_int(lib3to2FixerTestCase):
    fixer = "int"

    def test_1(self):
        b = """x = int(x)"""
        a = """x = long(x)"""
        self.check(b, a)

    def test_2(self):
        b = """y = isinstance(x, int)"""
        a = """y = isinstance(x, long)"""
        self.check(b, a)

    def test_unchanged(self):
        s = """int = True"""
        self.unchanged(s)

        s = """s.int = True"""
        self.unchanged(s)

        s = """def int(): pass"""
        self.unchanged(s)

        s = """class int(): pass"""
        self.unchanged(s)

        s = """def f(int): pass"""
        self.unchanged(s)

        s = """def f(g, int): pass"""
        self.unchanged(s)

        s = """def f(x, int=True): pass"""
        self.unchanged(s)

    def test_prefix_preservation(self):
        b = """x =   int(  x  )"""
        a = """x =   long(  x  )"""
        self.check(b, a)

    def test_literal_1(self):
        b = """5"""
        a = """5L"""
        self.check(b, a)

    def test_literal_2(self):
        b = """a = 12"""
        a = """a = 12L"""
        self.check(b, a)

    def test_literal_3(self):
        b = """0"""
        a = """0L"""
        self.check(b, a)

    def test_complex_1(self):
        b = """5 + 4j"""
        a = """5L + 4j"""
        self.check(b, a)

    def test_complex_2(self):
        b = """35  +  2j"""
        a = """35L  +  2j"""
        self.check(b, a)
