from lib3to2.tests.support import lib3to2FixerTestCase

class Test_range(lib3to2FixerTestCase):
    fixer = "range"

    def test_notbuiltin_list(self):
        b = "x.list(range(10))"
        a = "x.list(xrange(10))"
        self.check(b, a)

    def test_prefix_preservation(self):
        b = """x =    range(  10  )"""
        a = """x =    xrange(  10  )"""
        self.check(b, a)

        b = """x = range(  1  ,  10   )"""
        a = """x = xrange(  1  ,  10   )"""
        self.check(b, a)

        b = """x = range(  0  ,  10 ,  2 )"""
        a = """x = xrange(  0  ,  10 ,  2 )"""
        self.check(b, a)

    def test_single_arg(self):
        b = """x = range(10)"""
        a = """x = xrange(10)"""
        self.check(b, a)

    def test_two_args(self):
        b = """x = range(1, 10)"""
        a = """x = xrange(1, 10)"""
        self.check(b, a)

    def test_three_args(self):
        b = """x = range(0, 10, 2)"""
        a = """x = xrange(0, 10, 2)"""
        self.check(b, a)

    def test_wrapped_in_list(self):
        b = """x = list(range(10, 3, 9))"""
        a = """x = range(10, 3, 9)"""
        self.check(b, a)

        b = """x = foo(list(range(10, 3, 9)))"""
        a = """x = foo(range(10, 3, 9))"""
        self.check(b, a)

        b = """x = list(range(10, 3, 9)) + [4]"""
        a = """x = range(10, 3, 9) + [4]"""
        self.check(b, a)

        b = """x = list(range(10))[::-1]"""
        a = """x = range(10)[::-1]"""
        self.check(b, a)

        b = """x = list(range(10))  [3]"""
        a = """x = range(10)  [3]"""
        self.check(b, a)

    def test_range_in_for(self):
        b = """for i in range(10):\n    j=i"""
        a = """for i in xrange(10):\n    j=i"""
        self.check(b, a)

        b = """[i for i in range(10)]"""
        a = """[i for i in xrange(10)]"""
        self.check(b, a)

