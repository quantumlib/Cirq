from lib3to2.tests.support import lib3to2FixerTestCase

class Test_input(lib3to2FixerTestCase):
    fixer = "input"

    def test_prefix_preservation(self):
        b = """x =    input(   )"""
        a = """x =    raw_input(   )"""
        self.check(b, a)

        b = """x = input(   ''   )"""
        a = """x = raw_input(   ''   )"""
        self.check(b, a)

    def test_1(self):
        b = """x = input()"""
        a = """x = raw_input()"""
        self.check(b, a)

    def test_2(self):
        b = """x = input('a')"""
        a = """x = raw_input('a')"""
        self.check(b, a)

    def test_3(self):
        b = """x = input('prompt')"""
        a = """x = raw_input('prompt')"""
        self.check(b, a)

    def test_4(self):
        b = """x = input(foo(a) + 6)"""
        a = """x = raw_input(foo(a) + 6)"""
        self.check(b, a)

    def test_5(self):
        b = """x = input(invite).split()"""
        a = """x = raw_input(invite).split()"""
        self.check(b, a)

    def test_6(self):
        b = """x = input(invite) . split ()"""
        a = """x = raw_input(invite) . split ()"""
        self.check(b, a)

    def test_7(self):
        b = "x = int(input())"
        a = "x = int(raw_input())"
        self.check(b, a)
