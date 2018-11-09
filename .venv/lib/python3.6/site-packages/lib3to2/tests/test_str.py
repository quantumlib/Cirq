from lib3to2.tests.support import lib3to2FixerTestCase

class Test_str(lib3to2FixerTestCase):
    fixer = "str"

    def test_str_call(self):
        b = """str(x, y, z)"""
        a = """unicode(x, y, z)"""
        self.check(b, a)

    def test_chr_call(self):
        b = """chr(a, t, m)"""
        a = """unichr(a, t, m)"""
        self.check(b, a)

    def test_str_literal_1(self):
        b = '''"x"'''
        a = '''u"x"'''
        self.check(b, a)

    def test_str_literal_2(self):
        b = """r'x'"""
        a = """ur'x'"""
        self.check(b, a)

    def test_str_literal_3(self):
        b = """R'''x'''"""
        a = """uR'''x'''"""
        self.check(b, a)

