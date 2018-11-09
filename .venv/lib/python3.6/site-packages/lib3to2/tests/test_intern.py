from lib3to2.tests.support import lib3to2FixerTestCase

class Test_intern(lib3to2FixerTestCase):
    fixer = "intern"

    #XXX: Does not remove unused "import sys" lines.
    def test_prefix_preservation(self):
        b = """import sys\nx =   sys.intern(  a  )"""
        a = """import sys\nx =   intern(  a  )"""
        self.check(b, a)

        b = """import sys\ny = sys.intern("b" # test
              )"""
        a = """import sys\ny = intern("b" # test
              )"""
        self.check(b, a)

        b = """import sys\nz = sys.intern(a+b+c.d,   )"""
        a = """import sys\nz = intern(a+b+c.d,   )"""
        self.check(b, a)

    def test(self):
        b = """from sys import intern\nx = intern(a)"""
        a = """\nx = intern(a)"""
        self.check(b, a)

        b = """import sys\nz = sys.intern(a+b+c.d,)"""
        a = """import sys\nz = intern(a+b+c.d,)"""
        self.check(b, a)

        b = """import sys\nsys.intern("y%s" % 5).replace("y", "")"""
        a = """import sys\nintern("y%s" % 5).replace("y", "")"""
        self.check(b, a)

    # These should not be refactored

    def test_multimports(self):
        b = """from sys import intern, path"""
        a = """from sys import path"""
        self.check(b, a)

        b = """from sys import path, intern"""
        a = """from sys import path"""
        self.check(b, a)

        b = """from sys import argv, intern, path"""
        a = """from sys import argv, path"""
        self.check(b, a)

    def test_unchanged(self):
        s = """intern(a=1)"""
        self.unchanged(s)

        s = """intern(f, g)"""
        self.unchanged(s)

        s = """intern(*h)"""
        self.unchanged(s)

        s = """intern(**i)"""
        self.unchanged(s)

        s = """intern()"""
        self.unchanged(s)
