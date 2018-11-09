from lib3to2.tests.support import lib3to2FixerTestCase

class Test_getcwd(lib3to2FixerTestCase):
    fixer = "getcwd"

    def test_prefix_preservation(self):
        b = """ls =    os.listdir(  os.getcwd()  )"""
        a = """ls =    os.listdir(  os.getcwdu()  )"""
        self.check(b, a)

        b = """whatdir = os.getcwd      (      )"""
        a = """whatdir = os.getcwdu      (      )"""
        self.check(b, a)
