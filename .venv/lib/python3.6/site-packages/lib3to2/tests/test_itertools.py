from lib3to2.tests.support import lib3to2FixerTestCase

class Test_itertoools(lib3to2FixerTestCase):
    fixer = "itertools"

    def test_map(self):
        b = """map(a, b)"""
        a = """from itertools import imap\nimap(a, b)"""
        self.check(b, a)

    def test_unchanged_nobuiltin(self):
        s = """obj.filter(a, b)"""
        self.unchanged(s)

        s = """
        def map():
            pass
        """
        self.unchanged(s)

    def test_filter(self):
        b = "a =    filter( a,  b)"
        a = "from itertools import ifilter\na =    ifilter( a,  b)"
        self.check(b, a)

    def test_zip(self):
        b = """for key, val in zip(a, b):\n\tdct[key] = val"""
        a = """from itertools import izip\nfor key, val in izip(a, b):\n\tdct[key] = val"""
        self.check(b, a)

    def test_filterfalse(self):
        b = """from itertools import function, filterfalse, other_function"""
        a = """from itertools import function, ifilterfalse, other_function"""
        self.check( b, a)

        b = """filterfalse(a, b)"""
        a = """ifilterfalse(a, b)"""
        self.check(b, a )
