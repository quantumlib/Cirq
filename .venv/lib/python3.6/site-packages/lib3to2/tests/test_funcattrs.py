from lib3to2.tests.support import lib3to2FixerTestCase

class Test_funcattrs(lib3to2FixerTestCase):
    fixer = "funcattrs"

    def test_doc_unchanged(self):
        b = """whats.up.__doc__"""
        self.unchanged(b)
    def test_defaults(self):
        b = """myFunc.__defaults__"""
        a = """myFunc.func_defaults"""
        self.check(b, a)
    def test_closure(self):
        b = """fore.__closure__"""
        a = """fore.func_closure"""
        self.check(b, a)
    def test_globals(self):
        b = """funkFunc.__globals__"""
        a = """funkFunc.func_globals"""
        self.check(b, a)
    def test_dict_unchanged(self):
        b = """tricky.__dict__"""
        self.unchanged(b)
    def test_name_unchanged(self):
        b = """sayMy.__name__"""
        self.unchanged(b)
