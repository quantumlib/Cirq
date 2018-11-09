from lib3to2.tests.support import lib3to2FixerTestCase

class Test_bool(lib3to2FixerTestCase):
    fixer = "bool"

    def test_1(self):
        b = """
            class A:
                def __bool__(self):
                    pass
            """
        a = """
            class A:
                def __nonzero__(self):
                    pass
            """
        self.check(b, a)

    def test_2(self):
        b = """
            class A(object):
                def __bool__(self):
                    pass
            """
        a = """
            class A(object):
                def __nonzero__(self):
                    pass
            """
        self.check(b, a)

    def test_unchanged_1(self):
        s = """
            class A(object):
                def __nonzero__(self):
                    pass
            """
        self.unchanged(s)

    def test_unchanged_2(self):
        s = """
            class A(object):
                def __bool__(self, a):
                    pass
            """
        self.unchanged(s)

    def test_unchanged_func(self):
        s = """
            def __bool__(thing):
                pass
            """
        self.unchanged(s)

