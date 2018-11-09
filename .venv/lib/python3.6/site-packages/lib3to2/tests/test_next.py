from lib3to2.tests.support import lib3to2FixerTestCase

class Test_next(lib3to2FixerTestCase):
    fixer = "next"

    def test_1(self):
        b = """next(it)"""
        a = """it.next()"""
        self.check(b, a)

    def test_2(self):
        b = """next(a.b.c.d)"""
        a = """a.b.c.d.next()"""
        self.check(b, a)

    def test_3(self):
        b = """next((a + b))"""
        a = """(a + b).next()"""
        self.check(b, a)

    def test_4(self):
        b = """next(a())"""
        a = """a().next()"""
        self.check(b, a)

    def test_5(self):
        b = """next(a()) + b"""
        a = """a().next() + b"""
        self.check(b, a)

    def test_6(self):
        b = """c(      next(a()) + b)"""
        a = """c(      a().next() + b)"""
        self.check(b, a)

    def test_prefix_preservation_1(self):
        b = """
            for a in b:
                foo(a)
                next(a)
            """
        a = """
            for a in b:
                foo(a)
                a.next()
            """
        self.check(b, a)

    def test_prefix_preservation_2(self):
        b = """
            for a in b:
                foo(a) # abc
                # def
                next(a)
            """
        a = """
            for a in b:
                foo(a) # abc
                # def
                a.next()
            """
        self.check(b, a)

    def test_prefix_preservation_3(self):
        b = """
            next = 5
            for a in b:
                foo(a)
                a.__next__()
            """

        a = """
            next = 5
            for a in b:
                foo(a)
                a.next()
            """
        self.check(b, a)

    def test_prefix_preservation_4(self):
        b = """
            next = 5
            for a in b:
                foo(a) # abc
                # def
                a.__next__()
            """
        a = """
            next = 5
            for a in b:
                foo(a) # abc
                # def
                a.next()
            """
        self.check(b, a)

    def test_prefix_preservation_5(self):
        b = """
            next = 5
            for a in b:
                foo(foo(a), # abc
                    a.__next__())
            """
        a = """
            next = 5
            for a in b:
                foo(foo(a), # abc
                    a.next())
            """
        self.check(b, a)

    def test_prefix_preservation_6(self):
        b = """
            for a in b:
                foo(foo(a), # abc
                    next(a))
            """
        a = """
            for a in b:
                foo(foo(a), # abc
                    a.next())
            """
        self.check(b, a)

    def test_method_1(self):
        b = """
            class A:
                def __next__(self):
                    pass
            """
        a = """
            class A:
                def next(self):
                    pass
            """
        self.check(b, a)

    def test_method_2(self):
        b = """
            class A(object):
                def __next__(self):
                    pass
            """
        a = """
            class A(object):
                def next(self):
                    pass
            """
        self.check(b, a)

    def test_method_3(self):
        b = """
            class A:
                def __next__(x):
                    pass
            """
        a = """
            class A:
                def next(x):
                    pass
            """
        self.check(b, a)

    def test_method_4(self):
        b = """
            class A:
                def __init__(self, foo):
                    self.foo = foo

                def __next__(self):
                    pass

                def __iter__(self):
                    return self
            """
        a = """
            class A:
                def __init__(self, foo):
                    self.foo = foo

                def next(self):
                    pass

                def __iter__(self):
                    return self
            """
        self.check(b, a)

    def test_noncall_access_1(self):
        b = """gnext = g.__next__"""
        a = """gnext = g.next"""
        self.check(b, a)

    def test_noncall_access_2(self):
        b = """f(g.__next__ + 5)"""
        a = """f(g.next + 5)"""
        self.check(b, a)

    def test_noncall_access_3(self):
        b = """f(g().__next__ + 5)"""
        a = """f(g().next + 5)"""
        self.check(b, a)

