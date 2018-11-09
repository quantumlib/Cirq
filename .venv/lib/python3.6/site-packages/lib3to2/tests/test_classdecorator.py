from lib3to2.tests.support import lib3to2FixerTestCase

class Test_classdecorator(lib3to2FixerTestCase):
    fixer = "classdecorator"

    def test_basic_functionality(self):

        b = """
        @decor
        class decorated(object):
            pass"""

        a = """
        class decorated(object):
            pass
        decorated = decor(decorated)"""

        self.check(b, a)

    def test_whitespace(self):

        b = """
        @decor
        class decorated(object):
            pass
        print("hello, there!")"""

        a = """
        class decorated(object):
            pass
        decorated = decor(decorated)

        print("hello, there!")"""

        self.check(b, a)

    def test_chained(self):

        b = """
        @f1
        @f2
        @f3
        class wow(object):
           do_cool_stuff_here()"""

        a = """
        class wow(object):
           do_cool_stuff_here()
        wow = f1(f2(f3(wow)))"""

        self.check(b, a)

    def test_dots_and_parens(self):

        b = """
        @should_work.with_dots(and_parens)
        @dotted.name
        @with_args(in_parens)
        class awesome(object):
            inconsequential_stuff()"""

        a = """
        class awesome(object):
            inconsequential_stuff()
        awesome = should_work.with_dots(and_parens)(dotted.name(with_args(in_parens)(awesome)))"""

        self.check(b, a)

    def test_indentation(self):

        b = """
        if 1:
            if 2:
                if 3:
                    @something
                    @something_else
                    class foo(bar):
                        do_stuff()
                elif 4:
                    pass"""
        a = """
        if 1:
            if 2:
                if 3:
                    class foo(bar):
                        do_stuff()
                    foo = something(something_else(foo))
                elif 4:
                    pass"""
