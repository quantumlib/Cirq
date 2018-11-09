from lib3to2.tests.support import lib3to2FixerTestCase

class Test_reduce(lib3to2FixerTestCase):
    fixer = "reduce"

    def test_functools_import(self):

        b = """
            from functools import reduce
            reduce(f, it)"""
        a = """
            reduce(f, it)"""
        self.check(b, a)

        b = """
            do_other_stuff; from functools import reduce
            reduce(f, it)"""
        a = """
            do_other_stuff
            reduce(f, it)"""
        self.check(b, a)

        b = """
            do_other_stuff; from functools import reduce; do_more_stuff
            reduce(f, it)"""
        a = """
            do_other_stuff; do_more_stuff
            reduce(f, it)"""
        self.check(b, a)

    def test_functools_reduce(self):

        b = """
            import functools
            functools.reduce(spam, ['spam', 'spam', 'baked beans', 'spam'])
            """
        a = """
            import functools
            reduce(spam, ['spam', 'spam', 'baked beans', 'spam'])
            """
        self.check(b, a)

    def test_prefix(self):

        b = """
            a  =  functools.reduce( self.thing,  self.children , f( 3 ))
            """
        a = """
            a  =  reduce( self.thing,  self.children , f( 3 ))
            """
        self.check(b, a)
