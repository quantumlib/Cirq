from lib3to2.tests.support import lib3to2FixerTestCase

class Test_throw(lib3to2FixerTestCase):

    fixer = 'throw'

    def test_unchanged(self):
        """
        Due to g.throw(E(V)) being valid in 2.5, this fixer fortunately doesn't
        need to touch code that constructs exception objects without explicit
        tracebacks.
        """

        s = """g.throw(E(V))"""
        self.unchanged(s)

        s = """omg.throw(E("What?"))"""
        self.unchanged(s)

    def test_what_doesnt_work(self):
        """
        These tests should fail, but don't.  TODO: Uncomment successfully.
        One potential way of making these work is a separate fix_exceptions
        with a lower run order than fix_throw, to communicate to fix_throw how
        to sort out that third argument.

        These items are currently outside the scope of 3to2.
        """

        b = """
        E = BaseException(V).with_traceback(T)
        gen.throw(E)
        """

        #a = """
        #E = BaseException(V)
        #gen.throw(E, V, T)
        #"""

        #self.check(b, a)
        self.unchanged(b)

        b = """
        E = BaseException(V)
        E.__traceback__ = S
        E.__traceback__ = T
        gen.throw(E)
        """

        #a = """
        #E = BaseException(V)
        #gen.throw(E, V, T)

        #self.check(b, a)
        self.unchanged(b)


    def test_traceback(self):
        """
        This stuff currently works, and is the opposite counterpart to the
        2to3 version of fix_throw.
        """
        b = """myGen.throw(E(V).with_traceback(T))"""
        a = """myGen.throw(E, V, T)"""
        self.check(b, a)

        b = """fling.throw(E().with_traceback(T))"""
        a = """fling.throw(E, None, T)"""
        self.check(b, a)

        b = """myVar.throw(E("Sorry, you cannot do that.").with_traceback(T))"""
        a = """myVar.throw(E, "Sorry, you cannot do that.", T)"""
        self.check(b, a)
