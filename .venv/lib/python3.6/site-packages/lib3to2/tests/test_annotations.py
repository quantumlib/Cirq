from lib3to2.tests.support import lib3to2FixerTestCase

class Test_annotations(lib3to2FixerTestCase):
    fixer = "annotations"

    def test_return_annotations_alone(self):
        b = "def foo() -> 'bar': pass"
        a = "def foo(): pass"
        self.check(b, a, ignore_warnings=True)

        b = """
        def foo() -> "bar":
            print "baz"
            print "what's next, again?"
        """
        a = """
        def foo():
            print "baz"
            print "what's next, again?"
        """
        self.check(b, a, ignore_warnings=True)

    def test_single_param_annotations(self):
        b = "def foo(bar:'baz'): pass"
        a = "def foo(bar): pass"
        self.check(b, a, ignore_warnings=True)

        b = """
        def foo(bar:"baz"="spam"):
            print "what's next, again?"
            print "whatever."
        """
        a = """
        def foo(bar="spam"):
            print "what's next, again?"
            print "whatever."
        """
        self.check(b, a, ignore_warnings=True)

    def test_multiple_param_annotations(self):
        b = "def foo(bar:'spam'=False, baz:'eggs'=True, ham:False='spaghetti'): pass"
        a = "def foo(bar=False, baz=True, ham='spaghetti'): pass"
        self.check(b, a, ignore_warnings=True)

        b = """
        def foo(bar:"spam"=False, baz:"eggs"=True, ham:False="spam"):
            print "this is filler, just doing a suite"
            print "suites require multiple lines."
        """
        a = """
        def foo(bar=False, baz=True, ham="spam"):
            print "this is filler, just doing a suite"
            print "suites require multiple lines."
        """
        self.check(b, a, ignore_warnings=True)

    def test_mixed_annotations(self):
        b = "def foo(bar=False, baz:'eggs'=True, ham:False='spaghetti') -> 'zombies': pass"
        a = "def foo(bar=False, baz=True, ham='spaghetti'): pass"
        self.check(b, a, ignore_warnings=True)

        b = """
        def foo(bar:"spam"=False, baz=True, ham:False="spam") -> 'air':
            print "this is filler, just doing a suite"
            print "suites require multiple lines."
        """
        a = """
        def foo(bar=False, baz=True, ham="spam"):
            print "this is filler, just doing a suite"
            print "suites require multiple lines."
        """
        self.check(b, a, ignore_warnings=True)

        b = "def foo(bar) -> 'brains': pass"
        a = "def foo(bar): pass"
        self.check(b, a, ignore_warnings=True)

    def test_unchanged(self):
        s = "def foo(): pass"
        self.unchanged(s)

        s = """
        def foo():
            pass
            pass
        """
        self.unchanged(s)

        s = """
        def foo(bar=baz):
            pass
            pass
        """
        self.unchanged(s)
