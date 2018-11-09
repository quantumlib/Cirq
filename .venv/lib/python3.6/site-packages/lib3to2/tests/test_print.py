from lib3to2.tests.support import lib3to2FixerTestCase

class Test_print(lib3to2FixerTestCase):
    fixer = "print"

    def test_generic(self):
        b = """print()"""
        a = """print"""
        self.check(b,a)

    def test_literal(self):
        b = """print('spam')"""
        a = """print 'spam'"""
        self.check(b,a)

    def test_not_builtin_unchanged(self):
        s = "this.shouldnt.be.changed.because.it.isnt.builtin.print()"
        self.unchanged(s)

    #XXX: Quoting this differently than triple-quotes, because with newline
    #XXX: setting, I can't quite get the triple-quoted versions to line up.
    def test_arbitrary_printing(self):
        b = "import dinosaur.skull\nimport sys\nprint"\
            "(skull.jaw, skull.jaw.biteforce, file=sys.stderr)"
        a = "import dinosaur.skull\nimport sys\nprint "\
            ">>sys.stderr, skull.jaw, skull.jaw.biteforce"
        self.check(b, a)

    def test_long_arglist(self):
        b = "print(spam, spam, spam, spam, spam, baked_beans, spam, spam,"\
            " spam, spam, sep=', spam, ', end=wonderful_spam)\nprint()"
        a = "import sys\nprint ', spam, '.join([unicode(spam), unicode(spam), unicode(spam), unicode(spam), unicode(spam), unicode(baked_beans),"\
            " unicode(spam), unicode(spam), unicode(spam), unicode(spam)]),; sys.stdout.write(wonderful_spam)\nprint"
        self.check(b, a, ignore_warnings=True)

    def test_nones(self):
        b = "print(1,2,3,end=None, sep=None, file=None)"
        a = "print 1,2,3"
        self.check(b, a)

    def test_argument_unpacking(self):
        s = "print(*args)"
        self.warns_unchanged(s, "-fprint does not support argument unpacking.  fix using -xprint and then again with  -fprintfunction.")
