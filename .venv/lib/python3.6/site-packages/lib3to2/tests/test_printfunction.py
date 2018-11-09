from lib3to2.tests.support import lib3to2FixerTestCase

class Test_printfunction(lib3to2FixerTestCase):
    fixer = "printfunction"

    def test_generic(self):
        b = """print()"""
        a = """from __future__ import print_function\nprint()"""
        self.check(b,a)

    def test_literal(self):
        b = """print('spam')"""
        a = """from __future__ import print_function\nprint('spam')"""
        self.check(b,a)

    def test_not_builtin_unchanged(self):
        s = "this.shouldnt.be.changed.because.it.isnt.builtin.print()"
        self.unchanged(s)

    #XXX: Quoting this differently than triple-quotes, because with newline
    #XXX: setting, I can't quite get the triple-quoted versions to line up.
    def test_arbitrary_printing(self):
        b = "import dinosaur.skull\nimport sys\nprint"\
            "(skull.jaw, skull.jaw.biteforce, file=sys.stderr)"
        a = "from __future__ import print_function\n"\
            "import dinosaur.skull\nimport sys\nprint"\
            "(skull.jaw, skull.jaw.biteforce, file=sys.stderr)"
        self.check(b, a)

    def test_long_arglist(self):
        b = "print(spam, spam, spam, spam, spam, baked_beans, spam, spam,"\
            "spam, spam, sep=', spam, ', end=wonderful_spam)\nprint()"
        a = "from __future__ import print_function\n"\
            "print(spam, spam, spam, spam, spam, baked_beans, spam, spam,"\
            "spam, spam, sep=', spam, ', end=wonderful_spam)\nprint()"
        self.check(b, a)
