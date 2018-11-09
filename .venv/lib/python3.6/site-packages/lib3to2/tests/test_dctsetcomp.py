from lib3to2.tests.support import lib3to2FixerTestCase

class Test_dctsetcomp(lib3to2FixerTestCase):
    fixer = "dctsetcomp"

    def test_dictcomp_straightforward(self):
        b = "{key:val for (key, val) in tuple_of_stuff}"
        a = "dict((key, val) for (key, val) in tuple_of_stuff)"
        self.check(b, a)

    def test_dictcomp_nestedstuff_noif(self):
        b = "{hashlify(spam):valuate(ham).whatsthis(eggs) for \
             (spam, ham, eggs) in spam_iterator}"
        a = "dict((hashlify(spam), valuate(ham).whatsthis(eggs)) for \
             (spam, ham, eggs) in spam_iterator)"
        self.check(b, a)

    def test_dictcomp_nestedstuff_withif(self):
        b = "{moo:(lambda new: None)(cow) for (moo, cow) in \
            farm_animal['cow'] if has_milk()}"
        a = "dict((moo, (lambda new: None)(cow)) for (moo, cow) in \
            farm_animal['cow'] if has_milk())"
        self.check(b, a)

    def test_setcomps(self):
        """
        setcomp fixer should keep everything inside the same
        and only replace the {} with a set() call on a gencomp
        """
        tests = []
        tests.append("milk.price for milk in find_milk(store)")
        tests.append("compute_nth_prime(generate_complicated_thing(\
            n.value(hashlifier))) for n in my_range_func(1, (how_far+offset))")
        tests.append("compute_nth_prime(generate_complicated_thing(\
            n.value(hashlifier))) for n in my_range_func(1, (how_far+offset))\
            if a==b.spam()")
        for comp in tests:
            b = "{%s}" % comp
            a = "set(%s)" % comp
        self.check(b, a)

    def test_prefixes(self):
        b = "spam = {foo for foo in bar}"
        a = "spam = set(foo for foo in bar)"
        self.check(b, a)

        b = "spam = {foo:bar for (foo, bar) in baz}"
        a = "spam = dict((foo, bar) for (foo, bar) in baz)"
        self.check(b, a)
