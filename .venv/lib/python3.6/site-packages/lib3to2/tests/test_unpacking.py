from lib3to2.tests.support import lib3to2FixerTestCase

class Test_unpacking(lib3to2FixerTestCase):

    fixer = 'unpacking'

    def test_unchanged(self):
        s = "def f(*args): pass"
        self.unchanged(s)

        s = "for i in range(s): pass"
        self.unchanged(s)

        s = "a, b, c = range(100)"
        self.unchanged(s)

    def test_forloop(self):
        b = """
        for a, b, c, *d, e in two_dim_array: pass"""
        a = """
        for _3to2iter in two_dim_array:
            _3to2list = list(_3to2iter)
            a, b, c, d, e, = _3to2list[:3] + [_3to2list[3:-1]] + _3to2list[-1:]
            pass"""
        self.check(b, a)

        b = """
        for a, b, *c in some_thing:
            do_stuff"""
        a = """
        for _3to2iter in some_thing:
            _3to2list = list(_3to2iter)
            a, b, c, = _3to2list[:2] + [_3to2list[2:]]
            do_stuff"""
        self.check(b, a)

        b = """
        for *a, b, c, d, e, f, g in some_thing:
            pass"""
        a = """
        for _3to2iter in some_thing:
            _3to2list = list(_3to2iter)
            a, b, c, d, e, f, g, = [_3to2list[:-6]] + _3to2list[-6:]
            pass"""
        self.check(b, a)

    def test_assignment(self):
        b = """
        a, *b, c = range(100)"""
        a = """
        _3to2list = list(range(100))
        a, b, c, = _3to2list[:1] + [_3to2list[1:-1]] + _3to2list[-1:]"""
        self.check(b, a)

        b = """
        a, b, c, d, *e, f, g = letters"""
        a = """
        _3to2list = list(letters)
        a, b, c, d, e, f, g, = _3to2list[:4] + [_3to2list[4:-2]] + _3to2list[-2:]"""
        self.check(b, a)

        b = """
        *e, f, g = letters"""
        a = """
        _3to2list = list(letters)
        e, f, g, = [_3to2list[:-2]] + _3to2list[-2:]"""
        self.check(b, a)

        b = """
        a, b, c, d, *e = stuff"""
        a = """
        _3to2list = list(stuff)
        a, b, c, d, e, = _3to2list[:4] + [_3to2list[4:]]"""
        self.check(b, a)

        b = """
        *z, = stuff"""
        a = """
        _3to2list = list(stuff)
        z, = [_3to2list[:]]"""
        self.check(b, a)

        b = """
        while True:
            a, *b, c = stuff
            other_stuff = make_more_stuff(a, b, c)"""

        a = """
        while True:
            _3to2list = list(stuff)
            a, b, c, = _3to2list[:1] + [_3to2list[1:-1]] + _3to2list[-1:]
            other_stuff = make_more_stuff(a, b, c)"""
        self.check(b, a)
