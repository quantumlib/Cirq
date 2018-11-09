from lib3to2.tests.support import lib3to2FixerTestCase

class Test_metaclass(lib3to2FixerTestCase):

    fixer = 'metaclass'

    def test_unchanged(self):
        self.unchanged("class X(): pass")
        self.unchanged("class X(object): pass")
        self.unchanged("class X(object1, object2): pass")
        self.unchanged("class X(object1, object2, object3): pass")

        s = """
        class X():
            def __metaclass__(self): pass
        """
        self.unchanged(s)

        s = """
        class X():
            a[23] = 74
        """
        self.unchanged(s)

    def test_comments(self):
        a = """
        class X(object):
            # hi
            __metaclass__ = AppleMeta
            pass
        """
        b = """
        class X(metaclass=AppleMeta):
            # hi
            pass
        """
        self.check(b, a)

        a = """
        class X(object):
            __metaclass__ = Meta
            pass
            # Bedtime!
        """
        b = """
        class X(metaclass=Meta):
            pass
            # Bedtime!
        """
        self.check(b, a)

    def test_meta_noparent_odd_body(self):
        # no-parent class, odd body
        a = """
        class X(object):
            __metaclass__ = Q
            pass
        """
        b = """
        class X(metaclass=Q):
            pass
        """
        self.check(b, a)

    def test_meta_oneparent_no_body(self):
        # one parent class, no body
        a = """
        class X(object):
            __metaclass__ = Q
            pass"""
        b = """
        class X(object, metaclass=Q): pass"""
        self.check(b, a)

    def test_meta_oneparent_simple_body_1(self):
        # one parent, simple body
        a = """
        class X(object):
            __metaclass__ = Meta
            bar = 7
        """
        b = """
        class X(object, metaclass=Meta):
            bar = 7
        """
        self.check(b, a)

    def test_meta_oneparent_simple_body_2(self):
        a = """
        class X(object):
            __metaclass__ = Meta
            x = 4; g = 23
        """
        b = """
        class X(metaclass=Meta):
            x = 4; g = 23
        """
        self.check(b, a)

    def test_meta_oneparent_simple_body_3(self):
        a = """
        class X(object):
            __metaclass__ = Meta
            bar = 7
        """
        b = """
        class X(object, metaclass=Meta):
            bar = 7
        """
        self.check(b, a)

    def test_meta_multiparent_simple_body_1(self):
        # multiple inheritance, simple body
        a = """
        class X(clsA, clsB):
            __metaclass__ = Meta
            bar = 7
        """
        b = """
        class X(clsA, clsB, metaclass=Meta):
            bar = 7
        """
        self.check(b, a)

    def test_meta_multiparent_simple_body_2(self):
        # keywords in the class statement
        a = """
        class m(a, arg=23):
            __metaclass__ = Meta
            pass"""
        b = """
        class m(a, arg=23, metaclass=Meta):
            pass"""
        self.check(b, a)

    def test_meta_expression_simple_body_1(self):
        a = """
        class X(expression(2 + 4)):
            __metaclass__ = Meta
            pass
        """
        b = """
        class X(expression(2 + 4), metaclass=Meta):
            pass
        """
        self.check(b, a)

    def test_meta_expression_simple_body_2(self):
        a = """
        class X(expression(2 + 4), x**4):
            __metaclass__ = Meta
            pass
        """
        b = """
        class X(expression(2 + 4), x**4, metaclass=Meta):
            pass
        """
        self.check(b, a)

    def test_meta_noparent_simple_body(self):

        a = """
        class X(object):
            __metaclass__ = Meta
            save.py = 23
            out = 5
        """
        b = """
        class X(metaclass=Meta):
            save.py = 23
            out = 5
        """
        self.check(b, a)
