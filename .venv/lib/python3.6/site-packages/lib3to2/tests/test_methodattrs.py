from lib3to2.tests.support import lib3to2FixerTestCase

class Test_methodattrs(lib3to2FixerTestCase):
    fixer = "methodattrs"

    attrs = ["func", "self"]

    def test_methodattrs(self):
        for attr in self.attrs:
            b = "a.__%s__" % attr
            a = "a.im_%s" % attr
            self.check(b, a)

            b = "self.foo.__%s__.foo_bar" % attr
            a = "self.foo.im_%s.foo_bar" % attr
            self.check(b, a)

        b = "dir(self.foo.__self__.__class__)"
        a = "dir(self.foo.im_self.__class__)"
        self.check(b, a)

    def test_unchanged(self):
        for attr in self.attrs:
            s = "foo(__%s__ + 5)" % attr
            self.unchanged(s)
