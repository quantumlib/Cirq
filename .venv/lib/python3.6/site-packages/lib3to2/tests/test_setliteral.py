from lib3to2.tests.support import lib3to2FixerTestCase

class Test_setliteral(lib3to2FixerTestCase):
    fixer = "setliteral"

    def test_unchanged_dict(self):
        s = """{"ghoul": 100, "zombie": 50, "gremlin": 40}"""
        self.unchanged(s)

        s = """{1: "spider", 2: "hills", 3: "bologna", None: "tapeworm"}"""
        self.unchanged(s)

        s = """{}"""
        self.unchanged(s)

        s = """{'a':'b'}"""
        self.unchanged(s)

    def test_simple_literal(self):
        b = """{'Rm 101'}"""
        a = """set(['Rm 101'])"""
        self.check(b, a)

    def test_multiple_items(self):
        b = """{'Rm 101',   'Rm 102',  spam,    ham,      eggs}"""
        a = """set(['Rm 101',   'Rm 102',  spam,    ham,      eggs])"""
        self.check(b, a)

        b = """{ a,  b,   c,    d,     e}"""
        a = """set([ a,  b,   c,    d,     e])"""
        self.check(b, a)

    def test_simple_set_comprehension(self):
        b = """{x for x in range(256)}"""
        a = """set([x for x in range(256)])"""
        self.check(b, a)

    def test_complex_set_comprehension(self):
        b = """{F(x) for x in range(256) if x%2}"""
        a = """set([F(x) for x in range(256) if x%2])"""
        self.check(b, a)

        b = """{(lambda x: 2000 + x)(x) for x, y in {(5, 400), (6, 600), (7, 900), (8, 1125), (9, 1000)}}"""
        a = """set([(lambda x: 2000 + x)(x) for x, y in set([(5, 400), (6, 600), (7, 900), (8, 1125), (9, 1000)])])"""
        self.check(b, a)
