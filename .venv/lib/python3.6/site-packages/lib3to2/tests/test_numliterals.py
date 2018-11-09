from lib3to2.tests.support import lib3to2FixerTestCase

class Test_numliterals(lib3to2FixerTestCase):
    fixer = "numliterals"

    def test_octal_1(self):
        b = """0o755"""
        a = """0755"""
        self.check(b, a)

    def test_octal_2(self):
        b = """0o777"""
        a = """0777"""
        self.check(b, a)

    def test_bin_1(self):
        b = """0b10010110"""
        a = """__builtins__.long("10010110", 2)"""
        self.check(b, a)

    def test_bin_2(self):
        b = """spam(0b1101011010110)"""
        a = """spam(__builtins__.long("1101011010110", 2))"""
        self.check(b, a)

    def test_comments_and_spacing_2(self):
        b = """b = 0o755 # spam"""
        a = """b = 0755 # spam"""
        self.check(b, a)

    def test_unchanged_str(self):
        s = """'0x1400'"""
        self.unchanged(s)

        s = """'0b011000'"""
        self.unchanged(s)

        s = """'0o755'"""
        self.unchanged(s)

    def test_unchanged_other(self):
        s = """5.0"""
        self.unchanged(s)

        s = """5.0e10"""
        self.unchanged(s)

        s = """5.4 + 4.9j"""
        self.unchanged(s)

        s = """4j"""
        self.unchanged(s)

        s = """4.4j"""
        self.unchanged(s)
