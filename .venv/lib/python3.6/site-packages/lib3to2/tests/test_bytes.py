from lib3to2.tests.support import lib3to2FixerTestCase

class Test_bytes(lib3to2FixerTestCase):
    fixer = "bytes"

    def test_bytes_call_1(self):
        b = """bytes(x)"""
        a = """str(x)"""
        self.check(b, a)

    def test_bytes_call_2(self):
        b = """a = bytes(x) + b"florist" """
        a = """a = str(x) + "florist" """
        self.check(b, a)

    def test_bytes_call_noargs(self):
        b = """bytes()"""
        a = """str()"""
        self.check(b, a)

    def test_bytes_call_args_1(self):
        b = """bytes(x, y, z)"""
        a = """str(x).encode(y, z)"""
        self.check(b, a)

    def test_bytes_call_args_2(self):
        b = """bytes(encoding="utf-8", source="dinosaur", errors="dont-care")"""
        a = """str("dinosaur").encode("utf-8", "dont-care")"""
        self.check(b, a)

    def test_bytes_literal_1(self):
        b = '''b"\x41"'''
        a = '''"\x41"'''
        self.check(b, a)

    def test_bytes_literal_2(self):
        b = """b'x'"""
        a = """'x'"""
        self.check(b, a)

    def test_bytes_literal_3(self):
        b = """BR'''\x13'''"""
        a = """R'''\x13'''"""
        self.check(b, a)

    def test_bytes_concatenation(self):
        b = """b'bytes' + b'bytes'"""
        a = """'bytes' + 'bytes'"""
        self.check(b, a)
