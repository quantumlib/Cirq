from lib3to2.tests.support import lib3to2FixerTestCase

class Test_collections(lib3to2FixerTestCase):
    fixer = "collections"

    def test_from_UserDict(self):

        b = """
        from collections import UserDict"""
        a = """
        from UserDict import UserDict"""
        self.check(b, a)

    def test_from_UserList(self):

        b = """
        from collections import UserList"""
        a = """
        from UserList import UserList"""
        self.check(b, a)

    def test_from_UserString(self):

        b = """
        from collections import UserString"""
        a = """
        from UserString import UserString"""
        self.check(b, a)

    def test_using_UserDict(self):

        b = """
        class Scapegoat(collections.UserDict):
            pass"""
        a = """import UserDict

class Scapegoat(UserDict.UserDict):
    pass"""
        self.check(b, a)

    def test_using_UserList(self):

        b = """
        class Scapegoat(collections.UserList):
            pass"""
        a = """import UserList

class Scapegoat(UserList.UserList):
    pass"""
        self.check(b, a)

    def test_using_UserString(self):

        b = """
        class Scapegoat(collections.UserString):
            pass"""
        a = """import UserString

class Scapegoat(UserString.UserString):
    pass"""
        self.check(b, a)
