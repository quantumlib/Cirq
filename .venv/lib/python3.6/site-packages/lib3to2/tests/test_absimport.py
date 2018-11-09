from lib3to2.tests.support import lib3to2FixerTestCase

class Test_absimport(lib3to2FixerTestCase):
    fixer = 'absimport'
  
    def test_import(self):
      a = 'import abc'
      b = 'from __future__ import absolute_import\nimport abc'
      
      self.check(a, b)
      
    def test_no_imports(self):
      a  = '2+2'
      
      self.unchanged(a)
