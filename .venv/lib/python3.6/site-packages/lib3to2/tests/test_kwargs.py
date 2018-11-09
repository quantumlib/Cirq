from lib3to2.tests.support import lib3to2FixerTestCase

class Test_kwargs(lib3to2FixerTestCase):
    fixer = 'kwargs'

    def test_basic_unchanged(self):
        s = """
        def spam(ham, eggs): funky()"""
        self.unchanged(s)


    def test_args_kwargs_unchanged(self):
        s = """
        def spam(ham, *args, **kwargs): funky()"""
        self.unchanged(s)


    def test_args_named_pos(self):
        b = """
        def spam(ham, *args, eggs, monkeys): funky()"""
        a = """
        def spam(ham, *args, **_3to2kwargs):
            monkeys = _3to2kwargs['monkeys']; del _3to2kwargs['monkeys']
            eggs = _3to2kwargs['eggs']; del _3to2kwargs['eggs']
            funky()"""
        self.check(b, a)


    def test_args_named_pos_catchall(self):
        b = """
        def spam(ham, *args, eggs, monkeys, **stuff): funky()"""
        a = """
        def spam(ham, *args, **stuff):
            monkeys = stuff['monkeys']; del stuff['monkeys']
            eggs = stuff['eggs']; del stuff['eggs']
            funky()"""
        self.check(b, a)


    def test_bare_star_named(self):
        b = """
        def spam(ham, *, eggs, monkeys):
            funky()"""
        a = """
        def spam(ham, **_3to2kwargs):
            monkeys = _3to2kwargs['monkeys']; del _3to2kwargs['monkeys']
            eggs = _3to2kwargs['eggs']; del _3to2kwargs['eggs']
            funky()"""
        self.check(b, a)


    def test_bare_star_named_simple_defaults(self):
        b = """
        def spam(ham, *, dinosaurs, eggs=3, monkeys=2):
            funky()"""
        a = """
        def spam(ham, **_3to2kwargs):
            if 'monkeys' in _3to2kwargs: monkeys = _3to2kwargs['monkeys']; del _3to2kwargs['monkeys']
            else: monkeys = 2
            if 'eggs' in _3to2kwargs: eggs = _3to2kwargs['eggs']; del _3to2kwargs['eggs']
            else: eggs = 3
            dinosaurs = _3to2kwargs['dinosaurs']; del _3to2kwargs['dinosaurs']
            funky()"""
        self.check(b, a)


    def test_bare_star_named_simple_defaults_catchall(self):
        b = """
        def spam(ham, *, dinosaurs, eggs=3, monkeys=2, **stuff):
            funky()"""
        a = """
        def spam(ham, **stuff):
            if 'monkeys' in stuff: monkeys = stuff['monkeys']; del stuff['monkeys']
            else: monkeys = 2
            if 'eggs' in stuff: eggs = stuff['eggs']; del stuff['eggs']
            else: eggs = 3
            dinosaurs = stuff['dinosaurs']; del stuff['dinosaurs']
            funky()"""
        self.check(b, a)

    def test_bare_star_named_complicated_defaults(self):
        b = """
        def spam(ham, *, dinosaurs, eggs=call_fn(lambda a: b), monkeys=[i.split() for i in something(args)]):
            funky()"""
        a = """
        def spam(ham, **_3to2kwargs):
            if 'monkeys' in _3to2kwargs: monkeys = _3to2kwargs['monkeys']; del _3to2kwargs['monkeys']
            else: monkeys = [i.split() for i in something(args)]
            if 'eggs' in _3to2kwargs: eggs = _3to2kwargs['eggs']; del _3to2kwargs['eggs']
            else: eggs = call_fn(lambda a: b)
            dinosaurs = _3to2kwargs['dinosaurs']; del _3to2kwargs['dinosaurs']
            funky()"""
        self.check(b, a)


    def test_bare_star_named_complicated_defaults_catchall(self):
        b = """
        def spam(ham, *, dinosaurs, eggs=call_fn(lambda a: b), monkeys=[i.split() for i in something(args)], **stuff):
            funky()"""
        a = """
        def spam(ham, **stuff):
            if 'monkeys' in stuff: monkeys = stuff['monkeys']; del stuff['monkeys']
            else: monkeys = [i.split() for i in something(args)]
            if 'eggs' in stuff: eggs = stuff['eggs']; del stuff['eggs']
            else: eggs = call_fn(lambda a: b)
            dinosaurs = stuff['dinosaurs']; del stuff['dinosaurs']
            funky()"""
        self.check(b, a)

