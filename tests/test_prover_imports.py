"""The prover's layers: every internal import of rankbound/*.py (also inside functions) goes down the layer order
below, and the import graph has no cycle.  Mechanical, AST based, beside test_check_imports.py (the checker's
independence).  A name taken from the package itself (`from . import NAME`, NAME not a module) is an import of
__init__.py, which sits at the top; a dynamic import (__import__, importlib) of a prover module is not allowed."""
import ast, glob, os, unittest
import rankbound

PKG = os.path.dirname(os.path.abspath(rankbound.__file__))

# from the bottom up; a module may import the modules of lower layers, and of its own layer if no cycle results
LAYERS = (
    ('core', 'messages', 'machine'),            # contracts, errors, the user's words, the machine's resources
    ('runtime',),                               # run identity, budgets, artifacts, processes, the work folder
    ('arith', 'curve', 'field', 'linalg'),      # exact arithmetic and the native kernels' formats
    ('witness',),                               # the witness formats and statements
    ('sieve', 'ctp', 'bundle'),                 # the sieve's relations; the Cassels-Tate pairing; bundles
    ('descent', 'selmer'),                      # the routes' mathematics
    ('runner',),                                # normalize, route, execute, verify, publish
    ('cli', '__init__', '__main__'),            # the command line and the package's facade
)
LAYER = {m: i for i, ms in enumerate(LAYERS) for m in ms}
OUTSIDE = ('check',)                            # the checker (test_check_imports.py)


def modules():
    return sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(PKG, '*.py')))


def edges(path, mods):
    """(lineno, target) of every internal import in the source path: a prover module, '__init__' for a name of the
    package itself, or 'dynamic' for __import__/importlib."""
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.level > 1:
                out.append((n.lineno, 'outside'))
                continue
            if n.level == 1:
                head = n.module
            elif n.module == 'rankbound' or (n.module or '').startswith('rankbound.'):
                head = n.module[len('rankbound.'):] if '.' in n.module else None
            else:
                continue
            if head is None:                    # from . import a, b: modules, or names of the package
                for a in n.names:
                    out.append((n.lineno, a.name if a.name in mods or a.name in OUTSIDE else '__init__'))
            else:
                out.append((n.lineno, head.split('.')[0]))
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name == 'rankbound':
                    out.append((n.lineno, '__init__'))
                elif a.name.startswith('rankbound.'):
                    out.append((n.lineno, a.name.split('.')[1]))
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ''
            if name in ('__import__', 'import_module'):
                out.append((n.lineno, 'dynamic'))
    return [(l, t) for l, t in out if t not in OUTSIDE]


def graph():
    mods = set(modules())
    return {m: edges(os.path.join(PKG, m + '.py'), mods) for m in mods}


def cycle(g):
    """A cycle of the graph {module: [(lineno, target)]} as a list of modules, or None."""
    state, stack = {}, []

    def visit(m):
        state[m] = 1
        stack.append(m)
        for _, t in g.get(m, ()):
            if t == m or t not in g:
                continue
            if state.get(t) == 1:
                return stack[stack.index(t):] + [t]
            if t not in state:
                c = visit(t)
                if c:
                    return c
        state[m] = 2
        stack.pop()
        return None
    for m in sorted(g):
        if m not in state:
            c = visit(m)
            if c:
                return c
    return None


class ProverLayers(unittest.TestCase):
    def test_every_module_has_a_layer(self):
        self.assertEqual(sorted(set(modules()) - set(LAYER)), [])

    def test_imports_go_down(self):
        bad = []
        for m, es in sorted(graph().items()):
            for line, t in es:
                if t in ('dynamic', 'outside') or t not in LAYER:
                    bad.append('%s.py:%d imports %s' % (m, line, t))
                elif t != m and LAYER[t] > LAYER[m]:
                    bad.append('%s.py:%d imports %s, a higher layer' % (m, line, t))
        self.assertEqual(bad, [])

    def test_no_cycle(self):
        self.assertIsNone(cycle(graph()))

    def test_the_test_detects(self):
        """The scanner finds an import inside a function, a name of the package, an absolute import and a dynamic
        one; cycle() finds a cycle through them."""
        import tempfile
        src = ('from . import arith, RankboundError\ndef f():\n    from .runner import x\n    import rankbound.sieve\n'
               '    __import__("x")\n')
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(src)
        try:
            got = edges(f.name, {'arith', 'runner', 'sieve'})
        finally:
            os.unlink(f.name)
        self.assertEqual([t for _, t in got], ['arith', '__init__', 'runner', 'sieve', 'dynamic'])
        self.assertEqual(cycle({'a': [(1, 'b')], 'b': [(2, 'c')], 'c': [(3, 'a')]}), ['a', 'b', 'c', 'a'])
        self.assertIsNone(cycle({'a': [(1, 'b')], 'b': [(2, 'c')], 'c': []}))


if __name__ == '__main__':
    unittest.main()
