"""The checker's independence: rankbound/check/ imports nothing of the prover.  Mechanical, AST based: every
import statement anywhere in the checker's Python sources (also inside functions) must be the standard library, a
module of rankbound/check itself, or absent; a relative import may not leave rankbound/check; dynamic imports
(__import__, importlib) are not allowed at all."""
import ast, glob, os, sys, unittest
import rankbound.check

CHECK = os.path.dirname(os.path.abspath(rankbound.check.__file__))
STD = getattr(sys, 'stdlib_module_names', None)


def imports(path):
    """(lineno, level, module, names) of every import statement and (lineno, 'dynamic') of every dynamic import."""
    tree = ast.parse(open(path).read(), path)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield n.lineno, 0, a.name, []
        elif isinstance(n, ast.ImportFrom):
            yield n.lineno, n.level, n.module or '', [a.name for a in n.names]
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ''
            if name in ('__import__', 'import_module'):
                yield n.lineno, -1, name, []


class CheckerBoundary(unittest.TestCase):
    def test_sources_found(self):
        self.assertTrue(glob.glob(os.path.join(CHECK, '*.py')))

    def test_checker_imports_nothing_of_the_prover(self):
        bad = []
        for path in sorted(glob.glob(os.path.join(CHECK, '**', '*.py'), recursive=True)):
            rel = os.path.relpath(path, CHECK)
            depth = rel.count(os.sep)                       # packages below rankbound/check
            for line, level, mod, names in imports(path):
                where = '%s:%d' % (rel, line)
                top = mod.split('.')[0]
                if level == -1:
                    bad.append('%s dynamic import %s' % (where, mod))
                elif level > depth + 1:
                    bad.append('%s relative import leaves rankbound/check (level %d)' % (where, level))
                elif level > 0:
                    continue                                # inside rankbound/check
                elif top == 'rankbound':
                    if mod != 'rankbound.check' and not mod.startswith('rankbound.check.'):
                        bad.append('%s imports %s' % (where, mod))
                elif STD is not None and top not in STD:
                    bad.append('%s imports non-standard module %s' % (where, mod))
        self.assertEqual(bad, [])

    def test_the_test_detects(self):
        """The scanner itself finds the three kinds of violation."""
        import tempfile
        src = 'import rankbound.sieve\nfrom .. import curve\ndef f():\n    __import__("x")\n'
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(src)
        try:
            got = list(imports(f.name))
        finally:
            os.unlink(f.name)
        self.assertEqual([(g[1], g[2]) for g in got], [(0, 'rankbound.sieve'), (2, ''), (-1, '__import__')])


if __name__ == '__main__':
    unittest.main()
