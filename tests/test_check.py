"""Fast tests of the checker (rankbound/check/): the statement of a witness, its independence from the prover (a copy
of rankbound/check/ alone checks a witness), hostile or broken witnesses, the command, and rankbound 3's names
(rankbound-*-witness-N) next to rankbound 2's (rankbound-*-witness-N)."""
import ast, gzip, hashlib, json, os, random, shutil, subprocess, sys, tempfile, unittest
from rankbound import check as checker
from rankbound.check import selmer as vsel
from rankbound.check import isogeny_statement, parity_bound, selmer_statement, statement
from helpers import PKG, SRC, TESTS, wit0, corpus

ROOT = os.path.dirname(TESTS)
MAIN = 'import sys; from rankbound.check import main; sys.exit(main(sys.argv[1:]))'      # rankbound check


def fails(c, d, name='t'):
    """Return the names of the failed checks of the witness dict c (written to the folder d)."""
    path = os.path.join(d, name + '.cert.json')
    with open(path, 'w') as f:
        json.dump(c, f)
    v = checker.check(path)
    assert v['verdict'] in ('accepted', 'rejected'), v['reason']
    return [x['check'] for x in v['checks'] if not x['ok']]


class TestClaims(unittest.TestCase):
    """The parity bound, and rank = r only for S = r."""

    def test_parity_bound(self):
        self.assertEqual(parity_bound(18, 1, 11, -1), 29)      # #728: 30 lowered to 29 by w = -1
        self.assertEqual(parity_bound(18, 1, 11, 1), 30)
        self.assertEqual(parity_bound(1, 1, 0, 1), 2)          # 571a1
        self.assertEqual(parity_bound(1, 1, 0, None), 2)

    def test_descent_parity(self):
        """The descent's bound follows the same parity rule, including a root number other than +-1."""
        from rankbound import descent
        for B in range(5):
            for w in (1, -1, 0, None):
                self.assertEqual(descent.bound([B, B + 3], w), (B, parity_bound(B, 0, 0, w)))

    def test_statement(self):
        F = {'u': 1, 'n': 11, 'w': -1}
        S, s = statement(18, F, 29)
        self.assertEqual(S, 29)
        self.assertEqual(s, 'Assuming the GRH, we have that rank E(Q) <= 29. As E(Q) has 29 independent rational '
                            'points, we therefore have rank E(Q) = 29.')
        self.assertIn('(ICARM #1; not proved in this certificate)', statement(18, F, 29, 'ICARM #1')[1])
        S, s = statement(19, F, 29)                            # S = 31 > r: an upper bound only
        self.assertEqual(S, 31); self.assertNotIn('rank E(Q) =', s)
        S, s = statement(16, F, 29)                            # S = 27 < r: never "rank = r"
        self.assertNotIn('rank E(Q) =', s); self.assertIn('CONTRADICTION', s)
        self.assertNotIn('rank E(Q) =', statement(18, F, None)[1])
        # rankbound 3's witnesses name themselves in the statement; rankbound 2's said "certificate"
        self.assertEqual(selmer_statement(2, 'GRH', 2, 'X'), 'Assuming the GRH, we have that rank E(Q) <= 2. As E(Q) '
                         'has 2 independent rational points (X; not proved in this witness), we therefore have rank E(Q) = 2.')
        self.assertIn('(X; not proved in this certificate)', selmer_statement(2, 'none', 2, 'X', 'certificate'))
        self.assertIn('(X; not proved in this witness)', isogeny_statement(1, -1, 1, 'X')[1])



class TestIsolation(unittest.TestCase):
    """rankbound/check/ imports only the standard library and itself; its GP is its own; `import rankbound.check`
    loads none of the prover's modules; and a copy of that folder alone (under an empty package rankbound, with
    nothing else of the prover), in a fresh interpreter, checks a witness of ICARM #0 (rank = 0; wit0)."""
    VDIR = os.path.join(PKG, 'check')

    def test_imports(self):
        stdlib = getattr(sys, 'stdlib_module_names', None)        # Python >= 3.10
        for name in sorted(os.listdir(self.VDIR)):
            if not name.endswith('.py'):
                continue
            with open(os.path.join(self.VDIR, name)) as f:
                tree = ast.parse(f.read(), name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertLessEqual(node.level, 1, '%s: import from above the package' % name)
                    mods = [node.module] if node.level == 0 else []
                elif isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                else:
                    continue
                for m in mods:
                    top = m.split('.')[0]
                    self.assertNotEqual(top, 'rankbound', '%s imports the prover (%s)' % (name, m))
                    if stdlib is not None:
                        self.assertIn(top, stdlib, '%s imports %s' % (name, m))

    @staticmethod
    def bare_libs(src):
        """Return the GP file names given bare (a file of rankbound/check, for run_gp and source) in src: the string
        elements of a libs=... tuple or list, of LIBS = ..., and of source(...)."""
        out = set()
        def names(v):
            return {e.value for e in getattr(v, 'elts', ()) if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call):
                out |= {k for kw in node.keywords if kw.arg == 'libs' for k in names(kw.value)}
                if getattr(node.func, 'id', getattr(node.func, 'attr', None)) == 'source':
                    out |= {a.value for a in node.args if isinstance(a, ast.Constant)}
            elif isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'LIBS' for t in node.targets):
                out |= names(node.value)
        return out

    def test_gp(self):
        """The checker's GP is its own: its one GP file check.gp reads no other file or calls bnfinit, bnfcertify,
        ellrank (results not certified), and its Python loads only it.  The prover loads none of it and imports
        nothing of rankbound.check but its command line (cli.py and __main__.py, for `rankbound check`; the prover
        checks its witnesses by `python3 -m rankbound check --json`)."""
        own = sorted(n for n in os.listdir(self.VDIR) if n.endswith('.gp'))
        self.assertEqual(own, ['check.gp'])
        self.assertEqual(sorted(n for n in os.listdir(self.VDIR) if n.endswith(('.py', '.gp', '.c'))),
                         sorted(checker.SOURCES))
        for name in sorted(n for n in os.listdir(self.VDIR) if n.endswith(('.py', '.gp'))):
            with open(os.path.join(self.VDIR, name)) as f:
                src = f.read()
            self.assertNotRegex(src, r'\b(bnfinit|bnfcertify|ellrank|bnfsunit|ellsaturation)\s*\(', name)   # uncertified
            if name.endswith('.gp'):
                self.assertNotRegex(src, r'\bread(vec|str)?\s*\(', name)
            else:
                self.assertLessEqual(self.bare_libs(src), set(own), name)
        seen = set()
        for d, _, files in os.walk(PKG):
            if os.path.abspath(d).startswith(os.path.abspath(self.VDIR)):
                continue
            for name in (n for n in files if n.endswith(('.py', '.gp'))):
                with open(os.path.join(d, name)) as f:
                    src = f.read()
                if name.endswith('.gp'):
                    self.assertNotRegex(src, r'\bread\s*\(', name)
                else:
                    seen |= self.bare_libs(src)
                    if name not in ('__main__.py', 'cli.py'):       # (they run `rankbound check`)
                        self.assertNotRegex(src, r'from \.+(check|verify)\b|import (check|verify)\b|rankbound\.check',
                                            name)
        self.assertEqual(seen, set())               # the prover runs no gp and imports nothing of the checker

    def test_lazy(self):
        code = ('import sys; sys.path.insert(0, %r); import rankbound.check; '
                'print(sorted(m for m in sys.modules if m.split(".")[0] == "rankbound"))' % SRC)
        r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "['rankbound', 'rankbound.check']")

    def test_main_lazy(self):
        """main(['--help']) returns 0 and loads the checker alone."""
        code = ('import sys; sys.path.insert(0, %r); from rankbound.check import main\n'
                'print(main(["--help"]))\nprint(sorted(m for m in sys.modules if m.split(".")[0] == "rankbound"))' % SRC)
        r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines()[-2:], ['0', "['rankbound', 'rankbound.check']"])

    def test_alone(self):
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, 'rankbound'))
            open(os.path.join(d, 'rankbound', '__init__.py'), 'w').close()       # no prover: an empty package
            shutil.copytree(self.VDIR, os.path.join(d, 'rankbound', 'check'), ignore=shutil.ignore_patterns('__pycache__'))
            with open(os.path.join(d, 'VERSION'), 'w') as f:
                f.write('b7a1b0fc\n')
            code = 'import sys; sys.path.insert(0, %r); ' % d + MAIN
            wit = wit0()
            r = subprocess.run([sys.executable, '-I', '-c', code, wit], cwd=d, capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            with open(wit, 'rb') as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            lines = r.stdout.splitlines()                 # the provenance line and the verdict: no per-check lines
            self.assertEqual(len(lines), 2, r.stdout)
            self.assertRegex(lines[0], r'^\d\d:\d\d:\d\d Checker %s with PARI/GP 2\.\d+\.\d+, \d{1,2} [A-Z][a-z]+ \d{4} at '
                             r'\d{1,2}:\d\d [AP]M\. Witness sha256 %s\.$' % (checker.checker_hash(), sha))
            self.assertRegex(lines[1], r'^\d\d:\d\d:\d\d icarm0 passes in \d+ seconds?\. Rank 0 \(assuming the GRH\)\.$')
            r = subprocess.run([sys.executable, '-I', '-c', code, '--json', wit], cwd=d, capture_output=True, text=True,
                               timeout=600)
            self.assertEqual(json.loads(r.stdout)['deploy'], 'b7a1b0fc')
        finally:
            shutil.rmtree(d)

    def test_deploy(self):
        """The deploy word: the first word of a VERSION file at the top of a checkout, or of a copy of the checker
        alone (above rankbound/), else '-'."""
        from unittest import mock
        with tempfile.TemporaryDirectory() as d:
            for here in (os.path.join(d, 'rankbound', 'check'),):
                os.makedirs(here)
                with mock.patch.object(checker, 'HERE', here):
                    self.assertEqual(checker.deployment(), '-')
                    with open(os.path.join(d, 'VERSION'), 'w') as f:
                        f.write('b7a1b0fc more\n')
                    self.assertEqual(checker.deployment(), 'b7a1b0fc')
                    os.remove(os.path.join(d, 'VERSION'))



class TestVerifierInput(unittest.TestCase):
    """A hostile or broken witness fails cleanly: no gp code from it runs, an out-of-range row index or a wrong
    relation is a failed check and not an exception, and one bad witness does not stop the others."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.d)            # gp runs here: the payload's path has no '/'
        with open(wit0()) as f:
            self.c0 = json.load(f)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.d)

    def write(self, name, c):
        path = os.path.join(self.d, name + '.cert.json')
        with open(path, 'w') as f:
            json.dump(c, f)
        return path

    def check(self, c, name):
        ok, res, _, _ = checker.verify(self.write(name, c))
        self.assertFalse(ok, name)
        return [x[0] for x in res if not x[1]]

    def test_injection(self):
        mark = os.path.join(self.d, 'MARK')
        evil = '0*#Str(system("touch MARK"))'
        i = next(k for k, l in enumerate(self.c0['relations']) if not l.startswith('f:'))
        j = next(k for k, l in enumerate(self.c0['relations']) if l.startswith('f:'))
        for what in ('a', 'b', 'free', 'ainvs', 'minimal_model', 'bad_primes', 'poly', 'dK', 'point', 'prime'):
            c = json.loads(json.dumps(self.c0))
            a, b = c['relations'][i].split(':')[0].split(',')
            hs = c['relations'][i].split(':')[1]
            if what in ('a', 'b'):
                c['relations'][i] = ('(%s+%s),%s:%s' % (a, evil, b, hs) if what == 'a' else '%s,(%s+%s):%s' % (a, b, evil, hs))
            elif what == 'free':
                c['relations'][j] += '+' + evil
            elif what in ('ainvs', 'minimal_model', 'bad_primes'):
                c['curve'][what][-1] += '+' + evil
            elif what in ('poly', 'dK'):
                fd = c['field']
                fd[what] = [fd['poly'][0] + '+' + evil] + fd['poly'][1:] if what == 'poly' else fd['dK'] + '+' + evil
            else:
                c['rank_lower_bound'] = {'r': 1, 'points': [['1', '2']], 'primes': [5]}
                if what == 'point':
                    c['rank_lower_bound']['points'][0][0] += '+' + evil
                else:
                    c['rank_lower_bound']['primes'][0] = '5+' + evil
            self.assertEqual(self.check(c, what), ['format'], what)
            self.assertFalse(os.path.exists(mark), what)

    def test_out_of_range(self):
        for what, (ops, piv) in {'op': ([[0, 10 ** 6]], None), 'negative': ([[-1, 0]], None),
                                 'pivot': ([], [10 ** 6]), 'bool': ([[True, 0]], None)}.items():
            c = json.loads(json.dumps(self.c0))
            c['ops'] = ops
            if piv is not None:
                c['pivots'] = piv
            self.assertIn(self.check(c, what), (['ops_in_range'], ['format']), what)

    def test_wrong_relation(self):
        c = json.loads(json.dumps(self.c0))
        i = next(k for k, l in enumerate(c['relations']) if not l.startswith('f:'))
        a, rest = c['relations'][i].split(',', 1)
        c['relations'][i] = '%d,%s' % (int(a) + 1, rest)
        self.assertEqual(self.check(c, 'rel'), ['rejected'])

    def test_batch(self):
        """A broken witness first, then a good one: the good one is still checked; exit status 1.  Without PARI/GP
        2.17 or later: one line on stderr and exit status 4 (through check(): test_errors)."""
        c = json.loads(json.dumps(self.c0))
        c['ops'] = [[0, 10 ** 6]]
        bad = self.write('a_bad', c)
        good = self.write('b_good', self.c0)
        r = subprocess.run([sys.executable, '-c', MAIN, bad, good], cwd=ROOT, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertRegex(r.stdout, r' icarm0 passes in \d+ seconds?\. Rank 0 \(assuming the GRH\)\.\n')
        self.assertIn(' icarm0 fails at the check ops_in_range.\n', r.stdout)
        self.assertTrue(r.stdout.endswith(' 1 of 2 witnesses pass. Failed: a_bad.cert.json.\n'), r.stdout)
        # without PARI/GP 2.17 or later (none on the PATH, or an older one): one line on stderr, exit status 4, no
        # verdicts
        with open(os.path.join(self.d, 'gp'), 'w') as f:
            f.write('#!/bin/sh\necho 2.15.4\n')
        os.chmod(os.path.join(self.d, 'gp'), 0o755)
        for path, found in ((os.path.join(self.d, 'none'), 'gp is not on the PATH'),
                            (self.d, 'found 2.15.4 at %s' % os.path.join(self.d, 'gp'))):
            r = subprocess.run([sys.executable, '-c', MAIN, bad, good], cwd=ROOT, capture_output=True, text=True,
                               timeout=600, env=dict(os.environ, PATH=path))
            self.assertEqual((r.returncode, r.stdout, r.stderr),
                             (4, '', 'rankbound check needs PARI/GP 2.17 or later; %s.\n' % found))

    def test_canonical(self):
        """One written form per number: '-0', '05', '+5', ' 5', '0/3' are refused."""
        for x in ('0', '7', '-7', '123456789012345678901234567890'):
            checker.form(x, 'num')
        for x in ('-0', '05', '+5', ' 5', '5 ', '', '1e3', 5):
            self.assertRaises(ValueError, checker.form, x, 'num')
        for x in ('0', '-3/4', '3'):
            checker.form(x, 'rat')
        for x in ('-0', '0/3', '3/0', '3/04'):
            self.assertRaises(ValueError, checker.form, x, 'rat')
        self.assertRaises(ValueError, vsel._gpvec, '-0,1:3')

    def test_unproved_r(self):
        """A rank lower bound with neither points and primes nor a source is malformed."""
        for rl in ({'r': 2}, {'r': 2, 'source': ''}, {'r': 2, 'source': ' '}, {'r': 2, 'source': 3}):
            c = json.loads(json.dumps(self.c0))
            c['rank_lower_bound'] = rl
            self.assertEqual(self.check(c, 'unproved'), ['format'], rl)



class TestBreak(unittest.TestCase):
    """Attempts to break the checker through check() (rankbound check --json): every malformed, hostile, mismatched
    or oversized input gets a verdict other than 'accepted', with a reason, never a traceback; inputs that only
    rewrite a valid proof (another model of the curve, a repeated relation or pivot) stay accepted."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        with open(wit0()) as f:
            self.raw = f.read()
        self.c0 = json.loads(self.raw)

    def tearDown(self):
        shutil.rmtree(self.d)

    def verdict(self, c, name='x', **kw):
        """Return check() of C (a dict, or the file's text or bytes) written as NAME.cert.json."""
        path = os.path.join(self.d, name + '.cert.json')
        with open(path, 'wb') as f:
            f.write(c if isinstance(c, bytes) else (c if isinstance(c, str) else json.dumps(c)).encode())
        v = checker.check(path, **kw)
        self.assertNotIn('Traceback', json.dumps(v))
        self.assertIn(v['verdict'], checker.EXIT)
        if v['verdict'] != 'accepted':
            self.assertTrue(v['reason'], v)
            self.assertIsNone(v['rank_upper_bound'])
        return v

    def changed(self, f):
        c = json.loads(self.raw)
        f(c)
        return c

    def test_accepted(self):
        v = self.verdict(self.raw)
        self.assertEqual((v['verdict'], v['rank_upper_bound'], v['rank'], v['r_proved'], v['name']),
                         ('accepted', 0, 0, True, 'icarm0'))
        self.assertEqual(v['sha256'], hashlib.sha256(self.raw.encode()).hexdigest())
        self.assertEqual(v['checker'], checker.checker_hash())
        self.assertIn('not BSD', v['hypothesis'])
        # the JSON line is the record of a check: the provenance (no host) and every check with its info
        self.assertEqual(list(v), ['verdict', 'reason', 'witness', 'sha256', 'name', 'curve', 'rank_upper_bound',
                                   'rank_lower_bound', 'r_proved', 'rank', 'G', 'u', 'n', 'w', 'GRH', 'method',
                                   'torsion', 'hypothesis', 'statement',
                                   'checks', 'checker', 'deploy', 'gp', 'date', 'seconds', 'rank_check'])
        self.assertEqual([(m['mode'], m['by']) for m in v['rank_check']], [('whole', 'memory')])
        self.assertRegex(v['date'], r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4}$')
        self.assertNotEqual(v['gp'], '?')
        info = {x['check']: x['info'] for x in v['checks']}
        self.assertIn('exact interval arithmetic', info['bdf_GB'])
        self.assertRegex(info['relations_checked'], r'^\d+ rows recomputed \(\d+ s, 1 jobs\)$')
        self.assertRegex(info['dimV'], r'^dim V = \d+ \(rank \d+, \d+ large columns, \d+ s\)$')

    def test_sqrt_sign(self):
        """A local point where L vanishes at PARI's square root r: the class is taken at (s, t, -r)."""
        with open(os.path.join(os.path.dirname(__file__), 'data', 'sqrt_sign.cert.json')) as f:
            v = self.verdict(f.read())
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 1))

    def test_not_json(self):
        """Not a witness file at all: garbage, empty, truncated anywhere, gzipped garbage, a truncated gzip, UTF-16,
        another JSON value.  The witness gzipped is read (NAME.witness.json.gz)."""
        raw = self.raw.encode()
        cuts = [0, 1, len(raw) // 3, len(raw) // 2, len(raw) - 2] + random.Random(1).sample(range(2, len(raw) - 2), 40)
        gz = gzip.compress(raw)
        self.assertEqual(self.verdict(gz, 'gz')['verdict'], 'accepted')
        cases = [b'\x00\xff garbage', b'', gzip.compress(b'\x00\xff garbage'), gz[:len(gz) // 2],
                 gzip.compress(raw[:len(raw) // 2]), self.raw.encode('utf-16'), b'\xef\xbb\xbf' + raw,
                 b'[1, 2]', b'42', b'null', b'"text"'] + [raw[:k] for k in cuts]
        for k, b in enumerate(cases):
            v = self.verdict(b, 'n%d' % k)
            self.assertEqual((v['verdict'], v['checks'][0]['check']), ('rejected', 'format'), (k, b[:40], v['reason']))

    def test_json_tricks(self):
        """Duplicate keys (a reader sees the first, json keeps the last), NaN and Infinity, deep nesting."""
        dup = self.raw.replace('"claims": {', '"claims": {"S": 5, ', 1)
        self.assertEqual(self.verdict(dup)['reason'], 'not a valid witness: the key "S" appears twice')
        nan = self.raw.replace('"GB": %d' % self.c0['field']['GB'], '"GB": NaN', 1)
        self.assertNotEqual(nan, self.raw)
        self.assertEqual(self.verdict(nan)['reason'], 'not a valid witness: NaN is not a JSON number')
        self.assertEqual(self.verdict('[' * 100000 + ']' * 100000)['verdict'], 'rejected')
        self.assertEqual(self.verdict('{"a": ' * 100000 + '1' + '}' * 100000)['verdict'], 'rejected')

    def test_types_and_keys(self):
        """Wrong types, missing or extra keys, non-canonical or exotic numbers, unsafe names and sources, r < 0."""
        def put(path, value):
            def f(c):
                d = c
                for k in path[:-1]:
                    d = d[k]
                d[path[-1]] = value
            return f
        cases = {
            'name int': put(['name'], 7), 'name newline': put(['name'], 'x\n12:00:00 VERIFY PASS icarm1'),
            'name long': put(['name'], 'a' * 65), 'name unicode': put(['name'], 'icarm٠'),
            'curve list': put(['curve'], [1, 2]), 'ainvs ints': put(['curve', 'ainvs'], [0, -1, 1, 0, 0]),
            'ainvs 4': put(['curve', 'ainvs'], ['0', '-1', '1', '0']), 'ainvs float': put(['curve', 'ainvs', 1], '-1.0'),
            'ainvs arabic digits': put(['curve', 'ainvs', 2], '١'), 'ainvs fullwidth': put(['curve', 'ainvs', 2], '１'),
            'GB string': put(['field', 'GB'], '10'), 'GB float': put(['field', 'GB'], 10.0), 'GB bool': put(['field', 'GB'], True),
            'r bool': put(['rank_lower_bound', 'r'], False), 'r float': put(['rank_lower_bound', 'r'], 0.0),
            'r negative': lambda c: c.update(rank_lower_bound={'r': -1, 'source': 'x'}),
            'source newline': lambda c: c.update(rank_lower_bound={'r': 1, 'source': 'ICARM\nVERIFY PASS'}),
            'source long': lambda c: c.update(rank_lower_bound={'r': 1, 'source': 'x' * 201}),
            'source and points': put(['rank_lower_bound', 'source'], 'x'),
            'relations dict': put(['relations'], {}), 'relation int': put(['relations', 0], 5),
            'op of 3': put(['ops'], [[0, 1, 2]]), 'pivot string': put(['pivots'], ['0']),
            'statement list': put(['statement'], ['x']), 'created int': put(['created'], 2026),
            'extra key': put(['note'], 'rank 40'), 'extra claim': put(['claims', 'rank'], 40),
            'S negative': put(['claims', 'S'], -1), 'G negative': put(['claims', 'G'], -1), 'w = 0': put(['local', 'w'], 0),
            'format': put(['format'], 'rankbound-selmer-certificate-1'),
            'format 3 renamed': put(['format'], 'rankbound-selmer-witness-3'),
        }
        for k in KEYS_REQUIRED:
            cases['no ' + k] = (lambda k: lambda c: c.pop(k))(k)
        for what, f in cases.items():
            v = self.verdict(self.changed(f), 'k')
            self.assertEqual((v['verdict'], v['checks'][0]['check']), ('rejected', 'format'), (what, v['reason']))

    def test_wrong_claims(self):
        """A bound below the truth (S, G or dim V changed, with or without the statement to match), a wrong field
        datum, a relation whose primes are wrong, a fake prime: rejected by the check that fails."""
        S = self.c0['claims']['S']
        def claims(**kw):
            return lambda c: c['claims'].update(kw)
        cases = {
            'S + 1': (claims(S=S + 1), 'claims'), 'G + 1': (claims(G=self.c0['claims']['G'] + 1), 'claims'),
            'dimV + 1': (claims(dimV=self.c0['claims']['dimV'] + 1), 'dimV'),
            'statement': (lambda c: c.update(statement=c['statement'].replace('<= 0', '<= 1')), 'statement'),
            'ns + 1': (lambda c: c['field'].update(ns=c['field']['ns'] + 1), 'ns'),
            'GB / 2': (lambda c: c['field'].update(GB=c['field']['GB'] // 2), 'bdf_GB'),
            'w': (lambda c: c['local'].update(w=-c['local']['w']), 'root_number'),
            'u': (lambda c: c['local'].update(u=3 - c['local']['u']), 'bk_u'),
            'dK': (lambda c: c['field'].update(dK=str(4 * int(c['field']['dK']))), 'maximal_order'),
            'bad prime': (lambda c: c['curve'].update(bad_primes=['13']), 'disc_factored'),
            # a good prime among the bad ones (11a3: disc -11) has v_p = 0; a repeat doubles the product
            'extra bad prime': (lambda c: c['curve'].update(bad_primes=c['curve']['bad_primes'] + ['7']), 'disc_factored'),
            'repeated bad prime': (lambda c: c['curve'].update(bad_primes=c['curve']['bad_primes'] * 2), 'disc_factored'),
            'minimal model': (lambda c: c['curve'].update(minimal_model=['0', '2', '1', '1', '0']), 'minimal_model'),
            'cited r above S': (lambda c: c.update(rank_lower_bound={'r': 2, 'source': 'a claim'}), 'statement'),
        }
        rel = next(k for k, l in enumerate(self.c0['relations']) if not l.startswith('f:') and ':' in l and l[-1] != ':')
        a, b = self.c0['relations'][rel].split(':')[0].split(',')
        cases['relation primes'] = (lambda c: c['relations'].__setitem__(rel, '%s,%s:2,3' % (a, b)), 'rejected')
        cases['fake free prime'] = (lambda c: c['relations'].append('f:15'), 'rejected')
        for what, (f, chk) in cases.items():
            v = self.verdict(self.changed(f), 'w')
            failed = [x['check'] for x in v['checks'] if not x['ok']]
            self.assertEqual(v['verdict'], 'rejected', (what, v))
            self.assertIn(chk, failed, (what, failed, v['reason']))
            self.assertTrue(v['reason'].startswith('failed check: %s' % failed[0]), (what, v['reason']))
        v = self.verdict(self.changed(cases['relation primes'][0]), 'w')
        self.assertRegex(v['reason'], r'^failed check: rejected: relation \d+ does not factor as claimed$')
        lines = []
        checker.check(os.path.join(self.d, 'w.cert.json'), log=lines.append)
        self.assertRegex(lines[-1], r'^icarm0 fails at the check rejected\. Relation \d+ does not factor as claimed\.$')
        self.assertEqual(self.verdict(self.changed(cases['S + 1'][0]))['reason'], 'failed check: claims')

    def test_mutations(self):
        """Random changes of one byte anywhere in the file (a fixed seed): each is 'accepted' (a change the proof does not
        depend on, e.g. in 'created') or 'rejected', never an error or a traceback."""
        rnd, raw, seen = random.Random(7), self.raw.encode(), {}
        for k in range(150):
            i = rnd.randrange(len(raw))
            b = raw[:i] + bytes([rnd.choice(b'0123456789abcdef-,:"[]{} \\x\n')]) + raw[i + 1:]
            v = self.verdict(b, 'm')
            self.assertIn(v['verdict'], ('accepted', 'rejected'), (i, v['reason']))
            seen[v['verdict']] = seen.get(v['verdict'], 0) + 1
        self.assertGreater(seen.get('rejected', 0), 100, seen)

    def test_harmless(self):
        """Rewrites that keep the proof valid stay accepted: a relation repeated at the end, a pivot repeated."""
        c = self.changed(lambda c: c['relations'].append(c['relations'][0]))
        self.assertEqual(self.verdict(c)['verdict'], 'accepted')
        if self.c0['pivots']:
            c = self.changed(lambda c: c['pivots'].append(c['pivots'][0]))
            self.assertEqual(self.verdict(c)['verdict'], 'accepted')

    def test_curve(self):
        """--curve: the witness must be for that curve, on any model (11a3 = ICARM #0: shifted x -> x + 1, scaled
        by u = 2), and not for another (571a1, a quadratic twist): rejected before gp runs."""
        for a in ([0, -1, 1, 0, 0], [0, 2, 1, 1, 0], [0, -4, 8, 0, 0]):
            self.assertEqual(self.verdict(self.c0, curve=a)['verdict'], 'accepted', a)
        for a in ([0, -1, 1, -929, -10595], [0, 1, 1, -1, 0]):
            v = self.verdict(self.c0, curve=a)
            self.assertEqual((v['verdict'], v['checks'][1]['check']), ('rejected', 'curve'), v['reason'])
            self.assertLess(v['seconds'], 5)
        same = checker.same_curve
        self.assertTrue(same([0, 0, 0, 0, 1], [0, 0, 0, 0, 64]))           # j = 0: 64 = 2^6
        self.assertFalse(same([0, 0, 0, 0, 1], [0, 0, 0, 0, 8]))           # 8: a cube, not a sixth power
        self.assertTrue(same([0, 0, 0, 1, 0], [0, 0, 0, 16, 0]))           # j = 1728: 16 = 2^4
        self.assertFalse(same([0, 0, 0, 1, 0], [0, 0, 0, 4, 0]))           # 4 = (sqrt 2)^4: over Q(sqrt 2) only
        self.assertFalse(same([0, 0, 0, 1, 0], [0, 0, 0, -4, 0]))
        big = 10 ** 600 + 7
        self.assertTrue(same([0, 0, 0, -3, big], [0, 0, 0, -3 * 5 ** 4, big * 5 ** 6]))
        for a in ([0, 0, 0, 0, 0], [0, 0, 0, -3, 2]):                   # singular (a cusp, a node): not a curve, no error
            self.assertFalse(same(a, a))
            c = self.changed(lambda c: c['curve'].update(ainvs=[str(x) for x in a]))
            v = self.verdict(c, curve=a)                                # malformed (a singular model), as the 2-isogeny format
            self.assertEqual((v['verdict'], [x['check'] for x in v['checks']]), ('rejected', ['format']), v['reason'])

    def test_limits(self):
        """Over the size or time limit, GB above Bach's bound, a huge prime of the points or a huge number: 'limit',
        not checked (and quickly)."""
        v = self.verdict(self.c0, max_mb=1e-4)
        self.assertEqual((v['verdict'], v['reason']), ('limit', 'x.cert.json was not checked, because it is larger than '
                                                                 '0.0001 MB, the limit set by --max-mb.'))
        self.assertIsNone(v['sha256'])
        v = self.verdict(self.c0, max_seconds=0.01)
        self.assertEqual((v['verdict'], v['reason']), ('limit', 'x.cert.json was not checked within 0 seconds, the limit '
                                                                 'set by --max-seconds.'))
        self.assertRegex(self.verdict(self.changed(lambda c: c['field'].update(GB=10 ** 30)))['reason'],
                         r"^x\.cert\.json was not checked, because it asks for GB = %d, above Bach's bound \d+\.$" % 10 ** 30)
        for what, f in {'GB': lambda c: c['field'].update(GB=10 ** 30),
                        'prime': lambda c: c.update(rank_lower_bound={'r': 1, 'points': [['1', '2']], 'primes': [2 ** 64 + 13]}),
                        'digits': lambda c: c['curve']['ainvs'].__setitem__(4, '9' * 5000),
                        'relation': lambda c: c['relations'].append('1,1:' + 'f' * 5000)}.items():
            v = self.verdict(self.changed(f), what)
            self.assertEqual(v['verdict'], 'limit', (what, v['reason']))
            self.assertLess(v['seconds'], 5, what)

    def test_errors(self):
        """No file, a folder, gp missing: 'error' (the check could not run), not a verdict on a witness."""
        for path in (os.path.join(self.d, 'none.cert.json'), self.d):
            v = checker.check(path)
            self.assertEqual(v['verdict'], 'error', path); self.assertIn('Error', v['reason'])
        p = os.path.join(self.d, 'x.cert.json')
        with open(p, 'w') as f:
            f.write(self.raw)
        v = checker.check(p, gp=os.path.join(self.d, 'no-gp'))
        self.assertEqual(v['verdict'], 'error')
        self.assertEqual(v['reason'], 'x.cert.json was not checked, because the checker could not run. ' + checker.NO_GP)
        self.assertEqual(v['gp'], '?')
        self.assertRaises(ValueError, checker.verify, p, cores=0)

    def test_files(self):
        """FILE arguments: a work folder (its witnesses one level down), a folder whose name has glob characters, a FIFO
        (a usage error, at once), --curve as a4,a6 or in [] with spaces, a JSON file that is no witness, a gzip bomb
        (limit, without reading it all)."""
        good = os.path.join(self.d, 'w[1]', 'icarm0')
        os.makedirs(good)
        with open(os.path.join(good, 'icarm0.cert.json'), 'w') as f:
            f.write(self.raw)
        def run(*args):
            return subprocess.run([sys.executable, '-c', MAIN] + list(args), cwd=ROOT, capture_output=True, text=True,
                                  timeout=600)
        for x in (os.path.join(self.d, 'w[1]'), good):
            r = run('--json', x)
            self.assertEqual((r.returncode, json.loads(r.stdout)['verdict']), (0, 'accepted'), x)
        empty = os.path.join(self.d, 'empty')
        os.makedirs(empty)
        r = run(empty)
        self.assertEqual((r.returncode, r.stderr), (2, 'rankbound check: error: No witness in %s, nor in its folders. See '
                                                       'rankbound check --help.\n' % empty))
        fifo = os.path.join(self.d, 'f.witness.json')
        os.mkfifo(fifo)
        r = run('--max-seconds', '3', fifo)
        self.assertEqual((r.returncode, r.stderr), (2, 'The file %s cannot be read (not a regular file).\n' % fifo))
        self.assertEqual(checker._ainvs('-1,0'), [0, 0, 0, -1, 0])
        self.assertEqual(checker._ainvs(' [0, -1, 1, 0, 0] '), [0, -1, 1, 0, 0])
        for bad in ('0,-1,1,0', '0,\uff11', '1_0,1', '+1,1', '0,-0'):
            self.assertRaises(Exception, checker._ainvs, bad)
        w = os.path.join(good, 'icarm0.cert.json')
        r = run('--json', '--curve', '[0, -1, 1, 0, 0]', w)
        self.assertEqual((r.returncode, json.loads(r.stdout)['verdict']), (0, 'accepted'))
        v = self.verdict({'curves': []})
        self.assertEqual(v['reason'], 'not a valid witness: ' + checker.NOT_WITNESS)
        lines = []
        checker.check(os.path.join(self.d, 'x.cert.json'), log=lines.append)
        self.assertEqual(lines[-1], 'x.cert.json is not a rankbound witness (no known "format").')
        bomb = os.path.join(self.d, 'bomb.witness.json.gz')
        with gzip.open(bomb, 'wb') as f:
            for _ in range(200):
                f.write(b' ' * 10 ** 6)
        v = checker.check(bomb)
        self.assertEqual((v['verdict'], v['reason']), ('limit', 'bomb.witness.json.gz was not checked, because its JSON '
                                                                'is more than 50 times the size of its gzip file, unlike a '
                                                                'witness.'))
        old = checker.JSON_MB
        checker.JSON_MB = 1
        try:
            with open(bomb, 'wb') as f:
                f.write(gzip.compress(os.urandom(1500000)))
            v = checker.check(bomb)
            self.assertEqual(v['reason'], 'bomb.witness.json.gz was not checked, because its JSON is larger than 1 MB, the '
                                          'limit without --max-mb.')
        finally:
            checker.JSON_MB = old

    def test_cli(self):
        """rankbound check --json: one JSON object per line and nothing else; the exit status of the worst verdict;
        usage errors (no witness, --curve with two, --cores 0, a malformed --curve) exit 2 without a traceback; a FILE
        that does not exist or cannot be opened: one line on stderr, exit 2, nothing checked (a malformed one fails)."""
        good = os.path.join(self.d, 'icarm0.cert.json')
        bad = os.path.join(self.d, 'bad.cert.json')
        with open(good, 'w') as f:
            f.write(self.raw)
        with open(bad, 'w') as f:
            f.write(self.raw[:100])
        def run(*args):
            return subprocess.run([sys.executable, '-c', MAIN] + list(args), cwd=ROOT, capture_output=True, text=True,
                                  timeout=600)
        r = run('--json', good)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['verdict'], 'accepted')
        r = run('--json', good, bad)
        self.assertEqual(r.returncode, 1)
        self.assertEqual([json.loads(l)['verdict'] for l in r.stdout.splitlines()], ['accepted', 'rejected'])
        r = run('--json', '--max-mb', '0.0001', good)
        self.assertEqual((r.returncode, json.loads(r.stdout)['verdict']), (3, 'limit'))
        r = run('--json', '--curve', '0,2,1,1,0', good)
        self.assertEqual((r.returncode, json.loads(r.stdout)['verdict']), (0, 'accepted'))
        r = run('--curve', '0,-1,1,-929,-10595', good)
        self.assertEqual(r.returncode, 1)
        self.assertIn(' icarm0 fails at the check curve. It is not a witness for the curve 0,-1,1,-929,-10595.\n', r.stdout)
        r = run(bad)
        self.assertEqual(r.returncode, 1)
        self.assertIn(' bad.cert.json fails at the check format. It is not valid JSON at line ', r.stdout)
        r = run('--max-mb', '0.0001', good)
        self.assertEqual(r.returncode, 3)
        self.assertTrue(r.stdout.endswith(' icarm0.cert.json was not checked, because it is larger than 0.0001 MB, the '
                                          'limit set by --max-mb.\n'), r.stdout)
        for args in ([os.path.join(self.d, 'empty')], ['--curve', '0,-1,1,0,0', good, bad], ['--cores', '0', good],
                     ['--curve', '0,-1,1,0', good], ['--curve', '0,-1,1,0,x', good], ['--max-seconds', 'nan', good]):
            if args[0].endswith('empty'):
                os.makedirs(args[0])
            r = run(*args)
            self.assertEqual(r.returncode, 2, (args, r.stdout, r.stderr))
            self.assertNotIn('Traceback', r.stderr)
        none = os.path.join(self.d, 'none.cert.json')
        for args in ([none], ['--json', good, none], [good, os.path.join(self.d, 'nofolder')]):
            r = run(*args)
            self.assertEqual((r.returncode, r.stdout, r.stderr), (2, '', 'The file %s cannot be read (No such file or '
                                                                   'directory).\n' % args[-1]), args)
        locked = os.path.join(self.d, 'locked.cert.json')
        with open(locked, 'w') as f:
            f.write(self.raw)
        os.chmod(locked, 0)
        try:
            if not os.access(locked, os.R_OK):          # (root reads it anyway)
                r = run(good, locked)
                self.assertEqual((r.returncode, r.stdout, r.stderr),
                                 (2, '', 'The file %s cannot be read (Permission denied).\n' % locked))
        finally:
            os.chmod(locked, 0o644)


KEYS_REQUIRED = [k for k in vsel.SPEC3 if not k.endswith('?')]


class TestFormats(unittest.TestCase):
    """Class group witnesses: format 3 records torsion and GRH "yes" (checked); format 2 (without them) is still
    accepted."""

    def test_formats(self):
        with open(wit0()) as f:
            c3 = json.load(f)
        self.assertEqual((c3['format'], c3['torsion'], c3['GRH']), ('rankbound-selmer-certificate-3', [5], 'yes'))
        with tempfile.TemporaryDirectory() as d:
            c2 = {k: v for k, v in c3.items() if k not in ('torsion', 'GRH')}
            c2['format'] = 'rankbound-selmer-certificate-2'
            self.assertEqual(fails(c2, d), [])
            path = os.path.join(d, 't.cert.json')
            v = checker.check(path)
            self.assertEqual((v['GRH'], v['torsion'], v['method']), ('yes', None, 'class group'))
            self.assertEqual(fails(dict(c3, torsion=[3]), d), ['torsion'])
            self.assertEqual(fails(dict(c3, GRH='no'), d), ['GRH'])
            self.assertEqual(fails(dict(c3, torsion=[2]), d), ['format'])        # E(Q)[2] = 0 here
            self.assertEqual(fails(dict(c2, torsion=[5], GRH='yes'), d), ['format'])


class TestNames(unittest.TestCase):
    """rankbound 3's witnesses (tests/data/*.witness.json: corpus witnesses with the format strings
    rankbound-selmer-witness-4 and rankbound-2descent-witness-1) pass as their rankbound 2 twins do, with the same
    bound; a statement citing r names the format's own word ("witness" or "certificate"), and the other word fails."""
    NEW = {'37a1': 1, '571a1': 0, 'n113': 0, '37a1_cited': 1}

    def test_new_names(self):
        C = corpus()
        with tempfile.TemporaryDirectory() as d:
            for name, S in self.NEW.items():
                path = os.path.join(TESTS, 'data', name + '.witness.json')
                v = checker.check(path)
                self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', S), (name, v['reason']))
                with open(path) as f:
                    c = json.load(f)
                self.assertIn('-witness-', c['format'])
                if name in C:           # the same witness under rankbound 2's name
                    self.assertEqual(dict(c, format=c['format'].replace('-witness-', '-certificate-')), C[name][1])
                    self.assertEqual(fails(C[name][1], d, 'old'), [])
            with open(os.path.join(TESTS, 'data', '37a1_cited.witness.json')) as f:
                c = json.load(f)
            self.assertIn('not proved in this witness', c['statement'])
            old = dict(c, format='rankbound-selmer-certificate-4')
            self.assertEqual(fails(old, d), ['statement'])              # rankbound 2's word is "certificate"
            old['statement'] = c['statement'].replace('this witness', 'this certificate')
            self.assertEqual(fails(old, d), [])
            self.assertEqual(fails(dict(old, format=c['format']), d), ['statement'])


if __name__ == '__main__':
    unittest.main()
