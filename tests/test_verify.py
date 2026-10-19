"""Fast tests of the verifier: the statement of a certificate, its independence from the prover (a copy of
rankbound/verify/ alone verifies a certificate), and hostile or broken certificates."""
import ast, gzip, hashlib, json, os, random, shutil, subprocess, sys, tempfile, unittest
from rankbound import verify as verifier
from rankbound.verify import parity_bound, statement
from helpers import TESTS, cert0


class TestClaims(unittest.TestCase):
    """The parity bound, and rank = r only for S = r."""

    def test_parity_bound(self):
        self.assertEqual(parity_bound(18, 1, 11, -1), 29)      # #728: 30 lowered to 29 by w = -1
        self.assertEqual(parity_bound(18, 1, 11, 1), 30)
        self.assertEqual(parity_bound(1, 1, 0, 1), 2)          # 571a1
        self.assertEqual(parity_bound(1, 1, 0, None), 2)

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



class TestIsolation(unittest.TestCase):
    """rankbound/verify/ imports only the standard library and itself; `import rankbound.verify` loads none of the
    prover's modules; and a copy of that folder alone (under an empty package rankbound, with nothing else of the
    prover), in a fresh interpreter, verifies a certificate of ICARM #0 (rank = 0; cert0)."""
    VDIR = os.path.join(os.path.dirname(TESTS), 'rankbound', 'verify')

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

    def test_lazy(self):
        code = ('import sys; sys.path.insert(0, %r); import rankbound.verify; '
                'print(sorted(m for m in sys.modules if m.split(".")[0] == "rankbound"))' % os.path.dirname(TESTS))
        r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "['rankbound', 'rankbound.verify']")

    def test_cli_lazy(self):
        """The command `rankbound verify` (rankbound/__main__.py) loads the verifier alone, not the prover's cli.py."""
        code = ('import sys; sys.path.insert(0, %r); from rankbound.__main__ import main\n'
                'try:\n    main(["verify", "--help"])\nexcept SystemExit:\n    pass\n'
                'print(sorted(m for m in sys.modules if m.split(".")[0] == "rankbound"))' % os.path.dirname(TESTS))
        r = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.splitlines()[-1],
                         "['rankbound', 'rankbound.__main__', 'rankbound.verify', 'rankbound.verify.__main__']")

    def test_alone(self):
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, 'rankbound'))
            open(os.path.join(d, 'rankbound', '__init__.py'), 'w').close()       # no prover: an empty package
            shutil.copytree(self.VDIR, os.path.join(d, 'rankbound', 'verify'), ignore=shutil.ignore_patterns('__pycache__'))
            with open(os.path.join(d, 'VERSION'), 'w') as f:
                f.write('b7a1b0fc\n')
            code = ('import sys; sys.path.insert(0, %r); from rankbound.verify.__main__ import main; main()' % d)
            cert = cert0()
            r = subprocess.run([sys.executable, '-I', '-c', code, cert], cwd=d, capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
            self.assertIn('PASS: icarm0 (', r.stdout)
            self.assertIn('rank E(Q) <= 0', r.stdout)
            with open(cert, 'rb') as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            first = r.stdout.splitlines()[0]              # the provenance line
            self.assertIn(' Verifier %s, deployment b7a1b0fc, PARI/GP ' % verifier.verifier_hash(), first)
            self.assertTrue(first.endswith('. Certificate sha256 %s.' % sha), first)
            self.assertEqual(len(r.stdout.splitlines()), 2, r.stdout)     # provenance and PASS: no per-check lines
        finally:
            shutil.rmtree(d)



class TestVerifierInput(unittest.TestCase):
    """A hostile or broken certificate fails cleanly: no gp code from it runs, an out-of-range row index or a wrong
    relation is a FAIL and not an exception, and one bad certificate does not stop the others."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.d)            # gp runs here: the payload's path has no '/'
        with open(cert0()) as f:
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
        ok, res, _ = verifier.verify(self.write(name, c), log=lambda *a: None)
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
        """A broken certificate first, then a good one: the good one is still checked; exit status 1."""
        c = json.loads(json.dumps(self.c0))
        c['ops'] = [[0, 10 ** 6]]
        bad = self.write('a_bad', c)
        good = self.write('b_good', self.c0)
        r = subprocess.run([sys.executable, '-m', 'rankbound.verify', bad, good], cwd=os.path.dirname(TESTS),
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('PASS: icarm0 (', r.stdout)
        self.assertIn('FAIL: icarm0 failed the check "ops_in_range" (', r.stdout)
        self.assertIn('1 of 2 certificates pass. Failed: a_bad.cert.json', r.stdout)
        # an error that is not a verdict (gp not found): 'error' for each certificate, its reason on one line (no
        # traceback), the batch goes on; exit status 4
        r = subprocess.run([sys.executable, '-m', 'rankbound.verify', bad, good], cwd=os.path.dirname(TESTS),
                           capture_output=True, text=True, timeout=600, env=dict(os.environ, PATH=self.d))
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count('NOT CHECKED: the verifier could not run on '), 2, r.stdout)
        self.assertEqual(r.stdout.count('(FileNotFoundError: '), 2, r.stdout)
        self.assertNotIn('Traceback', r.stdout + r.stderr)
        self.assertIn('0 of 2 certificates pass. Not checked: a_bad.cert.json, b_good.cert.json', r.stdout)

    def test_canonical(self):
        """One written form per number: '-0', '05', '+5', ' 5', '0/3' are refused."""
        for x in ('0', '7', '-7', '123456789012345678901234567890'):
            self.assertEqual(verifier._str(x), x)
        for x in ('-0', '05', '+5', ' 5', '5 ', '', '1e3', 5):
            self.assertRaises(ValueError, verifier._str, x)
        for x in ('0', '-3/4', '3'):
            self.assertEqual(verifier._str(x, verifier.RAT), x)
        for x in ('-0', '0/3', '3/0', '3/04'):
            self.assertRaises(ValueError, verifier._str, x, verifier.RAT)
        self.assertRaises(ValueError, verifier._gpvec, '-0,1:3')

    def test_unproved_r(self):
        """A rank lower bound with neither points and primes nor a source is malformed."""
        for rl in ({'r': 2}, {'r': 2, 'source': ''}, {'r': 2, 'source': ' '}, {'r': 2, 'source': 3}):
            c = json.loads(json.dumps(self.c0))
            c['rank_lower_bound'] = rl
            self.assertEqual(self.check(c, 'unproved'), ['format'], rl)



class TestBreak(unittest.TestCase):
    """Attempts to break the verifier through check() (rankbound verify --json): every malformed, hostile, mismatched
    or oversized input gets a verdict other than 'accepted', with a reason, never a traceback; inputs that only
    rewrite a valid proof (another model of the curve, a repeated relation or pivot) stay accepted."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        with open(cert0()) as f:
            self.raw = f.read()
        self.c0 = json.loads(self.raw)

    def tearDown(self):
        shutil.rmtree(self.d)

    def verdict(self, c, name='x', **kw):
        """Return check() of C (a dict, or the file's text or bytes) written as NAME.cert.json."""
        path = os.path.join(self.d, name + '.cert.json')
        with open(path, 'wb') as f:
            f.write(c if isinstance(c, bytes) else (c if isinstance(c, str) else json.dumps(c)).encode())
        v = verifier.check(path, **kw)
        self.assertNotIn('Traceback', json.dumps(v))
        self.assertIn(v['verdict'], verifier.EXIT)
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
        self.assertEqual(v['verifier'], verifier.verifier_hash())
        self.assertIn('not BSD', v['hypothesis'])
        # the JSON line is the record of a verification: the provenance (no host) and every check with its info
        self.assertEqual(list(v), ['verdict', 'reason', 'certificate', 'sha256', 'name', 'curve', 'rank_upper_bound',
                                   'rank_lower_bound', 'r_proved', 'rank', 'G', 'u', 'n', 'w', 'hypothesis', 'statement',
                                   'checks', 'verifier', 'deploy', 'gp', 'date', 'seconds'])
        self.assertRegex(v['date'], r'^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4}$')
        self.assertNotEqual(v['gp'], '?')
        info = {x['check']: x['info'] for x in v['checks']}
        self.assertIn('exact interval arithmetic', info['bdf_GB'])
        self.assertRegex(info['relations_checked'], r'^\d+ rows recomputed by PARI \(\d+ s, 1 jobs\)$')
        self.assertRegex(info['dimV'], r'^dim V = \d+ \(rank \d+, \d+ large columns, \d+ s\)$')

    def test_not_json(self):
        """Not a certificate file at all: garbage, empty, truncated anywhere, gzipped, UTF-16, another JSON value."""
        raw = self.raw.encode()
        cuts = [0, 1, len(raw) // 3, len(raw) // 2, len(raw) - 2] + random.Random(1).sample(range(2, len(raw) - 2), 40)
        cases = [b'\x00\xff garbage', b'', gzip.compress(raw), self.raw.encode('utf-16'), b'\xef\xbb\xbf' + raw,
                 b'[1, 2]', b'42', b'null', b'"text"'] + [raw[:k] for k in cuts]
        for k, b in enumerate(cases):
            v = self.verdict(b, 'n%d' % k)
            self.assertEqual((v['verdict'], v['checks'][0]['check']), ('rejected', 'format'), (k, b[:40], v['reason']))

    def test_json_tricks(self):
        """Duplicate keys (a reader sees the first, json keeps the last), NaN and Infinity, deep nesting."""
        dup = self.raw.replace('"claims": {', '"claims": {"S": 5, ', 1)
        self.assertEqual(self.verdict(dup)['reason'], 'not a valid certificate: the key "S" appears twice')
        nan = self.raw.replace('"GB": %d' % self.c0['field']['GB'], '"GB": NaN', 1)
        self.assertNotEqual(nan, self.raw)
        self.assertEqual(self.verdict(nan)['reason'], 'not a valid certificate: NaN is not a JSON number')
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
            'format': put(['format'], 'rankbound-selmer-certificate-1'),
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
            'S - 1': (claims(S=S - 1), 'claims'), 'G - 1': (claims(G=self.c0['claims']['G'] - 1), 'claims'),
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
        self.assertRegex(v['reason'], r'^failed check: rejected: relation \d+: prod H != \|F\(a,b\)\|$')
        lines = []
        verifier.check(os.path.join(self.d, 'w.cert.json'), log=lines.append)
        self.assertRegex(lines[-1], r'^FAIL: icarm0 failed a check \(\d+ s\): relation \d+: prod H != \|F\(a,b\)\|\.$')
        self.assertEqual(self.verdict(self.changed(cases['S - 1'][0]))['reason'], 'failed check: claims')

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
        """--curve: the certificate must be for that curve, on any model (11a3 = ICARM #0: shifted x -> x + 1, scaled
        by u = 2), and not for another (571a1, a quadratic twist): rejected before gp runs."""
        for a in ([0, -1, 1, 0, 0], [0, 2, 1, 1, 0], [0, -4, 8, 0, 0]):
            self.assertEqual(self.verdict(self.c0, curve=a)['verdict'], 'accepted', a)
        for a in ([0, -1, 1, -929, -10595], [0, 1, 1, -1, 0]):
            v = self.verdict(self.c0, curve=a)
            self.assertEqual((v['verdict'], v['checks'][1]['check']), ('rejected', 'curve'), v['reason'])
            self.assertLess(v['seconds'], 5)
        same = verifier.same_curve
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
            v = self.verdict(c, curve=a)
            self.assertEqual((v['verdict'], v['checks'][1]['check']), ('rejected', 'curve'), v['reason'])

    def test_limits(self):
        """Over the size or time limit, GB above Bach's bound, a huge prime of the points or a huge number: 'limit',
        not checked (and quickly)."""
        v = self.verdict(self.c0, max_mb=1e-4)
        self.assertEqual((v['verdict'], v['reason']), ('limit', 'x.cert.json is larger than --max-mb (0.0001 MB)'))
        self.assertIsNone(v['sha256'])
        v = self.verdict(self.c0, max_seconds=0.01)
        self.assertEqual((v['verdict'], v['reason']), ('limit', 'x.cert.json was not checked within 0 s (--max-seconds)'))
        for what, f in {'GB': lambda c: c['field'].update(GB=10 ** 30),
                        'prime': lambda c: c.update(rank_lower_bound={'r': 1, 'points': [['1', '2']], 'primes': [2 ** 64 + 13]}),
                        'digits': lambda c: c['curve']['ainvs'].__setitem__(4, '9' * 5000),
                        'relation': lambda c: c['relations'].append('1,1:' + 'f' * 5000)}.items():
            v = self.verdict(self.changed(f), what)
            self.assertEqual(v['verdict'], 'limit', (what, v['reason']))
            self.assertLess(v['seconds'], 5, what)

    def test_errors(self):
        """No file, a folder, gp missing: 'error' (the check could not run), not a verdict on a certificate."""
        for path in (os.path.join(self.d, 'none.cert.json'), self.d):
            v = verifier.check(path)
            self.assertEqual(v['verdict'], 'error', path); self.assertIn('Error', v['reason'])
        p = os.path.join(self.d, 'x.cert.json')
        with open(p, 'w') as f:
            f.write(self.raw)
        v = verifier.check(p, gp=os.path.join(self.d, 'no-gp'))
        self.assertEqual(v['verdict'], 'error'); self.assertIn('FileNotFoundError', v['reason'])
        self.assertTrue(v['reason'].startswith('the verifier could not run on x.cert.json ('), v['reason'])
        self.assertEqual(v['gp'], '?')
        self.assertRaises(ValueError, verifier.verify, p, cores=0)

    def test_cli(self):
        """rankbound verify --json: one JSON object per line and nothing else; the exit status of the worst verdict;
        usage errors (no certificate, --curve with two, --cores 0, a malformed --curve) exit 2 without a traceback."""
        good = os.path.join(self.d, 'icarm0.cert.json')
        bad = os.path.join(self.d, 'bad.cert.json')
        with open(good, 'w') as f:
            f.write(self.raw)
        with open(bad, 'w') as f:
            f.write(self.raw[:100])
        cli = [sys.executable, '-m', 'rankbound', 'verify']
        def run(*args):
            return subprocess.run(cli + list(args), cwd=os.path.dirname(TESTS), capture_output=True, text=True, timeout=600)
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
        self.assertIn('FAIL: the certificate for icarm0 is not for the curve 0,-1,1,-929,-10595.', r.stdout)
        r = run(bad)
        self.assertEqual(r.returncode, 1)
        self.assertIn('FAIL: bad.cert.json is not a valid certificate: it is not valid JSON at line ', r.stdout)
        for args in ([os.path.join(self.d, 'empty')], ['--curve', '0,-1,1,0,0', good, bad], ['--cores', '0', good],
                     ['--curve', '0,-1,1,0', good], ['--curve', '0,-1,1,0,x', good], ['--max-seconds', 'nan', good]):
            if args[0].endswith('empty'):
                os.makedirs(args[0])
            r = run(*args)
            self.assertEqual(r.returncode, 2, (args, r.stdout, r.stderr))
            self.assertNotIn('Traceback', r.stderr)


KEYS_REQUIRED = [k for k in verifier.KEYS if k != 'rank_lower_bound']


if __name__ == '__main__':
    unittest.main()
