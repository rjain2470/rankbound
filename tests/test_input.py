"""Fast tests of what the prover is given: command-line arguments, curve lists, values pasted into gp, given
relations and points, and the default name."""
import contextlib, dataclasses, gzip, io, json, os, sys, tempfile, time, unittest
from rankbound import verify as verifier
from rankbound import field, rules, runner, sieve
from helpers import TESTS, cert0


class TestArguments(unittest.TestCase):
    def test_positive(self):
        """Bad --cores, --mem and --max-hours, and the same keyword values, are refused before anything is written."""
        from rankbound import __main__ as cli, rankbound
        with tempfile.TemporaryDirectory() as d:        # nothing may be written here
            for args in (['bound', '--dir', d, '0,0,1,-1,0', '--cores', '0'], ['bound', '--dir', d, '1,1', '--mem', 'nan'],
                         ['solve', d, '--cores', '-2'], ['sieve', d, '--max-hours', '-1'],
                         ['bound', '--dir', d, '--cores', 'two']):
                with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()) as err:
                    cli.main(args)
                self.assertEqual(e.exception.code, 2, args)
                self.assertRegex(err.getvalue(), 'argument --(cores|mem|max-hours): (must be (positive|>= 0)|not a number)\n',
                                 args)
            for kw in ({'cores': 0}, {'cores': 1.5}, {'mem_gb': 0}, {'mem_gb': float('inf')}):
                self.assertRaisesRegex(ValueError, 'must be a positive', rankbound, [0, 0, 1, -1, 0],
                                       workdir=os.path.join(d, 'w'), **kw)
            self.assertRaisesRegex(ValueError, '^cores must be a positive integer.$', runner.sieve_run,
                                   os.path.join(d, 'w'), jobs=0)
            self.assertEqual(os.listdir(d), [])

    def test_shard(self):
        """--shard K/N needs integers 0 <= K < N."""
        from rankbound import __main__ as cli
        with tempfile.TemporaryDirectory() as d:
            for k in ('0/0', '2/1', '1/1', 'x', '-1/2', '1/-2', '0/1/2', '0.5/2'):
                with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()) as err:
                    cli.main(['sieve', d, '--shard=' + k])      # =: Python 3.9 reads '--shard -1/2' as an option
                self.assertEqual(e.exception.code, 2, k)
                self.assertIn('must be K/N with 0 <= K < N', err.getvalue(), k)
            self.assertEqual(os.listdir(d), [])

    def test_names(self):
        """A curve's name must be one the verifier accepts, and not . or ..; a bad one is refused before gp runs, from
        --name, a curve list or rankbound(name=)."""
        from rankbound import __main__ as cli, rankbound
        bad = ('../x', 'a/b', 'a b', '.', '..', '', 'x' * 65, 'icarm\u0660', '/abs')
        for name in bad:
            self.assertRaisesRegex(ValueError, 'name', field.check_name, name)
        for name in ('icarm728', 'E28', 'a.b-c_1', '...'):
            self.assertEqual(field.check_name(name), name)
        with tempfile.TemporaryDirectory() as d:
            for name in bad[:5]:
                with self.assertRaises(SystemExit) as e:
                    cli.main(['bound', '--dir', d, '0,0,1,-1,0', '--name', name])
                self.assertEqual(str(e.exception), 'The name %r is not allowed: use at most 64 letters, digits, _ . and -.'
                                 % name)
                self.assertRaisesRegex(ValueError, 'name', rankbound, [0, 0, 1, -1, 0], name=name,
                                       workdir=os.path.join(d, 'w'))
                path = os.path.join(d, 'list.json')
                with open(path, 'w') as f:
                    json.dump({'curves': [dict(TestGpInput.GOOD, name=name)]}, f)
                self.assertRaisesRegex(ValueError, '^list.json, curve 1: The name', field.from_list, path)
                os.remove(path)
            self.assertEqual(os.listdir(d), [])
            self.assertFalse(os.path.exists(os.path.join(os.path.dirname(d), 'x')))

    def test_spawn_stderr(self):
        """A slice's stderr goes to a file, not a pipe that nobody reads until the process exits: a siever that writes
        more than a pipe holds (64 KB) does not block, and slice_failed reads all of it."""
        p = sieve.spawn([sys.executable, '-c', 'import sys; sys.stderr.write("x" * 300000 + "end"); sys.exit(1)'])
        try:
            self.assertEqual(p.wait(timeout=60), 1)
            with self.assertRaisesRegex(RuntimeError, '^The siever failed: x+end$'):
                runner.slice_failed(None, p, 'F', 1, 2, 'x', {'done': 0, 'crash': 0}, lambda *a: None)
        finally:
            p.stderr.close()



class TestGpInput(unittest.TestCase):
    """Every value of a curve list that is pasted into gp must be a JSON integer or a canonical decimal string (the
    verifier's form): gp evaluates what it is given."""
    GOOD = {'id': 1, 'ainvs': ['0', '-1', '1', '-929', '-10595'], 'bad_primes': ['571'], 'r': 0}
    EVIL = '0*#Str(system("touch MARK"))'

    def load(self, **kw):
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, 'list.json')
            with open(path, 'w') as f:
                json.dump({'curves': [dict(self.GOOD, **kw)]}, f)
            return field.from_list(path)

    def test_curve_list(self):
        self.assertEqual(self.load()[0].ainvs, self.GOOD['ainvs'])
        self.assertEqual(self.load(ainvs=[0, -1, 1, -929, -10595])[0].ainvs, self.GOOD['ainvs'])    # JSON integers
        self.assertEqual(self.load(ainvs=[-929, -10595])[0].ainvs, ['-929', '-10595'])
        proof = {'points': [['-3/4', '7']], 'primes': [5]}
        self.assertEqual(self.load(r=1, proof=proof)[0].proof, dict(proof, ainvs=self.GOOD['ainvs']))   # its model
        self.assertEqual(self.load(r=1, proof=dict(proof, ainvs=[0, 0, 1, -1, 0]))[0].proof['ainvs'], [0, 0, 1, -1, 0])
        for bad in ({'ainvs': ['0', '-1', '1', '-929', '-10595+' + self.EVIL]}, {'ainvs': [0, 1, 2]},
                    {'ainvs': ['0', '-1', '1', '-929', '-10595.0']}, {'ainvs': [0, -1, 1, -929, 1.5]},
                    {'bad_primes': ['571)']}, {'bad_primes': [' 571']}, {'bad_primes': ['0571']},
                    {'ainvs': '01234'}, {'bad_primes': '571'}, {'minimal_model': '01234'}, {'composites': '3997'},
                    {'ainvs': [0, 0, 0, 0, 0]}, {'ainvs': [0, 1, 0, 0, 0]},
                    {'minimal_model': ['0', '-1', '1', '-929', '1e3']}, {'r': '0'}, {'r': True}, {'r': -1},
                    {'proof': {'points': [['1', '2+' + self.EVIL]], 'primes': [5]}},
                    {'proof': {'points': [['1', '2']], 'primes': ['5']}}, {'proof': {'points': [['1']], 'primes': [5]}},
                    {'proof': {'points': [['1', '2']], 'primes': [5], 'ainvs': ['0', '0', '1', '-1', '0;' + self.EVIL]}}):
            with self.assertRaisesRegex(ValueError, '^list.json, curve 1: ', msg=bad):
                self.load(**bad)
        for key in ('id', 'ainvs', 'bad_primes'):         # the required keys: a missing one is named
            with tempfile.TemporaryDirectory() as t:
                path = os.path.join(t, 'list.json')
                with open(path, 'w') as f:
                    json.dump({'curves': [{k: v for k, v in self.GOOD.items() if k != key}]}, f)
                with self.assertRaisesRegex(ValueError, '^Entry 1 of list.json has no "id".$' if key == 'id' else
                                            '^Curve 1 of list.json has no "%s".$' % key):
                    field.from_list(path)

    def test_prover_gp(self):
        """from_ainvs, compute and rank_proof refuse a non-canonical value before gp runs."""
        from rankbound import linalg
        self.assertRaises(ValueError, field.from_ainvs, [0, -1, 1, -929, -10595], bad_primes=['571+' + self.EVIL])
        self.assertRaises(ValueError, field.from_ainvs, ['0', '-1', '1', '-929', ' -10595'])
        # a string is not read as a list of digits; a singular curve gets a clear message
        self.assertRaisesRegex(ValueError, 'ainvs must be a list', field.from_ainvs, '01234')
        self.assertRaisesRegex(ValueError, 'bad_primes must be a list', field.from_ainvs, [0, -1, 1, -929, -10595], '571')
        for ai in ([0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0]):
            self.assertRaisesRegex(ValueError, 'singular curve', field.from_ainvs, ai)
        self.assertRaisesRegex(ValueError, 'singular curve', field.compute, ['0'] * 5, [])    # a Curve made by hand
        self.assertEqual(field.disc([0, -1, 1, -929, -10595]), -571)
        self.assertEqual(field.disc([0, 0, 1, -1, 0]), 37)
        self.assertRaises(ValueError, field.compute, ['0', '-1', '1', '-929', '-10595'], ['571;quit'])
        good = {'points': [['1', '2']], 'primes': [3, 5]}
        for ai, bad, proof in ((self.GOOD['ainvs'], ['571'], {'points': [['1', '2;' + self.EVIL]], 'primes': [3, 5]}),
                               (['0', '-1', '1', '-929', 'x'], ['571'], good),
                               (self.GOOD['ainvs'], ['571)'], good),
                               (self.GOOD['ainvs'], ['571'], dict(good, ainvs=['0', '-1', '1', '-929', '1e3']))):
            self.assertRaises(ValueError, linalg.rank_proof, ai, bad, 1, proof)



class TestGiven(unittest.TestCase):
    """Relations and points given to rankbound(): their forms, the checks before anything is written, and the
    parameters of a run without a siever."""
    POLY = [-5, 0, 1, 1]          # an irreducible cubic: F(a, b) = a^3 + a^2 b - 5 b^3

    def test_parse(self):
        from rankbound import linalg
        P = linalg.parse_relation
        self.assertEqual(P('3,-2:2,b'), (3, -2, [2, 11]))
        self.assertEqual(P('3,-2'), (3, -2, None))
        self.assertEqual(P('1,0:'), (1, 0, []))
        self.assertIsNone(P('f:7'))
        self.assertEqual(P((3, -2)), (3, -2, None))
        self.assertEqual(P([3, -2, [2, 11]]), (3, -2, [2, 11]))
        for bad in ('3;-2', '03,1', '3,1:0b', '3,1:B', '3,1:2:', (3,), (3, '2'), (True, 1), (3, 2, [2.0]), 3, 'f:x',
                    'f:07'):
            self.assertRaises(ValueError, P, bad)
        # one digest per set of relations, whatever the form or order, repeats or not; free relations are left out
        a = linalg.relation_input(['3,-2', '1,1:2,b', 'f:7'])
        self.assertEqual(a.rels, [(3, -2, None), (1, 1, [2, 11])])
        self.assertEqual(linalg.relation_input([(3, -2), (1, 1, [2, 11])]).digest, a.digest)
        self.assertEqual(linalg.relation_input([(1, 1, [11, 2]), (3, -2), '3,-2']).digest, a.digest)
        self.assertNotEqual(linalg.relation_input([(3, -2)]).digest, a.digest)
        self.assertRaisesRegex(ValueError, '^No relations given.$', linalg.relation_input, ['f:7'])
        with tempfile.TemporaryDirectory() as d:
            for name, op in (('r.txt', open), ('r.txt.gz', gzip.open)):
                p = os.path.join(d, name)
                with op(p, 'wt') as f:
                    f.write('# a comment\n3,-2\n\n1,1:2,b\n')
                self.assertEqual(linalg.relation_input(p), a)
            p = os.path.join(d, 'x.cert.json')         # a certificate: its relations, without the free ones
            with open(p, 'w') as f:
                json.dump({'format': verifier.FORMAT, 'relations': ['f:7', '3,-2', '1,1:2,b']}, f)
            self.assertEqual(linalg.relation_input(p), a)
            for c in ({'format': verifier.FORMAT}, {'format': verifier.FORMAT, 'relations': {}}, [1], {'format': 'x'}):
                with open(p, 'w') as f:          # not a certificate with relations
                    json.dump(c, f)
                self.assertRaisesRegex(ValueError, 'x.cert.json is not a certificate with relations.$',
                                       linalg.read_relations, p)
            with open(os.path.join(d, 'r.txt'), 'w') as f:
                f.write('3,-2\nx\n')
            self.assertRaisesRegex(ValueError, '^In .*r.txt, relation \'x\' is not of the form "a,b" or "a,b:hexprimes".$',
                                   linalg.read_relations, os.path.join(d, 'r.txt'))

    def test_given_lines(self):
        """Missing primes are factored by PARI; wrong primes (not multiplying to |F(a, b)|, or composite) and (a, b) not
        coprime are refused, naming the relation."""
        from rankbound import linalg
        F = {'poly': self.POLY}
        v = abs(field.norm(self.POLY, 3, -2))       # 27 - 18 + 40 = 49
        self.assertEqual(v, 49)
        lines, nf = linalg.given_lines(F, [(3, -2, None), (3, -2, [7, 7]), (1, 0, None), (2, 1, None)])
        self.assertEqual((lines[:3], nf), (['3,-2:7,7', '3,-2:7,7', '1,0:'], 3))
        self.assertEqual(sieve.parse_line(lines[3])[2], [7])                          # 8 + 4 - 5 = 7
        for rel, err in (((3, -2, [49]), '^The primes given for relation 1 \\(3,-2\\) are not the prime factors'),
                         ((3, -2, [7]), 'are not the prime factors'), ((3, -2, [1, 7, 7]), 'are not the prime factors'),
                         ((2, 4, None), '^In relation 1 \\(2,4\\), a and b are not coprime.$'), ((0, 0, None), 'not coprime')):
            with self.assertRaisesRegex(ValueError, err):
                linalg.given_lines(F, [rel])

    def test_points(self):
        from fractions import Fraction
        c = field.Curve(name='e37', ainvs=['0', '0', '1', '-1', '0'], bad_primes=['37'])
        p = field.with_points(c, [(0, 0), ('-3/4', Fraction(6, 4))])
        self.assertEqual((p.r, p.proof), (2, {'points': [['0', '0'], ['-3/4', '3/2']], 'ainvs': c.ainvs}))
        self.assertEqual(field.with_points(c, [], None).r, 0)
        self.assertRaisesRegex(ValueError, '^r = 1 but 2 points were given.$', field.with_points, c, [(0, 0), (1, 0)], 1)
        for bad in ([(0,)], [(0, 1.5)], [('0', '06/4')], [('0', '1;x')]):
            self.assertRaises(ValueError, field.with_points, c, bad)
        self.assertIsNone(field.check_proof(None))
        self.assertEqual(field.check_proof(p.proof), p.proof)         # no primes: the prover searches them
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'p.txt')
            with open(path, 'w') as f:
                f.write('# P1, P2\n0 0\n-3/4,3/2\n')
            self.assertEqual(field.read_points(path), [['0', '0'], ['-3/4', '3/2']])
            with open(path, 'w') as f:
                f.write('0 0 1\n')
            self.assertRaisesRegex(ValueError, 'p.txt, line 1: not a point "x y".$', field.read_points, path)

    def test_params(self):
        """A run without a siever frees up to 2^lpb0 >= GB (every small column) and above the relations' primes (up to
        2^GIVEN_LPB_MAX); the runner's other readers of params.json find what they need."""
        for GB, top, lpb in ((10, 7, 4), (10, 251, 8), (16, 3, 4), (17, 3, 5), (1000, 2 ** 40, sieve.GIVEN_LPB_MAX)):
            prm = sieve.given_params({'GB': GB}, top)
            self.assertEqual(prm['lpb0'], lpb, (GB, top))
            self.assertGreaterEqual(prm['lim0'], GB)
            self.assertIsInstance(sieve.siever(prm), sieve.Given)
        self.assertRaisesRegex(RuntimeError, '^This folder was started from given relations; to sieve, start a new '
                               'folder with --keep-sieving.$', sieve.Given().start_slice, '', prm, 'F', 1, 2, '')
        self.assertEqual(rules.slice_cpu(prm), rules.SLICE_MIN)



class TestCurveList(unittest.TestCase):
    def test_icarm_list(self):
        """An ICARM curve list with one curve, ICARM #728 (from its certificate: tests/data/reference.json)."""
        with open(os.path.join(TESTS, 'data', 'reference.json')) as f:
            c = json.load(f)['icarm728']
        cur, lb = c['curve'], c['rank_lower_bound']
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, 'icarm_curves_728.json')
            with open(path, 'w') as f:
                json.dump({'curves': [dict(cur, icarm_id=728, name=c['name'], r=lb['r'], proof=lb)]}, f)
            a = field.from_list(path)
            self.assertEqual(len(a), 1)
            c = a[0]
            self.assertEqual((c.name, c.r, len(c.proof['points']), c.source),
                             ('icarm728', 29, 29, 'ICARM #728'))
            self.assertEqual(len(c.minimal_model), 5)
            self.assertEqual(field.Curve.from_dict(c.to_dict()), c)
            self.assertEqual(field.from_list(path, {728}), a)
            with self.assertRaises(ValueError):
                field.from_list(path, {692})

    def test_list_options(self):
        """--bad-primes, --name and --r (without --points) go with a single curve: with --curves they are refused,
        not ignored."""
        from rankbound.__main__ import main
        for extra in (['--name', 'x'], ['--bad-primes', '2,3'], ['--r', '1']):
            with tempfile.TemporaryDirectory() as t, self.assertRaises(SystemExit) as e:
                main(['estimate', '--dir', t, '--curves', os.path.join(t, 'none.json')] + extra)
            self.assertEqual(str(e.exception), '%s works with a single curve; a curve list gives it per curve.' % extra[0])
        with tempfile.TemporaryDirectory() as t, self.assertRaises(SystemExit) as e:
            main(['estimate', '--dir', t, '--curves', os.path.join(t, 'none.json'), '--name', 'x', '--r', '1'])
        self.assertEqual(str(e.exception), '--name and --r work with a single curve; a curve list gives them per curve.')



class TestDefaultName(unittest.TestCase):
    """A curve given without a name is named by its reduced minimal model (0_-1_1_0_0; a hash when that is longer
    than 64 characters), on the command line and in rankbound(), whose default workdir is rankbound_work/NAME;
    Curve.key() of a Curve without its minimal model (as from_list gives when the list has none) computes it."""

    def test_default_name(self):
        import rankbound as rb
        self.assertEqual(field.default_name(['0', '-1', '1', '0', '0']), '0_-1_1_0_0')
        big = ['1', '-1', '1', '-20067762415575526585033208209338542750', '30422586260593757248460328202047405873200']
        self.assertRegex(field.default_name(big), '^[0-9a-f]{12}$')
        self.assertEqual(field.check_name(field.default_name(big)), field.default_name(big))
        with open(cert0()) as f:
            cur = json.load(f)['curve']
        self.assertEqual(field.from_ainvs([0, 0, 0, -432, 8208]).name, '0_-1_1_0_0')      # a short model of #0
        self.assertEqual(field.from_ainvs([0, 0, 0, -432, 8208], name='e').name, 'e')
        self.assertEqual(field.Curve(name='x', ainvs=cur['ainvs'], bad_primes=cur['bad_primes']).key(), '0_-1_1_0_0')
        seen = []

        class Stop(Exception):
            pass

        class J:
            def __init__(self, d):
                seen.append(d)
                raise Stop

        old, rb.run.Job = rb.run.Job, J          # rankbound() imports Job on use
        try:
            self.assertRaises(Stop, rb.rankbound, [0, 0, 0, -432, 8208], compute=True)     # compute: #0 is in the index
            self.assertRaises(Stop, rb.rankbound, [0, 0, 0, -432, 8208], compute=True, r=0)
        finally:
            rb.run.Job = old
        self.assertEqual(seen, [os.path.join('rankbound_work', '0_-1_1_0_0'), os.path.join('rankbound_work', '0_-1_1_0_0_r0')])

    def test_cli_folders(self):
        """A curve given without --name gets its own folder for each r, points and relations, as in rankbound();
        --name and a curve list's names are kept."""
        from argparse import Namespace
        from rankbound import cli, linalg
        E = field.from_ainvs([0, 0, 1, -1, 0])                                          # 37a1
        rels = linalg.relation_input(['1,0', '2,1'])

        def f(c, rels=None, **kw):
            a = Namespace(**dict(dict(dir='d', name=None, curves=None, keep_sieving=False), **kw))
            return os.path.relpath(cli.folder(a, c, rels), 'd')
        n = E.name
        self.assertEqual(f(E), n)
        self.assertEqual(f(dataclasses.replace(E, r=0)), n + '_r0')
        self.assertEqual(f(field.with_points(E, [(0, 0)], None)), n + '_r1p')
        self.assertEqual(f(E, rels), n + '_rel' + rels.digest[:8])
        self.assertEqual(f(E, rels, keep_sieving=True), n + '_rel' + rels.digest[:8] + '_sieve')
        self.assertEqual(f(dataclasses.replace(E, r=0), rels, name='g'), n)            # --name: c.name is the name
        self.assertEqual(f(dataclasses.replace(E, r=0), rels, curves='list.json'), n)


class TestOutsider(unittest.TestCase):
    """Bad inputs to the command line and the API: each is refused with a clear reason before anything is written."""
    E37 = [0, 0, 1, -1, 0]

    def write(self, d, name, data, mode='w'):
        path = os.path.join(d, name)
        with open(path, mode) as f:
            f.write(json.dumps(data) if isinstance(data, dict) else data)
        return path

    def main(self, args):
        """Return cli.main(ARGS)'s SystemExit message or return value, with stderr discarded."""
        from rankbound import __main__ as cli
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                return cli.main(args)
            except SystemExit as e:
                return str(e.code)

    def test_curve_list(self):
        """A string proof, a repeated id or name, no curves, or a file that is not JSON, UTF-8 or gzip: refused,
        naming the file."""
        good = dict(TestGpInput.GOOD)
        with tempfile.TemporaryDirectory() as d:
            for data, want in (({'curves': [dict(good, proof='x')]}, 'list.json, curve 1: "proof" must be {'),
                               ({'curves': [good, dict(good, ainvs=self.E37)]}, '^list.json has two curves with the id 1.$'),
                               ({'curves': [dict(good, name='a'), dict(good, id=2, name='a')]},
                                '^list.json has two curves named a.$'),
                               ({'curves': []}, '^list.json has no curves.$'),
                               ([1], '^list.json is not a curve list; see the README for the format.$')):
                self.assertRaisesRegex(ValueError, want, field.from_list, self.write(d, 'list.json', json.dumps(data)))
            for name, data, want in (('a.json', '{not json', 'a.json is not valid JSON at line 1, column 2.$'),
                                     ('b.json', '{"curves": []}'.encode('utf-16'), 'b.json is not UTF-8 text.$'),
                                     ('c.json.gz', 'hello', 'c.json.gz is not a valid gzip file.$')):
                path = self.write(d, name, data, 'wb' if isinstance(data, bytes) else 'w')
                self.assertRaisesRegex(ValueError, want, field.from_list, path)

    def test_options(self):
        """A curve with --curves, no curve, bad --ids, --keep-sieving without --relations, and a missing folder:
        refused with a message."""
        with tempfile.TemporaryDirectory() as d:
            lst = self.write(d, 'list.json', {'curves': [TestGpInput.GOOD]})
            run = os.path.join(d, 'run')
            self.assertEqual(self.main(['bound', '--dir', run, '--curves', lst, '0,0,1,-1,0']),
                             'Give a curve or --curves FILE, not both.')
            self.assertEqual(self.main(['bound', '--dir', run]), 'Give a curve (e.g. 0,1,1,-2,0) or --curves FILE.')
            self.assertEqual(self.main(['bound', '--dir', run, '--curves', lst, '--ids', '1,abc']),
                             'list.json has no curve with the id abc.')
            self.assertEqual(self.main(['bound', '--dir', run, '--curves', lst, '--ids', '1,']),
                             "--ids '1,' contains an empty id.")
            self.assertEqual(self.main(['bound', '--dir', run, '--curves', lst, '--ids', ' ']),
                             '--ids is empty. Leave it out to run the whole list.')
            self.assertEqual(self.main(['bound', '--dir', run, '0,0,1,-1,0', '--keep-sieving']),
                             '--keep-sieving needs --relations.')
            self.assertEqual(self.main(['status', os.path.join(d, 'none')]), '%s is not a folder.' % os.path.join(d, 'none'))
            self.assertEqual(self.main(['solve', os.path.join(d, 'none')]),
                             '%s is not a rankbound work folder. Start it with rankbound bound or rankbound estimate.'
                             % os.path.join(d, 'none'))
            self.assertFalse(os.path.exists(run))

    def test_numbers(self):
        """Numbers of more than 4,000 digits, the verifier's limit, are refused by the prover too."""
        from rankbound import linalg
        for x in ('9' * 4001, 10 ** 4001, -10 ** 5000):
            self.assertRaisesRegex(ValueError, 'more than 4000 digits', field.canon, x, 'a4')
        self.assertEqual(field.canon('9' * 4000, 'a4'), '9' * 4000)
        self.assertRaisesRegex(ValueError, 'more than 4000 digits', field.from_ainvs, [0, 0, 0, '9' * 5000, 1])
        self.assertRaisesRegex(ValueError, "^An a-invariant has more than 4000 digits, the verifier's limit.$",
                               field.from_ainvs, [0, 0, 0, '9' * 5000, 1])
        self.assertRaisesRegex(ValueError, "^Relation '9{20}...' is too long for the verifier.$", linalg.parse_relation,
                               '9' * 4001 + ',1')
        self.assertRaisesRegex(ValueError, 'is too long for the verifier', linalg.parse_relation, (10 ** 5000, 1))
        self.assertRaisesRegex(ValueError, '^Relation 1 is too long for the verifier.$', linalg.given_lines,
                               {'poly': [1, 0, 0, 1]}, [(10 ** 1500, 1, None)])

    def test_bad_primes(self):
        """A bad prime below 2 is refused as not prime; so is an empty name."""
        for p in ('1', '-37', '0'):
            self.assertRaisesRegex(ValueError, '^The bad prime %s is not prime.$' % p, field.from_ainvs, self.E37, [p])
        self.assertRaisesRegex(ValueError, '^The bad prime -37 is not prime.$', field.compute, ['0', '0', '1', '-1', '0'],
                               ['-37'])
        self.assertRaisesRegex(ValueError, 'name', field.from_ainvs, self.E37, None, None, '')

    def test_files(self):
        """--points and --relations files that are binary, not JSON, or not valid gzip or xz: refused, naming the file
        and the reason."""
        from rankbound import linalg
        with tempfile.TemporaryDirectory() as d:
            b = self.write(d, 'x.bin', b'\x00\xff\xfe\x80\n', 'wb')
            self.assertRaisesRegex(ValueError, 'x.bin is not UTF-8 text.$', field.read_points, b)
            self.assertRaisesRegex(ValueError, 'x.bin is not UTF-8 text.$', linalg.read_relations, b)
            for name, want in (('c.cert.json', 'is not valid JSON at line 1, column 1.'), ('r.gz', 'is not a valid gzip file.'),
                               ('r.xz', 'is not a valid xz file.'), ('c.cert.json.xz', 'is not a valid xz file.')):
                self.assertRaisesRegex(ValueError, name + ' ' + want + '$', linalg.read_relations,
                                       self.write(d, name, 'hello'))

    def test_wrong_r(self):
        """r above ns + u + n is refused before the state folder exists; a bound S below r (ICARM #0 with r = 2) stops
        with 'Check r' and no certificate."""
        from rankbound import RankboundError, rankbound
        from rankbound.run import Job
        from helpers import ICARM0, ICARM0_RELS
        with tempfile.TemporaryDirectory() as d:
            for r in (5, 10 ** 60):
                w = os.path.join(d, 'w%d' % len(str(r)))
                self.assertRaisesRegex(ValueError, '^r = %d is impossible for this curve: under GRH its rank is at most 4. '
                                       'Check r.$' % r,
                                       Job(w).prepare, field.from_ainvs(self.E37, r=r))
                self.assertFalse(os.path.exists(w))
            w = os.path.join(d, 'i0')
            with self.assertRaisesRegex(RankboundError, '^The relations prove rank <= 0 under GRH, but r = 2 was given. No '
                                        'certificate written. Check r, or run without it.$'):
                rankbound(ICARM0, relations=ICARM0_RELS, r=2, verify=False, workdir=w)
            self.assertEqual([f for f in os.listdir(w) if f.endswith('.cert.json')], [])
            self.assertFalse(os.path.exists(os.path.join(w, 'result.json')))

    def test_minimal_model(self):
        """A curve list's minimal_model that is not the curve's is refused before the state folder exists."""
        from rankbound.run import Job
        c = field.Curve(name='c', ainvs=['0', '0', '1', '-1', '0'], bad_primes=['37'], minimal_model=['0', '0', '0', '1', '1'])
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'c')
            self.assertRaisesRegex(ValueError, r'^The minimal_model given for c is wrong; it should be \[0,0,1,-1,0\]. '
                                   'Correct it or leave it out.$', Job(w).prepare, c)
            self.assertFalse(os.path.exists(w))

    def test_api(self):
        """rankbound() given bad max_hours, workdir, points, relations or sieve, or a hand-made Curve with bad fields:
        a ValueError before anything is written."""
        from rankbound import rankbound
        from rankbound.run import Job
        good = field.Curve(name='c', ainvs=['0', '0', '1', '-1', '0'], bad_primes=['37'])
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'w')
            for kw, want in (({'max_hours': float('nan')}, '^max_hours must be None or a number >= 0.$'),
                             ({'max_hours': -1}, 'max_hours'), ({'max_hours': 'x'}, 'max_hours'),
                             ({'workdir': 5}, '^workdir must be a folder name.$'),
                             ({'points': 5}, '^points must be a list.$'), ({'points': [None]}, '^Each point must be a pair'),
                             ({'relations': 5}, '^relations must be a file name or a list.$'),
                             ({'relations': b'1,1'}, 'relations must be a file name or a list'),
                             ({'sieve': False}, '^sieve=False needs relations=....$')):
                self.assertRaisesRegex(ValueError, want, rankbound, self.E37, **dict({'workdir': w}, **kw))
            for c, want in ((dataclasses.replace(good, proof='x'), '^"proof" must be {'),
                            (dataclasses.replace(good, r='1'), '^r must be an integer >= 0.$'),
                            (dataclasses.replace(good, composites='3997'), '^composites must be a list.$'),
                            (dataclasses.replace(good, source=1), '^source must be a string.$')):
                self.assertRaisesRegex(ValueError, want, rankbound, c, workdir=w)
                self.assertRaisesRegex(ValueError, want, Job(w).prepare, c)
            self.assertRaisesRegex(ValueError, '^Expected a Curve.$', Job(w).prepare, 5)
            self.assertRaisesRegex(ValueError, '^ainvs must be a list.$', rankbound, 5, workdir=w)
            self.assertEqual(os.listdir(d), [])

    def test_exit_status(self):
        """`rankbound bound` and `estimate` exit 1 when a curve of a list is skipped or fails, or a certificate fails
        its verification."""
        from rankbound import cli
        with tempfile.TemporaryDirectory() as d:
            lst = self.write(d, 'list.json', {'curves': [TestGpInput.GOOD, dict(TestGpInput.GOOD, id=2, ainvs=[0, 0, 0, -1, 0],
                                                                                 bad_primes=['2'])]})
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(self.main(['estimate', '--dir', os.path.join(d, 'run'), '--curves', lst]), 1)
            self.assertIn('INFO:rankbound:Skipped curve2. This curve has a rational point of order 2, so the method does '
                          'not apply (use a 2-isogeny descent, e.g. mwrank).', out.output)
            self.assertTrue(any(l.startswith('INFO:rankbound:1 curves; estimated cost ') and l.endswith(' Skipped: curve2.')
                                for l in out.output), out.output)
        res = {'name': 'x', 'outcome': 'goal', 'G': 0, 'rank_bound': 2, 'sieve_core_hours': 0.014,
               'estimate_core_hours': 0, 'rank_lower_bound': 2}
        with self.assertLogs('rankbound', 'INFO') as out:
            self.assertEqual([cli.report(dict(res, verified=v, r_proved=p)) for v, p in ((True, True), (False, False),
                                                                                         (None, True))], [0, 1, 0])
        self.assertEqual(out.output, ['INFO:rankbound:RESULT x: rank = 2 under GRH, certificate verified (0.01 core-h)',
                                      'INFO:rankbound:RESULT x: rank <= 2 under GRH, certificate NOT verified (0.01 core-h)',
                                      'INFO:rankbound:RESULT x: rank = 2 under GRH, certificate not verified (0.01 core-h)'])


class TestKnown(unittest.TestCase):
    """A curve that certificates.tsv lists with a certificate is not computed again: its row is given (outcome
    'known'), unless --compute (compute=True); points, relations or a curve list always compute."""
    I0 = '0,-1,1,0,0'                                 # ICARM #0 (11a3), its reduced minimal model
    SHORT = [0, 0, 0, -432, 8208]                     # a short model of it: y^2 = x^3 - 27 c4 x - 54 c6
    HEAD = ['name', 'icarm_id', 'file', 'rank lower bound', 'rank upper bound under GRH', 'rank = r?', 'sha256',
            'verified', 'minimal model']

    def index(self, d, rows):
        path = os.path.join(d, 'certificates.tsv')
        with open(path, 'w') as f:
            f.write('\t'.join(self.HEAD) + '\n' + ''.join('\t'.join(r) + '\n' for r in rows))
        return path

    def row(self, name, S, eq, mm=None, file=True, r='0'):
        return [name, '', name + '.cert.json' if file else '', r, S, eq, 'ab' * 32, 'checked', mm or self.I0]

    def use(self, path):
        from rankbound import known
        old, known.INDEX = known.INDEX, path
        self.addCleanup(setattr, known, 'INDEX', old)

    def test_lookup(self):
        """Exact match on the minimal model; rows without a certificate or not matching the curve list are not used;
        of two rows of one curve the least S, then r proved; no index (pip install) or an index
        without the column: None."""
        from rankbound import known
        with tempfile.TemporaryDirectory() as d:
            p = self.index(d, [self.row('a', '2', 'r cited'), self.row('b', '2', 'yes'), self.row('c', '0', 'yes', file=False),
                               self.row('x', '1', 'not matching the curve list: r', mm='1,1,1,1,1'),
                               self.row('y', '3', 'yes', mm='0,-1,1,-10,-20')])
            got = known.lookup(['0', '-1', '1', '0', '0'], p)
            self.assertEqual(got['name'], 'b')
            self.assertIsNone(known.lookup(['1', '1', '1', '1', '1'], p))
            self.assertIsNone(known.lookup(['0', '-1', '1', '-10', '-21'], p))
            self.assertEqual(known.lookup([0, -1, 1, -10, -20], p)['name'], 'y')
            self.assertIsNone(known.lookup(['0', '-1', '1', '0', '0'], os.path.join(d, 'none.tsv')))
            with open(p, 'w') as f:
                f.write('\t'.join(self.HEAD[:-1]) + '\n' + '\t'.join(self.row('b', '2', 'yes')[:-1]) + '\n')
            self.assertIsNone(known.lookup(['0', '-1', '1', '0', '0'], p))
        st = lambda *a, **k: known.statement(dict(zip(self.HEAD, self.row(*a, **k))))
        self.assertEqual(st('b', '3', 'yes', r='3'), 'rank E(Q) = 3 under GRH')
        self.assertEqual(st('b', '0', 'r cited', r='0'), 'rank E(Q) = 0 under GRH')
        self.assertEqual(st('b', '3', 'r cited', r='3'), 'rank E(Q) <= 3 under GRH (r = 3 is cited, not proved in the '
                                                         'certificate)')
        self.assertEqual(st('b', '4', 'no', r='2'), '2 <= rank E(Q) <= 4 under GRH')

    def test_cli(self):
        """`rankbound bound CURVE` (any model) prints the row and exits 0 with no work folder; --compute, --points,
        --relations and a curve list compute (here: prepared, then stopped by --max-hours 0, or certified)."""
        from rankbound import __main__ as cli
        from helpers import ICARM0_RELS
        job = os.environ.pop('SLURM_JOB_ID', None)      # no time limit from a Slurm job
        if job is not None:
            self.addCleanup(os.environ.__setitem__, 'SLURM_JOB_ID', job)
        with tempfile.TemporaryDirectory() as d:
            self.use(self.index(d, [self.row('icarm0', '0', 'yes')]))
            run = os.path.join(d, 'run')
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(cli.main(['bound', '--dir', run, ','.join(map(str, self.SHORT))]), 0)
            self.assertEqual(out.output, [
                'INFO:rankbound:icarm0 is already certified: rank E(Q) = 0 under GRH. Nothing was computed.',
                'INFO:rankbound:  Certificate icarm0.cert.json (sha256 %s). Check it with: rankbound verify icarm0.cert.json'
                % ('ab' * 32),
                'INFO:rankbound:  To compute the upper bound anyway: rankbound bound 0,0,0,-432,8208 --compute'])
            self.assertFalse(os.path.exists(run))
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(cli.main(['bound', '--dir', run, self.I0, '--compute', '--max-hours', '0']), 0)
            self.assertEqual(out.output[-1], 'INFO:rankbound:Stopped at the time limit. Run the same command again to resume.')
            self.assertTrue(os.path.exists(os.path.join(run, '0_-1_1_0_0', 'field.json')))
            rels = os.path.join(d, 'rels.txt')
            with open(rels, 'w') as f:
                f.write('\n'.join(ICARM0_RELS) + '\n')
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(cli.main(['bound', '--dir', run, self.I0, '--relations', rels, '--name', 'g']), 0)
            self.assertEqual(out.output[-1], 'INFO:rankbound:RESULT g: rank = 0 under GRH, certificate verified (0.00 core-h)')
            for f in ('verify.log', 'verify.json'):         # the verifier's lines and verdict: not in the run's log
                self.assertTrue(os.path.exists(os.path.join(run, 'g', f)), f)
            with open(os.path.join(run, 'g', 'verify.json')) as f:
                self.assertEqual(json.loads(f.read())['verdict'], 'accepted')
            self.assertFalse(any('PASS' in l for l in out.output), out.output)
            lst = os.path.join(d, 'list.json')
            with open(lst, 'w') as f:
                json.dump({'curves': [{'id': 0, 'ainvs': self.SHORT, 'bad_primes': ['11']}]}, f)
            with self.assertLogs('rankbound', 'INFO'):
                self.assertEqual(cli.main(['bound', '--dir', run, '--curves', lst, '--max-hours', '0']), 0)
            self.assertTrue(os.path.exists(os.path.join(run, 'curve0', 'field.json')))

    def test_api(self):
        """rankbound(): outcome 'known' (G, u, n, w None, verified None, the certificate's name), from any model; a
        Curve's own minimal_model is not trusted (computed again); compute=True, relations or points compute."""
        from rankbound import Stopped, rankbound
        from helpers import ICARM0_RELS
        with tempfile.TemporaryDirectory() as d:
            self.use(self.index(d, [self.row('icarm0', '0', 'yes')]))
            with self.assertLogs('rankbound', 'INFO'):
                res = rankbound(self.SHORT, workdir=os.path.join(d, 'w'))
            self.assertEqual((res.outcome, res.name, res.rank_bound, res.G, res.verified, res.certificate,
                              res.proved_equal, res.r_proved, res.minimal_model),
                             ('known', 'icarm0', 0, None, None, 'icarm0.cert.json', True, True, self.I0.split(',')))
            self.assertFalse(os.path.exists(os.path.join(d, 'w')))
            wrong = field.Curve(name='c', ainvs=[str(x) for x in self.SHORT], bad_primes=['2', '3', '11'],
                                minimal_model=['1', '1', '1', '1', '1'])
            with self.assertLogs('rankbound', 'INFO'):
                self.assertEqual(rankbound(wrong, workdir=os.path.join(d, 'w')).outcome, 'known')
            other = field.Curve(name='c', ainvs=['0', '0', '1', '-1', '0'], bad_primes=['37'],
                                minimal_model=self.I0.split(','))          # 37a1, with #0's minimal model: computed
            self.assertRaisesRegex(ValueError, '^The minimal_model given for c is wrong', rankbound, other, workdir=os.path.join(d, 'w2'))
            with self.assertLogs('rankbound', 'INFO'):
                self.assertRaises(Stopped, rankbound, self.SHORT, compute=True, workdir=os.path.join(d, 'w3'), max_hours=0)
                res = rankbound(self.SHORT, relations=ICARM0_RELS, workdir=os.path.join(d, 'w4'), verify=False)
            self.assertEqual((res.outcome, res.rank_bound, res.G), ('goal', 0, 0))

    def test_index(self):
        """The shipped index: every row with a certificate has the 5 a-invariants of a reduced minimal model, and
        ICARM #0's model finds its row."""
        from rankbound import known
        with open(known.INDEX) as f:
            head = f.readline().rstrip('\n').split('\t')
            rows = [dict(zip(head, l.rstrip('\n').split('\t'))) for l in f]
        self.assertEqual(head[-1], 'minimal model')
        for r in rows:
            if r['file']:
                self.assertEqual(len(r['minimal model'].split(',')), 5, r['name'])
                self.assertTrue(all(field.NUM.fullmatch(x) for x in r['minimal model'].split(',')), r['name'])
        self.assertEqual(known.lookup(self.I0.split(','))['name'], 'icarm0')
        # one row per curve; ICARM #11 (rank 28) is found by its minimal model alone, without factoring
        from rankbound import __main__ as cli
        job = os.environ.pop('SLURM_JOB_ID', None)      # no time limit from a Slurm job
        if job is not None:
            self.addCleanup(os.environ.__setitem__, 'SLURM_JOB_ID', job)
        self.assertEqual(len({r['minimal model'] for r in rows if r['file']}), len([r for r in rows if r['file']]))
        mm = next(r for r in rows if r['name'] == 'icarm11')['minimal model']
        with tempfile.TemporaryDirectory() as d, self.assertLogs('rankbound', 'INFO') as out:
            t = time.time()
            self.assertEqual(cli.main(['bound', '--dir', os.path.join(d, 'run'), mm]), 0)
            self.assertLess(time.time() - t, 30)
            self.assertEqual(os.listdir(d), [])
        self.assertEqual(out.output[0], 'INFO:rankbound:icarm11 is already certified: rank E(Q) = 28 under GRH. Nothing was '
                                        'computed.')
        self.assertEqual(len(out.output), 3)


if __name__ == '__main__':
    unittest.main()
