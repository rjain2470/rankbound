"""The factor step (curve.py): GMP-ECM (a stand-in, tests/data/fake_ecm.py, and the real binary when it is installed)
and the built-in rho and ECM; a known split, a composite that must give up at the time limit, and the resume from factor.json;
whole runs on a curve whose discriminant trial division leaves composite."""
import fcntl, json, os, shutil, subprocess, sys, tempfile, time, unittest
from unittest import mock
from rankbound import FactorisationNeeded, Locked, RankboundError, Stopped, arith, cli, curve as cv, rankbound
from rankbound.runtime import read_json
from rankbound.check import run_gp
from helpers import SRC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = [sys.executable, '-m', 'rankbound']
ENV = dict({k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}, PYTHONPATH=SRC)
FAKE = os.path.join(ROOT, 'tests', 'data', 'fake_ecm.py')


def nextprime(n):
    while not arith.is_prime(n):
        n += 1
    return n


P12, P20, P30, P40 = nextprime(10 ** 11), nextprime(10 ** 19), nextprime(10 ** 29), nextprime(10 ** 39)
HARD = P40 * nextprime(P40 + 10 ** 20)      # C79, two 40-digit primes: neither engine splits it in seconds
# y^2 = x^3 + 100481 x + 10017069: |disc| = 2^4 * 11 * 17331101 * 35497001; trial division to 10^6 leaves the C15
E15 = [100481, 10017069]
C15 = 17331101 * 35497001


def log_lines(path):
    with open(path) as f:
        return f.read().splitlines()


class TestFakeECM(unittest.TestCase):
    """The GMP-ECM path, with a stand-in that finds a given prime from a given B1 on; PARI's first try on small
    composites is off (PARI_DIGITS 0) except in test_small."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ecm = os.path.join(self.tmp, 'ecm')
        with open(self.ecm, 'w') as f:
            f.write('#!%s\nimport runpy; runpy.run_path(%r, run_name="__main__")\n' % (sys.executable, FAKE))
        os.chmod(self.ecm, 0o755)
        self.calls = os.path.join(self.tmp, 'calls.txt')
        open(self.calls, 'w').close()
        self.env = mock.patch.dict(os.environ, {'FAKE_ECM_LOG': self.calls, 'FAKE_ECM_SPC': '0.002',
                                                'FAKE_ECM_FIND': '', 'FAKE_ECM_FAIL': ''})
        self.env.start()
        self.small = mock.patch.object(cv, 'PARI_DIGITS', 0)
        self.small.start()
        self.path = os.path.join(self.tmp, 'w', 'factor.json')
        self.log = []

    def tearDown(self):
        self.small.stop()
        self.env.stop()
        shutil.rmtree(self.tmp)

    def split(self, comps, **kw):
        return cv.split(self.path, comps, ecm=self.ecm, log=self.log.append, **kw)

    def test_known_split(self):
        """A C41 = p12 * p30 found at level 20, and the square of a prime, split without ECM; the result is kept, so
        a second call runs no engine."""
        os.environ['FAKE_ECM_FIND'] = '%d:11000' % P12
        self.assertEqual(self.split([P12 * P30, P30 ** 2], hours=0.01), [P12, P30])
        self.assertEqual(log_lines(self.calls), ['-q -one -maxmem 800 -c 74 11000 41'])
        self.assertIn('C41 = p12 * p30 (GMP-ECM, level 20; ', '\n'.join(self.log))
        self.assertEqual(self.split([P12 * P30, P30 ** 2], hours=0.01), [P12, P30])
        self.assertEqual(len(log_lines(self.calls)), 1)

    def test_small(self):
        """A composite of two primes just above 10^6, on which GMP-ECM finds only the whole number (exit 8), is split
        by PARI's first try; a C79 on which PARI's try runs out goes on to ECM, where finding the whole number
        credits no curves."""
        os.environ['FAKE_ECM_WHOLE'] = '1'
        n = 1000003 * 1000033
        with mock.patch.object(cv, 'PARI_DIGITS', 60):
            self.assertEqual(self.split([n], hours=0.01), [1000003, 1000033])
            self.assertEqual(log_lines(self.calls), [])
            self.assertIn("C13 = p7 * p7 (built-in rho and ECM", '\n'.join(self.log))
            with mock.patch.object(cv, 'PARI_S', 1), mock.patch.object(cv, 'PARI_DIGITS', 80):
                with self.assertRaises(FactorisationNeeded):
                    self.split([HARD], hours=4 / 3600)
        self.assertTrue(log_lines(self.calls))
        r = read_json(self.path)['numbers'][str(HARD)]
        self.assertGreater(r['pari_s'], 0.5)
        self.assertEqual(r['done'], {})
        self.assertIn('GMP-ECM, level 20 found the whole of C79 (no split; nothing credited).', self.log)

    def test_give_up_and_resume(self):
        """A composite not split in 3 s gives up with FactorisationNeeded, naming it and the levels done; the effort
        is in factor.json, and the next call continues from it (no level is repeated) and splits it."""
        n = P30 * P40
        t = time.time()
        with self.assertRaises(FactorisationNeeded) as e:
            self.split([n], hours=3 / 3600)
        self.assertLess(time.time() - t, 6)
        msg = self.log[-1]
        self.assertRegex(msg, r'^GMP-ECM stopped after [0-9]+ s \([0-9.]+ core-h in all\): 1 composite left, C69 \(P-1 not '
                              r'yet; level (25|30|35) finished.*: no factor below ~(25|30|35) digits is likely\)\.$')
        [c] = e.exception.composites
        self.assertEqual((c['n'], c['digits']), (str(n), 69))
        self.assertEqual((c['done'].get('0'), c['done'].get('1')), (74, 221))
        st = read_json(self.path)
        done = st['numbers'][str(n)]['done']
        self.assertEqual(done, c['done'])
        first = len(log_lines(self.calls))
        os.environ['FAKE_ECM_FIND'] = '%d:250000' % P30
        self.assertEqual(self.split([n], hours=0.01), [P30, P40])
        again = log_lines(self.calls)[first:]
        self.assertTrue(again)
        for L in done:                      # the levels finished before are not run again
            if done[L] >= cv.LEVELS[int(L)][2]:
                self.assertFalse([l for l in again if ' %.0f ' % cv.LEVELS[int(L)][1] in l], (L, again))

    def test_parallel(self):
        """With 3 cores, one composite gets 3 calls of level 20 at once (25 + 25 + 24 curves); when one finds a
        factor, the other two are stopped at once and credited nothing."""
        os.environ.update(FAKE_ECM_FIND='%d:11000' % P12, FAKE_ECM_SPC='0.2', FAKE_ECM_ONCE=os.path.join(self.tmp, 'once'))
        t = time.time()
        self.assertEqual(self.split([P12 * P40], hours=0.01, cores=3), [P12, P40])
        self.assertLess(time.time() - t, 3)         # each of the other calls would take 5 s
        self.assertEqual(sorted(l.split()[5:7] for l in log_lines(self.calls)),
                         [['24', '11000'], ['25', '11000'], ['25', '11000']])
        r = read_json(self.path)['numbers'][str(P12 * P40)]
        self.assertEqual((r['status'], r['done']), ('split', {}))

    def test_pm1(self):
        """P-1 runs once per composite, after levels 20-30, when PM1_MIN_S is left; a part of a split composite
        inherits P-1 and the levels done."""
        n = P20 * P30 * P40
        os.environ['FAKE_ECM_FIND'] = '%d:1000000-1000000' % P20       # level 35 only
        with mock.patch.object(cv, 'PM1_MIN_S', 0):
            with self.assertRaises(FactorisationNeeded):
                self.split([n], hours=6 / 3600)
        calls = log_lines(self.calls)
        pm1 = [i for i, l in enumerate(calls) if '-pm1' in l]
        self.assertEqual(len(pm1), 1, calls)
        before = [float(l.split()[-2]) for l in calls[:pm1[0]]]
        self.assertEqual((sorted(set(before)), before), ([11000, 50000, 250000], sorted(before)))
        st = read_json(self.path)
        part = st['numbers'][str(P30 * P40)]
        self.assertTrue(part['pm1'])
        self.assertEqual({k: part['done'][k] for k in '012'}, {'0': 74, '1': 221, '2': 453})

    def test_engine_failure(self):
        """Three failures of one kind of call on one composite stop the step with RankboundError; nothing is
        credited."""
        os.environ['FAKE_ECM_FAIL'] = '1'
        with self.assertRaisesRegex(RankboundError, '^GMP-ECM, level 20 failed 3 times on C79: GMP-ECM exit 1'):
            self.split([HARD], hours=0.01)
        self.assertEqual(read_json(self.path)['numbers'][str(HARD)]['done'], {})

    def test_stopped(self):
        """The budget (a run's --max-hours) ending before the ECM limit gives Stopped, with the effort saved."""
        from rankbound import runtime
        with self.assertRaises(Stopped):
            self.split([HARD], hours=1, budget=runtime.Budget(seconds=1.5, stop=lambda: Stopped(self.tmp)))
        self.assertGreater(read_json(self.path)['core_s'], 0)

    def test_locked(self):
        """Another process factoring in the same folder: Locked."""
        os.makedirs(os.path.dirname(self.path))
        fd = os.open(self.path + '.lock', os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            with self.assertRaises(Locked):
                self.split([HARD], hours=0.01)
        finally:
            os.close(fd)

    def test_find_ecm(self):
        """RANKBOUND_ECM names the binary; 'none' means PARI; a path that is not executable is refused."""
        with mock.patch.dict(os.environ, {'RANKBOUND_ECM': self.ecm}):
            self.assertEqual(cv.find_ecm(), self.ecm)
        with mock.patch.dict(os.environ, {'RANKBOUND_ECM': 'none'}):
            self.assertIsNone(cv.find_ecm())
        with mock.patch.dict(os.environ, {'RANKBOUND_ECM': FAKE + '.missing'}):
            self.assertRaisesRegex(RankboundError, 'is not a program', cv.find_ecm)

    def test_cli_resume(self):
        """`rankbound bound --estimate` on E15 with GMP-ECM that finds nothing gives up (exit 1, the message); run again
        with a factor in reach, it continues from factor.json and finishes."""
        job = os.environ.pop('SLURM_JOB_ID', None)      # no time limit from a Slurm job
        if job is not None:
            self.addCleanup(os.environ.__setitem__, 'SLURM_JOB_ID', job)
        os.environ.update(RANKBOUND_ECM=self.ecm, FAKE_ECM_SPC='0.05')
        cmd = ['bound', '--estimate', ','.join(map(str, E15)), '--dir', os.path.join(self.tmp, 'cli'), '--ecm-hours']
        with self.assertLogs('rankbound', 'INFO') as lg, self.assertRaises(SystemExit) as e:
            cli.main(cmd + ['0.0006'])
        self.assertIn('Factoring a 15-digit number with GMP-ECM on 1 core for at most 2 seconds.', lg.output[0])
        self.assertRegex(str(e.exception.code), r'^Curve 0_0_0_100481_10017069 is skipped. Its discriminant could not be factored in ')
        first = len(log_lines(self.calls))
        self.assertGreater(first, 0)
        os.environ['FAKE_ECM_FIND'] = '17331101:11000'
        with self.assertLogs('rankbound', 'INFO') as lg:
            self.assertEqual(cli.main(cmd + ['0.01']), 0)
        out = '\n'.join(lg.output)
        self.assertIn('Found 15-digit = 8-digit * 8-digit prime after ', out)
        self.assertIn('has no rational 2-torsion. Sieving for relations should take ', out)
        self.assertGreater(len(log_lines(self.calls)), first)


class TestPari(unittest.TestCase):
    """The path without GMP-ECM: the built-in rho and ECM on each composite, for the time limit."""

    def test_split_and_give_up(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cv.split(os.path.join(d, 'a.json'), [P12 * P30], hours=0.01, ecm=None, log=lambda m: None),
                             [P12, P30])
            t = time.time()
            log = []
            with self.assertRaises(FactorisationNeeded) as e:
                cv.split(os.path.join(d, 'b.json'), [HARD], hours=3 / 3600, ecm=None, log=log.append)
            self.assertLess(time.time() - t, 6)
            self.assertRegex(log[-1], r"^Factoring stopped after [0-9]+ s .*C79 \(built-in rho and ECM [0-9]+ s\)\.$")
            self.assertRegex(str(e.exception), r"^Curve \w+ is skipped. Its discriminant could not be factored in 3 "
                                               r"seconds. Give its primes with --bad-primes, more time with --ecm-hours, "
                                               r"or install GMP-ECM.$")
            tok = cv.FACTOR_NOTE.set(True)          # --estimate's short factoring, said
            try:
                with self.assertLogs('rankbound', 'INFO') as lg, self.assertRaises(FactorisationNeeded):
                    cv.split(os.path.join(d, 'c.json'), [HARD], hours=1 / 3600, ecm=None, log=log.append)
            finally:
                cv.FACTOR_NOTE.reset(tok)
            self.assertEqual(lg.records[0].getMessage(), 'Factoring a 79-digit number for at most 1 second, since GMP-ECM '
                             'is not installed (the time --estimate gives it; --ecm-hours H for more).')


@unittest.skipUnless(cv.find_ecm(), 'GMP-ECM not installed (ecm on PATH or RANKBOUND_ECM)')
class TestGmpEcm(unittest.TestCase):
    """The real GMP-ECM binary: a known split and a give-up."""

    def test_real(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'factor.json')
            self.assertEqual(cv.split(path, [P20 * P40], hours=0.05, log=print), [P20, P40])
            r = read_json(path)['numbers'][str(P20 * P40)]
            self.assertEqual((r['status'], r['found'][0]['factor']), ('split', str(P20)))
            with self.assertRaises(FactorisationNeeded) as e:
                cv.split(os.path.join(d, 'hard.json'), [HARD], hours=5 / 3600, log=print)
            self.assertTrue(e.exception.composites[0]['done'], e.exception)


class TestCurve(unittest.TestCase):
    """The curve's own factoring (curve.CURVE_GP) and whole runs through the factor step."""

    def test_curve_gp(self):
        """Without PARI time (factor_seconds=0) the C15 is left as a composite, or refused without composites=True;
        with it (1.0's 600 s) it is factored there; a prime power above 10^6 (a quadratic twist by p) is split
        without factor()."""
        c = cv.from_ainvs(E15, factor_seconds=0, composites=True)
        self.assertEqual((c.bad_primes, c.composites), (['2', '11'], [str(C15)]))
        with self.assertRaises(FactorisationNeeded) as e:
            cv.from_ainvs(E15, factor_seconds=0)
        self.assertEqual(e.exception.composites, [{'n': str(C15), 'digits': 15}])
        self.assertEqual(cv.from_ainvs(E15).bad_primes, ['2', '11', '17331101', '35497001'])
        p = 1000003                             # E15 twisted by p: p^6 divides the discriminant
        t = cv.from_ainvs([100481 * p ** 2, 10017069 * p ** 3], factor_seconds=0, composites=True)
        self.assertIn(str(p), t.bad_primes)
        self.assertEqual(t.composites, [str(C15)])

    def test_run(self):
        """rankbound() on E15 factors the C15 (factor.json), proves rank <= 0, and returns the conductor (= PARI's
        ellglobalred); with ecm_hours=0 the 600 s of 1.0 factor it in CURVE_GP, with no factor step (sieve=True)."""
        with tempfile.TemporaryDirectory() as d:
            res = rankbound(E15, workdir=os.path.join(d, 'w'), compute=True)
            self.assertEqual((res.rank_bound, res.checked), (0, True))
            N = run_gp('print(ellglobalred(ellinit([%d, %d]))[1]);' % tuple(E15)).strip()
            self.assertEqual(res.conductor, N)
            st = read_json(os.path.join(res.workdir, 'factor.json'))
            self.assertEqual(st['numbers'][str(C15)]['status'], 'split')
            with open(os.path.join(res.workdir, 'curve.json')) as f:
                self.assertEqual(json.load(f)['bad_primes'], ['2', '11', '17331101', '35497001'])
            res0 = rankbound(E15, workdir=os.path.join(d, 'w0'), ecm_hours=0, compute=True)
            self.assertEqual(res0.rank_bound, 0)
            self.assertFalse(os.path.exists(os.path.join(res0.workdir, 'factor.json')))
            for bad in (-1, float('nan'), float('inf'), True, '1'):
                self.assertRaisesRegex(ValueError, '^ecm_hours must be a number >= 0.$', rankbound, E15, ecm_hours=bad)

    def test_bad_composites(self):
        """Composites 0, 1, -15 or one not dividing the discriminant are refused with a ValueError before any
        factoring."""
        from rankbound.selmer import Job
        with tempfile.TemporaryDirectory() as d:
            for c, msg in (('0', 'is not an integer above 1'), ('1', 'is not an integer above 1'),
                           ('-15', 'is not an integer above 1'), ('15', 'does not divide the discriminant')):
                cur = cv.Curve(name='c', ainvs=[str(x) for x in E15], bad_primes=['2', '11'], composites=[c])
                self.assertRaisesRegex(ValueError, '^The composite %s %s.$' % (c, msg), Job(os.path.join(d, c)).prepare, cur)
                self.assertFalse(os.path.exists(os.path.join(d, c, 'factor.json')))

    def test_list(self):
        """A curve list's composites go to the factor step; with --ecm-hours 0 the curve is refused as in 1.0."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'list.json')
            with open(path, 'w') as f:
                json.dump({'curves': [{'id': 1, 'ainvs': E15, 'bad_primes': ['2', '11'], 'composites': [str(C15)]}]}, f)
            r = subprocess.run(CLI + ['bound', '--estimate', '--dir', os.path.join(d, 'a'), '--curves', path], env=ENV,
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('Found 15-digit = 8-digit * 8-digit prime', r.stderr)
            r = subprocess.run(CLI + ['bound', '--estimate', '--dir', os.path.join(d, 'b'), '--curves', path, '--ecm-hours', '0'],
                               env=ENV, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 1)
            self.assertIn('The discriminant is not fully factored.', r.stderr)

    def test_list_without_bad_primes(self):
        """List curves without bad_primes are factored one by one, as a single curve is: E15 runs, and y^2 = x^3 - x
        (E(Q)[2] != 0) goes to the 2-descent.  E15 goes by our SPLIT, so --estimate factors its C15 (factor.json) and
        prepares no sieve."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'list.json')
            with open(path, 'w') as f:
                json.dump({'curves': [{'id': 1, 'ainvs': E15}, {'id': 2, 'ainvs': [0, 0, 0, -1, 0]}]}, f)
            r = subprocess.run(CLI + ['bound', '--estimate', '--dir', os.path.join(d, 'a'), '--curves', path],
                               env=dict(ENV, RANKBOUND_ECM='none'), capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('Found 15-digit = 8-digit * 8-digit prime after', r.stderr)
            self.assertIn('curve1 has no rational 2-torsion. Sieving for relations should take ', r.stdout)
            self.assertIn('curve2 has rational 2-torsion, so a 2-isogeny descent bounds its rank.\n', r.stdout)
            self.assertRegex(r.stdout, r'2 curves: 1 estimated at [0-9.]+ core-hours in all\.\n')
            st = read_json(os.path.join(d, 'a', 'curve1', 'factor.json'))
            self.assertEqual(st['numbers'][str(C15)]['status'], 'split')


class TestSaturation(unittest.TestCase):
    def test_halved(self):
        """Independent points dependent in E(Q)/2E(Q) (389a1: 2P and Q, 2P and 2Q, 2(P + Q) and Q) are 2-saturated by
        halving and prove rank 2; the regression is ICARM #813, #3551, ... (r 26-30, ICARM's points)."""
        from fractions import Fraction as F
        a = [0, 1, 1, -2, 0]
        P, Q = (F(-1), F(1)), (F(0), F(0))
        for pts in ([cv.mul(a, P, 2), Q], [cv.mul(a, P, 2), cv.mul(a, Q, 2)], [cv.mul(a, cv.add(a, P, Q), 2), Q]):
            out, primes = cv.rkproof(a, a, [389], 2, [list(z) for z in pts], [], 10 ** 4)
            self.assertEqual(len(primes), 2)
            self.assertTrue(all(cv.on_curve(a, z) for z in out))
        self.assertIsNone(cv._halve(a, P))      # P is not a double


class TestFinishedFolder(unittest.TestCase):
    def test_other_points(self):
        """A work folder (of every method) holds the work of the same curve with other points (or points where it
        had none), but not its result: a route's begin accepts it, and only the runner's binding (RunStore.begin
        with rebind, tests/test_stop_resume.py) drops the result of the other points."""
        from rankbound import runner, selmer, descent as tower
        a = ['0', '0', '1', '-1', '0']
        cited = cv.Curve(name='c37', ainvs=a, bad_primes=['37'], r=1, minimal_model=a, torsion=[])
        pts = cv.with_points(cited, [['0', '0']])
        pts2 = cv.with_points(cited, [['1', '0']])
        for cls in (selmer.Selmer, tower.Descent, selmer.Job):
            for old, new, ok in ((cited, cited, True), (pts, pts, True), (cited, pts, False), (pts, cited, False),
                                 (pts, pts2, False)):
                with tempfile.TemporaryDirectory() as d:
                    runner.bind_folder(cls(d), old)
                    runner.bind_folder(cls(d), new)
                    self.assertEqual(read_json(os.path.join(d, 'run.json'))['key']['lower']['proof'] is None, old is cited)


class TestTable(unittest.TestCase):
    def test_known(self):
        """witnesses.tsv: the columns torsion and GRH; a row with GRH "no" is stated unconditionally."""
        from rankbound import curve as cv
        row = {cv.COL_UB: '1', cv.COL_LB: '1', cv.COL_EQ: 'yes', cv.COL_GRH: 'no', cv.COL_TOR: '[2, 2]'}
        self.assertEqual(cv.table_statement(row), 'rank E(Q) = 1 unconditionally')
        self.assertEqual(cv.table_statement(dict(row, GRH='yes')), 'rank E(Q) = 1 under GRH')
        self.assertEqual(cv.table_statement(dict(row, **{cv.COL_EQ: 'no', cv.COL_UB: '3'})), '1 <= rank E(Q) <= 3 unconditionally')
        with open(cv.WITNESSES) as f:
            head = f.readline().rstrip('\n').split('\t')
            rows = [dict(zip(head, l.rstrip('\n').split('\t'))) for l in f]
        self.assertEqual(head[-3:], ['torsion', 'GRH', 'minimal model'])
        self.assertEqual({r['GRH'] for r in rows if r['file']}, {'yes', 'no'})
        two = lambda t: any(int(x) % 2 == 0 for x in t.strip('[]').split(',') if x.strip())
        self.assertTrue(all((r['GRH'] == 'no') == two(r['torsion']) for r in rows if r['file']))
        self.assertEqual(cv.table_row(['1', '0', '1', '4', '-6'])['GRH'], 'no')            # 14a1 = ICARM #865
        self.assertEqual(cv.table_row(['0', '-1', '1', '0', '0'])['torsion'], '[5]')       # 11a3

if __name__ == '__main__':
    unittest.main()
