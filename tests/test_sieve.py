"""The sieve (sieve.py, sieve.c): a failed slice, the table and the trial (the siever faked), the band-only supply, a
slice's file, field.gp's skewed size; the siever's roots of F mod p against sieve.roots, every relation of a run
rechecked (F(a, b) = +-the product of its primes, each prime and below 2^lpb, gcd(a, b) = 1, b > 0, a special-q of the
range among them), the siever's interface on a state folder; the job's rules, folder and runs (below)."""
import glob, gzip, itertools, json, math, os, random, shutil, signal, subprocess, sys, tempfile, time, unittest
from rankbound import check as vf
from rankbound import RankboundError, Stopped, rankbound
from rankbound.selmer import Job
from rankbound.sieve import State
from rankbound import arith, curve as cv, field, linalg, machine, runtime, sieve, witness
from rankbound.check import parity_bound
from helpers import SRC, bare_solver, run_route
jobs = sieve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestSlice(unittest.TestCase):
    def test_failed(self):
        """A slice whose siever exited nonzero is an error, with the last line of its error output."""
        class P:
            def poll(self):
                return 1
            def error(self):
                return 'x\nsome error\n'
        with self.assertRaisesRegex(RuntimeError, '^The siever failed. some error$'):
            jobs.slice_failed(P())

    def test_finish(self):
        """A finished slice: its relations in sdir/NAME.rels.gz (side 1 dropped), its statistics in NAME.json."""
        with tempfile.TemporaryDirectory() as d:
            lines = ['3,2:5,7:\n', '-5,3:b:\n', '# Total cpu time 0.50s\n', '# Total 2 reports [0.25s/r, 2.0r/sq]\n']
            self.assertEqual(sieve.finish_slice(d, 'F_100', {'poly': 'F'}, lines), (0.5, 2))
            with gzip.open(os.path.join(d, 'F_100.rels.gz'), 'rt') as f:
                self.assertEqual(f.read().splitlines(), ['3,2:5,7', '-5,3:b'])
            self.assertEqual([linalg.parse_line(l) for l in ('3,2:5,7', '1,0:')], [(3, 2, [5, 7]), (1, 0, [])])

    def test_prime(self):
        self.assertEqual([n for n in range(60) if arith.is_prime(n)],
                         [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])
        self.assertFalse(arith.is_prime(3215031751))                 # a strong pseudoprime to bases 2, 3, 5, 7
        self.assertTrue(arith.is_prime(2 ** 61 - 1))


class TestTable(unittest.TestCase):
    F = {'GB': 13004, 'ns': 1577, 'log2_skew': 2.0, 'log2_dK': 46.5 / math.log10(2), 'poly': [7, -5, 3, 2]}

    def fake(self, rels_per_q=lambda lpb: 30):
        tried = []
        def rate(T, prm, nq=10):
            tried.append((prm['lpb0'], prm['N']))
            return {'rate': 100.0, 'sec_per_q': 0.3, 'rels_per_q': rels_per_q(prm['lpb0'])}
        return tried, rate

    def test_table(self):
        """Inside the table (log10 |d_K| up to 80, log2 skew <= 30): lpb = ceil(log2 GB) + 2, at least LPB_MIN, the
        table's N, no trial sieve; with trial=True, outside the table, and after a dried-up sieve (lpb_min): the
        trial, which tries N_TRIAL at its lpb."""
        F = self.F
        tried, rate = self.fake()
        real, sieve.Trial.rate = sieve.Trial.rate, rate
        try:
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.tune(d, F)
            self.assertEqual((prm['lpb0'], prm['N'], list(prm['polys']), prm['trial']['table']), (16, 21, ['F'], True))
            self.assertEqual(tried, [])
            with tempfile.TemporaryDirectory() as d:        # a small field: lpb LPB_MIN, above ceil(log2 GB) + 2
                prm = sieve.tune(d, dict(F, GB=1500, log2_dK=20 / math.log10(2)))
            self.assertEqual((prm['lpb0'], prm['N'], prm['trial']['table'], tried), (sieve.LPB_MIN, 17, True, []))
            for G, kw in ((F, {'trial': True}), (dict(F, log2_dK=81 / math.log10(2)), {}),
                          (dict(F, log2_skew=31.0), {}), (F, {'lpb_min': 18})):
                del tried[:]
                with tempfile.TemporaryDirectory() as d:
                    prm = sieve.tune(d, G, **kw)
                self.assertTrue(tried, kw)
                self.assertNotIn('table', prm['trial'])
                self.assertEqual(sorted({n for l, n in tried if l == prm['lpb0']}), list(sieve.N_TRIAL))
                self.assertGreaterEqual(prm['lpb0'], kw.get('lpb_min', 0))
        finally:
            sieve.Trial.rate = real

    def test_supply(self):
        """The trial's lpb supplies the relations from its band [2^(lpb-1), 2^lpb) alone: with too few relations
        per special-q the trial goes up in lpb until the band suffices."""
        def supply(l, rq):
            return rq * (sieve.li(2 ** l) - sieve.li(2 ** (l - 1))) * sieve.DUP / sieve.need(l)
        rq = lambda lpb: 0.6 * (lpb - 15)
        tried, rate = self.fake(rq)
        real, sieve.Trial.rate = sieve.Trial.rate, rate
        try:
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.tune(d, self.F, trial=True)
        finally:
            sieve.Trial.rate = real
        lpb = prm['lpb0']
        self.assertGreaterEqual(supply(lpb, rq(lpb)), sieve.SUPPLY_MIN)
        self.assertTrue(any(l < lpb for l, _ in tried))
        self.assertTrue(all(supply(l, rq(l)) < sieve.SUPPLY_MIN for l, _ in tried if l < lpb))


    def test_poly_too_big(self):
        """A coefficient the siever cannot parse (ICARM #568's, ~10^430) is out of reach before any trial."""
        F = dict(self.F, poly=[1, 10 ** 430, 0, 1])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'more than the sieve can take'):
                sieve.tune(d, F)
            self.assertFalse(os.path.exists(os.path.join(d, 'params.json')))


class TestSkewSize(unittest.TestCase):
    def test_sfsize(self):
        """field.gp's sfsize scans only the steps near the least size (~40 ms -> ~1 ms a call); the same
        [size, skew] as the full scan k = -400..400 on random skewed cubics and extremes."""
        from rankbound.check import run_gp
        body = r"""{
sfold(f) = my(best = oo, sk = 1); for(k = -400, 400, my(s = 2^(k / 4.), m = 0);
  for(i = 0, 3, my(ci = polcoef(f, i)); if(ci, m = max(m, abs(ci) * s^(i - 1.5)))); if(m < best, best = m; sk = s));
  [log(best) / log(2), sk];
}
{
setrand(5); bad = 0;
for(t = 1, 150, my(e = random(60), sk = random(80) - 40, z = vector(4, i, (random(2^(e + 1)) - 2^e) * (random(4) > 0)));
  for(i = 1, 4, z[i] = round(z[i] * 2.^(sk * (i - 2.5) / 3))); if(!z[4], z[4] = 1); if(!z[1], z[1] = 7);
  my(f = Pol(Vecrev(z))); if(sfold(f) != sfsize(f), bad++));
foreach([x^3 + 10^60, 10^60 * x^3 + 1, x^3 + x + 1, 2*x^3 - 3*x^2 + 5*x - 7, x^3 + 2^300], f, if(sfold(f) != sfsize(f), bad++));
print("BAD ", bad);
}"""
        out = run_gp(body, libs=(os.path.join(ROOT, 'tests', 'data', 'oracle', 'field.gp'),))
        self.assertIn('BAD 0', out)


POLYS = ([-51467, 51467, 37355, 37355],                        # ICARM #4528's field, log10 d_K 20.2
         [3743484470, -120770739562, -1909331149, 17397990103],    # #3603's, 44.1
         [7, -5, 3, 30])                                         # 2, 3, 5 divide c3: projective roots


def write_poly(d, f, skew=1):
    pp = os.path.join(d, 'F.poly')
    sieve.write_poly({'poly': f, 'log2_skew': math.log2(skew)}, pp)
    return pp


class TestRoots(unittest.TestCase):
    def test_roots(self):
        for f in POLYS:
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, 'r')
                subprocess.run([sieve.BIN, 'roots', ','.join(map(str, f)), '3000', out], check=True)
                got = {}
                with open(out) as g:
                    lines = g.readlines()
                for l in lines:
                    if l[0] != '#':
                        p, rs = l.split(':')
                        got[int(p)] = sorted(int(x) for x in rs.split(','))
                want = {p: sorted(sieve.roots(f, p)) for p in arith.primes(2, 3001)}
                self.assertEqual(got, {p: r for p, r in want.items() if r}, f)


class TestRelations(unittest.TestCase):
    def test_relations(self):
        for f, lpb, I, q0 in ((POLYS[0], 14, 10, 8192), (POLYS[1], 16, 11, 32768)):
            with tempfile.TemporaryDirectory() as d:
                pp = write_poly(d, f, 2)
                prm = {'lim0': 2 ** lpb, 'lpb0': lpb, 'mfb0': lpb, 'N': 2 * I - 1, 'q0': q0}
                _, fb0, _ = sieve.LSieve().prepare(d, prm, 'F')
                r = subprocess.run(sieve.cmd(pp, fb0[:-3], prm) + ['-q0', str(q0), '-q1', str(q0 + 60)],
                                   capture_output=True, text=True, check=True)
                n = 0
                for l in r.stdout.splitlines():
                    if l[0] == '#':
                        continue
                    ab, s0, s1 = l.split(':')
                    a, b = map(int, ab.split(','))
                    P = [int(h, 16) for h in s0.split(',')]
                    N = f[0] * b ** 3 + f[1] * a * b * b + f[2] * a * a * b + f[3] * a ** 3
                    self.assertTrue(b > 0 and math.gcd(a, b) == 1 and abs(N) == math.prod(P) and s1 == '', l)
                    self.assertTrue(all(arith.is_prime(p) and p < 2 ** lpb for p in P), l)
                    self.assertTrue(any(q0 <= p < q0 + 60 for p in P), l)
                    n += 1
                g = sieve.stats_of(''.join(l + '\n' for l in r.stdout.splitlines() if l[0] == '#'))
                self.assertEqual(g['reports'], n)
                self.assertGreater(n, 1000)


class TestInterface(unittest.TestCase):
    def test_slices(self):
        with tempfile.TemporaryDirectory() as d:
            write_poly(d, POLYS[0], 2)
            prm = {'lim0': 2 ** 14, 'lpb0': 14, 'mfb0': 14, 'N': 19, 'q0': 8192}
            sv = sieve.LSieve()
            try:
                pp, fb0, _ = sv.prepare(d, prm, 'F')
                with gzip.open(fb0, 'rt') as g:            # makefb's format, for the free relations
                    self.assertTrue(any(l.startswith('2: ') for l in g))
                running = [sv.start_slice(d, prm, 'F', 8192, 8250), sv.start_slice(d, prm, 'F', 8250, 8300)]
                while any(s.poll() is None for s in running):
                    sv.wait()
                for s in running:
                    self.assertEqual(s.poll(), 0)
                    g = sieve.stats_of(''.join(l for l in s.lines() if l[0] == '#'))
                    rels = list(sieve.relations(s.lines()))
                    self.assertEqual(g['reports'], len(rels)); self.assertGreater(len(rels), 100)
                    s.release()
                s = sv.start_slice(d, prm, 'F', 8300, 8320)       # a process sieves slice after slice
                while s.poll() is None:
                    sv.wait()
                self.assertEqual((s.poll(), len(sv.servers)), (0, 2))
                t = sieve.trial(d, prm, 'F', nq=3)
                self.assertGreater(t['rate'], 0); self.assertGreater(t['rels_per_q'], 100)
            finally:
                sv.close()


# the recorded checks of ICARM #728, #692, #356 (time and solve_s omitted)
H = {
    'icarm728': {'target': 18, 'history': [
        {'relations': 128873, 'rows': 154745, 'excess': 15143, 'bound': 18, 'dimV': 8870, 'rank': 15119, 'pivots': 15119},
    ], 'params': {'lpb0': 21, 'q0': 1048576, 'polys': {'F': {'shift': 0, 'skew': 1, 'weight': 1.0}},
        'model': {'kappa': 0.91, 'dup': 0.9, 'relations_needed': 115672, 'rate': 941.0, 'sec_per_q': 0.11074999999999999, 'rels_per_q': 115.8, 'core_hours': 0.034}}},
    'icarm692': {'target': 20, 'history': [
        {'relations': 96802, 'rows': 122806, 'excess': -10568},
        {'relations': 123202, 'rows': 149206, 'excess': 6809, 'bound': 20, 'dimV': 12272, 'rank': 20157, 'pivots': 20157},
    ], 'params': {'lpb0': 21, 'q0': 1048576, 'polys': {'F': {'shift': 0, 'skew': 3444, 'weight': 1.0}},
        'model': {'kappa': 0.91, 'dup': 0.9, 'relations_needed': 115672, 'rate': 199.4, 'sec_per_q': 0.0975, 'rels_per_q': 21.6, 'core_hours': 0.161}}},
    'icarm356': {'target': 22, 'history': [
        {'relations': 182101, 'rows': 232051, 'excess': -24109},
        {'relations': 250519, 'rows': 300469, 'excess': 23054, 'bound': 22, 'dimV': 14774, 'rank': 35611, 'pivots': 35611},
    ], 'params': {'lpb0': 22, 'q0': 2097152, 'polys': {'F': {'shift': 0, 'skew': 1073741824, 'weight': 1.0}, 'F_r1': {'shift': -5725295679, 'skew': 67108864, 'weight': 0.929}},
        'model': {'kappa': 0.91, 'dup': 0.9, 'relations_needed': 219966, 'rate': 61.1, 'sec_per_q': 0.19424999999999998, 'rels_per_q': 13.675, 'core_hours': 1.0}}},
}


def alive(pid):
    """Whether the process pid exists and is not a zombie (Linux: /proc; macOS reaps an orphan at once)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open('/proc/%d/stat' % pid) as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'
    except (OSError, IndexError):
        return True


class TestParentKilled(unittest.TestCase):
    PARENT = """import sys, time
sys.path.insert(0, %r)
from rankbound import sieve
p = sieve.spawn([sieve.BIN, 'sieve', '-poly', '7,-5,3,30', '-lim', '100000', '-lpb', '24', '-mfb', '48', '-N', '20',
                 '-q0', '100003', '-q1', '4000000000'])
print(p.pid, flush=True)
time.sleep(600)
"""

    def test_parent_killed(self):
        """A siever (sieve.spawn, sieving a range that would take hours) whose Python parent is killed by SIGKILL
        exits within a few seconds: PR_SET_PDEATHSIG on Linux, its thread that watches the parent elsewhere."""
        parent = subprocess.Popen([sys.executable, '-c', self.PARENT % SRC], stdout=subprocess.PIPE, text=True)
        pid = None
        try:
            pid = int(parent.stdout.readline())
            time.sleep(1.5)
            self.assertTrue(alive(pid))
            parent.kill()
            parent.wait()
            t = time.time()
            while alive(pid) and time.time() - t < 15:
                time.sleep(0.05)
            self.assertFalse(alive(pid))
            self.assertLess(time.time() - t, 5)
        finally:
            parent.kill()
            parent.wait()
            parent.stdout.close()
            if pid is not None:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass


class TestRetunedSlices(unittest.TestCase):
    def test_old_slices_load(self):
        """The slices sieved before a re-tune (their names from the old parameters) are loaded with the new schedule's,
        first (#846 on fe59b86: 115k relations left out after the re-tune, the sieve ran on to 1.3M)."""
        old = {'polys': {'F': {'weight': 1.0, 'shift': 0}}, 'model': {'core_hours': 0.0, 'sec_per_q': jobs.SLICE_MIN / 10},
               'q0': 512, 'lpb0': 10}
        new = dict(old, q0=2048, lpb0=12)
        with tempfile.TemporaryDirectory() as d:
            S = jobs.State(d)
            names = [jobs.slice_name(p, q0) for _, p, q0, _ in list(jobs.schedule(old))[:6]]
            for k, n in enumerate(names):
                jobs.finish_slice(S.sdir, n, {'poly': 'F'}, iter(['%d,1:2,3\n' % (k + 1)]), S.fp)
            self.assertFalse(set(names) & {jobs.slice_name(p, q0) for _, p, q0, _ in jobs.schedule(new)})
            V = bare_solver(S, prm=new)
            V.load(upto=10 ** 6)
            self.assertEqual((V.nrel, sorted(V.order)), (6, sorted(names)))


class TestAlarm(unittest.TestCase):
    def test_alarm(self):
        """The 2-descent's stop at the run's limit (runtime.alarm on the run's Budget, as Job._witness uses it): Python work and a
        program waited for are both stopped, with the approved line, and a program waited for is killed; no limit, or a
        limit not reached: nothing happens."""
        def alarm(seconds, d):
            b = runtime.Budget(seconds=seconds, stop=lambda: Stopped(d), label='run')
            return runtime.alarm(b, on_fire=lambda: jobs.say('late_witness'))
        with tempfile.TemporaryDirectory() as d:
            for work in (lambda: [x for x in iter(int, 1)], lambda: runtime.run(['sleep', '30'])):
                t = time.time()
                with self.assertLogs('rankbound', 'INFO') as out, self.assertRaises(Stopped):
                    with alarm(0.3, d):
                        work()
                self.assertLess(time.time() - t, 10)
                self.assertEqual(out.output, ['INFO:rankbound:Not enough time is left to write the witness. It is '
                                              'written first when you resume.'])
            with alarm(None, d), alarm(60, d):
                time.sleep(0.5)
            time.sleep(0.2)             # the timer is off again


class TestRetune(unittest.TestCase):
    def test_dried_up(self):
        """A sieve that dries up before the excess turns positive is re-tuned at lpb + 2, keeping params_lpb22.json and
        the slices; at LPB_MAX, or with given relations, the error stands.  Job.tune is a stand-in."""
        from rankbound.sieve import DriedUp
        with tempfile.TemporaryDirectory() as d:
            J = Job(d)
            jobs.State(d).save_params({'lpb0': 22})
            asked = []
            def tune(cores=1, lpb_min=None):
                asked.append(lpb_min)
                jobs.State(d).save_params({'lpb0': lpb_min})
                return {'lpb0': lpb_min}
            J.tune = tune
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(J.retune(DriedUp('dry'))['lpb0'], 24)
            self.assertEqual(out.output, ['INFO:rankbound:Too few relations at these settings, so the sieve is retuned. '
                                          'The relations found so far are kept.'])
            self.assertTrue(os.path.exists(os.path.join(d, 'params_lpb22.json')))
            for prm in ({'lpb0': sieve.LPB_MAX}, {'lpb0': 20, 'siever': 'given'}):
                jobs.State(d).save_params(prm)
                self.assertRaises(DriedUp, J.retune, DriedUp('dried up'))
                self.assertEqual(jobs.read_json(os.path.join(d, 'params.json')), prm)
            self.assertEqual(asked, [24])


class TestRules(unittest.TestCase):
    """The schedule is deterministic and interleaves polynomials by weight; next_check and decide on recorded runs."""

    def test_schedule(self):
        prm = H['icarm356']['params']
        a = list(itertools.islice(jobs.schedule(prm), 300))
        self.assertEqual(a, list(itertools.islice(jobs.schedule(prm), 300)))
        self.assertEqual([j for j, *_ in a], list(range(300)))
        w = {n: v['weight'] for n, v in prm['polys'].items()}
        for n in w:                      # slices by weight; contiguous ranges
            mine = [(q0, q1) for _, p, q0, q1 in a if p == n]
            self.assertAlmostEqual(len(mine) / 300, w[n] / sum(w.values()), delta=0.01)
            self.assertEqual(mine[0][0], prm['q0'])
            self.assertTrue(all(x[1] == y[0] for x, y in zip(mine, mine[1:])))

    def test_q_ranges(self):
        """Up from q0 to 2^lpb, down from q0 to Q_FLOOR, then none: every special-q once; the schedule ends when every
        polynomial's ranges are used up."""
        prm = {'lpb0': 13}
        a = list(itertools.islice(jobs.q_ranges(prm, 4096, 1000), 20))
        self.assertEqual(a[:5], [(4096, 5096), (5096, 6096), (6096, 7096), (7096, 8096), (8096, 8192)])
        self.assertEqual(a[5:], [(3096, 4096), (2096, 3096), (1096, 2096), (jobs.Q_FLOOR, 1096)])    # then none
        cover = sorted(a)
        self.assertEqual(cover[0][0], jobs.Q_FLOOR)
        self.assertTrue(all(x[1] == y[0] for x, y in zip(cover, cover[1:])))      # no gap, no overlap
        prm = {'lpb0': 12, 'q0': 2048, 'model': {'core_hours': 0, 'sec_per_q': 1.0},
               'polys': {'F': {'weight': 1.0}, 'G': {'weight': 0.5}}}
        s = list(jobs.schedule(prm))                    # finite: each polynomial's q from 2^10 to 2^12, once
        for n in 'FG':
            cover = sorted((q0, q1) for _, p, q0, q1 in s if p == n)
            self.assertEqual((cover[0][0], cover[-1][1]), (jobs.Q_FLOOR, 4096))
            self.assertTrue(all(x[1] == y[0] for x, y in zip(cover, cover[1:])))

    def test_recorded_runs(self):
        for name, d in sorted(H.items()):
            h, prm, tgt = d['history'], d['params'], d['target']
            self.assertEqual(jobs.next_check([], prm), jobs.FIRST_CHECK * prm['model']['relations_needed'], name)
            for i in range(1, len(h)):
                self.assertIsNone(jobs.decide(h[:i], tgt), name)      # every earlier check: go on
                self.assertLessEqual(jobs.next_check(h[:i], prm), h[i]['relations'] * 1.0 + 1, name)
            self.assertEqual(jobs.decide(h, tgt), 'goal', name)
        # #692: one purge-only check at excess -10568, then the next check from the default slope
        h = H['icarm692']['history']
        self.assertAlmostEqual(jobs.next_check(h[:1], H['icarm692']['params']),
                               96802 + max(0.03 * 96802, 1.05 * 10568 / 0.6 + 0.01 * 96802))

    def test_slope(self):
        """Below excess 0 the next check comes from the measured slope dE/drelation (when above 0.05), with at least 3%
        more relations; after a check with a bound, at NEXT times the relations."""
        prm = H['icarm692']['params']
        h = [{'relations': 100000, 'excess': -20000}, {'relations': 120000, 'excess': -10000}]     # slope 0.5
        self.assertAlmostEqual(jobs.next_check(h, prm), 120000 + 1.05 * 10000 / 0.5 + 1200)
        h[1]['excess'] = -19990                                                                 # slope 0.0005: default
        self.assertAlmostEqual(jobs.next_check(h, prm), 120000 + 1.05 * 19990 / jobs.SLOPE_DEFAULT + 1200)
        h[1]['excess'] = -10                                                                    # at least 3% more
        self.assertAlmostEqual(jobs.next_check(h, prm), 120000 * 1.03)
        self.assertAlmostEqual(jobs.next_check(h + [{'relations': 130000, 'excess': 50, 'bound': 3}], prm), 130000 * jobs.NEXT)

    def test_plateau(self):
        """The give-up rule on relations (a history without costs): the same best bound since GIVE_UP times the
        relations is a plateau; while above the target, GIVE_UP_ABOVE times; at the target, the goal."""
        h = [{'relations': 1000, 'excess': 5, 'bound': 4}, {'relations': 1100, 'excess': 9, 'bound': 4}]
        self.assertIsNone(jobs.decide(h, None))
        h.append({'relations': 2000, 'excess': 20, 'bound': 4})
        self.assertEqual(jobs.decide(h, None), 'plateau')
        self.assertIsNone(jobs.decide(h, 3))                     # above the target 3: 3x
        self.assertEqual(jobs.decide(h + [{'relations': 3000, 'excess': 20, 'bound': 4}], 3), 'plateau')
        self.assertEqual(jobs.decide(h, 4), 'goal')
        self.assertIsNone(jobs.decide(h + [{'relations': 2400, 'excess': -1}], None))

    def test_plateau_next(self):
        """NEXT^3 < GIVE_UP <= NEXT^4 < GIVE_UP_ABOVE: with checks at the relation counts next_check asks for, the
        give-up rule takes 4 more checks at the same bound (at most); the bound's run starts at its first check."""
        self.assertTrue(jobs.NEXT ** 3 < jobs.GIVE_UP <= jobs.NEXT ** 4 < jobs.GIVE_UP_ABOVE)
        h = [{'relations': 100000, 'excess': 50, 'bound': 7, 'keep': 1000}]
        for k in range(4):
            self.assertIsNone(jobs.decide(h, None), k)
            h.append({'relations': math.ceil(jobs.next_check(h, {})), 'excess': 900, 'bound': 7, 'keep': 1000})
        self.assertEqual(jobs.decide(h, None), 'plateau')
        h[0]['bound'] = 9                                   # 7 since the second check only
        self.assertIsNone(jobs.decide(h, None))

    def test_plateau_lb(self):
        """While the bound is above the characters' lower bound lb, giving up needs GIVE_UP_ABOVE times the cost (here
        relations), else GIVE_UP times (on the benchmark: G = lb + 1 at 1.3x, lb at 1.5-2.7x)."""
        h = lambda lb, n: [{'relations': 100, 'bound': 11, 'lb': lb}, {'relations': n, 'bound': 11, 'lb': lb}]
        self.assertEqual([jobs.decide(h(lb, n), None) for lb, n in ((10, 290), (10, 300), (None, 199), (None, 200),
                                                                   (11, 200))],
                         [None, 'plateau', None, 'plateau', 'plateau'])

    def test_goal_stop(self):
        """The 2-descent certifies the sieve's relations: the goal stops a run only with r, where S_bk at the goal is r (or r + 1
        by parity), which the 2-descent cannot beat; without r, or with u + n above it, the 2-descent may still do better."""
        F = lambda r, u=1, n=0, w=1: {'rank_lower_bound': r, 'u': u, 'n': n, 'w': w}
        self.assertEqual([jobs.goal_stop(F(r, u, n, w)) for r, u, n, w in
                          ((None, 1, 0, 1), (2, 1, 0, 1), (1, 1, 0, 1), (0, 2, 3, 1), (16, 1, 4, 1), (3, 2, 5, 1))],
                         [False, True, True, False, True, False])

    def test_exact(self):
        """'exact': the bound gives the rank bound of the characters' lower bound lb (no relations give a better one,
        under GRH), also when G > lb; the goal first; without lb or a rank function, the plateau rule alone."""
        rank = lambda G: G + 1 if G % 2 == 0 else G         # the largest odd S <= G + 1 (u + n = 1, w = -1)
        h = [{'relations': 1000, 'excess': 5, 'bound': 9, 'lb': 6}]
        self.assertIsNone(jobs.decide(h, 3, rank))          # S 9 against 7
        h.append({'relations': 1030, 'excess': 9, 'bound': 7, 'lb': 6})
        self.assertEqual(jobs.decide(h, 3, rank), 'exact')  # S 7 = rank(6)
        self.assertIsNone(jobs.decide(h, 3))                # no rank function
        self.assertIsNone(jobs.decide(h[:1] + [dict(h[1], lb=None)], 3, rank))
        self.assertEqual(jobs.decide(h, 7, rank), 'goal')
        h.append({'relations': 2060, 'excess': 9, 'bound': 7})
        self.assertEqual(jobs.decide(h, None, rank), 'plateau')    # no lb: 7 since 1030, 2060 >= 2 x 1030

    def test_next_check_too_dense(self):
        """A check whose dense phase did not fit (no bound, excess >= 0) is followed by one at NEXT times the
        relations, not by the slope rule of a negative excess."""
        h = [{'relations': 100000, 'excess': 4000, 'dense_MB': 9e5}]
        self.assertAlmostEqual(jobs.next_check(h, {}), 100000 * jobs.NEXT)

    def test_goal(self):
        """The stopping goal is the largest G that gives the least S possible: g* + 1 when S = r can be reached; with
        w != (-1)^r the G that gives S = r + 1; 0 or 1 when u + n > r + 1 (g* + 1 < 0: S = r is out of reach)."""
        F = lambda r, u, n, w: {'rank_lower_bound': r, 'u': u, 'n': n, 'w': w}
        self.assertEqual(field.goal(F(29, 1, 11, -1)), 18)          # #728: g* + 1
        self.assertEqual(field.goal(F(29, 1, 11, 1)), 19)           # parity mismatch: G = 19 gives S = 30 = r + 1
        self.assertEqual(field.goal(F(0, 1, 0, 1)), 0)              # 571a1 with r = 0: g* + 1 = 0
        self.assertEqual(field.goal(F(1, 2, 2, -1)), 0)             # g* + 1 = -2 < 0: S(0) = 3 < S(1) = 5
        self.assertEqual(field.goal(F(1, 2, 3, -1)), 1)             # g* + 1 = -3 < 0: S(0) = S(1) = 5
        self.assertEqual(field.goal(F(None, 1, 0, 1)), 0)           # S(0) = 0 < S(1) = 2
        for r, u, n, w in itertools.product(list(range(5)) + [None], (1, 2), range(6), (1, -1)):
            g, S = field.goal(F(r, u, n, w)), (lambda G: parity_bound(G, u, n, w))
            best = S(0) if r is None else max(r if w == (-1) ** r else r + 1, S(0))   # the least S any G >= 0 gives
            self.assertEqual(S(g), best)                              # reaching the goal gives it ...
            self.assertGreater(S(g + 1), best)                        # ... and the goal is the largest such G
        h = [{'relations': 1000, 'excess': 5, 'bound': 0}]
        self.assertEqual(jobs.decide(h, 0), 'goal')                 # G = 0 stops at once

    def test_tune_lopsided(self):
        """A field with no relation below lpb 25 (ICARM #3587): lpb_A goes from 23 to 25, and 0.9 relations per q at
        lpb 25 (#6782's production rate) cannot supply lpb 25 from its band: 27 (trial faked)."""
        F = {'GB': 340848, 'ns': 29306, 'log2_skew': 52.2, 'poly': [7, -5, 3, 2]}
        def rate(self, prm, nq=10):
            r = 0.0 if prm['lpb0'] < 25 else 3.0 * 2 ** (prm['lpb0'] - 25)
            return {'rate': r, 'sec_per_q': 0.3, 'rels_per_q': r * 0.3}
        real, sieve.Trial.rate = sieve.Trial.rate, rate
        try:
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.trial_params(d, F, 23, 2.0)
            self.assertEqual((prm['trial']['lpb_A'], prm['lpb0']), (25, 27))
            with tempfile.TemporaryDirectory() as d:      # no relation at any lpb up to LPB_MAX: a clear error
                sieve.Trial.rate = lambda *a, **k: {'rate': 0.0, 'sec_per_q': 0.0, 'rels_per_q': 0.0}
                self.assertRaisesRegex(RuntimeError, 'too few relations', sieve.trial_params, d, F, 23, 2.0)
        finally:
            sieve.Trial.rate = real

    def test_tune_lpb_min(self):
        """lpb_min, as in Job.retune: no lpb below it is chosen, although lpb 17 is the cheapest; a small field
        is sieved at LPB_MIN at least (trial faked)."""
        F = {'GB': 2 ** 15, 'ns': 3000, 'log2_skew': 2.0, 'poly': [7, -5, 3, 2]}
        real, tried = sieve.Trial.rate, []
        def rate(self, prm, nq=10):
            tried.append(prm['lpb0'])
            return {'rate': 100.0 * 1.2 ** -prm['lpb0'], 'sec_per_q': 0.3, 'rels_per_q': 1000.0}
        try:
            sieve.Trial.rate = rate
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(sieve.trial_params(d, F, 19, 0.5)['lpb0'], 17)
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.trial_params(d, F, 19, 0.5, lpb_min=23)
            self.assertEqual((prm['lpb0'], prm['trial']['lpb_A'], prm['trial']['lpb_min']), (23, 23, 23))
            del tried[:]
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.tune(d, dict(F, GB=600), mem_gb=0.5, lpb_min=12)
            self.assertEqual(prm['lpb0'], sieve.LPB_MIN); self.assertGreaterEqual(min(tried), sieve.LPB_MIN)
        finally:
            sieve.Trial.rate = real

    def test_tune_supply(self):
        """An lpb whose special-q cannot supply the relations needed is no candidate: the cheapest lpb 17 and 19 are
        passed over for 21; with too few relations per q at every lpb up to LPB_MAX, a clear error (trial faked)."""
        F = {'GB': 2 ** 15, 'ns': 3000, 'log2_skew': 2.0, 'poly': [7, -5, 3, 2]}
        rpq = {17: 0.01, 19: 0.1, 21: 2.0}
        real = sieve.Trial.rate
        try:
            sieve.Trial.rate = lambda self, prm, nq=10: {'rate': 100.0, 'sec_per_q': 0.3, 'rels_per_q': rpq.get(prm['lpb0'], 0.01)}
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.trial_params(d, F, 19, 0.5)
            self.assertEqual(prm['lpb0'], 21)
            self.assertGreaterEqual(prm['model']['supply'], sieve.SUPPLY_MIN)
            rpq = {}
            with tempfile.TemporaryDirectory() as d:
                self.assertRaisesRegex(RuntimeError, 'too few relations', sieve.trial_params, d, F, 19, 0.5)
        finally:
            sieve.Trial.rate = real



class TestTimeLimit(unittest.TestCase):
    def test_job_end(self):
        """The Slurm job's end from SLURM_JOB_END_TIME, else from `squeue -h -j ID -o %e` (here a fake squeue);
        none outside Slurm or without a time limit.  Inside a Slurm job (SLURM_JOB_ID) the deadline is that end less
        JOB_MARGIN; outside one, or when it cannot be read, max_hours from now."""
        self.assertEqual(machine.job_end({'SLURM_JOB_END_TIME': '1760000000', 'SLURM_JOB_ID': '7'}), 1760000000.0)
        self.assertIsNone(machine.job_end({}))
        with tempfile.TemporaryDirectory() as d:
            sq = os.path.join(d, 'squeue')
            def fake(out):
                with open(sq, 'w') as f:
                    f.write('#!/bin/sh\necho "$@" > %s/args\necho "%s"\n' % (d, out))
                os.chmod(sq, 0o755)
            fake('2026-10-15T04:00:00')
            t = machine.job_end({'SLURM_JOB_ID': '123'}, squeue=sq)
            self.assertEqual(time.localtime(t)[:6], (2026, 10, 15, 4, 0, 0))
            with open(os.path.join(d, 'args')) as f:
                self.assertEqual(f.read().split(), ['-h', '-j', '123', '-o', '%e'])
            # an array task: its own line, JOB_TASK (the array's job id lists every task of the array)
            arr = {'SLURM_JOB_ID': '124', 'SLURM_ARRAY_JOB_ID': '123', 'SLURM_ARRAY_TASK_ID': '1'}
            self.assertEqual(machine.job_end(arr, squeue=sq), t)
            with open(os.path.join(d, 'args')) as f:
                self.assertEqual(f.read().split(), ['-h', '-j', '123_1', '-o', '%e'])
            for out in ('NONE', 'Unknown', '', '2026-10-15T04:00:00\n2026-10-15T05:00:00'):   # no time limit, the job
                fake(out)                                                                    # unknown, or two lines
                self.assertIsNone(machine.job_end({'SLURM_JOB_ID': '123'}, squeue=sq))
            self.assertIsNone(machine.job_end({'SLURM_JOB_ID': '123'}, squeue=os.path.join(d, 'none')))
        end = time.time() + 3600
        with self.assertLogs('rankbound', 'INFO') as out:
            self.assertAlmostEqual(runtime.run_seconds(11.3, True, {'SLURM_JOB_END_TIME': '%d' % end}),
                                   int(end) - runtime.JOB_MARGIN - time.time(), delta=5)
            for job, env in ((True, {}), (False, {'SLURM_JOB_END_TIME': '%d' % end})):
                self.assertAlmostEqual(runtime.run_seconds(2, job, env), 7200, delta=5)
            self.assertIsNone(runtime.run_seconds(None, True, {}))
        from rankbound import messages as cli
        self.assertEqual(out.output[0], 'INFO:rankbound:This job ends at %s, so rankbound will stop at %s.' % (
            cli.clock(int(end)), cli.clock(int(end) - runtime.JOB_MARGIN)))
        self.assertEqual(out.output[1:], ['INFO:rankbound:Could not read the end time of the Slurm job, so rankbound stops '
                                          'after 2.0 hours as set by --max-hours.',
                                          'INFO:rankbound:Could not read the end time of the Slurm job, so rankbound runs '
                                          'without a time limit.'])

    def test_check_plan(self):
        """check_need: each check's solves together, in core-seconds, scaled by the square of the relation ratio (never
        down), over the threads; a late solve with f2elim's estimate; None before a full check.  plan_check: a check
        that would not end in the time left waits for the next run ('later'), unless the run has FRESH of its window
        left; one that needs more than the window, 'too_long'; no limit: 'now'."""
        h = [{'relations': 1000, 'excess': -50, 'solve_s': 2.0},
             {'relations': 2000, 'excess': 10, 'bound': 9, 'solve_s': 100.0},       # one check: 100 + 60 + 40
             {'relations': 2000, 'excess': 10, 'bound': 9, 'keep': 10000, 'solve_s': 60.0},
             {'relations': 2000, 'excess': 10, 'keep_tried': -1, 'dense_MB': 9e5, 'solve_s': 40.0}]
        N = jobs.check_need
        self.assertIsNone(N([], 5000))
        self.assertIsNone(N(h[:1], 5000))                                  # purge checks only
        self.assertEqual(N(h, 2000), 200.0)
        self.assertEqual(N(h, 3000), 200.0 * 1.5 ** 2)
        self.assertEqual(N(h, 1000), 200.0)                                # never scaled down
        self.assertEqual(N(h, 2000, 4), 200.0)                             # threads not recorded: these
        self.assertEqual(N([dict(e, threads=4) for e in h], 2000, 1), 800.0)
        late = h + [{'relations': 2400, 'excess': 10, 'late': True, 'need_s': 5000, 'solve_s': 3.0}]
        self.assertEqual(N(late, 2400), 5000.0)
        self.assertEqual(N(late[:-1] + [dict(late[-1], need_s=0)], 2400), 200.0 * 1.2 ** 2)
        P = jobs.plan_check
        self.assertEqual(P(450.0, None, None), 'now')
        self.assertEqual(P(450.0, 450.0, 1000.0), 'now')
        self.assertEqual(P(450.0, 449.0, 1000.0), 'later')
        self.assertEqual(P(950.0, 900.0, 1000.0), 'now')                   # FRESH of the window: at once
        self.assertEqual(P(1001.0, 1000.0, 1000.0), 'too_long')
        self.assertEqual(P(None, 500.0, 1000.0), 'now')                    # unknown: FIRST_FULL of the window
        self.assertEqual(P(None, 499.0, 1000.0), 'later')

    def test_run_check(self):
        """checking.json (the check's slices, before it starts) stays for a check that was killed (SystemExit,
        KeyboardInterrupt), stopped (Stopped) or late: the next run's only work; a check that fails with an error or
        ends with or without an outcome removes it; an outcome is written to STOP."""
        class V:
            nrel, threads, hist, order = 1300, 4, [], ['F_1', 'F_2']
            def __init__(self, exc=None, out=None):
                self.exc, self.out = exc, out
            def check(self, work, dry=False):
                if self.exc:
                    raise self.exc
                return self.out
        with tempfile.TemporaryDirectory() as d:
            S = jobs.State(d)
            with self.assertRaises(RuntimeError):
                jobs.run_check(S, V(RuntimeError('f2elim failed (exit 2)')), {})
            self.assertIsNone(S.pending())
            for exc in (SystemExit(143), KeyboardInterrupt(), Stopped(d)):
                with self.assertRaises(type(exc)):
                    jobs.run_check(S, V(exc), {})
                u = S.pending()
                self.assertEqual((u['relations'], u['slices'], u['threads']), (1300, ['F_1', 'F_2'], 4))
            self.assertEqual(jobs.run_check(S, V(out='late'), {}), 'late')
            self.assertEqual(S.pending()['slices'], ['F_1', 'F_2'])
            self.assertIsNone(jobs.run_check(S, V(), {}))
            self.assertIsNone(S.pending())
            self.assertFalse(S.stopped())
            self.assertEqual(jobs.run_check(S, V(out='plateau'), {}), 'plateau')
            self.assertEqual((S.pending(), S.stop_outcome()), (None, 'plateau'))
            # another process finished the run: no check here (la/ holds that process's elimination)
            self.assertIsNone(jobs.run_check(S, V(RuntimeError('checked after STOP')), {}))



class TestEmptySlices(unittest.TestCase):
    def test_count(self):
        """MAX_EMPTY slices in a row without a relation stop the run; one relation resets the count."""
        t = {'done': 0}
        for k in range(jobs.MAX_EMPTY - 1):
            jobs.count_slice(t, 0)
        jobs.count_slice(t, 5)
        for k in range(jobs.MAX_EMPTY - 1):
            jobs.count_slice(t, 0)
        self.assertRaisesRegex(RuntimeError, 'too few relations', jobs.count_slice, t, 0)



class TestLocked(unittest.TestCase):
    def test_lock(self):
        """A second process on the same shard of a state (or a check while another holds run.lock) raises Locked;
        the lock is free again when the holder closes it."""
        from rankbound import Locked
        with tempfile.TemporaryDirectory() as d:
            S = jobs.State(d)
            fd = S.lock('shard0_1.lock')
            try:
                self.assertRaises(Locked, jobs.sieve_run, d)
                self.assertRaises(Locked, S.lock, 'shard0_1.lock')
            finally:
                os.close(fd)
            os.close(S.lock('shard0_1.lock'))

    def test_local_and_shards(self):
        """A local run and the shards of a sharded run never work on one state at once (the local run deletes the
        shards' slices in progress): sieve.lock, shared by shards, exclusive for a local run.  The shard lock is free
        again after the refusal."""
        from rankbound import Locked
        with tempfile.TemporaryDirectory() as d:
            S = jobs.State(d)
            a, b = S.lock('sieve.lock', shared=True), S.lock('sieve.lock', shared=True)      # two shards
            try:
                self.assertRaises(Locked, jobs.sieve_run, d)                               # a local run
            finally:
                os.close(a); os.close(b)
            os.close(S.lock('shard0_1.lock'))
            fd = S.lock('sieve.lock')                                                     # a local run
            try:
                self.assertRaises(Locked, jobs.sieve_run, d, shard=(1, 2))                  # a shard
            finally:
                os.close(fd)
            os.close(S.lock('shard1_2.lock'))



class TestShardEnd(unittest.TestCase):
    PRM = {'polys': {'F': {'weight': 1.0, 'shift': 0}}, 'model': {'core_hours': 0.0, 'sec_per_q': jobs.SLICE_MIN / 10},
           'q0': 1024, 'lpb0': 11}

    def test_others_left(self):
        """Shard 0's last check waits for every other shard's share of the schedule (slices j mod n), whether that shard
        is running (holds its lock) or not (queued or failed)."""
        with tempfile.TemporaryDirectory() as d:
            S = jobs.State(d)
            sched = list(jobs.schedule(self.PRM))
            self.assertGreater(len(sched), 4)
            self.assertEqual(jobs.others_left(S, self.PRM, 0, 2), ([], [1]))
            fd = S.lock('shard1_2.lock')
            try:
                self.assertEqual(jobs.others_left(S, self.PRM, 0, 2), ([1], []))
            finally:
                os.close(fd)
            for j, poly, q0, _ in sched:
                if j % 2 == 1:
                    jobs.finish_slice(S.sdir, jobs.slice_name(poly, q0), {'poly': poly}, iter([]), S.fp)
            self.assertEqual(jobs.others_left(S, self.PRM, 0, 2), ([], []))      # shard 0's own share does not count
            self.assertEqual(jobs.others_left(S, self.PRM, 1, 2), ([], [0]))

    def test_cut_short_slice(self):
        """A committed slice cut short later (a node crash) is not valid against its manifest: it is not loaded, its
        manifest is set aside and the slice is sieved again (runtime.load)."""
        with tempfile.TemporaryDirectory() as d:
            S = jobs.State(d)
            p = os.path.join(S.sdir, 'F_5.rels.gz')
            jobs.finish_slice(S.sdir, 'F_5', {'poly': 'F'}, ('%d,1:2,3\n' % a for a in range(3000)), S.fp)
            with open(p, 'rb') as f:
                data = f.read()
            with open(p, 'wb') as f:
                f.write(data[:len(data) // 2])
            V = bare_solver(S)
            V.load()
            self.assertEqual((V.nrel, V.order, S.done('F_5')), (0, [], False))
            self.assertTrue(os.path.exists(os.path.join(S.sdir, 'F_5.manifest.json.damaged')))

    def test_damaged_slice_json(self):
        """A committed slice's NAME.json cut short, or without its relation count, stops the run with an error naming
        it and what to do."""
        from rankbound import RankboundError
        with tempfile.TemporaryDirectory() as d:
            S = jobs.State(d)
            jobs.finish_slice(S.sdir, 'F_5', {'poly': 'F'}, iter(['1,2:3,5\n']), S.fp)
            p = os.path.join(S.sdir, 'F_5.json')
            for text in ('{"cpu": 1', '{"cpu": 1}'):
                with open(p, 'w') as f:
                    f.write(text)
                with self.assertRaisesRegex(RankboundError, 'F_5.json is damaged. Delete it and .*F_5.rels.gz, and that part is sieved again.'):
                    S.slice_stats()
            with open(p, 'w') as f:
                f.write('{"cpu": 1.5, "relations": 7}')
            self.assertEqual(S.slice_stats(), (1.5, 7))



class TestSieveRun(unittest.TestCase):
    """sieve_run's control flow in this process, on a schedule of 15 slices (TestShardEnd.PRM): a stand-in siever
    whose slices are a subprocess that writes R relations, and a stand-in Solver that records its checks (relations,
    dry) and answers 'plateau' to a dry one; the runner's sleeps cut to 0.02 s and POLL to 0.05 s."""
    PRM = TestShardEnd.PRM
    R = 20

    class Slice:                # finished at once, with R relations
        pid = os.getpid()
        def poll(self):
            return 0
        def lines(self):
            return iter(['%d,1:2,3:\n' % a for a in range(TestSieveRun.R)] + ['# Total cpu time 0.01s\n'])
        def release(self):
            pass
        kill = stop = release
        def end(self, q1):
            return q1

    class Siever:
        def start_slice(self, state, prm, poly, q0, q1):
            return TestSieveRun.Slice()
        def wait(self):
            time.sleep(0.02)
        def close(self):
            pass

    def setUp(self):
        self.checks, self.nxt = [], 10 ** 9
        test = self

        class V:
            threads, pids, budget, late_need, had, last = 1, (), None, None, None, None
            @classmethod
            def of(cls, S, F, prm, jobs, norm=None):
                v = cls(); v.S, v.hist, v.nrel, v.order = S, S.history(), 0, []
                return v
            def in_order(self):
                return iter(self.S.finished())
            def load(self, names=None, upto=None):
                self.order = list(names) if names is not None else self.S.finished()
                self.nrel = TestSieveRun.R * len(self.order)
            def check(self, work, dry=False):
                test.checks.append((self.nrel, dry))
                self.announce()
                return 'plateau' if dry else None
            def close(self):
                pass
        real = (jobs.open_state, jobs.Solver, sieve.siever, jobs.time, jobs.POLL, jobs.next_check)
        def restore():
            jobs.open_state, jobs.Solver, sieve.siever, jobs.time, jobs.POLL, jobs.next_check = real
        self.addCleanup(restore)
        jobs.open_state = lambda S: ({}, self.PRM)
        jobs.next_check = lambda h, prm: test.nxt
        jobs.Solver, sieve.siever = V, lambda prm: self.Siever()
        jobs.time = type('T', (), {'time': time.time, 'strftime': time.strftime, 'localtime': time.localtime,
                                     'sleep': staticmethod(lambda s: time.sleep(min(s, 0.02)))})
        jobs.POLL = 0.05
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.S = jobs.State(self.d)
        self.sched = list(jobs.schedule(self.PRM))
        self.assertEqual(len(self.sched), 15)

    def finished(self):
        return sorted(self.S.finished()) == sorted(jobs.slice_name(p, q0) for _, p, q0, _ in self.sched)

    def test_schedule_end(self):
        """A local run whose schedule ends before a check is due: one last check, dry, on every slice; its plateau
        is written to STOP; no partial output left."""
        with self.assertLogs('rankbound', 'INFO'):
            out, V = jobs.sieve_run(self.d, jobs=2)
        self.assertEqual((out, self.checks), ('plateau', [(15 * self.R, True)]))
        self.assertEqual(self.S.stop_outcome(), 'plateau')
        self.assertTrue(self.finished())

    def test_later(self):
        """A check due (here at 30 relations) that would not end in the time left (the history has a full check of
        500 s at these relations; 360 s left of a 1000 s window: not fresh) is not run: the run stops sieving at once
        and ends without an outcome, the check's slices in checking.json (said once).  The next run's only work
        before sieving is that check, on exactly those slices; then it sieves on to the schedule's end, whose last
        check (dry) gives the plateau."""
        self.S.save_history([{'relations': 2 * self.R, 'rows': 1, 'excess': 1, 'bound': 5, 'solve_s': 500.0}])
        self.nxt = 30
        with self.assertLogs('rankbound', 'INFO') as out:
            res, V = jobs.sieve_run(self.d, jobs=2, budget=runtime.Budget(seconds=360), window=1000)
        self.assertEqual((res, self.checks), (None, []))
        pend = self.S.pending()
        self.assertEqual((pend['slices'], pend['relations']), (V.order, V.nrel))
        self.assertGreaterEqual(V.nrel, 30)
        self.assertFalse(self.finished() or self.S.stopped())
        n0 = len(self.S.finished())
        self.assertLessEqual(n0, len(V.order) + 2)          # no sieving past the pending check (2 cores: in flight)
        self.assertEqual(out.output.count('INFO:rankbound:Not enough time is left for the next check. It runs first when '
                                          'you resume.'), 1, out.output)
        self.nxt = 10 ** 9
        with self.assertLogs('rankbound', 'INFO') as out:
            res, V = jobs.sieve_run(self.d, jobs=2, budget=runtime.Budget(seconds=10 ** 5))
        self.assertIn('INFO:rankbound:The check left from the last run runs first, on the same {:,} relations.'.format(
            pend['relations']), out.output)
        self.assertEqual((res, self.checks), ('plateau', [(pend['relations'], False), (15 * self.R, True)]))
        self.assertIsNone(self.S.pending())
        self.assertTrue(self.finished())

    def test_too_long(self):
        """A check that needs more than the whole window (f2elim's estimate of an earlier late solve, 10^6 s) stops
        the run at once with 'too_long' (STOP: the window, the threads, the need), its slices kept for a run with more."""
        self.S.save_history([{'relations': 2 * self.R, 'rows': 1, 'excess': 1, 'late': True, 'need_s': 1e6,
                              'solve_s': 1.0}])
        self.nxt = 30
        with self.assertLogs('rankbound', 'INFO'):
            res, V = jobs.sieve_run(self.d, jobs=2, budget=runtime.Budget(seconds=360))
        st = self.S.stop_record()
        self.assertEqual((res, self.checks, st['outcome'], st['threads'], st['need_s']), ('too_long', [], 'too_long', 2,
                                                                                         10 ** 6))
        self.assertAlmostEqual(st['window'], 360, delta=5)
        self.assertEqual(self.S.pending()['slices'], V.order)

    def test_shard(self):
        """Shard 1 of 2 sieves its share until the folder has the relations of the next check, and exits without a
        check; shard 2 not running (no lock) and short of relations: 'Submit it again'.  With a check pending, a
        shard does not sieve."""
        self.nxt = 3 * self.R
        with self.assertLogs('rankbound', 'INFO') as out:
            res, V = jobs.sieve_run(self.d, shard=(0, 2), jobs=1, budget=runtime.Budget(seconds=360))
        self.assertEqual((res, V, self.checks), (None, None, []))
        self.assertGreaterEqual(self.S.slice_stats()[1], self.nxt)
        self.assertIn('INFO:rankbound:Shard 1 of 2 is done.', out.output)
        self.nxt = 10 ** 9
        with self.assertLogs('rankbound', 'INFO') as out:
            jobs.sieve_run(self.d, shard=(0, 2), jobs=2, budget=runtime.Budget(seconds=360))
        self.assertEqual(out.output[-2:], ['INFO:rankbound:Shard 1 of 2 is done.',
                                           'INFO:rankbound:Shard 2 is not running. Submit it again.'])
        n = len(self.S.finished())
        self.S.begin_check(['F_1024'], 20)
        with self.assertLogs('rankbound', 'INFO') as out:
            self.assertEqual(jobs.sieve_run(self.d, shard=(1, 2), jobs=2), (None, None))
        self.assertEqual(out.output, ['INFO:rankbound:A check is waiting, so this shard does not sieve. Run the job '
                                      'that checks first.'])
        self.assertEqual(len(self.S.finished()), n)


class TestMatrixWrite(unittest.TestCase):
    def test_append(self):
        """A second write of a Matrix to the same file appends the rows added since and rewrites the header: the same
        bytes as a whole write; a file touched by another writer is written whole."""
        from rankbound import linalg
        M = linalg.Matrix({'J': []}, [(2, 0), (3, 1)])
        rnd = random.Random(5)
        with tempfile.TemporaryDirectory() as d:
            p, q = os.path.join(d, 'a.bin'), os.path.join(d, 'b.bin')
            for k in range(4):
                for _ in range(10):
                    M._add([(rnd.randrange(50), 0) for _ in range(3)], 'x')
                M.write(p)
                linalg.Matrix.write(type('W', (), {'rows': M.rows, 'col': M.col, 'pad': 0, 'written': None})(), q)
                with open(p, 'rb') as f, open(q, 'rb') as g:
                    self.assertEqual(f.read(), g.read(), k)
            with open(p, 'ab') as f:
                f.write(b'junk')
            M._add([(1, 0)], 'x')
            M.write(p)
            linalg.Matrix.write(type('W', (), {'rows': M.rows, 'col': M.col, 'pad': 0, 'written': None})(), q)
            with open(p, 'rb') as f, open(q, 'rb') as g:
                self.assertEqual(f.read(), g.read())


class TestSolveReuse(unittest.TestCase):
    def test_reuse(self):
        """linalg.solve records the matrix's fingerprint with f2elim's output and reuses elim.ops/piv for the same
        matrix (certify after the proving check: no second elimination); another matrix runs f2elim again."""
        from rankbound import linalg
        if not os.path.exists(linalg.ENGINE):
            self.skipTest('f2elim not built')
        class M:
            ns, pad, written = 2, 0, None
            def __init__(self, rows):
                self.rows = [linalg.array.array('I', r) for r in rows]
                self.col = range(1 + max(max(r) for r in rows))
            def held(self):
                return []
            def write(self, path):
                linalg.Matrix.write(self, path)
        calls = []
        real, real_sha = linalg.runtime.subprocess.Popen, linalg.sha256_file
        def counting(*a, **k):
            calls.append(a[0][0]); return real(*a, **k)
        linalg.runtime.subprocess.Popen = counting
        try:
            with tempfile.TemporaryDirectory() as d:
                m1 = M([[0, 2], [1, 2], [0, 1, 3], [3, 1], [0]])
                r1 = linalg.solve(m1, d, 0)
                r2 = linalg.solve(m1, d, 0, gate=True, threads=4)
                self.assertEqual(len(calls), 1)
                self.assertEqual((r1.bound, r1.dimV, r1.rank), (r2.bound, r2.dimV, r2.rank))
                r3 = linalg.solve(M([[0, 2], [1, 2]]), d, 0)
                self.assertEqual(len(calls), 2)
                self.assertNotEqual(r3.rank, r1.rank)
                # another f2elim binary (its sha256 is in the key): the elimination runs again
                linalg.solve(m1, d, 0)
                self.assertEqual(len(calls), 3)
                linalg.sha256_file = lambda p: 'another build' if p == linalg.ENGINE else real_sha(p)
                linalg.solve(m1, d, 0)
                self.assertEqual(len(calls), 4)
                linalg.sha256_file = real_sha
                # a truncated elim.piv (a full disk, say) is not reused
                linalg.solve(m1, d, 0)
                self.assertEqual(len(calls), 5)
                with open(os.path.join(d, 'elim.piv'), 'r+b') as f:
                    f.truncate(2)
                self.assertEqual(linalg.solve(m1, d, 0).bound, r1.bound)
                self.assertEqual(len(calls), 6)
        finally:
            linalg.runtime.subprocess.Popen, linalg.sha256_file = real, real_sha

    def test_reuse_resumed(self):
        """A resumed certify's matrix (the certified solve's rows, loaded again: fewer columns than the run had
        when it solved them) is written with that solve's column count (Matrix.pad from best.json's ncols), so its
        elimination is reused without f2elim; and the record keeps the fields jobs.Solver._best added (keep, shrunk)."""
        from rankbound import linalg
        if not os.path.exists(linalg.ENGINE):
            self.skipTest('f2elim not built')
        class M:
            ns, pad, written = 2, 0, None
            def __init__(self, rows, nc):
                self.rows = [linalg.array.array('I', r) for r in rows]
                self.col = range(nc)
            def held(self):
                return []
            def write(self, path):
                linalg.Matrix.write(self, path)
        calls = []
        real = linalg.runtime.subprocess.Popen
        def counting(*a, **k):
            calls.append(a[0][0]); return real(*a, **k)
        linalg.runtime.subprocess.Popen = counting
        try:
            with tempfile.TemporaryDirectory() as d:
                rows = [[0, 2], [1, 2], [0, 1, 3], [3, 1], [0]]
                r1 = linalg.solve(M(rows, 9), d, 0, name='best')      # the run's matrix had 9 columns by then
                rec = os.path.join(d, 'best.json')
                b = runtime.read_json(rec)
                self.assertEqual((b['nrows'], b['ncols'], len(calls)), (5, 9, 1))
                runtime.write_json(rec, dict(b, keep=-1, shrunk=True), None)       # as Solver._best and shrink write it
                m = M(rows, 4)                                       # resumed: only the columns of these rows
                self.assertNotEqual(linalg.ncols(m), 9)
                m.pad = runtime.read_json(rec)['ncols']
                r2 = linalg.solve(m, d, 0, name='best')
                self.assertEqual((len(calls), r2.bound), (1, r1.bound))      # reused
                # without the pad, the matrix differs: f2elim runs again, and the record keeps keep and shrunk
                linalg.solve(M(rows, 4), d, 0, name='best')
                self.assertEqual(len(calls), 2)
                b = runtime.read_json(rec)
                self.assertEqual((b['keep'], b['shrunk'], b['ncols']), (-1, True, 4))
        finally:
            linalg.runtime.subprocess.Popen = real

    def test_too_dense(self):
        """A dense phase above dense_max: f2elim exits 3, solve gives no bound (too_dense) instead of raising, and the
        elim.ops / elim.piv of the solve before are left whole (f2elim writes temporary files)."""
        from rankbound import linalg
        if not os.path.exists(linalg.ENGINE):
            self.skipTest('f2elim not built')
        rnd = random.Random(3)
        def matrix():
            M = linalg.Matrix({'J': []}, [(i, 0) for i in range(30)])
            for _ in range(200):
                M._add([(c, 0) for c in rnd.sample(range(30), 3)] + [('L', c) for c in rnd.sample(range(150), 2)], 'x')
            return M
        with tempfile.TemporaryDirectory() as d:
            r1 = linalg.solve(matrix(), d, 0, dense=0)
            self.assertIsNotNone(r1.bound); self.assertFalse(r1.too_dense)
            with open(r1.ops, 'rb') as f, open(r1.piv, 'rb') as g:
                before = f.read(), g.read()
            r2 = linalg.solve(matrix(), d, 0, dense=0, dense_max=1)
            self.assertTrue(r2.too_dense); self.assertIsNone(r2.bound); self.assertIsNotNone(r2.excess)
            self.assertIsNotNone(r2.dense_mb)
            with open(r1.ops, 'rb') as f, open(r1.piv, 'rb') as g:
                self.assertEqual((f.read(), g.read()), before)
            self.assertEqual([x for x in os.listdir(d) if x.endswith('.tmp')], [])
            # elim.json still records r1's solve: its matrix again reuses the files
            self.assertTrue(os.path.exists(os.path.join(d, 'elim.json')))
            ino = os.stat(r1.ops).st_ino
            rnd.seed(3)
            self.assertEqual(linalg.solve(matrix(), d, 0, dense=0).bound, r1.bound)
            self.assertEqual(os.stat(r1.ops).st_ino, ino)                   # not run again: not renamed into place

    def test_killed(self):
        """An f2elim killed by SIGKILL (an out-of-memory kill) gives no bound (too_dense, killed), not an error, and
        the check records it.  The dense limit leaves out what the paused sieve processes and the prover hold (rss,
        from /proc where there is one)."""
        from rankbound import linalg
        real = linalg.ENGINE
        with tempfile.TemporaryDirectory() as d:
            fake = os.path.join(d, 'f2elim')
            with open(fake, 'w') as f:
                f.write('#!/bin/sh\necho "purge rows 2 removed 0 lcols 1 nnz 4 excess 1 time 0.0"\nkill -9 $$\n')
            os.chmod(fake, 0o755)
            M = linalg.Matrix({'J': []}, [(0, 0), (1, 0)])
            M._add([(0, 0), ('L', 1)], 'x'); M._add([(1, 0), ('L', 1)], 'y')
            linalg.ENGINE = fake
            try:
                lines = []
                r = linalg.solve(M, os.path.join(d, 'w'), 0, log=lines.append)
                self.assertTrue(r.too_dense and r.killed); self.assertIsNone(r.bound); self.assertEqual(r.excess, 1)
                self.assertEqual(lines[-1], 'The elimination ran out of memory and was stopped; no bound from this check.')
            finally:
                linalg.ENGINE = real
        old = dict(os.environ)
        try:
            os.environ['SLURM_MEM_PER_NODE'] = '1024'         # 1 GB
            self.assertEqual(linalg.dense_limit(), int(linalg.DENSE_SHARE * (1 << 30)))
            self.assertEqual(linalg.dense_limit(1 << 29), int(linalg.DENSE_SHARE * (1 << 29)))
            self.assertEqual(linalg.dense_limit(2 << 30), 1)                    # never 0: that is no limit
            os.environ['SLURM_MEM_PER_NODE'] = '0'            # --mem=0: all of the node's memory
            self.assertGreater(linalg.dense_limit(), 1 << 20)
        finally:
            os.environ.clear(); os.environ.update(old)
        if os.path.exists('/proc/self/status'):
            self.assertGreater(linalg.rss([os.getpid()]), 1 << 20)
        else:
            self.assertEqual(linalg.rss([os.getpid()]), 0)
        self.assertEqual(linalg.rss([2 ** 22 + 12345]), 0)                       # no such process

    def test_write_failure(self):
        """f2elim checks its writes (a full disk; here a file size limit, as with `ulimit -f`): exit 2 and no
        elim.ops."""
        import resource, signal
        from rankbound import linalg
        if not os.path.exists(linalg.ENGINE):
            self.skipTest('f2elim not built')
        M = linalg.Matrix({'J': []}, [(i, 0) for i in range(3)])
        for i in range(300):                 # a cycle of weight-2 large columns: ~300 merges
            M._add([(i % 3, 0), ('L', i), ('L', (i + 1) % 300)], 'x')       # of 8 bytes each
        def limit():
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            resource.setrlimit(resource.RLIMIT_FSIZE, (256, 256))
        with tempfile.TemporaryDirectory() as d:
            inp, out = os.path.join(d, 'm.bin'), os.path.join(d, 'elim')
            M.write(inp)
            r = subprocess.run([linalg.ENGINE, inp, '3', '0', out, '0', '64', '0', '-1', '1', '0', '300'],
                               capture_output=True, text=True, preexec_fn=limit)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn('write to', r.stderr)
            self.assertFalse(os.path.exists(out + '.ops'))
            r = subprocess.run([linalg.ENGINE, inp, '3', '0', out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)              # every argument is required
            self.assertIn('usage', r.stderr)

    def test_progress(self):
        """f2elim's progress lines on stderr (one every PROGRESS seconds; progress=0: every step), for the merge and
        the dense phase.  They are not logged, and the bound is the same with and without them."""
        from rankbound import linalg
        if not os.path.exists(linalg.ENGINE):
            self.skipTest('f2elim not built')
        M = linalg.Matrix({'J': []}, [(i, 0) for i in range(100)])
        rnd = random.Random(5)
        for i in range(300):                 # a cycle of weight-2 large columns: merged away
            M._add([(i % 100, 0), ((i * 7 + 1) % 100, 0), ('L', i), ('L', (i + 1) % 300)], 'x')
        for i in range(150):                 # rows on the 100 small columns only: two dense panels
            M._add([(c, 0) for c in rnd.sample(range(100), 5)], 'x')
        with tempfile.TemporaryDirectory() as d:
            lines = []
            r = linalg.solve(M, d, 0, dense=0, log=lines.append, progress=0)
            self.assertEqual(lines, [])
            inp = os.path.join(d, 'm.bin')
            M.write(inp)
            p = subprocess.run([linalg.ENGINE, inp, '100', '0', os.path.join(d, 'e'), '0', str(linalg.WMAX), '0', '1000',
                                '1', '0', '0'], capture_output=True, text=True)
            err = p.stderr.splitlines()
            self.assertTrue(any(l.startswith('f2elim: merge: ') and 'of 300 large columns eliminated' in l
                                for l in err), err)
            dense = [l for l in err if l.startswith('f2elim: dense: ')]
            self.assertEqual([l.split(' done')[0] for l in dense],
                             ['f2elim: dense: columns 64 of 100', 'f2elim: dense: columns 100 of 100'])
            self.assertIn('done (100.0%)', dense[-1])
            quiet = []
            os.remove(os.path.join(d, 'elim.json'))
            self.assertEqual(linalg.solve(M, d, 0, dense=0, log=quiet.append).bound, r.bound)
            self.assertEqual(quiet, [])



class TestCheck(unittest.TestCase):
    """Solver.check with linalg.solve and purge_excess faked: one choice per check (untrimmed with characters while
    the dense phase is cheap, else trimmed to excess ns for the rest of the run), the purge gate, a trimmed plateau's
    escalation, the deadline (late), the last check (dry)."""
    NS = 100

    def solver(self, d, answers, excess=5):
        """A bare Solver whose solves return answers (by keep: a SolveResult field dict, or a list taken in turn)."""
        from rankbound import linalg
        calls = []
        def solve(M, work, target, keep=1000, **kw):
            calls.append((keep, kw.get('dense_s'), bool(kw.get('chars'))))
            a = answers[keep] if keep in answers else answers['*']
            a = a.pop(0) if isinstance(a, list) else a
            base = dict(bound=None, dimV=None, rank=None, pivots=None, excess=50, ops=os.path.join(work, 'x.ops'),
                        piv=os.path.join(work, 'x.piv'))
            return linalg.SolveResult(**dict(base, **a))
        real = (linalg.solve, linalg.purge_excess)
        linalg.solve, linalg.purge_excess = solve, lambda M, work, target: excess
        self.addCleanup(lambda: setattr(linalg, 'solve', real[0]) or setattr(linalg, 'purge_excess', real[1]))
        M = linalg.Matrix({'J': []}, [(i, 0) for i in range(self.NS)])
        V = bare_solver(jobs.State(d), M, nrel=1000, top=2 ** 10, threads=1, pids=(), prm={'lpb0': 10},
                        F={'poly': [-2, 0, 0, 1], 'u': 0, 'n': 0, 'w': 1, 'dK': '-108'}, target=-1, goal_stop=False)
        return V, calls

    def test_cheap_then_trimmed(self):
        """Untrimmed with characters and the time and memory limits while cheap ('exact' at G = lb); a check whose
        untrimmed dense phase is not cheap is solved again at once trimmed to ns, and the run stays trimmed."""
        with tempfile.TemporaryDirectory() as d:
            V, calls = self.solver(d, {-1: [dict(bound=5, crank=6), dict(too_dense=True, dense_s=900.0)],
                                       self.NS: dict(bound=4)})
            self.assertIsNone(V.check(d))
            self.assertEqual((calls, V.hist[-1]['keep'], V.hist[-1]['lb']), ([(-1, 60, True)], -1, 4))
            V.nrel = 1200
            self.assertIsNone(V.check(d))
            self.assertEqual(calls[1:], [(-1, 60, True), (self.NS, None, True)])
            self.assertEqual([(h.get('keep'), h.get('bound'), h.get('dense_s')) for h in V.hist[1:]],
                             [(-1, None, 900.0), (self.NS, 4, None)])
            V.nrel = 1500
            self.assertIsNone(V.check(d))                   # trimmed from now on: no purge gate, no untrimmed try
            self.assertEqual(calls[3:], [(self.NS, None, True)])
        with tempfile.TemporaryDirectory() as d:          # 'exact': G = lb
            V, calls = self.solver(d, {-1: dict(bound=5, crank=7)})
            V.F['dK'] = '-108'
            self.assertEqual(V.check(d), 'exact')

    def test_gate(self):
        """A negative purge excess (untrimmed or trimmed) is a purge-only entry, without an elimination and without
        the lines that announce one (#846 on fe59b86: 12 "Solving ..." lines, no solve)."""
        for hist in ([], [{'relations': 700, 'rows': 900, 'excess': 50, 'bound': 4, 'keep': self.NS}]):
            with tempfile.TemporaryDirectory() as d:
                V, calls = self.solver(d, {'*': dict(bound=5)}, excess=-300)
                V.hist, said = list(hist), []
                V.announce = lambda: said.append(1)
                self.assertIsNone(V.check(d))
                self.assertEqual((calls, said, V.hist[-1]['excess'], V.hist[-1]['gated']), ([], [], -300, True))
                V, calls = self.solver(d, {'*': dict(bound=5)}, excess=10)
                V.hist, V.announce = list(hist), lambda: said.append(1)
                V.check(d)
                self.assertEqual((len(calls), said), (1, [1]))

    def test_escalation(self):
        """A trimmed plateau is solved again at 10x the excess, then untrimmed: a plateau only if the bound stays; a
        better bound goes on; a larger elimination that does not fit leaves the plateau; one that cannot end in time
        is 'late' (the next run starts at the last trim)."""
        hist = [{'relations': 400, 'rows': 900, 'excess': 50, 'bound': 4, 'keep': self.NS}]
        for answers, want, keeps in (({'*': dict(bound=4, excess=5000)}, 'plateau', [self.NS, 10 * self.NS, -1]),
                                     ({'*': dict(bound=4, excess=5000), 1000: dict(bound=3)}, None, [self.NS, 1000]),
                                     ({'*': dict(bound=4, excess=5000), 1000: dict(too_dense=True)}, 'plateau', [self.NS, 1000]),
                                     ({'*': dict(bound=4, excess=5000), 1000: dict(late=30.0)}, 'late', [self.NS, 1000])):
            with tempfile.TemporaryDirectory() as d:
                V, calls = self.solver(d, answers)
                V.hist = [dict(h) for h in hist]
                with self.assertLogs('rankbound', 'INFO') as out:
                    self.assertEqual(V.check(d), want, answers)
                self.assertEqual([c[0] for c in calls], keeps, answers)
                self.assertIn('INFO:rankbound:The bound stopped improving, so one last check uses every relation.',
                              out.output)

    def test_late_and_dry(self):
        """A check that cannot end in time: 'late', no bound, f2elim's estimate recorded (need_s, late_need); the last
        check (dry) without a bound: DriedUp; with one: a plateau."""
        with tempfile.TemporaryDirectory() as d:
            V, calls = self.solver(d, {-1: dict(late=1234.0)})
            self.assertEqual(V.check(d), 'late')
            self.assertEqual((V.hist[-1]['late'], V.hist[-1]['need_s'], V.late_need), (True, 1234, 1234.0))
        with tempfile.TemporaryDirectory() as d:
            V, calls = self.solver(d, {-1: dict(bound=None)})
            self.assertRaises(jobs.DriedUp, V.check, d, dry=True)
            V, calls = self.solver(d, {-1: dict(bound=7)})
            self.assertEqual(V.check(d, dry=True), 'plateau')

    def test_purge_excess(self):
        """linalg.purge_excess equals the excess of a full solve's purge (f2elim's one-shot mode), for target 0 and
        a positive target, on random matrices with and without a positive excess."""
        import random
        from rankbound import linalg
        rnd = random.Random(5)
        for trial in range(6):
            ns, nl = 40, rnd.choice((30, 300))
            M = linalg.Matrix({'J': []}, [(p, 0) for p in range(ns)])
            for i in range(rnd.choice((60, 400))):
                labs = {(rnd.randrange(ns), 0) for _ in range(3)} | {('L', rnd.randrange(nl)) for _ in range(2)}
                M._add(labs, 'r%d' % i, i, 1)
            for target in (0, 5):
                with tempfile.TemporaryDirectory() as d:
                    r = linalg.solve(M, d, target, gate=True)
                with tempfile.TemporaryDirectory() as d:
                    self.assertEqual(linalg.purge_excess(M, d, target), r.excess, (trial, target))


class TestSolverInit(unittest.TestCase):
    def test_init(self):
        """Solver(): with lim0 < 2^lpb0 the free relations go up to 2^lpb0, from a factor base written up to there
        (sieve roots); a run on given relations takes every free relation of its split file."""
        from rankbound import linalg
        F = {'poly': [-5, 0, 1, 1], 'log2_skew': 0.0, 'J': [], 'ramified': [], 'u': 1, 'n': 0, 'w': 1, 'rank_lower_bound': None}
        with tempfile.TemporaryDirectory() as d:
            sieve.write_poly(F, os.path.join(d, 'F.poly'))
            S = jobs.State(d)
            V = jobs.Solver(S, F, [], {'lpb0': 12, 'lim0': 1024})
            self.assertTrue(os.path.exists(os.path.join(d, 'F.lsroots_4096.gz')))
            n = len(list(linalg.free_rows(F, os.path.join(d, 'F.lsroots_4096.gz'), 4096)))
            self.assertEqual(V.nfree, n)
            self.assertGreater(n, len(list(linalg.free_rows(F, sieve.factor_base(d, 'F', 1024), 1024))))
            shutil.copy(os.path.join(d, 'F.lsroots_1024.gz'), os.path.join(d, 'F.split_1024.gz'))
            V = jobs.Solver(S, F, [], {'siever': 'given', 'lpb0': 8, 'lim0': 1024})      # its split file
            self.assertEqual(V.nfree, len(list(linalg.free_rows(F, os.path.join(d, 'F.split_1024.gz'), 1024))))

class TestState(unittest.TestCase):
    MM = ['0', '-1', '1', '-929', '-10595']

    def test_load_order(self):
        """Certify rebuilds the matrix of the last full solve: la/order.txt (in the artifact 'best') keeps the slices
        in the order they were loaded (F_5 before F_13; sorted() gives F_13 first), Solver.load(names) adds them in
        that order (the same rows, so the same fingerprint and elimination), and an order naming a missing slice, or
        one not committed by its manifest, is not used."""
        with tempfile.TemporaryDirectory() as d:
            S = jobs.State(d)
            for name, lines in (('F_5', ['3,1:2,b', '5,2:7']), ('F_13', ['5,2:7', '-4,3:d'])):
                jobs.finish_slice(S.sdir, name, {'poly': 'F'}, iter(l + '\n' for l in lines), S.fp)
            V = bare_solver(S)
            V.load(['F_5', 'F_13'])
            self.assertEqual((list(V.M.src), V.order, V.nrel), (['3,1:2,b', '5,2:7', '-4,3:d'], ['F_5', 'F_13'], 3))
            U = bare_solver(S)
            U.load()
            self.assertEqual(U.order, ['F_13', 'F_5'])
            la = os.path.join(d, 'la'); os.makedirs(la)
            self.assertEqual(S.best(la), (None, None))
            def save(names):
                with open(os.path.join(la, 'order.txt'), 'w') as f:
                    f.writelines(n + '\n' for n in names)
            save(V.order)
            self.assertEqual(S.best(la), (None, None))          # not committed
            S.commit_best(la)
            W = bare_solver(S)
            W.load(S.best(la)[0])
            self.assertEqual(list(W.M.src), list(V.M.src))
            save(['F_5', 'F_7'])                                # changed under its manifest: damaged
            self.assertEqual(S.best(la), (None, None))
            S.commit_best(la)
            self.assertEqual(S.best(la), (None, None))          # F_7 is not finished

# ========== whole runs: 571a1 (rank 0, Sha[2] = (Z/2)^2), 37a1 with points on two models, ICARM #1 (FULL)

CLI = [sys.executable, '-m', 'rankbound']
# outside any Slurm job, so that no time limit comes from a job's end
ENV = dict({k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}, PYTHONPATH=SRC)
N571 = '0_-1_1_-929_-10595'                         # 571a1's default name: its minimal model
E571 = [0, -1, 1, -929, -10595]


# ICARM #1 (r = 12): GB 3917, lpb 14; G = 10, rank <= 12
ICARM1 = dict(ainvs=[0, 0, 1, -6349808647, 193146346911036], bad_primes=[60259, 550469, 11241887, 722983930261], r=12)


class TestEndToEnd(unittest.TestCase):
    """One run of 571a1 through rankbound() (setUpClass), shared by the tests below."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(); cwd = os.getcwd()
        try:
            os.chdir(cls.tmp)
            cls.res = rankbound(E571, cores=1, compute=True)       # the 2-descent on the sieve's relations, then the Cassels-Tate step
        finally:
            os.chdir(cwd)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_api(self):
        """rankbound() on any model: rank_bound 0 (rank 0, Sha[2] = (Z/2)^2: the 2-descent's 2, then the Cassels-Tate pairing of
        rank 2), checked, 'exact' (G = 1, the characters' bound); a second call answers from run.json; the default
        workdir is ./rankbound_work/NAME, NAME the minimal model's a-invariants.  A curve with E(Q)[2] != 0 goes to the
        2-descent (rank 0 unconditionally for y^2 = x^3 - x), or the table."""
        res = self.res
        d = res.details
        self.assertEqual((res.rank_bound, res.G, d['u'], d['n'], d['claims']['S']), (0, 1, 1, 0, 2))
        self.assertTrue(res.checked)
        self.assertEqual((d['outcome'], res.method, res.grh), ('exact', '2-Selmer group', True))
        self.assertEqual((res.rank_lower_bound, res.r_proved, res.proved_equal), (None, False, True))     # S = 0
        self.assertEqual(os.path.dirname(res.workdir), os.path.realpath(os.path.join(self.tmp, 'rankbound_work')))
        self.assertEqual((os.path.basename(res.workdir), res.name), (N571, N571))
        t = time.time()
        again = rankbound(E571, workdir=res.workdir, compute=True)
        self.assertEqual((again.rank_bound, again.witness, again.details['earlier']), (0, res.witness, True))
        self.assertLess(time.time() - t, 15)
        res = rankbound([0, 0, 0, -1, 0], workdir=os.path.join(self.tmp, 'e2'), compute=True)    # y^2 = x^3 - x
        self.assertEqual((res.rank_bound, res.grh, res.method, res.torsion, res.checked, res.proved_equal),
                         (0, False, '2-isogeny descent', [2, 2], True, True))
        res = rankbound([0, 0, 0, -1, 0], workdir=os.path.join(self.tmp, 'e3'))     # listed in witnesses.tsv
        self.assertEqual((res.rank_bound, res.grh, res.method, res.torsion, res.proved_equal, res.known),
                         (0, False, '2-isogeny descent', [2, 2], True, True))

    def test_witness(self):
        """A checked format-5 witness NAME.witness.json.gz (the 2-descent on the sieve's relations: v = U, dim Sel_2 = 2), with its
        keys in order and the ideals (p) of S among its relations; another format, a changed relation, pivot list,
        claim, element, second elimination (ops_s, pivots_s) or character fails."""
        wit = self.res.witness
        self.assertEqual(wit, os.path.join(self.res.workdir, N571 + '.witness.json.gz'))
        with open(os.path.join(self.res.workdir, 'check.json')) as f:      # the checker's verdict (check --json)
            v = json.loads(f.read())
        self.assertEqual((v['verdict'], v['witness'], v['rank_upper_bound']), ('accepted', wit, 0))
        self.assertTrue(os.path.getsize(os.path.join(self.res.workdir, 'check.log')))
        raw = vf.load(wit)
        self.assertEqual(list(raw), ['name', 'curve', 'statement', 'torsion', 'hypothesis', 'field', 'weierstrass', 'theta',
                                     's_primes', 's_columns', 's_dropped', 'local', 'characters', 'claims', 'created',
                                     'relations', 'ops', 'pivots', 'pivots_s', 'elements', 'cassels_tate', 'format'])
        c0 = witness.decode(raw)          # format 6's lists (format 5's packing) as the prover holds them (ops_s: the same as ops)
        self.assertEqual((c0['torsion'], c0['hypothesis'], c0['format']), ([], 'GRH', 'rankbound-selmer-witness-6'))
        self.assertFalse(any('certificate' in k for k in vf.load(wit)))
        cl = c0['claims']
        self.assertEqual((cl['v'], cl['U'], cl['dim_sel_cap_V'], cl['S'], cl['G']), (cl['U'], cl['U'], 2, 2, 1))
        self.assertTrue({'f:%s' % p for p in c0['s_primes']} <= set(c0['relations']))       # (p) for p in S
        # a relation with an odd prime q outside S that no relation of element 0 has: added to it, an odd valuation
        primes = lambda l: [int(h, 16) for h in l.split(':')[1].split(',') if h] if not l.startswith('f:') else []
        e0 = set(c0['elements'][0])
        near = {q for j in e0 for q in primes(c0['relations'][j])}
        far = next(j for j, l in enumerate(c0['relations']) if j not in e0 and not l.startswith(('f:', 'e:')) and any(
            primes(l).count(q) % 2 and q not in near and q != 571 and q != 2 for q in primes(l)))
        for what in ('format', 'rel', 'ops', 'claim', 'element', 'ops_s', 'char'):
            c = json.loads(json.dumps(c0))
            if what == 'format':
                c['format'] = 'x'
            elif what == 'rel':
                i = next(k for k, l in enumerate(c['relations']) if not l.startswith(('f:', 'e:')))
                a, rest = c['relations'][i].split(',', 1); c['relations'][i] = '%d,%s' % (int(a) + 1, rest)
            elif what == 'ops':
                c['pivots'] = c['pivots'][:-1]; c['claims']['dimV'] += 1; c['claims']['G'] -= 1
            elif what == 'claim':
                c['claims']['S'] -= 2
            elif what == 'element':
                c['elements'][0] = sorted(e0 | {far})
            elif what == 'ops_s':       # pivots_s with a repeated row (delta 0): malformed
                c = json.loads(json.dumps(raw)); c['pivots_s'].append(0)
            else:
                q, r = c['characters']['primes'][0]; c['characters']['primes'][0] = [q, (r + 1) % q]
            t = os.path.join(self.tmp, what + '.witness.json')
            if what == 'ops_s':
                with open(t, 'w') as f:
                    json.dump(c, f)
            else:
                witness.write(c, t)
            self.assertNotEqual(vf.check(t)['verdict'], 'accepted', what)

    def fresh(self, name, keep=()):
        """A copy of the run's folder at DIR/name/N571 after prepare and tune (no slice, check or result unless kept);
        return DIR/name/N571."""
        st = os.path.join(self.tmp, name, N571)
        drop = {'slices', 'la', 'history.json', 'STOP', 'check.json', 'check.log', 'selmer.json', 'selmer.json.gz',
                'selmer.manifest.json'} - set(keep)
        shutil.copytree(self.res.workdir, st, ignore=lambda d, fs: [f for f in fs if f in drop or
                                                                     f.endswith(('.witness.json', '.witness.json.gz', '.lock'))])
        jobs.write_json(os.path.join(st, 'run.json'), dict(jobs.read_json(os.path.join(st, 'run.json')), result=None))
        return st

    def end_on_failure(self, p):
        """SIGTERM to the run P if it still runs when the test ends, so that it stops its sieve processes."""
        def end():
            if p.poll() is None:
                p.terminate()
                try:
                    p.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill(); p.communicate()
        self.addCleanup(end)

    def wait_slices(self, B, p):
        """Wait until the run P has a finished slice in its folder B."""
        t = time.time()
        part = lambda: glob.glob(os.path.join(B, 'slices', '*.rels.gz'))
        while not part() and p.poll() is None and time.time() - t < 60:
            time.sleep(0.05)
        if p.poll() is not None:
            self.fail('it exited (%s) before it sieved: %s' % (p.returncode, p.communicate()[0]))
        self.assertTrue(part())

    def slow(self, B):
        """Raise sec_per_q so that a slice is one special-q (a run that sieves for a while)."""
        p = os.path.join(B, 'params.json')
        prm = jobs.read_json(p)
        prm['polys']['F']['sec_per_q'] = 1e3
        jobs.write_json(p, prm)

    def test_shard_stop_sigterm(self):
        """A shard abandons its slices within seconds of STOP and leaves no partial output; SIGTERM to a run (scancel,
        a Slurm job's end) exits 143 within seconds, its sieve processes (in their own sessions) stopped, what they
        had sieved kept as slices."""
        for k, how in ((1, 'stop'), (0, 'term')):
            B, tmp = self.fresh(how), tempfile.mkdtemp(dir=self.tmp)
            self.slow(B)
            p = subprocess.Popen(CLI + ['bound', ','.join(map(str, E571)), '--compute', '--dir', os.path.dirname(B),
                                        '--cores', '2'] + (['--shard', '2/2'] if k else []), cwd=self.tmp,
                                 env=dict(ENV, TMPDIR=tmp), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.end_on_failure(p)
            self.wait_slices(B, p)
            if how == 'stop':
                State(B).write_stop('goal')
            else:
                p.send_signal(signal.SIGTERM)
            t = time.time()
            out = p.communicate(timeout=60)[0]
            self.assertEqual(p.returncode, 0 if how == 'stop' else 128 + signal.SIGTERM, out)
            self.assertLess(time.time() - t, 10)
            self.assertEqual(os.listdir(tmp), [])
            self.assertIsNone(jobs.read_json(os.path.join(B, 'run.json'))['result'])

    def test_time_limit(self):
        """A check that needs more than a run's window (an earlier late solve's estimate) stops the run at once with
        'too_long'.  A check longer than the time left starts at once in a run that has FRESH of its window; a check
        left to the next run (checking.json) is its first act, on its slices."""
        B = self.fresh('limit', keep=('slices',))
        S = State(B)
        S.save_history([{'relations': 1, 'rows': 1, 'excess': 1, 'late': True, 'need_s': 10 ** 6, 'solve_s': 1}])
        with self.assertLogs('rankbound', 'INFO'):
            self.assertEqual(jobs.sieve_run(B, 1, budget=runtime.Budget(seconds=60))[0], 'too_long')
        self.assertEqual((len(S.history()), S.stop_outcome()), (1, 'too_long'))
        self.assertGreaterEqual(S.stop_record()['need_s'], 10 ** 6)
        B = self.fresh('fresh', keep=('slices',))
        S = State(B)
        S.save_history([{'relations': 10 ** 9, 'rows': 1, 'excess': -1, 'solve_s': 1},
                        {'relations': 10 ** 9, 'rows': 1, 'excess': 1, 'bound': 5, 'solve_s': 99}])
        S.begin_check(S.finished(), 1)
        with self.assertLogs('rankbound', 'INFO') as cm:
            self.assertEqual(jobs.sieve_run(B, 1, budget=runtime.Budget(seconds=95), window=100)[0], 'exact')
        self.assertTrue(any('Solving the linear system of' in l for l in cm.output), cm.output)       # the check ran
        self.assertIn('INFO:rankbound:The check left from the last run runs first, on the same {:,} relations.'.format(
            S.history()[-1]['relations']), cm.output)
        self.assertEqual(S.history()[-1]['bound'], self.res.G)
        self.assertIsNone(S.pending())

    def test_resume_witness(self):
        """A stopped folder without a witness: a run again rebuilds the certified check's matrix in la/order.txt's
        order (the artifact la/best) and reuses its elimination (la/best.*); stopped in the Cassels-Tate step (selmer.json), it reuses the 2-descent's
        witness; a budget already ended: Stopped at once.  The same claims every time."""
        from rankbound import ctp
        B = self.fresh('stopped', keep=('slices', 'la', 'history.json', 'STOP'))
        rec = os.path.join(B, 'la', 'best.json')
        t = os.stat(rec).st_mtime_ns
        self.assertRaises(Stopped, Job(B).witness, 1, runtime.Budget.for_run(seconds=0, workdir=B))
        real = ctp.after_selmer
        def stop(*a, **k):
            raise Stopped(B)
        ctp.after_selmer = stop
        try:
            with self.assertLogs('rankbound', 'INFO'):
                self.assertRaises(Stopped, Job(B).witness)              # the 2-descent written, then stopped in the Cassels-Tate step
        finally:
            ctp.after_selmer = real
        self.assertEqual(os.stat(rec).st_mtime_ns, t)                   # no second elimination
        self.assertTrue(any(os.path.exists(os.path.join(B, f)) for f in ('selmer.json', 'selmer.manifest.json')))
        with self.assertLogs('rankbound', 'INFO') as out:
            p = Job(B).witness()
        self.assertIn('INFO:rankbound:Rank at most 0. Verification method: Cassels-Tate pairing (assuming the GRH).',
                      out.output)
        if getattr(p, 'data', None) is not None:           # the runner's write
            from rankbound import witness
            witness.write(p.data, p.witness)
        self.assertEqual((p.rank_bound, vf.load(p.witness)['claims']), (0, vf.load(self.res.witness)['claims']))
        self.assertEqual(vf.check(p.witness)['verdict'], 'accepted')

    def test_curve_list(self):
        """`rankbound bound --curves` in a Slurm job says a finished folder's result again; a folder past its time
        limit stops without a result; `rankbound status` gives one row per work folder."""
        L = os.path.join(self.tmp, 'list.json')
        with open(L, 'w') as f:
            json.dump({'curves': [{'id': 1, 'ainvs': E571, 'bad_primes': ['571']}]}, f)
        D = os.path.join(self.tmp, 'list')
        shutil.copytree(self.res.workdir, os.path.join(D, 'curve1'), ignore=lambda d, fs: [f for f in fs if f.endswith('.lock')])
        k = jobs.read_json(os.path.join(D, 'curve1', 'run.json'))
        key = dict(k['key'], name='curve1')             # the request's identity holds the witness's name
        jobs.write_json(os.path.join(D, 'curve1', 'run.json'), dict(k, name='curve1', key=key, result_key=key))
        env = dict(ENV, SLURM_JOB_ID='1', SLURM_JOB_END_TIME='%d' % (time.time() + 3600))
        r = subprocess.run(CLI + ['bound', '--dir', D, '--curves', L, '--ids', '1', '--cores', '1', '--max-hours', '1'],
                           cwd=self.tmp, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertRegex(r.stderr, r'This job ends at \d+:\d\d [AP]M, so rankbound will stop at \d+:\d\d [AP]M.')
        self.assertIn('Already done earlier. Rank 0, assuming the GRH. Witness: ', r.stdout)
        late = os.path.join(self.tmp, 'late')
        shutil.copytree(self.fresh('late0'), os.path.join(late, 'curve1'))
        k = jobs.read_json(os.path.join(late, 'curve1', 'run.json'))
        jobs.write_json(os.path.join(late, 'curve1', 'run.json'), dict(k, name='curve1', key=dict(k['key'], name='curve1')))
        r = subprocess.run(CLI + ['bound', '--dir', late, '--curves', L, '--max-hours', '0'], cwd=self.tmp, env=ENV,
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)          # stopped at the time limit
        self.assertIn('The time limit has already passed, so there is nothing to do.', r.stderr)
        self.assertIn('Stopped at the time limit. Run the same command again to resume.', r.stdout)
        self.assertNotIn('Witness', r.stdout)
        r = subprocess.run(CLI + ['status', D], cwd=self.tmp, env=ENV, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rows = [l.split() for l in r.stdout.splitlines()]
        self.assertEqual(rows[0], ['name', 'relations', 'core-hours', 'estimate', 'rank', 'bound', 'status'])
        self.assertEqual((len(rows), rows[1][0], rows[1][4:]), (2, 'curve1', ['0', 'finished', 'and', 'checked']))

    def test_relations(self):
        """The witness's relations fed back, gzip-compressed or as bare pairs (a, b) on the command line, prove the
        same bound without a sieve, and a second call answers from run.json.  Other relations in the same folder, a
        wrong relation: refused."""
        wit = self.res.witness
        gz = os.path.join(self.tmp, 'given.witness.json.gz')
        shutil.copy(wit, gz)                   # the prover's witness is gzipped
        with self.assertRaisesRegex(RankboundError, 'was started for another curve or other relations'):
            rankbound(E571, relations=gz, workdir=self.res.workdir)
        W = os.path.join(self.tmp, 'given')
        r = rankbound(E571, relations=gz, workdir=W)
        self.assertEqual((r.G, r.rank_bound, r.checked, r.details['outcome']), (self.res.G, 0, True, 'relations'))
        with open(os.path.join(W, 'params.json')) as f:
            self.assertEqual(json.load(f)['siever'], 'given')
        self.assertEqual(State(W).finished(), ['given'])
        self.assertEqual(rankbound(E571, relations=gz, workdir=W).details['earlier'], True)
        pairs = [l.split(':')[0] for l in vf.load(wit)['relations'] if not l.startswith(('f:', 'e:'))]
        with self.assertRaisesRegex(RankboundError, 'was started for another curve'):
            rankbound(E571, relations=pairs[1:], workdir=W)
        txt = os.path.join(self.tmp, 'pairs.txt')
        with open(txt, 'w') as f:
            f.write('# 571a1: (a, b) of a - b alpha\n' + '\n'.join(pairs) + '\n')
        p = subprocess.run(CLI + ['bound', ','.join(map(str, E571)), '--relations', txt, '--dir', 'pairs'],
                           cwd=self.tmp, env=ENV, capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('Rank at most 2. Verification method: 2-Selmer group (assuming the GRH).', p.stderr)
        self.assertRegex(p.stdout, r'Rank 0, assuming the GRH. Witness: pairs/%s/%s.witness.json.gz \(checked\)' % (N571, N571))
        a, b = map(int, pairs[0].split(','))
        with self.assertRaisesRegex(ValueError, '^The primes given for relation 1 '):
            rankbound(E571, relations=[(a, b, [2])], workdir=os.path.join(self.tmp, 'wrong'))

    def test_batch(self):
        """`rankbound bound --estimate` over a curve list: a curve that cannot be prepared (a composite in its
        discriminant) is skipped with a reason, the others are estimated; the exit status is then 1."""
        path = os.path.join(self.tmp, 'two.json')
        with open(path, 'w') as f:
            json.dump({'curves': [{'id': 1, 'ainvs': E571, 'bad_primes': ['571'], 'r': 0},
                                  {'id': 2, 'ainvs': E571, 'bad_primes': [], 'composites': ['3997'], 'r': 0}]}, f)
        r = subprocess.run(CLI + ['bound', '--estimate', '--dir', os.path.join(self.tmp, 'batch'), '--curves', path],
                           cwd=self.tmp, env=ENV, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('Curve curve2 failed. The composite 3997 does not divide the discriminant.', r.stdout)
        self.assertRegex(r.stdout, r'1 curves, estimated [0-9.]+ core-hours in all\. Skipped curve2\.')
        self.assertIn('curve1 has no rational 2-torsion. Sieving for relations should take ', r.stdout)


E37 = ['0', '0', '1', '-1', '0']                   # 37a1 (minimal), rank 1: P = (0, 0)
E37_SHORT = ['0', '0', '0', '-1296', '11664']        # a short Weierstrass model of it (u = 1/6): P = (0, +-108)


class TestModels(unittest.TestCase):
    """Witnesses on the reduced minimal model, whatever model the curve and its points are given on; 37a1's
    witness (rankbound(points=...), setUpClass) shared by the tests below."""
    MM_POINTS = ([['0', '0']], [['0', '-1']])        # (0, +-108) of the short model on the minimal one

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.r37 = rankbound(E37, points=[(0, 0)], workdir=os.path.join(cls.d, 'e37'), compute=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def test_points_of_another_model(self):
        """A state prepared from the minimal model, run from a list entry on the short model with its point (0, 108):
        the witness is on the minimal model, with the point mapped onto it and primes chosen there, and proves
        rank = 1.  The relations are the first witness.s, without a sieve."""
        st = os.path.join(self.d, 'list37')
        path = os.path.join(self.d, 'list.json')
        primes = [p for p in range(2, 100) if all(p % q for q in range(2, p))]
        with open(path, 'w') as f:
            json.dump({'curves': [{'id': 37, 'name': 'e37', 'ainvs': E37_SHORT, 'bad_primes': ['37'], 'r': 1,
                                   'proof': {'points': [['0', '108']], 'primes': primes}}]}, f)
        cur = cv.from_list(path)[0]
        self.assertEqual(cur.proof['ainvs'], E37_SHORT)           # the model the list's points are on
        with self.assertLogs('rankbound', 'INFO'):
            r = run_route(Job(st), cur, relations=self.r37.witness)
        self.assertEqual((r['checked'], r['rank_bound']), (True, 1))
        c = vf.load(r['witness'])
        self.assertEqual(c['curve']['ainvs'], E37)
        rl = c['rank_lower_bound']
        self.assertEqual(rl['r'], 1)
        self.assertIn(rl['points'], self.MM_POINTS)
        with open(os.path.join(st, 'check.json')) as f:
            self.assertIn({'check': 'rank_lower_bound', 'ok': True}, [{k: c[k] for k in ('check', 'ok')}
                                                                     for c in json.loads(f.read())['checks']])
        self.assertIn('rank E(Q) = 1', r['statement'])

    def test_points(self):
        """rankbound(points=...) proves rank = 1 for 37a1 with a prime it finds.  `rankbound bound --points
        --relations` on the short model gives a witness on the minimal model.  A point off the curve is refused before
        anything is sieved."""
        r, d = self.r37, self.d
        self.assertEqual((r.rank_bound, r.rank_lower_bound, r.proved_equal, r.checked, r.details['outcome']),
                         (1, 1, True, True, 'goal'))
        rl = vf.load(r.witness)['rank_lower_bound']
        self.assertEqual((rl['points'], len(rl['primes'])), ([['0', '0']], 1))
        pts = os.path.join(d, 'points.txt')
        with open(pts, 'w') as f:
            f.write('0 108\n')
        p = subprocess.run(CLI + ['bound', '--dir', os.path.join(d, 'short'), ','.join(E37_SHORT), '--points', pts,
                                  '--relations', r.witness], cwd=d, env=ENV, capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('Rank at least 1.', p.stderr)
        self.assertRegex(p.stdout, r'Rank 1, assuming the GRH. Witness: \S+ \(checked\)')
        c = vf.load(glob.glob(os.path.join(d, 'short', '*', '*.witness.json.gz'))[0])
        self.assertEqual(c['curve']['ainvs'], E37)
        self.assertIn(c['rank_lower_bound']['points'], self.MM_POINTS)
        self.assertIn('rank E(Q) = 1', c['statement'])
        W = os.path.join(d, 'off')
        with self.assertRaisesRegex(RankboundError, '^Point 1 is not on the curve.$'):
            rankbound(E37, points=[(0, 1)], workdir=W)
        self.assertFalse(os.path.exists(os.path.join(W, 'params.json')))      # no trial sieve, no sieve

    def test_no_elements(self):
        """With r given and S_bk = r (37a1: G = 0, S_bk = 1 = r), the 2-descent's elements are skipped (selmer.SKIP): a checked
        format-4 witness with elements [], v = 0, no ops_s, no characters or local data, S = min(U, S_bk) = S_bk;
        a claim v = 1, a lower S, or local data off S fails.  With SKIP off the same bound comes with elements."""
        from rankbound import selmer
        c = vf.load(self.r37.witness)
        cl = c['claims']
        self.assertEqual((c['elements'], c['characters'], cl['v'], cl['dim_sel_cap_V'], cl['S'], cl['S_bk'], cl['k']),
                         ([], {'real': [], 'primes': []}, 0, 0, 1, 1, cl['G']))
        self.assertNotIn('ops_s', c)
        self.assertEqual((c['local']['inf'], c['local']['p']), ([], {}))
        self.assertEqual((c['s_primes'], c['s_dropped']), (['2'], [[37, 1]]))      # 37a1: c_37 = 1, S = {2}
        for what in ('v', 'S', 'local'):
            b = json.loads(json.dumps(c))
            if what == 'v':
                b['claims']['v'] = 1
            elif what == 'S':
                b['claims']['S'] = 0
            else:
                b['local']['p']['5'] = {'points': [], 'prec': 1}
            t = os.path.join(self.d, 'noel_%s.witness.json' % what)
            with open(t, 'w') as f:
                json.dump(b, f)
            self.assertNotEqual(vf.check(t)['verdict'], 'accepted', what)
        selmer.SKIP = False
        try:
            r = rankbound(E37, points=[(0, 0)], workdir=os.path.join(self.d, 'e37full'), compute=True)
        finally:
            selmer.SKIP = True
        f = vf.load(r.witness)
        self.assertEqual((r.rank_bound, r.checked, f['claims']['v'] > 0), (1, True, True))

    def test_cited_r(self):
        """r given without points is cited: S = r = 1 proves rank <= 1 only, so r_proved and proved_equal are False and
        the statement says r is cited (with points both are True) (here on the first witness's relations, no
        sieve)."""
        self.assertEqual((self.r37.r_proved, self.r37.proved_equal), (True, True))
        r = rankbound(E37, r=1, relations=self.r37.witness, workdir=os.path.join(self.d, 'cited'))
        self.assertEqual((r.rank_bound, r.rank_lower_bound, r.r_proved, r.proved_equal, r.checked),
                         (1, 1, False, False, True))
        self.assertIn('(cited; not proved in this witness)', r.statement)

    def test_refused_proof(self):
        """A proof that Job.prepare refuses leaves the folder's curve.json as it was.  When K is a cyclic cubic field (x^3 - 3x + 1: d_K = 81), no prime has
        #E(F_p) = 2 mod 4: r = 1 is proved by a point of infinite order (the checker's ellorder), r >= 2 refused with that reason."""
        p = os.path.join(self.r37.workdir, 'curve.json')
        with open(p) as f:
            before = f.read()
        with self.assertRaisesRegex(RankboundError, '^Point 1 is not on the curve.$'):
            Job(os.path.join(self.d, 'off2')).prepare(cv.with_points(cv.from_ainvs(E37), [(0, 1)]))
        with open(p) as f:
            self.assertEqual(f.read(), before)
        r = rankbound([0, 0, 0, -3, 1], points=[(0, 1)], workdir=os.path.join(self.d, 'cyclic'))
        self.assertEqual((r.rank_bound, r.rank_lower_bound, r.r_proved, r.checked), (1, 1, True, True))
        from fractions import Fraction
        P = (Fraction(0), Fraction(1))
        two = [(0, 1), tuple(str(x) for x in cv.mul([0, 0, 0, -3, 1], P, 2))]
        cyclic = r'^rankbound cannot prove the rank of this curve from points because its cubic field is cyclic. Give the rank with --r instead. It will be cited, not proved.$'
        with self.assertRaisesRegex(RankboundError, cyclic):
            rankbound([0, 0, 0, -3, 1], points=two, workdir=os.path.join(self.d, 'cyclic2'))
        # the same on SPLIT (--uncond), before the 2-descent and before curve.json (it once failed after the 2-descent, "look dependent")
        w = os.path.join(self.d, 'cyclic_split')
        with self.assertRaisesRegex(RankboundError, cyclic):
            rankbound([0, 0, 0, -3, 1], points=two, workdir=w, uncond=True)
        self.assertFalse(os.path.exists(os.path.join(w, 'curve.json')))
        r = rankbound([0, 0, 0, -3, 1], r=1, compute=True, workdir=os.path.join(self.d, 'cyclic_r'))     # r cited: fine
        self.assertEqual((r.rank_bound, r.rank_lower_bound, r.r_proved, r.checked), (1, 1, False, True))


@unittest.skipUnless(os.environ.get('RANKBOUND_TEST_FULL'), 'a whole run on a field of the table: RANKBOUND_TEST_FULL')
class TestIcarm1(unittest.TestCase):
    def test_icarm1(self):
        """A whole run on a field in the table: ICARM #1, G = 10, rank <= 12 = r, checked, with two sieve processes."""
        d = tempfile.mkdtemp()
        try:
            J = Job(os.path.join(d, 'icarm1'))
            with self.assertLogs('rankbound', 'INFO'):
                r = run_route(J, cv.from_ainvs(ICARM1['ainvs'], ICARM1['bad_primes'], ICARM1['r'], 'icarm1'), cores=2)
            self.assertEqual((r['outcome'], r['G'], r['rank_bound'], r['checked']), ('goal', 10, 12, True))
            with open(os.path.join(J.dir, 'params.json')) as f:
                prm = json.load(f)
            self.assertEqual(prm['lpb0'], 14)
        finally:
            shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
