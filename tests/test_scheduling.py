"""The sieve's scheduling with many cores: slices sized by the cores in use, shards that stop near a check's relation
count, a run that checks what its folder already has before it sieves, the checker's time in the request's spending,
and the estimates of a check's time."""
import io, json, os, shutil, sys, tempfile, textwrap, threading, time, unittest
from unittest import mock
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rankbound import cli, runner, runtime, selmer, sieve as jobs  # noqa: E402
from rankbound.core import UNCONDITIONAL, Request, Resources  # noqa: E402
from rankbound.sieve import State  # noqa: E402
from test_lifecycle import CHECKER, FakeRoute  # noqa: E402
from test_stopping import E571, N571  # noqa: E402

# #846's parameters as its trial chose them on an EPYC 7H12 (lpb 22, 2^23 positions per special-q)
PRM846 = {'lpb0': 22, 'mfb0': 22, 'lim0': 2 ** 22, 'N': 23, 'q0': 2 ** 21,
          'polys': {'F': {'weight': 1.0, 'sec_per_q': 0.02525}},
          'model': {'kappa': 0.91, 'dup': 0.9, 'relations_needed': 219966, 'rate': 98.9, 'sec_per_q': 0.02525,
                    'rels_per_q': 2.775, 'core_hours': 0.6177}}


def tmpdir(test):
    d = tempfile.mkdtemp(prefix='rb-schedule-')
    test.addCleanup(shutil.rmtree, d, True)
    return d


def in_flight(prm, c):
    """The relations and CPU seconds of the first c slices of prm's schedule (the slices of c processes at once), by
    the model: its special-q are the width over log q0."""
    m, rels, cpu = prm['model'], 0.0, 0.0
    for j, poly, q0, q1 in jobs.schedule(prm):
        if j >= c:
            break
        nq = (q1 - q0) / jobs.math.log(prm['q0'])
        rels += nq * m['rels_per_q']
        cpu += nq * m['sec_per_q']
    return rels, cpu


class TestSliceSize(unittest.TestCase):
    def test_slices_in_flight(self):
        """With c processes (params' slice_cores) the c slices in flight at once hold at most SLICE_FLIGHT of the
        relations needed and of the estimated sieve time (before: 64 slices of #846 held 2.4x its relations, and 64
        cores sieved 514,284 relations for a check of 251,861); one process keeps its slices."""
        m = PRM846['model']
        for c in (64, 126, 1008):
            rels, cpu = in_flight(dict(PRM846, slice_cores=c), c)
            self.assertLessEqual(rels, 1.05 * jobs.SLICE_FLIGHT * m['relations_needed'], c)
            self.assertLessEqual(cpu, 1.05 * jobs.SLICE_FLIGHT * m['core_hours'] * 3600, c)
            self.assertGreater(rels, 0.5 * jobs.SLICE_FLIGHT * m['relations_needed'], c)      # not needlessly small
        self.assertEqual(list(jobs.schedule(dict(PRM846, slice_cores=1)))[:50], list(jobs.schedule(PRM846))[:50])

    def test_sized_once(self):
        """size_slices records the processes in params.json once, before the first slice: a later run or shard with
        other cores keeps the same schedule; a folder with the schedule's first slice sieved keeps its slices."""
        d = tmpdir(self)
        S = State(d)
        S.save_params(dict(PRM846))
        jobs.size_slices(d, 2 * 63)
        self.assertEqual(S.params()['slice_cores'], 126)
        first = list(jobs.schedule(S.params()))[:20]
        jobs.size_slices(d, 1)
        self.assertEqual(list(jobs.schedule(S.params()))[:20], first)
        e = tmpdir(self)
        T = State(e)
        T.save_params(dict(PRM846))
        jobs.finish_slice(T.sdir, jobs.slice_name('F', PRM846['q0']), {'poly': 'F'}, iter(['3,1:2\n']), T.fp)
        jobs.size_slices(e, 64)
        self.assertNotIn('slice_cores', T.params())


class TestShardStop(unittest.TestCase):
    def test_found_in_flight(self):
        """A slice in progress counts the relation lines its siever has sent (Slice.found), not its comments."""
        script = textwrap.dedent('''
            import sys
            for line in sys.stdin:
                print('1,2:3'); print('# a comment'); print('4,5:6'); print('# done', flush=True)
            ''')
        wake = threading.Event()
        sv = jobs.Server([sys.executable, '-c', script], 'F', wake)
        self.addCleanup(sv.close)
        s = sv.sieve(1, 2)
        t = time.monotonic()
        while s.poll() is None and time.monotonic() - t < 30:
            wake.wait(0.1)
        self.assertEqual((s.poll(), s.found), (0, 2))

    def test_done_with_slices_in_flight(self):
        """A shard is done once the folder's finished slices and the slices in progress (its own, as many again for
        each other shard running) reach the next check's relations: it then ends, and its slices in progress keep what
        they have (save_slices).  Before, a shard counted finished slices only: with 882 cores the slices in flight
        alone held 7 times the first check of #6782, and the shards sieved 2.55x its relations."""
        d = tmpdir(self)
        S = State(d)
        for k in range(2):
            jobs.finish_slice(S.sdir, 'F_%d' % k, {'poly': 'F'}, iter(['%d,%d:2\n' % (a, k + 1) for a in range(10)]),
                              S.fp)
        fly = [type('Slice', (), {'found': 15})()]
        self.assertFalse(jobs.shard_done(S, 0, 2, 50, jobs.Count(S), fly))      # 20 + 15
        fd = S.lock('shard1_2.lock')                                           # shard 2 is running
        try:
            self.assertTrue(jobs.shard_done(S, 0, 2, 50, jobs.Count(S), fly))   # 20 + 2 x 15
        finally:
            os.close(fd)
        self.assertTrue(jobs.shard_done(S, 0, 2, 20, jobs.Count(S)))

    def test_done_on_distinct_relations(self):
        """A shard stops on distinct relations, as the next check counts them (linalg.seen_key), not on the slices'
        lines: a relation found again at another special-q counts once, and the slices in progress count at the
        share of new relations in the slices read last (Count.rate), or the model's DUP before any.  Before, 2 shards
        of #846 stopped at 254k lines holding 165k distinct relations, and the `bound` after them purged twice and
        sieved on from 165k to 259k."""
        d = tmpdir(self)
        S = State(d)
        jobs.finish_slice(S.sdir, 'F_1', {'poly': 'F'}, iter(['%d,1:2\n' % a for a in range(100)]), S.fp)
        c = jobs.Count(S)
        self.assertEqual((c.total(), c.rate()), (100, 1.0))
        jobs.finish_slice(S.sdir, 'F_2', {'poly': 'F'}, iter(['%d,1:2\n' % a for a in range(50, 150)]), S.fp)
        self.assertEqual((c.total(), c.rate()), (150, 0.5))         # 200 lines, 150 distinct; the last slice: 50 new
        self.assertFalse(jobs.shard_done(S, 0, 2, 160, jobs.Count(S)))              # 200 lines, but 150 relations
        fly = [type('Slice', (), {'found': 30})()]
        self.assertTrue(jobs.shard_done(S, 0, 2, 165, jobs.Count(S), fly))          # 150 + 0.5 x 30
        self.assertFalse(jobs.shard_done(S, 0, 2, 166, jobs.Count(S), fly))
        e = jobs.Count(State(tmpdir(self)))
        self.assertEqual((e.total(), e.rate()), (0, jobs.DUP))      # nothing read: the model's rate


class TestShardBudget(unittest.TestCase):
    def test_shard_sets_the_budget(self):
        """A folder made only by shards has the request budget, and its shards stop at it: the first run with an
        estimate fixes it, a shard or not (RunStore.set_budget, under key.lock), at 3x the estimate (at least 1
        core-hour) or --budget; a later shard keeps it, a larger --budget raises it.  Before, only a run that proves
        set it."""
        d = tmpdir(self)
        seen = []
        est = [0.5]
        with mock.patch.object(selmer.Job, 'estimate', lambda *a, **k: est[0]), \
                mock.patch.object(jobs, 'size_slices', lambda *a: None), \
                mock.patch.object(jobs, 'sieve_run', lambda *a, **k: seen.append(k['meter'].budget)):
            selmer.Job(d).sieve_shard(None, shard=(1, 2))
            est[0] = 0.7
            selmer.Job(d).sieve_shard(None, shard=(2, 2))
            selmer.Job(d).sieve_shard(None, shard=(1, 2), budget_core_hours=4.0)
        self.assertEqual(seen, [1.5, 1.5, 4.0])
        self.assertEqual(runtime.RunStore(d).info()['budget_core_hours'], 4.0)


class TestCheckFirst(unittest.TestCase):
    def test_load_order_with_a_gap(self):
        """The slices finished when a run starts come first in its load order, a gap before them (a slice another
        shard did not sieve, or one lost to a kill) after them."""
        d = tmpdir(self)
        S = State(d)
        prm = {'polys': {'F': {'weight': 1.0}}, 'model': {'core_hours': 0.0, 'sec_per_q': jobs.SLICE_MIN / 10},
               'q0': 1024, 'lpb0': 11}
        names = [jobs.slice_name(p, q0) for _, p, q0, _ in jobs.schedule(prm)]
        for n in names[1:4]:
            jobs.finish_slice(S.sdir, n, {'poly': 'F'}, iter(['3,1:2\n']), S.fp)
        first = frozenset(S.finished())
        self.assertEqual(list(jobs.load_order(S, prm, first))[:5], names[1:4] + [names[0]] + names[4:5])
        self.assertEqual(list(jobs.load_order(S, prm)), names)

    def test_bound_after_one_shard(self):
        """A `bound` on a folder that already has the relations of its next check, with a gap in them (here shard 1
        of 2 never ran): it checks them first and sieves nothing."""
        w = tmpdir(self)
        E = ','.join(map(str, E571))
        with self.assertLogs('rankbound', 'INFO'), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(['bound', E, '--shard', '2/2', '--dir', w, '--compute']), 0)
        S = State(os.path.join(w, N571))
        names = S.finished()
        self.assertNotIn(jobs.slice_name('F', S.params()['q0']), names)       # the gap: shard 1's first slice
        out = io.StringIO()
        with self.assertLogs('rankbound', 'INFO') as lg, redirect_stdout(out):
            self.assertEqual(cli.main(['bound', E, '--r', '0', '--dir', w, '--compute', '--json']), 0)
        res = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual(res['rank_bound'], 0)
        self.assertEqual(S.finished(), names, lg.output)                     # nothing sieved
        self.assertTrue(any(l.startswith('INFO:rankbound:The work folder already has the ') for l in lg.output),
                        lg.output)


BURN = 'import time\nt = time.process_time()\nwhile time.process_time() - t < %s: pass\n'


class TestCheckerSpent(unittest.TestCase):
    def setUp(self):
        self.d = tmpdir(self)
        self.work = os.path.join(self.d, 'work')
        self.mode = os.path.join(self.d, 'mode')
        os.environ['FAKE_MODE'] = self.mode
        self.addCleanup(os.environ.pop, 'FAKE_MODE', None)

    def run_(self, mode, burn):
        with open(self.mode, 'w') as f:
            f.write(mode)
        ck = os.path.join(self.d, 'checker%s.py' % burn)
        with open(ck, 'w') as f:
            f.write(BURN % burn + CHECKER)
        req = Request(source=('0', '0', '0', '-1', '0'), name='E1', output=self.work, hypothesis=UNCONDITIONAL)
        return runner.execute(None, req, Resources(), runner.DESCENT, FakeRoute(), checker=[sys.executable, ck])

    def test_check_only_run(self):
        """The checker's CPU time (a child process) goes to run.json's spent, also in a run that only checks the
        witness left by the last one."""
        with self.assertRaises(RuntimeError):
            self.run_('crash', 0)                       # the witness is written, its check failed
        st = runtime.RunStore(self.work)
        before = st.spent()
        self.assertEqual(before['runs'], 1)
        res = self.run_('accept', 1.5)                  # only the check, 1.5 CPU seconds
        self.assertEqual((res.checked, res.origin), (True, 'resumed'))
        after = st.spent()
        self.assertGreaterEqual(after['core_hours'] - before['core_hours'], 1.4 / 3600)
        self.assertEqual(after['runs'], 2)

    def test_counted_once(self):
        """The runner's Meter and a route's (selmer.Job.prove, a shard) are one per folder: nested, the same one, and
        the run is counted once."""
        st = runtime.RunStore(tmpdir(self))
        with runtime.metered(st) as a:
            with runtime.metered(st) as b:
                self.assertIs(a, b)
            self.assertFalse(a.closed)
        self.assertTrue(a.closed)
        self.assertEqual(st.spent()['runs'], 1)


class TestEstimate(unittest.TestCase):
    def test_846(self):
        """#846's estimate is near the CPU its sieve took at 64 cores (1.23 core-hours to its check, slices sized by
        the cores; before, 0.62 was estimated), and the default request budget, max(3 x the estimate, 1 core-hour),
        is twice what the whole run spent, the checker included (1.60), so it never relies on the run's first check."""
        h = jobs.sieve_estimate(PRM846)
        self.assertTrue(0.75 * 1.227 <= h <= 1.5 * 1.227, h)
        st = runtime.RunStore(tmpdir(self))
        self.assertGreaterEqual(st.set_budget(h), 2 * 1.598)


class TestCheckNeed(unittest.TestCase):
    def test_late_solve_redone(self):
        """A late solve redone at the same relations does not count with the one that ended (#6782's next check was
        put at 2.2 hours: 137 + 2,592 + 2,825 s at 126 threads, scaled; the one that ended took 2,825 s), and an
        untrimmed solve given up as too dense is, alone, no measure (it was counted as the check: 4.8 core-hours
        against 90.7)."""
        bail = {'relations': 5691576, 'excess': 577778, 'keep': -1, 'dense_MB': 107691.0, 'dense_s': 3420.0,
                'solve_s': 137.3, 'threads': 126}
        late = {'relations': 5691576, 'excess': 577778, 'keep': 27059, 'late': True, 'need_s': 0, 'solve_s': 2592.3,
                'threads': 126}
        ended = {'relations': 5691576, 'excess': 577778, 'keep': 27059, 'bound': 30, 'solve_s': 2825.0, 'threads': 126}
        N = jobs.check_need
        self.assertIsNone(N([bail], 5691576, 126))
        self.assertAlmostEqual(N([bail, late], 5691576, 126), 2592.3)
        self.assertAlmostEqual(N([bail, late, ended], 5691576, 126), 137.3 + 2825.0)
        self.assertAlmostEqual(N([bail, late, ended], 1.2 * 5691576, 126), (137.3 + 2825.0) * 1.44)


if __name__ == '__main__':
    unittest.main()
