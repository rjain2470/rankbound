"""The sieve's stopping rules (sieve.py), the request budget (runtime.RunStore, Meter), the one Budget in the sieve,
linalg and arith, the artifacts of the sieve's folder (runtime.transaction).

- exact stops: the goal, 'exact' (G = lb), the norm-bound target (stop_target's norm, the route's: a stand-in here);
- the give-up rule on cost: C >= 2 C_imp, 3 C_imp while G is above lb or the target; five benchmark curves reach
  G = lb under it;
- no sieving past a pending check; a check that never fits ends within two runs with 'too_long' (the #6782 loop);
- the request budget across three resumes ends with 'budget'; a resume does not reset it;
- anytime: a final stop certifies the best solve it has, or raises a final Stopped (outcome, reason) without one;
- the old paths are gone (check_plan, PLATEAU, Folder.check/finish, Job.certify)."""
import inspect, itertools, os, shutil, tempfile, time, unittest
from rankbound import arith, linalg, runtime, sieve
from rankbound import check as vf
from rankbound import Stopped, rankbound
from rankbound.messages import text
from rankbound.selmer import Job
from rankbound.sieve import State

jobs = sieve
E571 = [0, -1, 1, -929, -10595]
N571 = '0_-1_1_-929_-10595'
PRM = {'polys': {'F': {'weight': 1.0, 'shift': 0}}, 'model': {'core_hours': 0.0, 'sec_per_q': jobs.SLICE_MIN / 10},
       'q0': 1024, 'lpb0': 11}          # a schedule of 15 slices (test_sieve's TestShardEnd.PRM)
R = 20                                   # relations per fake slice


class FakeRun:
    """sieve_run in this process with a stand-in siever (a slice: R relations of its own, after `delay` seconds) and a
    stand-in Solver whose checks answer what `answer(V, dry)` says (default: None, a plateau when dry); checks records
    (relations, dry)."""

    def setUp(self):
        self.checks, self.nxt, self.delay = [], 10 ** 9, 0.0
        self.answer = lambda V, dry: 'plateau' if dry else None
        test = self

        serial = itertools.count(1)

        class Slice:
            pid, found = os.getpid(), 0
            def __init__(self):
                self.t, self.b = time.monotonic(), next(serial)
            def poll(self):
                return 0 if time.monotonic() - self.t >= test.delay else None
            def lines(self):
                return iter(['%d,%d:2,3:\n' % (a, self.b) for a in range(R)] + ['# Total cpu time 0.01s\n'])
            def release(self):
                pass
            kill = stop = release
            def end(self, q1):
                return q1

        class Siever:
            def start_slice(self, state, prm, poly, q0, q1):
                return Slice()
            def wait(self):
                time.sleep(0.01)
            def close(self):
                pass

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
                self.nrel = R * len(self.order)
            def check(self, work, dry=False):
                test.checks.append((self.nrel, dry))
                self.announce()
                b = self.budget
                self.had = b.left() if b is not None else None
                return test.answer(self, dry)

        real = (jobs.open_state, jobs.Solver, jobs.siever, jobs.next_check)
        self.addCleanup(lambda: setattr(jobs, 'open_state', real[0]) or setattr(jobs, 'Solver', real[1]) or
                        setattr(jobs, 'siever', real[2]) or setattr(jobs, 'next_check', real[3]))
        jobs.open_state = lambda S: ({}, PRM)
        jobs.next_check = lambda h, prm: test.nxt
        jobs.Solver, jobs.siever = V, lambda prm: Siever()
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.S = State(self.d)


class TestGiveUp(unittest.TestCase):
    def test_cost_rule(self):
        """C >= GIVE_UP C_imp gives up (C_imp: the cost when the best G was first reached); above lb or the target,
        GIVE_UP_ABOVE C_imp; an improvement moves C_imp; costs, not relations, once every full solve has one."""
        e = lambda n, G, c, **kw: dict(relations=n, bound=G, cost_ch=c, **kw)
        h = [e(100, 5, 1.0), e(5000, 5, 1.99)]                  # relations do not count: cost 1.99 < 2
        self.assertIsNone(jobs.decide(h, None))
        self.assertEqual(jobs.decide(h + [e(5100, 5, 2.0)], None), 'plateau')
        self.assertIsNone(jobs.decide(h + [e(5100, 5, 2.0, lb=4)], None))           # G > lb: 3x
        self.assertEqual(jobs.decide(h + [e(5100, 5, 3.0, lb=4)], None), 'plateau')
        self.assertIsNone(jobs.decide(h + [e(5100, 5, 2.5)], 4))                     # the target 4 not reached: 3x
        self.assertEqual(jobs.decide(h + [e(5100, 4, 2.5)], 4), 'goal')
        g = h + [e(6000, 4, 2.5), e(7000, 4, 4.9)]               # improved at 2.5: gives up at 5.0
        self.assertIsNone(jobs.decide(g, None))
        self.assertEqual(jobs.decide(g + [e(8000, 4, 5.0)], None), 'plateau')
        self.assertIsNone(jobs.decide([e(100, 5, 0.0), e(150, 5, 0.0)], None))       # nothing measured: relations
        self.assertIsNone(jobs.decide([e(100, 5, 1.0)], None))                      # one solve: never

    def test_reach_lb(self):
        """Five benchmark curves: G = lb + 1 from the plateau's start, lb reached at these relations; a
        1.3x rule would stop all five at lb + 1, the give-up rule (cost ~ relations) waits for every one of them."""
        for name, G, start, reach in (('94', 11, 7886, 21142), ('381', 16, 21893, 34879), ('1980', 16, 15308, 22782),
                                      ('4205', 18, 32616, 63066), ('4233', 17, 27079, 39989)):
            h = [dict(relations=start, bound=G, lb=G - 1, cost_ch=start / 1e4)]
            n = start
            while n * jobs.NEXT < reach:
                n = n * jobs.NEXT
                h.append(dict(relations=round(n), bound=G, lb=G - 1, cost_ch=n / 1e4))
                self.assertIsNone(jobs.decide(h, None), (name, n))
            self.assertGreaterEqual(1.3 * start, h[1]['relations'] if len(h) > 1 else reach, name)  # where a 1.3x rule stops

    def test_record_cost(self):
        """Solver._record: each entry's check_ch (solve_s x threads) and cost_ch (the sieve's core-hours so far and
        every solve's): the request's cost of sieving and checks together, which the give-up rule measures."""
        from helpers import bare_solver
        with tempfile.TemporaryDirectory() as d:
            V = bare_solver(State(d), threads=4)
            V.sieve_ch = 2.0
            V.nrel = 100
            V._record(linalg.SolveResult(bound=5, dimV=0, rank=0, pivots=0, excess=1, ops='', piv=''), time.time(),
                      {'solve_s': 900.0, 'bound': 5})
            V.sieve_ch = 3.0
            V._record(linalg.SolveResult(bound=5, dimV=0, rank=0, pivots=0, excess=1, ops='', piv=''), time.time(),
                      {'solve_s': 1800.0, 'bound': 5})
            self.assertEqual([(x['check_ch'], x['cost_ch'], x['threads']) for x in V.hist], [(1.0, 3.0, 4), (2.0, 6.0, 4)])
            self.assertEqual(State(d).history(), V.hist)


class TestNormTarget(unittest.TestCase):
    def test_norm(self):
        """The stop target is the larger of the goal (where goal_stop allows it) and the norm-bound target norm
        (selmer.norm_target, which the route passes); without one (no r) the goal."""
        self.assertEqual((jobs.stop_target(2, True), jobs.stop_target(2, False)), (2, None))
        self.assertEqual((jobs.stop_target(2, True, 7), jobs.stop_target(9, True, 7), jobs.stop_target(2, False, 7)),
                         (7, 9, 7))
        self.assertEqual((jobs.stop_target(2, True, -1), jobs.stop_target(2, False, -1)), (2, None))
        h = [dict(relations=100, bound=7, cost_ch=1.0)]
        self.assertEqual(jobs.decide(h, 7), 'goal')                 # G <= the norm target: an exact stop


class TestLoop(FakeRun, unittest.TestCase):
    def test_never_fits(self):
        """The #6782 loop (a late check, sieving on past it, a larger check next time) cannot happen: a check that
        never fits (f2elim late, its need unknown) is the next run's only work, on the same slices, and that run,
        which has the whole window for it, ends with 'too_long'; a third run reports it again without a check."""
        self.nxt = 30
        self.answer = lambda V, dry: 'late'
        runs = []
        with self.assertLogs('rankbound', 'INFO'):
            runs.append(jobs.sieve_run(self.d, jobs=2, budget=runtime.Budget(seconds=360), window=1000)[0])
        n = len(self.S.finished())
        pend = self.S.pending()
        self.assertEqual((runs, self.checks), ([None], []))                 # not tried: it would not end
        with self.assertLogs('rankbound', 'INFO'):
            runs.append(jobs.sieve_run(self.d, jobs=2, budget=runtime.Budget(seconds=1000))[0])
        self.assertEqual(runs, [None, 'too_long'])
        self.assertEqual(self.checks, [(pend['relations'], False)])           # the same relations
        self.assertEqual(len(self.S.finished()), n)                         # no sieving meanwhile
        st = self.S.stop_record()
        self.assertEqual(st['need_s'], 0)                                    # f2elim did not say: the window
        self.assertAlmostEqual(st['window'], 1000, delta=10)
        self.assertEqual(jobs.too_long_text(None, st['window'], st['threads']),
                         'The last check did not finish in the 17 minutes it had on 2 cores. Give it more cores or more '
                         'time.')
        self.assertEqual(jobs.sieve_run(self.d, jobs=2, budget=runtime.Budget(seconds=1000))[0], 'too_long')
        self.assertEqual(len(self.checks), 1)

    def test_stopped_in_fresh_check(self):
        """A check stopped by the run's end (its program stopped) in a run that had the whole window for it: STOP
        'too_long', and without a solve to certify, a final Stopped (outcome 'too_long') at once (no third run)."""
        self.nxt = 30
        def stop(V, dry):
            raise Stopped(self.d)
        self.answer = stop
        with self.assertRaises(Stopped) as cm:
            jobs.sieve_run(self.d, jobs=1, budget=runtime.Budget(seconds=600))
        self.assertEqual((cm.exception.outcome, self.S.stop_outcome()), ('too_long', 'too_long'))
        self.assertIsNotNone(self.S.pending())


    def test_due_at_start(self):
        """A resumed run whose finished slices already hold the next check's relations: that check is pending from
        the start (checking.json written before loading), the run's only work before it sieves; late, it is
        'too_long' at once (it had the whole run)."""
        for k in range(3):
            jobs.finish_slice(self.S.sdir, jobs.slice_name('F', 1024 + 100 * k), {'poly': 'F'},
                              iter(['%d,1:2,3:\n' % a for a in range(R)]), self.S.fp)
        self.nxt = 30
        self.answer = lambda V, dry: 'late'
        with self.assertLogs('rankbound', 'INFO') as out:
            self.assertEqual(jobs.sieve_run(self.d, jobs=1, budget=runtime.Budget(seconds=600))[0], 'too_long')
        self.assertEqual(self.checks, [(40, False)])                        # the prefix to 30 relations: 2 slices
        self.assertEqual(len(self.S.finished()), 3)                         # nothing sieved
        self.assertIn('INFO:rankbound:' + text('due_check', n=40), out.output)

    def test_load_stopped(self):
        """A pending check whose loading reaches the run's end: 'too_long' (it had the whole run), a final Stopped without a
        solve; no third run."""
        self.S.begin_check(['F_1024'], 20)
        real = jobs.Solver.load
        def load(V, names=None, upto=None):
            raise Stopped(self.d)
        jobs.Solver.load = load
        self.addCleanup(setattr, jobs.Solver, 'load', real)
        with self.assertRaises(Stopped):
            jobs.sieve_run(self.d, jobs=1, budget=runtime.Budget(seconds=600))
        self.assertEqual(self.S.stop_outcome(), 'too_long')


class TestBudget(FakeRun, unittest.TestCase):
    def test_set_budget(self):
        """budget_core_hours: set at the first run with an estimate, 3x it (BUDGET_FACTOR) or the given one; a resume
        does not change it, except to a larger given one (a smaller one is reported); no estimate and none given: no cap."""
        with tempfile.TemporaryDirectory() as d:
            st = runtime.RunStore(d)
            st.begin({'k': 1})
            self.assertIsNone(st.set_budget(0.0))
            self.assertEqual(st.set_budget(2.0), 6.0)
            self.assertEqual(st.set_budget(5.0), 6.0)
            with self.assertLogs('rankbound', 'INFO') as lg:        # a smaller one is not used, and says so
                self.assertEqual(st.set_budget(2.0, 4.0), 6.0)
            self.assertEqual(lg.output, ['INFO:rankbound:' + text('budget_kept', budget='6.0 core-hours',
                                                                  given='4.0 core-hours')])
            self.assertEqual(st.set_budget(2.0, 10.0), 10.0)
            self.assertEqual(st.info()['budget_core_hours'], 10.0)
        with tempfile.TemporaryDirectory() as d:     # a small curve's estimate: the floor, not 3x a few seconds
            st = runtime.RunStore(d)
            st.begin({'k': 1})
            self.assertEqual(st.set_budget(1e-4), runtime.BUDGET_FLOOR)

    def test_meter_writes(self):
        """run.json's spent is written while a run goes on (every Meter.EVERY seconds, from a thread: a long check
        or the witness included) and at its end, which counts the run."""
        st = runtime.RunStore(self.d)
        st.begin({'k': 1})
        old = runtime.Meter.EVERY
        runtime.Meter.EVERY = 0.1
        self.addCleanup(setattr, runtime.Meter, 'EVERY', old)
        m = runtime.Meter(st, cpu=lambda: time.monotonic() * 3600)     # a CPU clock of 1 core-hour a second
        time.sleep(0.35)
        mid = st.spent()
        m.close()
        end = st.spent()
        self.assertTrue(0.2 < mid['core_hours'] < end['core_hours'] < 1.0 and (mid['runs'], end['runs']) == (0, 1),
                        (mid, end))

    def test_three_resumes(self):
        """The request budget spent across three resumes ends the third with 'budget' (STOP, with the budget): every
        run's spending is in run.json (spent), and a resume does not reset it."""
        st = runtime.RunStore(self.d)
        st.begin({'k': 1})
        cpu = lambda: time.monotonic() * 36000      # a CPU clock of 10 core-hours a second
        st.set_budget(25.0 / 3)             # 3x the estimate: 25 core-hours, ~2.5 s, so two runs of 1 s leave room
        self.delay, self.nxt = 0.15, R      # a check after every slice: the budget ends a run after its first one
        self.S.save_history([{'relations': R, 'rows': 1, 'excess': 1, 'bound': 5, 'solve_s': 1e-4}])   # quick checks
        outs = []
        for k in range(3):
            meter = runtime.Meter(st, cpu=cpu)
            try:
                with self.assertLogs('rankbound', 'INFO') as out:
                    outs.append(jobs.sieve_run(self.d, jobs=1, budget=runtime.Budget(seconds=1.0), meter=meter)[0])
            finally:
                meter.close()
        self.assertEqual(outs, [None, None, 'budget'])
        self.assertEqual((self.S.stop_outcome(), self.S.stop_record()['budget_core_hours']), ('budget', 25.0))
        self.assertGreaterEqual(self.S.stop_record()['spent_core_hours'], 25.0)
        self.assertFalse(any('budget' in l for l in out.output), out.output)    # the reason is the runner's to show
        sp = st.spent()
        self.assertEqual(sp['runs'], 3)
        self.assertGreaterEqual(sp['core_hours'], 25.0)
        self.assertLess(sp['core_hours'], 35.0)


class TestEndToEnd(unittest.TestCase):
    """One run of 571a1 (setUpClass): the anytime certification and the final stops on its folder."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(); cwd = os.getcwd()
        try:
            os.chdir(cls.tmp)
            cls.res = rankbound(E571, cores=1, compute=True)
        finally:
            os.chdir(cwd)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def copy(self, name, drop=()):
        st = os.path.join(self.tmp, name, N571)
        shutil.copytree(self.res.workdir, st, ignore=lambda d, fs: [f for f in fs if f in drop or f.endswith(
            ('.witness.json', '.witness.json.gz', '.lock')) or f in ('selmer.json', 'selmer.json.gz', 'selmer.manifest.json', 'check.json')])
        return st

    def test_children(self):
        """Every program of the run (the sievers, f2elim, f2kernel) was reaped: none left in runtime's registry."""
        self.assertEqual(runtime._CHILDREN, set())

    def test_artifacts(self):
        """The run's folder: params, history and every slice with a valid manifest; la/best valid."""
        S = State(self.res.workdir)
        for name in ('params', 'history'):
            self.assertEqual(runtime.inspect(S.path, name, None, S.fp)[0], 'valid', name)
        self.assertTrue(S.finished() and all(runtime.inspect(S.sdir, n, jobs.SLICE_SCHEMA, S.fp)[0] == 'valid'
                                             for n in S.finished()))
        self.assertIsNotNone(S.best(os.path.join(S.path, 'la'))[0])
        h = S.history()
        self.assertTrue(all(e.get('cost_ch') is not None and e.get('threads') == 1 for e in h), h)

    def test_damaged_artifacts(self):
        """A damaged history.json, params.json or checking.json is not reused (set aside: rebuilt)."""
        B = self.copy('damaged')
        S = State(B)
        S.begin_check(S.finished(), 5)
        for f in ('history.json', 'params.json', 'checking.json'):
            with open(os.path.join(B, f), 'a') as g:
                g.write(' ')
        self.assertEqual((S.history(), S.params(), S.pending()), ([], None, None))
        self.assertTrue(os.path.exists(os.path.join(B, 'history.manifest.json.damaged')))

    def test_final_certifies_best(self):
        """A final stop ('too_long', 'budget') with a solve (the artifact la/best) certifies it: a witness with the
        run's claims, the outcome the stop's; with ProofData's final and reason (entry's
        fields; before that merge, in its record); without one, a final Stopped with the reason."""
        for oc in ('too_long', 'budget'):
            B = self.copy('final_' + oc)
            S = State(B)
            S.write_stop(oc, window=60, threads=1, need_s=7200, budget_core_hours=1.0)
            with self.assertLogs('rankbound', 'INFO'):
                p = Job(B).witness(1, None, oc)
            self.assertEqual((p.outcome, p.rank_bound), (oc, self.res.rank_bound))
            fin = (getattr(p, 'final', None), getattr(p, 'reason', None)) if hasattr(p, 'final') else \
                (p.record.get('final'), p.record.get('reason'))
            self.assertEqual(fin[0], oc)
            self.assertTrue(fin[1].startswith('The next check needs about 2.0 hours' if oc == 'too_long' else
                                              'The budget of 1.0 core-hours for this curve is spent'), fin[1])
            if getattr(p, 'data', None) is not None:       # the runner's write
                from rankbound import witness
                witness.write(p.data, p.witness)
            self.assertEqual(vf.load(p.witness)['claims'], vf.load(self.res.witness)['claims'])
        for oc, msg in (('too_long', 'The next check needs about 2.0 hours on 1 core, more than the 60 seconds a run '
                                     'has here. Run it with more time (--max-hours) or more cores.'),
                        ('budget', 'The budget of 1.0 core-hours for this curve was spent before any check gave a '
                                   'bound. Give a larger one with --budget to go on.')):
            B = self.copy('none_' + oc, drop=('la',))
            State(B).write_stop(oc, window=60, threads=1, need_s=7200, budget_core_hours=1.0)
            with self.assertRaises(Stopped) as cm:
                Job(B).witness(1, None, oc)
            self.assertEqual((cm.exception.outcome, cm.exception.reason), (oc, msg))

    def test_retry(self):
        """A final stop is cleared for a run that has more: 'too_long' with a longer window or more cores, 'budget'
        with a larger budget; else it stands."""
        B = self.copy('retry')
        S, J = State(B), Job(B)
        S.write_stop('too_long', window=600, threads=2)
        self.assertFalse(J.retry(S, 600, 2, None) or J.retry(S, 620, 1, None))
        self.assertTrue(S.stopped())
        self.assertTrue(J.retry(S, 700, 2, None))
        self.assertFalse(S.stopped())
        S.write_stop('too_long', window=600, threads=2)
        self.assertTrue(J.retry(S, 600, 4, None))
        S.write_stop('budget', budget_core_hours=5.0)
        self.assertFalse(J.retry(S, None, 1, 5.0))
        self.assertTrue(J.retry(S, None, 1, 6.0))
        S.write_stop('plateau')
        self.assertFalse(J.retry(S, None, 64, 1e9))


class TestRemoved(unittest.TestCase):
    def test_old_paths(self):
        """One Budget: no until or deadline floats in the sieve's, linalg's and arith's interfaces; the relation-based
        plateau, the earlier check plan, due.json and the old publishers are gone."""
        for f in (jobs.sieve_run, jobs.solve_given, Job.witness, linalg.solve, linalg.run_engine, arith._split,
                  arith._rho, arith._ecm):
            ps = inspect.signature(f).parameters
            self.assertFalse({'until', 'deadline'} & set(ps) - ({'deadline'} if f is linalg.run_engine else set()), f)
        for name in ('check_plan', 'check_seconds', 'PLATEAU', 'PLATEAU_LB'):
            self.assertFalse(hasattr(jobs, name), name)
        for name in ('mark_due', 'due', 'unfinished_check', 'stop_check', 'load_order', 'save_order'):
            self.assertFalse(hasattr(jobs.State, name), name)
        self.assertFalse(hasattr(Job, 'certify') or hasattr(runtime.Folder, 'check') or
                         hasattr(runtime.Folder, 'finish'))
        self.assertFalse(hasattr(jobs.Solver, 'until'))
        src = inspect.getsource(jobs) + inspect.getsource(linalg) + inspect.getsource(arith)
        self.assertNotIn('.wall()', src)
        self.assertNotIn('threading.Timer', src)
        self.assertNotIn('subprocess.Popen', src)
        self.assertNotIn('subprocess.run(', src)


if __name__ == '__main__':
    unittest.main()
