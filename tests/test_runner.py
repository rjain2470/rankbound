"""Fast tests of the run: its rules on recorded histories, the time limit, empty slices, the locks and shards, and
sieve_run's control flow with a stand-in siever and solver."""
import gzip, itertools, math, os, shutil, subprocess, sys, tempfile, time, unittest
from rankbound import field, rules, run, runner, sieve, state
from rankbound.verify import parity_bound
from helpers import bare_solver

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


class TestStopFromElsewhere(unittest.TestCase):
    def test_stop_from_elsewhere(self):
        """A local run whose sieve ends without an outcome because another process (`rankbound solve`) wrote STOP
        meanwhile certifies (or returns that process's result), instead of 'stopped by the time limit; run again';
        without STOP it is a time limit (Stopped).  Job.sieve and certify are stand-ins."""
        with tempfile.TemporaryDirectory() as d:
            J = run.Job(d)
            J.prm = {}                                  # tuned here: run() goes to the sieve
            J.certify = lambda verify, jobs: 'certified'
            J.sieve = lambda cores, until=None: None
            self.assertRaises(run.Stopped, J.run)       # no STOP: the time limit
            def other(cores, until=None, result=False):
                state.State(d).write_stop('goal')
                if result:
                    run.write_json(os.path.join(d, 'result.json'), {'name': 'x', 'statement': 'by the other'})
            J.sieve = other
            self.assertEqual(J.run(), 'certified')
            os.remove(os.path.join(d, 'STOP'))
            J.sieve = lambda cores, until=None: other(cores, until, result=True)
            self.assertEqual(J.run()['statement'], 'by the other')


class TestRetune(unittest.TestCase):
    def test_dried_up(self):
        """A sieve that dries up before the excess turns positive is re-tuned at lpb >= 24, keeping params_lpb22.json
        and the slices, and the run goes on; again at 24 -> 26.  At LPB_MAX, or with given relations, the error
        stands.  Job.sieve, tune and certify are stand-ins."""
        from rankbound.solver import DriedUp
        with tempfile.TemporaryDirectory() as d:
            J = run.Job(d)
            J.prm = {}                                  # tuned here: run() goes to the sieve
            J.certify = lambda verify, jobs: 'certified'
            run.write_json(os.path.join(d, 'params.json'), {'lpb0': 22})
            os.makedirs(os.path.join(d, 'slices')); open(os.path.join(d, 'slices', 'F_r1_2097152.json'), 'w').close()
            asked, dry = [], [2]
            def tune(mem_gb=1.0, cores=1, lpb_min=None):
                asked.append(lpb_min)
                prm = {'lpb0': lpb_min}
                run.write_json(os.path.join(d, 'params.json'), prm)
                J.prm = prm
                return prm
            def sieve_(cores, until=None):
                if dry[0]:
                    dry[0] -= 1
                    raise DriedUp(sieve.TOO_FEW)
                state.State(d).write_stop('goal')
                return 'goal'
            J.tune, J.sieve = tune, sieve_
            self.assertEqual(J.run(), 'certified')
            self.assertEqual(asked, [24, 26])
            self.assertEqual(run.read_json(os.path.join(d, 'params.json'))['lpb0'], 26)
            self.assertTrue(os.path.exists(os.path.join(d, 'params_lpb22.json')))
            self.assertTrue(os.path.exists(os.path.join(d, 'params_lpb24.json')))
            self.assertTrue(os.path.exists(os.path.join(d, 'slices', 'F_r1_2097152.json')))
            os.remove(os.path.join(d, 'STOP'))
            prm = {'lpb0': sieve.LPB_MAX}
            run.write_json(os.path.join(d, 'params.json'), prm)
            dry[0], asked[:] = 1, []
            self.assertRaisesRegex(DriedUp, 'too few relations for this curve, even at its largest setting', J.run)
            self.assertEqual((asked, run.read_json(os.path.join(d, 'params.json'))), ([], prm))
            prm = {'lpb0': 20, 'siever': 'given'}      # run() refuses it earlier (check_sieve)
            run.write_json(os.path.join(d, 'params.json'), prm)
            self.assertRaises(DriedUp, J.retune, DriedUp('dried up'))
            self.assertEqual((asked, run.read_json(os.path.join(d, 'params.json'))), ([], prm))


class TestRules(unittest.TestCase):
    """The schedule is deterministic and interleaves polynomials by weight; next_check and decide on recorded runs."""

    def test_schedule(self):
        prm = H['icarm356']['params']
        a = list(itertools.islice(rules.schedule(prm), 300))
        self.assertEqual(a, list(itertools.islice(rules.schedule(prm), 300)))
        self.assertEqual([j for j, *_ in a], list(range(300)))
        w = {n: v['weight'] for n, v in prm['polys'].items()}
        for n in w:                      # slices by weight; contiguous ranges
            mine = [(q0, q1) for _, p, q0, q1 in a if p == n]
            self.assertAlmostEqual(len(mine) / 300, w[n] / sum(w.values()), delta=0.01)
            self.assertEqual(mine[0][0], prm['q0'])
            self.assertTrue(all(x[1] == y[0] for x, y in zip(mine, mine[1:])))

    def test_q_ranges(self):
        """Up from q0 to 2^lpb, down from q0 to Q_FLOOR, then none: every special-q once.  The line siever's lines go
        up for ever; the schedule ends when every polynomial's ranges are used up."""
        prm = {'lpb0': 13}
        a = list(itertools.islice(rules.q_ranges(prm, 4096, 1000), 20))
        self.assertEqual(a[:5], [(4096, 5096), (5096, 6096), (6096, 7096), (7096, 8096), (8096, 8192)])
        self.assertEqual(a[5:], [(3096, 4096), (2096, 3096), (1096, 2096), (rules.Q_FLOOR, 1096)])    # then none
        cover = sorted(a)
        self.assertEqual(cover[0][0], rules.Q_FLOOR)
        self.assertTrue(all(x[1] == y[0] for x, y in zip(cover, cover[1:])))      # no gap, no overlap
        self.assertEqual(list(itertools.islice(rules.q_ranges({'siever': 'line', 'lpb0': 10}, 1, 3), 3)),
                         [(1, 4), (4, 7), (7, 10)])
        prm = {'lpb0': 12, 'q0': 2048, 'model': {'core_hours': 0, 'sec_per_q': 1.0},
               'polys': {'F': {'weight': 1.0}, 'G': {'weight': 0.5}}}
        s = list(rules.schedule(prm))                    # finite: each polynomial's q from 2^10 to 2^12, once
        for n in 'FG':
            cover = sorted((q0, q1) for _, p, q0, q1 in s if p == n)
            self.assertEqual((cover[0][0], cover[-1][1]), (rules.Q_FLOOR, 4096))
            self.assertTrue(all(x[1] == y[0] for x, y in zip(cover, cover[1:])))

    def test_recorded_runs(self):
        for name, d in sorted(H.items()):
            h, prm, tgt = d['history'], d['params'], d['target']
            self.assertEqual(rules.next_check([], prm), 0.8 * prm['model']['relations_needed'], name)
            for i in range(1, len(h)):
                self.assertIsNone(rules.decide(h[:i], tgt), name)      # every earlier check: go on
                self.assertLessEqual(rules.next_check(h[:i], prm), h[i]['relations'] * 1.0 + 1, name)
            self.assertEqual(rules.decide(h, tgt), 'goal', name)
        # #692: one purge-only check at excess -10568, then the next check from the default slope
        h = H['icarm692']['history']
        self.assertAlmostEqual(rules.next_check(h[:1], H['icarm692']['params']),
                               96802 + max(0.03 * 96802, 1.05 * 10568 / 0.6 + 0.01 * 96802))

    def test_slope(self):
        """Below excess 0 the next check comes from the measured slope dE/drelation (when above 0.05), with at least 3%
        more relations; after a full solve, at NEXT_FULL times the relations."""
        prm = H['icarm692']['params']
        h = [{'relations': 100000, 'excess': -20000}, {'relations': 120000, 'excess': -10000}]     # slope 0.5
        self.assertAlmostEqual(rules.next_check(h, prm), 120000 + 1.05 * 10000 / 0.5 + 1200)
        h[1]['excess'] = -19990                                                                 # slope 0.0005: default
        self.assertAlmostEqual(rules.next_check(h, prm), 120000 + 1.05 * 19990 / rules.SLOPE_DEFAULT + 1200)
        h[1]['excess'] = -10                                                                    # at least 3% more
        self.assertAlmostEqual(rules.next_check(h, prm), 120000 * 1.03)
        self.assertAlmostEqual(rules.next_check(h + [{'relations': 130000, 'excess': 50, 'bound': 3}], prm), 130000 * rules.NEXT_FULL)

    def test_plateau(self):
        h = [{'relations': 1000, 'excess': 5, 'bound': 4}, {'relations': 1100, 'excess': 9, 'bound': 4}]
        self.assertIsNone(rules.decide(h, None))
        h.append({'relations': 1300, 'excess': 20, 'bound': 4})
        self.assertEqual(rules.decide(h, None), 'plateau')
        self.assertEqual(rules.decide(h, 3), 'plateau')
        self.assertEqual(rules.decide(h, 4), 'goal')
        self.assertIsNone(rules.decide(h + [{'relations': 1400, 'excess': -1}], None))

    def test_plateau_next_full(self):
        """PLATEAU < NEXT_FULL: two consecutive full solves with the same bound, the second at the relation count
        next_check asked for, are a plateau (with PLATEAU above NEXT_FULL a third solve would be needed)."""
        self.assertLess(rules.PLATEAU, rules.NEXT_FULL)
        h = [{'relations': 100000, 'excess': 50, 'bound': 7}]
        h.append({'relations': math.ceil(rules.next_check(h, {})), 'excess': 900, 'bound': 7})
        self.assertEqual(rules.decide(h, 3), 'plateau')

    def test_next_check_too_dense(self):
        """A check whose dense phase did not fit (no bound, excess >= 0) is followed by one at NEXT_FULL times the
        relations, not by the slope rule of a negative excess."""
        h = [{'relations': 100000, 'excess': 4000, 'dense_MB': 9e5}]
        self.assertAlmostEqual(rules.next_check(h, {}), 100000 * rules.NEXT_FULL)

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
        self.assertEqual(rules.decide(h, 0), 'goal')                 # G = 0 stops at once

    def test_aim_text(self):
        """prepare's Aim line in each of its cases (the common one, S = r, is the only one the end-to-end runs
        reach): a parity mismatch, S = r out of reach, no r."""
        F = lambda r, u, n, w: {'rank_lower_bound': r, 'u': u, 'n': n, 'w': w, 'g_target': field.target({'u': u, 'n': n}, r)}
        self.assertEqual(run.aim_text(F(29, 1, 11, -1)), 'Aim: rank <= 29 = r, which proves rank = 29.')
        self.assertEqual(run.aim_text(F(29, 1, 11, 1)), 'Aim: rank <= 30 = r + 1 (the root number rules out r).')
        self.assertEqual(run.aim_text(F(1, 2, 3, -1)), 'Aim: rank <= 5; r = 1 cannot be reached here (u + n = 5).')
        self.assertEqual(run.aim_text(F(1, 2, 3, 1)),         # both the parity and u + n
                         'Aim: rank <= 4; r = 1 cannot be reached here (u + n = 5).')
        self.assertEqual(run.aim_text(F(None, 1, 0, 1)),
                         'Aim: rank <= 0 (the least possible). Stops there or when the bound stops improving.')
        self.assertEqual(run.aim_text(F(None, 1, 0, -1)),
                         'Aim: rank <= 1 (the least possible). Stops there or when the bound stops improving.')

    def test_tune_lopsided(self):
        """A field with no relation below lpb 25 (ICARM #3587): lpb_A goes from 23 to 25 and the shifted polynomial
        wins, not F by a tie of zero rates (trial faked: no CADO)."""
        F = {'GB': 340848, 'ns': 29306, 'log2_skew': 52.2, 'poly': [7, -5, 3, 2], 'real_roots': [1, 2]}
        def rate(self, name, prm, nq=10):
            r = 0.0 if name == 'F' or prm['lpb0'] < 25 else {'F_r1': 1.0, 'F_r2': 3.0}[name] * 2 ** (prm['lpb0'] - 25)
            return {'rate': r, 'sec_per_q': 0.3, 'rels_per_q': r * 0.3, 'mem_MB': 100}
        real, sieve.Trial.rate = sieve.Trial.rate, rate
        try:
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.las_params(d, F, 23, 2.0, lambda *a: None)
        finally:
            sieve.Trial.rate = real
        self.assertEqual((prm['trial']['lpb_A'], prm['lpb0']), (25, 25))
        self.assertEqual(sorted(prm['polys']), ['F_r2'])
        with tempfile.TemporaryDirectory() as d:          # no relation at any lpb up to LPB_MAX: a clear error
            sieve.Trial.rate = lambda *a, **k: {'rate': 0.0, 'sec_per_q': 0.0, 'rels_per_q': 0.0, 'mem_MB': 0}
            try:
                self.assertRaisesRegex(RuntimeError, 'too few relations', sieve.las_params, d, F, 23, 2.0, lambda *a: None)
            finally:
                sieve.Trial.rate = real

    def test_tune_lpb_min(self):
        """lpb_min, as in run.Job.retune: no lpb below it is chosen, although lpb 17 is the cheapest; a field of the
        line siever goes to las, at least at LAS_LPB_MIN (trial faked: no CADO)."""
        F = {'GB': 2 ** 15, 'ns': 3000, 'log2_skew': 2.0, 'poly': [7, -5, 3, 2], 'real_roots': [1]}
        real, tried = sieve.Trial.rate, []
        def rate(self, name, prm, nq=10):
            tried.append(prm['lpb0'])
            return {'rate': 100.0 * 1.2 ** -prm['lpb0'], 'sec_per_q': 0.3, 'mem_MB': 100, 'rels_per_q': 1000.0}
        try:
            sieve.Trial.rate = rate
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(sieve.las_params(d, F, 19, 0.5, lambda *a: None)['lpb0'], 17)
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.las_params(d, F, 19, 0.5, lambda *a: None, lpb_min=23)
            self.assertEqual((prm['lpb0'], prm['trial']['lpb_A'], prm['trial']['lpb_min']), (23, 23, 23))
            G = dict(F, GB=600)                                         # lpb_A 14 < LINE_BELOW: the line siever
            del tried[:]
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.tune(d, G, mem_gb=0.5, log=lambda *a: None, lpb_min=12)
            self.assertNotEqual(prm.get('siever'), 'line')
            self.assertEqual(prm['lpb0'], sieve.LAS_LPB_MIN); self.assertGreaterEqual(min(tried), sieve.LAS_LPB_MIN)
        finally:
            sieve.Trial.rate = real

    def test_tune_supply(self):
        """An lpb whose special-q cannot supply the relations needed is no candidate: the cheapest lpb 17 and 19 are
        passed over for 21; with too few relations per q at every lpb up to LPB_MAX, a clear error (trial faked)."""
        F = {'GB': 2 ** 15, 'ns': 3000, 'log2_skew': 2.0, 'poly': [7, -5, 3, 2], 'real_roots': [1]}
        rpq = {17: 0.01, 19: 0.1, 21: 1.0}
        real = sieve.Trial.rate
        try:
            sieve.Trial.rate = lambda self, name, prm, nq=10: {'rate': 100.0, 'sec_per_q': 0.3, 'mem_MB': 100,
                                                               'rels_per_q': rpq.get(prm['lpb0'], 0.01)}
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.las_params(d, F, 19, 0.5, lambda *a: None)
            self.assertEqual(prm['lpb0'], 21)
            self.assertGreaterEqual(prm['model']['supply'], sieve.SUPPLY_MIN)
            rpq = {}
            with tempfile.TemporaryDirectory() as d:
                self.assertRaisesRegex(RuntimeError, 'too few relations', sieve.las_params, d, F, 19, 0.5, lambda *a: None)
        finally:
            sieve.Trial.rate = real



class TestTimeLimit(unittest.TestCase):
    def test_job_end(self):
        """The Slurm job's end from SLURM_JOB_END_TIME, else from `squeue -h -j ID -o %e` (here a fake squeue);
        none outside Slurm or without a time limit.  Inside a Slurm job (SLURM_JOB_ID) the deadline is that end less
        JOB_MARGIN; outside one, or when it cannot be read, max_hours from now."""
        self.assertEqual(rules.job_end({'SLURM_JOB_END_TIME': '1760000000', 'SLURM_JOB_ID': '7'}), 1760000000.0)
        self.assertIsNone(rules.job_end({}))
        with tempfile.TemporaryDirectory() as d:
            sq = os.path.join(d, 'squeue')
            def fake(out):
                with open(sq, 'w') as f:
                    f.write('#!/bin/sh\necho "$@" > %s/args\necho "%s"\n' % (d, out))
                os.chmod(sq, 0o755)
            fake('2026-10-15T04:00:00')
            t = rules.job_end({'SLURM_JOB_ID': '123'}, squeue=sq)
            self.assertEqual(time.localtime(t)[:6], (2026, 10, 15, 4, 0, 0))
            with open(os.path.join(d, 'args')) as f:
                self.assertEqual(f.read().split(), ['-h', '-j', '123', '-o', '%e'])
            # an array task: its own line, JOB_TASK (the array's job id lists every task of the array)
            arr = {'SLURM_JOB_ID': '124', 'SLURM_ARRAY_JOB_ID': '123', 'SLURM_ARRAY_TASK_ID': '1'}
            self.assertEqual(rules.job_end(arr, squeue=sq), t)
            with open(os.path.join(d, 'args')) as f:
                self.assertEqual(f.read().split(), ['-h', '-j', '123_1', '-o', '%e'])
            for out in ('NONE', 'Unknown', '', '2026-10-15T04:00:00\n2026-10-15T05:00:00'):   # no time limit, the job
                fake(out)                                                                    # unknown, or two lines
                self.assertIsNone(rules.job_end({'SLURM_JOB_ID': '123'}, squeue=sq))
            self.assertIsNone(rules.job_end({'SLURM_JOB_ID': '123'}, squeue=os.path.join(d, 'none')))
        quiet = lambda *a: None
        end = time.time() + 3600
        self.assertEqual(rules.deadline(11.3, True, {'SLURM_JOB_END_TIME': '%d' % end}, quiet), int(end) - rules.JOB_MARGIN)
        for job, env in ((True, {}), (False, {'SLURM_JOB_END_TIME': '%d' % end})):
            self.assertAlmostEqual(rules.deadline(2, job, env, quiet), time.time() + 7200, delta=5)
        self.assertIsNone(rules.deadline(None, True, {}, quiet))
        # automatic: inside a Slurm job (SLURM_JOB_ID), its end; outside one, max_hours; the lines it logs
        said = []
        self.assertEqual(rules.deadline(2, None, {'SLURM_JOB_ID': '7', 'SLURM_JOB_END_TIME': '%d' % end}, said.append),
                         int(end) - rules.JOB_MARGIN)
        self.assertEqual(said, ['Will stop at %s, 5 minutes before the Slurm job ends.' % time.strftime(
            '%H:%M', time.localtime(int(end) - rules.JOB_MARGIN))])
        self.assertAlmostEqual(rules.deadline(2, None, {'SLURM_JOB_END_TIME': '%d' % end}, said.append),
                               time.time() + 7200, delta=5)
        self.assertEqual(len(said), 1)
        rules.deadline(2.5, None, {'SLURM_JOB_ID': '7'}, said.append)       # squeue fails here (no such job, or none)
        rules.deadline(None, None, {'SLURM_JOB_ID': '7'}, said.append)
        self.assertEqual(said[1:], ["Could not read the Slurm job's end time; stopping after --max-hours (2.5 h) instead.",
                                    "Could not read the Slurm job's end time; running without a time limit."])
        self.assertAlmostEqual(rules.earlier(end, 0.5), time.time() + 1800, delta=5)
        self.assertEqual(rules.earlier(end, None), end)
        self.assertIsNone(rules.earlier(None, None))

    def test_check_plan(self):
        """check_seconds: the longest elimination so far, scaled by the square of the relation ratio, or half the run
        before the first.  check_plan: a check that would not finish waits for the next run, unless this process has
        just started; one that did not finish in an earlier run with as much time stops the run ('too long')."""
        h = [{'relations': 1000, 'excess': -50, 'solve_s': 2.0},
             {'relations': 2000, 'excess': 10, 'bound': 9, 'solve_s': 100.0},       # one check: 100 + 60 + 40
             {'relations': 2000, 'excess': 10, 'bound': 9, 'keep': 10000, 'solve_s': 60.0},
             {'relations': 2000, 'excess': 10, 'keep_tried': -1, 'dense_MB': 9e5, 'solve_s': 40.0}]
        self.assertEqual(rules.check_seconds([], 5000), 0)
        self.assertEqual(rules.check_seconds([], 5000, 3600), 1800)
        self.assertEqual(rules.check_seconds(h, 2000), 200.0)
        self.assertEqual(rules.check_seconds(h, 3000), 200.0 * 1.5 ** 2)
        self.assertEqual(rules.check_seconds(h, 3000, 10 ** 6), 200.0 * 1.5 ** 2)   # a full check known: no share
        self.assertEqual(rules.check_seconds(h[:1], 500), 2.0)              # never scaled down
        self.assertEqual(rules.check_seconds(h[:1], 500, 3600), 1800)       # purge checks only: half the run
        self.assertEqual(rules.check_seconds(h[:1], 50000, 100), 2.0 * 50 ** 2)
        P = rules.check_plan
        self.assertEqual(P(h, 3000, None, 0.0, 1000.0, False), ('now', 0.0))
        self.assertEqual(P(h, 3000, 1450.0, 0.0, 1000.0, False), ('now', 450.0))
        self.assertEqual(P(h, 3000, 1449.0, 0.0, 1000.0, False), ('later', 450.0))
        self.assertEqual(P(h, 3000, 1449.0, 0.0, 1000.0, True), ('now', 450.0))        # just started: at once
        old = {'relations': 3000, 'start': 0.0, 'until': 1000.0}          # had 1000 s, did not finish
        self.assertEqual(P(h, 3000, 2000.0, 1000.0, 1000.0, True, old), ('too long', 1100.0))
        self.assertEqual(P(h, 3000, 2000.0, 1000.0, 1000.0, False, old), ('later', 1100.0))
        self.assertEqual(P(h, 3000, 2200.0, 1000.0, 1000.0, True, old), ('now', 1100.0))  # a longer run tries again
        self.assertEqual(P(h, 3000, 1449.0, 0.0, 1000.0, True, dict(old, until=None)), ('now', 450.0))
        old8 = dict(old, threads=8)                                       # had 8 threads: 'too long' with 8, not 16
        self.assertEqual(P(h, 3000, 2000.0, 1000.0, 1000.0, True, old8, 8), ('too long', 1100.0))
        self.assertEqual(P(h, 3000, 2000.0, 1000.0, 1000.0, True, old8, 16), ('now', 1100.0))
        self.assertEqual(P(h, 3000, 2000.0, 1000.0, 1000.0, False, old8, 16), ('later', 1100.0))
        # stopped (Ctrl-C) at or after its run's limit: out of time; well before it: an interruption, left out
        late = dict(old, stopped=1000.0 - rules.JOB_MARGIN + 1)
        self.assertEqual(P(h, 3000, 2000.0, 1000.0, 1000.0, True, late), ('too long', 1100.0))
        early = dict(old, stopped=60.0)
        self.assertEqual(P(h, 3000, 2000.0, 1000.0, 1000.0, True, early), ('now', 450.0))

    def test_run_check(self):
        """check.json stays only for a check that was killed (SystemExit, KeyboardInterrupt: not an Exception), with
        the elimination's threads, and only Ctrl-C marks it 'stopped' (a SIGTERM's SystemExit stays a timeout); a check
        that fails with an error (f2elim's, a dried-up sieve) or finishes removes it.  A finished check clears due.json
        (a check left to the next run)."""
        class V:
            nrel, threads, hist = 1300, 4, []
            def __init__(self, exc=None):
                self.exc = exc
            def check(self, work, dry=False):
                if self.exc:
                    raise self.exc
        with tempfile.TemporaryDirectory() as d:
            S = state.State(d)
            S.mark_due(1200)
            until = time.time() + 3600
            with self.assertRaises(RuntimeError):
                runner.run_check(S, V(RuntimeError('f2elim failed (exit 2)')), {}, until=until)
            self.assertIsNone(S.unfinished_check())
            self.assertTrue(S.due())
            with self.assertRaises(SystemExit):
                runner.run_check(S, V(SystemExit(143)), {}, until=until)
            u = S.unfinished_check()
            self.assertEqual((u['relations'], u['until'], u['threads']), (1300, until, 4))
            self.assertNotIn('stopped', u)                                      # SIGTERM: a timeout
            with self.assertRaises(KeyboardInterrupt):
                runner.run_check(S, V(KeyboardInterrupt()), {}, until=until)
            u = S.unfinished_check()
            self.assertEqual((u['relations'], u['until'], u['threads']), (1300, until, 4))
            self.assertAlmostEqual(u['stopped'], time.time(), delta=30)       # when: an interruption, not a timeout
            self.assertIsNone(runner.run_check(S, V(), {}, until=until))
            self.assertIsNone(S.unfinished_check())
            self.assertFalse(S.due())
            # another process (rankbound solve) finished the run: no check here (la/ holds that process's elimination)
            S.write_stop('goal')
            self.assertIsNone(runner.run_check(S, V(RuntimeError('checked after STOP')), {}, until=until))



class TestEmptySlices(unittest.TestCase):
    def test_count(self):
        """MAX_EMPTY slices in a row without a relation stop the run; one relation resets the count."""
        t = {'done': 0}
        for k in range(rules.MAX_EMPTY - 1):
            rules.count_slice(t, 0)
        rules.count_slice(t, 5)
        for k in range(rules.MAX_EMPTY - 1):
            rules.count_slice(t, 0)
        self.assertRaisesRegex(RuntimeError, 'too few relations', rules.count_slice, t, 0)



class TestLocked(unittest.TestCase):
    def test_lock(self):
        """A second process on the same shard of a state (or a check while another holds run.lock) raises Locked;
        the lock is free again when the holder closes it."""
        from rankbound import Locked
        with tempfile.TemporaryDirectory() as d:
            S = state.State(d)
            fd = S.lock('shard0_1.lock')
            try:
                self.assertRaises(Locked, runner.sieve_run, d)
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
            S = state.State(d)
            a, b = S.lock('sieve.lock', shared=True), S.lock('sieve.lock', shared=True)      # two shards
            try:
                self.assertRaises(Locked, runner.sieve_run, d)                               # a local run
            finally:
                os.close(a); os.close(b)
            os.close(S.lock('shard0_1.lock'))
            fd = S.lock('sieve.lock')                                                     # a local run
            try:
                self.assertRaises(Locked, runner.sieve_run, d, shard=(1, 2))                  # a shard
            finally:
                os.close(fd)
            os.close(S.lock('shard1_2.lock'))



class TestShardEnd(unittest.TestCase):
    PRM = {'polys': {'F': {'weight': 1.0, 'shift': 0}}, 'model': {'core_hours': 0.0, 'sec_per_q': 1.0}, 'q0': 1024,
           'lpb0': 11}

    def test_others_left(self):
        """Shard 0's last check waits for every other shard's share of the schedule (slices j mod n), whether that shard
        is running (holds its lock) or not (queued or failed)."""
        with tempfile.TemporaryDirectory() as d:
            S = state.State(d)
            sched = list(rules.schedule(self.PRM))
            self.assertGreater(len(sched), 4)
            self.assertEqual(runner.others_left(S, self.PRM, 0, 2), ([], [1]))
            fd = S.lock('shard1_2.lock')
            try:
                self.assertEqual(runner.others_left(S, self.PRM, 0, 2), ([1], []))
            finally:
                os.close(fd)
            for j, poly, q0, _ in sched:
                if j % 2 == 1:
                    with gzip.open(os.path.join(S.sdir, state.slice_name(poly, q0) + '.rels.gz'), 'wt'):
                        pass
            self.assertEqual(runner.others_left(S, self.PRM, 0, 2), ([], []))      # shard 0's own share does not count
            self.assertEqual(runner.others_left(S, self.PRM, 1, 2), ([], [0]))

    def test_kill_las_tmp(self):
        """A stopped slice leaves no partial output: las writes OUT.tmp.PID (here a stand-in that does the same), which
        kill() removes."""
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'F_9000.las')
            p = subprocess.Popen([sys.executable, '-c', 'import os, sys, time\nopen(sys.argv[1] + ".tmp.%d" % '
                                  'os.getpid(), "w").write("x")\ntime.sleep(60)', out],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
            t = time.time()
            while not os.listdir(d) and time.time() - t < 20:
                time.sleep(0.05)
            self.assertEqual(os.listdir(d), ['F_9000.las.tmp.%d' % p.pid])
            runner.kill(p, out)
            self.assertEqual(os.listdir(d), [])

    def test_cut_short_slice(self):
        """A slice file cut short (a node crash) stops the solver with an error naming it and what to do."""
        from rankbound import RankboundError
        with tempfile.TemporaryDirectory() as d:
            S = state.State(d)
            p = os.path.join(S.sdir, 'F_5.rels.gz')
            with gzip.open(p, 'wt') as f:
                f.writelines('%d,1:2,3\n' % a for a in range(3000))
            with open(p, 'rb') as f:
                data = f.read()
            with open(p, 'wb') as f:
                f.write(data[:len(data) // 2])
            V = bare_solver(S)
            with self.assertRaisesRegex(RankboundError, 'F_5.rels.gz is damaged. Delete it and .*F_5.json; that part is sieved again.'):
                V.load()

    def test_damaged_slice_json(self):
        """A slice's NAME.json cut short, or without its relation count, stops the run with an error naming it and
        what to do."""
        from rankbound import RankboundError
        with tempfile.TemporaryDirectory() as d:
            S = state.State(d)
            p = os.path.join(S.sdir, 'F_5.json')
            for text in ('{"cpu": 1', '{"cpu": 1}'):
                with open(p, 'w') as f:
                    f.write(text)
                with self.assertRaisesRegex(RankboundError, 'F_5.json is damaged. Delete it and .*F_5.rels.gz; that part is sieved again.'):
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

    class Siever:
        def start_slice(self, state, prm, poly, q0, q1, out):
            text = ''.join('%d,1:2,3:\n' % a for a in range(TestSieveRun.R)) + '# Total cpu time 0.01s\n'
            return subprocess.Popen([sys.executable, '-c', 'import sys; open(sys.argv[1], "w").write(sys.argv[2])', out,
                                     text], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)

    def setUp(self):
        self.checks, self.nxt = [], 10 ** 9
        test = self

        class V:
            threads, pids = 1, ()
            @classmethod
            def of(cls, S, F, prm, log, jobs):
                v = cls(); v.S, v.hist, v.nrel = S, S.history(), 0
                return v
            def load(self):
                self.nrel = self.S.slice_stats()[1]
            def next_check(self):
                return test.nxt
            def check(self, work, dry=False):
                test.checks.append((self.nrel, dry))
                return 'plateau' if dry else None
        real = (runner.open_state, runner.Solver, sieve.siever, runner.time, runner.POLL)
        def restore():
            runner.open_state, runner.Solver, sieve.siever, runner.time, runner.POLL = real
        self.addCleanup(restore)
        runner.open_state = lambda S: ({}, self.PRM)
        runner.Solver, sieve.siever = V, lambda prm: self.Siever()
        runner.time = type('T', (), {'time': time.time, 'strftime': time.strftime, 'localtime': time.localtime,
                                     'sleep': staticmethod(lambda s: time.sleep(min(s, 0.02)))})
        runner.POLL = 0.05
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.S = state.State(self.d)
        self.sched = list(rules.schedule(self.PRM))
        self.assertEqual(len(self.sched), 15)

    def finished(self):
        return sorted(self.S.finished()) == sorted(state.slice_name(p, q0) for _, p, q0, _ in self.sched)

    def test_schedule_end(self):
        """A local run whose schedule ends before a check is due: one last check, dry, on every slice; its plateau
        is written to STOP; no partial output left."""
        log = []
        out, V = runner.sieve_run(self.d, jobs=2, log=log.append)
        self.assertEqual((out, self.checks), ('plateau', [(15 * self.R, True)]))
        self.assertEqual(self.S.stop_outcome(), 'plateau')
        self.assertTrue(self.finished())
        self.assertEqual([f for f in os.listdir(self.S.sdir) if '.las' in f], [])

    def test_later(self):
        """A check due (here at 30 relations) that would not finish in the time left (the history has a full check of
        10^6 s) in a process that has sieved already waits for the next run (due.json, logged once): this run sieves on
        to the schedule's end without a check and stops without an outcome.  The next run starts with that check, before
        it sieves the 5 slices removed here, dry (its relations are below the next check, 10^9) and so the last one."""
        self.S.save_history([{'relations': 1, 'rows': 1, 'excess': 1, 'bound': 5, 'solve_s': 1e6}])
        self.nxt, log = 30, []
        out, V = runner.sieve_run(self.d, jobs=2, max_hours=0.1, log=log.append)
        self.assertEqual((out, self.checks), (None, []))
        self.assertTrue(self.S.due() and self.finished() and not self.S.stopped())
        self.assertEqual(len([l for l in log if l == 'Not enough time left for the next check; it runs first when you resume.']), 1, log)
        for _, poly, q0, _ in self.sched[-5:]:
            for ext in ('.rels.gz', '.json'):
                os.remove(os.path.join(self.S.sdir, state.slice_name(poly, q0) + ext))
        self.nxt = 10 ** 9
        out, V = runner.sieve_run(self.d, jobs=2, max_hours=0.1, log=log.append)
        self.assertEqual((out, self.checks), ('plateau', [(10 * self.R, True)]))
        self.assertFalse(self.S.due())
        self.assertEqual(len(self.S.finished()), 10)

    def test_shard_waits(self):
        """Shard 0 of 2, its share sieved, waits for shard 1's share: logged once while shard 1 runs (holds its lock),
        once more when it is not running (queued or stopped: resubmit it); no check meanwhile.  When shard 1's slices
        are there, shard 0's last check is dry, on all 15 slices."""
        log, fd = [], [self.S.lock('shard1_2.lock')]
        def step(msg):
            log.append(msg)
            if 'waiting for shard 1.' in msg and fd:
                os.close(fd.pop())                              # shard 1 stops before its share is sieved
            elif msg == 'Shard 1 is not running: resubmit it.':     # it is resubmitted and sieves its share
                for j, poly, q0, q1 in self.sched:
                    if j % 2:
                        out = os.path.join(self.S.sdir, state.slice_name(poly, q0) + '.las')
                        with open(out, 'w') as f:
                            f.write('%d,1:2,3:\n' % j * self.R)
                        self.S.finish_slice(self.PRM, poly, q0, q1, out)
        out, V = runner.sieve_run(self.d, shard=(0, 2), jobs=2, max_hours=0.1, log=step)
        waits = [l for l in log if 'waiting' in l or 'not running' in l]
        self.assertEqual(waits, ['Shard 0 of 2 is done; waiting for shard 1.'] * 2 + ['Shard 1 is not running: resubmit it.'],
                         log)
        self.assertEqual((out, self.checks), ('plateau', [(15 * self.R, True)]))
        self.assertTrue(self.finished())


if __name__ == '__main__':
    unittest.main()
