"""Fast tests of the checks: f2elim's reuse, limits and failures, the trim's escalation, Solver() itself, and the
state folder's load order and curve."""
import gzip, os, random, subprocess, tempfile, unittest
from rankbound import field, rules, run, sieve, solver, state
from helpers import bare_solver


class TestSolveReuse(unittest.TestCase):
    def test_reuse(self):
        """linalg.solve records the matrix's fingerprint with f2elim's output and reuses elim.ops/piv for the same
        matrix (certify after the proving check: no second elimination); another matrix runs f2elim again."""
        from rankbound import linalg
        if not os.path.exists(linalg.ENGINE):
            self.skipTest('f2elim not built')
        class M:
            ns = 2
            def __init__(self, rows):
                self.rows = [linalg.array.array('I', r) for r in rows]
                self.col = range(1 + max(max(r) for r in rows))
            def write(self, path):
                linalg.Matrix.write(self, path)
        calls = []
        real, real_sha = linalg.subprocess.Popen, linalg.sha256_file
        def counting(*a, **k):
            calls.append(a[0][0]); return real(*a, **k)
        linalg.subprocess.Popen = counting
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
            linalg.subprocess.Popen, linalg.sha256_file = real, real_sha

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
                V = bare_solver(state.State(d), M, nrel=2)
                self.assertIsNone(V.check(os.path.join(d, 'w')))
                self.assertEqual((V.hist[-1]['killed'], V.hist[-1]['dense_MB']), (True, None))
                self.assertEqual(rules.next_check(V.hist, {}), 2 * rules.NEXT_FULL)
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



class TestTrimEscalation(unittest.TestCase):
    def test_check(self):
        """An apparent plateau is re-solved with 10x the trim's excess, then without trim; it is a plateau only if the
        untrimmed bound is the same."""
        from rankbound import linalg
        real = linalg.solve
        def bare(d, target, hist):
            return bare_solver(state.State(d), type('M', (), {'rows': []})(), nrel=200, target=target, hist=hist,
                               order=['F_5', 'F_13'])
        def result(bound):          # None: the dense phase does not fit
            if bound is None:
                return linalg.SolveResult(None, None, None, None, 500000, '', '', too_dense=True, dense_mb=9e5)
            return linalg.SolveResult(bound, 0, 0, 0, 500000, '', '')
        def case(bounds, full=False):
            calls = []
            def solve(M, work, target, gate=False, keep=1000, **kw):
                calls.append(keep)
                return result(bounds[keep])
            linalg.solve = solve
            with tempfile.TemporaryDirectory() as d:
                V = bare(d,15, [{'relations': 100, 'bound': bounds[1000], 'excess': 1}])
                out = (V.check(d), calls, V.keep)
                return out + (V, state.State(d).history()) if full else out
        try:
            self.assertEqual(case({1000: 60, 10000: 60, 100000: 15}), ('goal', [1000, 10000, 100000], 100000))
            self.assertEqual(case({1000: 60, 10000: 40}), (None, [1000, 10000], 10000))
            self.assertEqual(case({1000: 60, 10000: 60, 100000: 60, -1: 60}), ('plateau', [1000, 10000, 100000, -1], -1))
            # the less trimmed matrix does not fit: a plateau at the trim that fits, which a resumed run keeps; the
            # last solve with a bound stays the one to certify
            out, calls, keep, V, h = case({1000: 60, 10000: 60, 100000: None}, full=True)
            self.assertEqual((out, calls, keep), ('plateau', [1000, 10000, 100000], 10000))
            self.assertEqual((V.last.bound, h[-1]['keep_tried'], h[-1]['dense_MB']), (60, 100000, 9e5))
            self.assertNotIn('keep', h[-1])
            self.assertEqual(next(x['keep'] for x in reversed(h) if 'keep' in x), 10000)
            # a check whose dense phase does not fit: no bound, recorded, go on (an error once the sieve has dried up)
            linalg.solve = lambda M, work, target, gate=False, keep=1000, **kw: result(None)
            with tempfile.TemporaryDirectory() as d:
                V = bare(d,15, [])
                self.assertIsNone(V.check(d))
                self.assertEqual(V.hist[-1]['dense_MB'], 9e5)
                self.assertIsNone(V.last)
                self.assertRaisesRegex(RuntimeError, 'more than is available here', V.check, d, dry=True)
            # purge only (excess < 0: f2elim's gate), the dense phase not reached: go on, but once the sieve has dried
            # up (or used every special-q) the run's last word is an error, not a plateau without a bound
            linalg.solve = lambda M, work, target, gate=False, keep=1000, **kw: linalg.SolveResult(
                None, None, None, None, -5, '', '')
            with tempfile.TemporaryDirectory() as d:
                V = bare(d,15, [])
                self.assertIsNone(V.check(d))
                self.assertRaisesRegex(RuntimeError, 'too few relations for this curve', V.check, d, dry=True)
                self.assertEqual(V.hist[-1]['excess'], -5)
            # the sieve has dried up (571a1's line siever): a plateau after one check, whatever the history; each full
            # solve records its slices in load order (la/order.txt, here WORK/order.txt)
            linalg.solve = lambda M, work, target, gate=False, keep=1000, **kw: result(1)
            with tempfile.TemporaryDirectory() as d:
                V = bare(d,0, [])
                self.assertIsNone(V.check(d))
                self.assertEqual(V.check(d, dry=True), 'plateau')
                with open(os.path.join(d, 'order.txt')) as f:
                    self.assertEqual(f.read(), 'F_5\nF_13\n')
        finally:
            linalg.solve = real

    def test_dry_untrimmed(self):
        """The last check (dry) goes on to no trim even when a less trimmed matrix already improves the bound."""
        from rankbound import linalg
        real = linalg.solve
        bounds, calls = {1000: 5, 10000: 3, 100000: 2, -1: 1}, []
        def solve(M, work, target, gate=False, keep=1000, **kw):
            calls.append(keep)
            return linalg.SolveResult(bounds[keep], 0, 0, 0, 500000, '', '')
        linalg.solve = solve
        try:
            with tempfile.TemporaryDirectory() as d:
                V = bare_solver(state.State(d), type('M', (), {'rows': []})(), nrel=200, target=0, order=['F_5'])
                self.assertEqual(V.check(d, dry=True), 'plateau')
                self.assertEqual((calls, V.last.bound, V.keep), ([1000, 10000, 100000, -1], 1, -1))
        finally:
            linalg.solve = real

    def test_dry(self):
        """dry(): the last DRY_SLICES slices of this process gave under DRY_RATIO of its mean per slice.  A steady
        yield is never dry, however many slices."""
        t = {'done': 0}
        for k in range(5000):
            rules.count_slice(t, 300 + k % 7)
        self.assertFalse(rules.dry(t))                           # steady: not dry
        for k in range(rules.DRY_SLICES):
            rules.count_slice(t, 2)
        self.assertTrue(rules.dry(t))                            # 40 relations in 20 slices, mean ~300
        rules.count_slice(t, 10 ** 4)
        self.assertFalse(rules.dry(t))
        t = {'done': 0}                                        # 571a1's line siever: 10k, then a few per slice
        for n in [10342, 1067, 394, 233, 135, 105, 76, 69, 31, 39, 33, 21, 24, 24, 12, 14, 17, 19, 13, 8, 8, 6, 11, 6]:
            rules.count_slice(t, n)
        self.assertTrue(rules.dry(t))



class TestSolverInit(unittest.TestCase):
    def test_init(self):
        """Solver(): with lim0 < 2^lpb0 the free relations go up to 2^lpb0, from a factor base makefb writes up to
        there; a run on given relations takes every free relation of its split file; the trim is that of the last
        solve with a bound."""
        from rankbound import linalg
        F = {'poly': [-5, 0, 1, 1], 'J': [], 'ramified': [], 'u': 1, 'n': 0, 'w': 1, 'rank_lower_bound': None}
        made = []

        class Siever:
            def prepare(self, st, prm, poly):
                return os.path.join(st, poly + '.poly'), fb0, None

        def makefb(pp, lim, out, side=0):
            made.append((os.path.basename(pp), lim, os.path.basename(out)))
            sieve.write_roots(F['poly'], lim, out)
            return out
        real = sieve.siever, sieve.makefb
        sieve.siever, sieve.makefb = lambda prm: Siever(), makefb
        quiet = lambda *a: None
        try:
            with tempfile.TemporaryDirectory() as d:
                fb0 = os.path.join(d, 'F.fb0_1024.gz')
                sieve.write_roots(F['poly'], 1024, fb0)
                S = state.State(d)
                V = solver.Solver(S, F, [], {'lpb0': 12, 'lim0': 1024}, quiet)
                self.assertEqual(made, [('F.poly', 4096, 'F.free_4096.gz')])
                n = len(list(linalg.free_rows(F, os.path.join(d, 'F.free_4096.gz'), 4096)))
                self.assertEqual(V.nfree, n)
                self.assertGreater(n, len(list(linalg.free_rows(F, fb0, 1024))))
                self.assertEqual((V.keep, V.hist), (solver.KEEP, []))
                V = solver.Solver(S, F, [], {'siever': 'given', 'lpb0': 8, 'lim0': 1024}, quiet)    # no makefb
                self.assertEqual((len(made), V.nfree), (1, len(list(linalg.free_rows(F, fb0, 1024)))))
                h = [{'relations': 10, 'excess': 5, 'bound': 3, 'keep': 10000},
                     {'relations': 10, 'excess': 5, 'keep_tried': 100000, 'dense_MB': 9e5}]
                S.save_history(h)
                V = solver.Solver(S, F, [], {'lpb0': 12, 'lim0': 4096}, quiet)          # lim0 = 2^lpb0: no makefb
                self.assertEqual((len(made), V.keep, V.hist), (1, 10000, h))
        finally:
            sieve.siever, sieve.makefb = real



class TestState(unittest.TestCase):
    MM = ['0', '-1', '1', '-929', '-10595']

    def test_load_order(self):
        """Certify rebuilds the matrix of the last full solve: la/order.txt keeps the slices in the order they were
        loaded (F_5 before F_13; sorted() gives F_13 first), Solver.load(names) adds them in that order (the same
        rows, so the same fingerprint and elimination), and an order naming a missing slice is not used."""
        with tempfile.TemporaryDirectory() as d:
            S = state.State(d)
            for name, lines in (('F_5', ['3,1:2,b', '5,2:7']), ('F_13', ['5,2:7', '-4,3:d'])):
                with gzip.open(os.path.join(S.sdir, name + '.rels.gz'), 'wt') as f:
                    f.writelines(l + '\n' for l in lines)
            V = bare_solver(S)
            V.load(['F_5', 'F_13'])
            self.assertEqual((V.M.src, V.order, V.nrel), (['3,1:2,b', '5,2:7', '-4,3:d'], ['F_5', 'F_13'], 3))
            U = bare_solver(S)
            U.load()
            self.assertEqual(U.order, ['F_13', 'F_5'])
            la = os.path.join(d, 'la'); os.makedirs(la)
            self.assertIsNone(S.load_order(la))
            S.save_order(la, V.order)
            W = bare_solver(S)
            W.load(S.load_order(la))
            self.assertEqual(W.M.src, V.M.src)
            S.save_order(la, ['F_5', 'F_7'])
            self.assertIsNone(S.load_order(la))

    def test_other_curve(self):
        """A state folder holds one curve: another minimal model, or another r, is refused; the same curve resumes."""
        from rankbound import RankboundError, write_json
        with tempfile.TemporaryDirectory() as d:
            write_json(os.path.join(d, 'field.json'), {'minimal_model': self.MM})
            write_json(os.path.join(d, 'curve.json'), {'name': 'curve', 'ainvs': [0, -1, 1, -929, -10595]})
            write_json(os.path.join(d, 'result.json'), {'name': 'curve', 'statement': 'S'})
            J = run.Job(d)
            same = field.Curve(name='curve', ainvs=self.MM, bad_primes=['571'], minimal_model=self.MM)
            self.assertEqual(J.run(same)['statement'], 'S')
            self.assertEqual(J.run(field.Curve(name='curve', ainvs=self.MM, bad_primes=['571']))['statement'], 'S')
            other = field.Curve(name='curve', ainvs=['0', '0', '1', '-1', '0'], bad_primes=['37'],
                                minimal_model=['0', '0', '1', '-1', '0'])
            self.assertRaisesRegex(RankboundError, 'was started for another curve', J.run, other)
            self.assertRaisesRegex(RankboundError, 'was started for another curve', J.prepare, other)
            # another r (e.g. points given later) is refused: field.json keeps the r it was prepared with
            write_json(os.path.join(d, 'field.json'), {'minimal_model': self.MM, 'rank_lower_bound': None})
            self.assertRaisesRegex(RankboundError, 'was started for another curve', J.run,
                                   field.Curve(name='curve', ainvs=self.MM, bad_primes=['571'], minimal_model=self.MM, r=1))
            self.assertEqual(J.run(same)['statement'], 'S')


if __name__ == '__main__':
    unittest.main()
