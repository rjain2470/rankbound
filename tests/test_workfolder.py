"""The work folder and the elimination's files: a shared work folder never returns or records another run's result;
orphan slice metadata is no finished work; complete reuse keys of linalg.solve and linalg.kernel; f2kernel keeps the
empty rows' vectors when Lanczos finds none; the purge and trim observe F2ELIM_DEADLINE (run_engine's watchdog), and a
late solve leaves an earlier solve's files as they were; the sieve's roots at LIM 0."""
import multiprocessing as mp, os, shutil, subprocess, tempfile, time, unittest
from unittest import mock
from rankbound import (Locked, RankboundError, TimeLimit, curve as cv, descent, linalg, rankbound, runner, runtime, selmer,
                       sieve)
from helpers import PKG, run_route
from rankbound.runtime import read_json, write_json

SIEVE = os.path.join(PKG, 'sieve')
HAVE_GP = shutil.which('gp') is not None


def _worker(ainvs, r, folder, barrier):
    barrier.wait()
    try:
        run_route(descent.Descent(folder), cv.from_ainvs(ainvs, r=r))
    except Exception:
        pass            # a refused second run (other_run, Locked) is safe


@unittest.skipUnless(HAVE_GP, 'needs PARI/GP')
class TestSharedFolder(unittest.TestCase):
    """[0,0,0,-1,0] r=0 (rank bound 0) and [0,0,0,-25,0] r=1 (rank bound 1) started together in one folder."""
    A, B = ([0, 0, 0, -1, 0], 0), ([0, 0, 0, -25, 0], 1)

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='rb_shared')

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_race(self):
        """Two processes at once: whatever the schedule, run.json's result is its key's, and the API
        returns the keyed curve's own bound (or refuses the other curve)."""
        ctx = mp.get_context('spawn')
        bound = {tuple(self.A[0]): 0, tuple(self.B[0]): 1}
        for attempt in range(6):
            folder = os.path.join(self.d, str(attempt))
            barrier = ctx.Barrier(2)
            ps = [ctx.Process(target=_worker, args=(c[0], c[1], folder, barrier)) for c in (self.A, self.B)]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
            st = read_json(os.path.join(folder, 'run.json'), {})
            if not st.get('result'):
                continue
            ainvs = tuple(int(x) for x in st['key']['curve'])
            self.assertEqual(st['result']['rank_bound'], bound[ainvs], (attempt, st))
            self.assertEqual(st['result']['name'], st['name'])
            self.assertEqual(st['result_key'], st['key'])
            for c in (self.A, self.B):
                if tuple(c[0]) == ainvs:
                    res = rankbound(c[0], r=c[1], workdir=folder, compute=True)
                    self.assertEqual(res.rank_bound, bound[ainvs])
                    self.assertTrue(res.checked)
                else:
                    self.assertRaises(RankboundError, rankbound, c[0], r=c[1], workdir=folder, compute=True)

    def test_mismatched_record(self):
        """The state once observed (B's key, A's checked result) is refused, never returned."""
        a = os.path.join(self.d, 'a')
        run_route(descent.Descent(a), cv.from_ainvs(self.A[0], r=self.A[1]))
        st = read_json(os.path.join(a, 'run.json'))
        b = os.path.join(self.d, 'b')
        B = cv.from_ainvs(self.B[0], r=self.B[1])
        runner.bind_folder(descent.Descent(b), B)
        sb = read_json(os.path.join(b, 'run.json'))
        sb.update(result=st['result'], stage='finished', rank_bound=0)          # no result_key: an older record
        write_json(os.path.join(b, 'run.json'), sb)
        with self.assertRaises(RankboundError):
            run_route(descent.Descent(b), B)
        sb['result_key'] = st['key']                                             # a result keyed for another run
        write_json(os.path.join(b, 'run.json'), sb)
        with self.assertRaises(RankboundError):
            run_route(descent.Descent(b), B)
        st2 = dict(st, result=dict(st['result']))
        del st2['result_key']                                                    # an older record of its own run
        write_json(os.path.join(a, 'run.json'), st2)
        self.assertEqual(run_route(descent.Descent(a), cv.from_ainvs(self.A[0], r=self.A[1]))['rank_bound'], 0)

    def test_finish_removed(self):
        """A result is published only by the runner (runner.conclude, under the key it was computed for: RunStore);
        the route's own publishers of before (Folder.check, Folder.finish, Job.certify) are gone."""
        for name in ('check', 'finish'):
            self.assertFalse(hasattr(runtime.Folder, name), name)
        self.assertFalse(hasattr(selmer.Job, 'certify'))

    def test_owner_lock(self):
        """A second run in a folder while one runs: Locked, before anything is read or written."""
        f = os.path.join(self.d, 'e')
        os.makedirs(f)
        fd = sieve.State(f).lock('bound.lock')
        try:
            with self.assertRaises(Locked):
                run_route(descent.Descent(f), cv.from_ainvs(self.A[0], r=self.A[1]))
            self.assertFalse(os.path.exists(os.path.join(f, 'run.json')))
        finally:
            os.close(fd)


class TestSlices(unittest.TestCase):
    LINES = ['1,2:3,5\n', '# Total cpu time 7.00s\n', '# Total 1 reports [7s/r, 1.0r/sq]\n']

    def test_orphan_metadata(self):
        """A crash between NAME.json and NAME.rels.gz (fault-injected) leaves no finished work; a malformed orphan is
        no damage either; sieving the slice again commits it."""
        d = tempfile.mkdtemp(prefix='rb_slices')
        try:
            S = sieve.State(d)
            real = os.replace
            def fail(a, b):
                if str(b).endswith('.rels.gz'):
                    raise OSError('injected')
                return real(a, b)
            with mock.patch('os.replace', side_effect=fail):
                self.assertRaises(OSError, S.finish_slice, 'f0', 100, 200, self.LINES)
            self.assertTrue(os.path.exists(os.path.join(S.sdir, 'f0_100.json')))
            self.assertEqual(S.finished(), [])
            self.assertEqual(S.slice_stats(), (0, 0))
            with open(os.path.join(S.sdir, 'f1_200.json'), 'w') as f:
                f.write('{"cpu": ')                  # cut short, without its rels.gz
            self.assertEqual(S.slice_stats(), (0, 0))
            self.assertEqual(S.finish_slice('f0', 100, 200, self.LINES), (7.0, 1))
            self.assertEqual(S.finished(), ['f0_100'])
            self.assertEqual(S.slice_stats(), (7.0, 1))
            os.remove(os.path.join(S.sdir, 'f0_100.json'))        # a committed slice without its metadata: damaged
            self.assertRaises(RankboundError, S.slice_stats)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_given_republished(self):
        """write_given: new metadata never describes an older given.rels.gz (crash after given.json)."""
        d = tempfile.mkdtemp(prefix='rb_slices')
        try:
            S = sieve.State(d)
            S.write_given(['1,0:'], 'old')
            self.assertEqual(S.given(), 'old')
            real = os.replace
            def fail(a, b):
                if str(b).endswith('.rels.gz'):
                    raise OSError('injected')
                return real(a, b)
            with mock.patch('os.replace', side_effect=fail):
                self.assertRaises(OSError, S.write_given, ['1,0:', '2,0:'], 'new')
            self.assertIsNone(S.given())
            self.assertEqual(S.slice_stats(), (0, 0))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestKeys(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='rb_keys')

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_elements(self):
        """K = Q(2^(1/3)): the units 1 and 1 - alpha have the same (empty) row; at (5, 3) 1 is a square and 1 - 3 is
        not.  The second solve in the first one's folder gives its fresh crank."""
        def M(a, b):
            m = linalg.Matrix({'J': []}, [(7, 0)])
            m._add([], '%d,%d:' % (a, b), a, b)
            return m
        w = os.path.join(self.d, 'w')
        r1 = linalg.solve(M(1, 0), w, 0, keep=-1, chars=[(5, 3)])
        r2 = linalg.solve(M(1, 1), w, 0, keep=-1, chars=[(5, 3)])
        fresh = linalg.solve(M(1, 1), os.path.join(self.d, 'f'), 0, keep=-1, chars=[(5, 3)])
        self.assertEqual((r1.crank, fresh.crank), (0, 1))
        self.assertEqual(r2.crank, fresh.crank)

    def test_held_ids(self):
        """The same matrix with another held column of the same count: the reused folder gives the fresh result."""
        labs = [(3, 0), (5, 0), (7, 0)]
        def M(bad):
            m = linalg.Matrix({'J': [], 'bad_primes': bad}, [(11, 0)])
            for r in ([(11, 0), labs[0]], [labs[1]], [labs[2]]):
                m._add(r, 'x')
            return m
        w = os.path.join(self.d, 'w')
        a = linalg.solve(M([5]), w, 0, keep=-1)
        b = linalg.solve(M([7]), w, 0, keep=-1)
        fresh = linalg.solve(M([7]), os.path.join(self.d, 'f'), 0, keep=-1)
        self.assertNotEqual(a.held, b.held)
        def files(r):
            out = []
            for p in (r.ops, r.piv, r.vs):
                with open(p, 'rb') as f:
                    out.append(f.read())
            return out
        self.assertEqual((b.bound, b.dimV, b.held, files(b)), (fresh.bound, fresh.dimV, fresh.held, files(fresh)))

    def test_words(self):
        """The same bytes of kernel.bin under WORDS 0 and 1 are different matrices."""
        w = os.path.join(self.d, 'w')
        self.assertEqual(linalg.kernel([[1, 0]], 64, 2, work=w).vectors, [[0]])
        second = linalg.kernel([[]], 64, 2, bits=[(1 << 32) | 2], work=w).vectors
        fresh = linalg.kernel([[]], 64, 2, bits=[(1 << 32) | 2], work=os.path.join(self.d, 'f')).vectors
        self.assertEqual(fresh, [])
        self.assertEqual(second, fresh)


class TestKernel(unittest.TestCase):
    """Row 0 = {0} (a known vector) and I + J on 1100 large columns (invertible: no further vector)."""
    n = 1100

    def rows(self):
        return [[c for c in range(1, self.n + 1) if c != i] for i in range(1, self.n + 1)]

    def test_known_vector_kept(self):
        K = linalg.kernel([[0]] + self.rows(), self.n + 1, 1)
        self.assertEqual(K.vectors, [[0]])
        self.assertIn(' method lanczos ', K.info)
        self.assertTrue(K.info.endswith(' status partial'), K.info)

    def test_invertible_block_alone(self):
        """No vector at all: still a failure (exit 5), never an empty kernel."""
        with self.assertRaises(RuntimeError):
            linalg.kernel(self.rows(), self.n + 1, 1)

    def test_dense_complete(self):
        with mock.patch.dict(os.environ, {'F2KERNEL_DENSE_N': '2000'}):
            K = linalg.kernel([[0]] + self.rows(), self.n + 1, 1)
        self.assertEqual(K.vectors, [[0]])
        self.assertTrue(K.info.endswith(' status complete'), K.info)


class TestDeadline(unittest.TestCase):
    """R_0 = {0}, R_i = {i-1, i} on large columns only: a quadratic purge (one row per pass)."""
    n = 40000

    def test_solve_deadline(self):
        d = tempfile.mkdtemp(prefix='rb_deadline')
        try:
            M = chain(self.n)
            t = time.time()
            r = linalg.solve(M, d, 0, keep=-1, budget=runtime.Budget(seconds=0.2))
            took = time.time() - t
            # with the queue purge (linear here) the solve can end in time; it is never late with a bound
            if r.late is not None or took > 1.0:
                self.assertIsNotNone(r.late)
                self.assertIsNone(r.bound)
            self.assertLess(took, 1.5)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_purge_only(self):
        """TARGET < 0 (the purge only): exit 4 and no output files past the deadline."""
        d = tempfile.mkdtemp(prefix='rb_deadline')
        try:
            M = chain(self.n)
            inp = os.path.join(d, 'matrix.bin')
            M.write(inp)
            t = time.time()
            rc, out, _ = linalg.run_engine([inp, '0', '-1', os.path.join(d, 'purge'), '1000', '64', '1', '-1', '1', '0',
                                            '300'], runtime.Budget(seconds=0.2), deadline=True)
            took = time.time() - t
            if rc != 0 or took > 1.0:       # with the queue purge it can end in time (exit 0); never late with exit 0
                self.assertEqual(rc, 4, out)
            self.assertLess(took, 1.5)
            self.assertEqual(sorted(os.listdir(d)), ['matrix.bin'])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_watchdog(self):
        """A program that ignores its deadline is stopped WATCHDOG_S after it and reaped (runtime.run), and the
        budget's exception is raised; with time left, it runs to its end."""
        t = time.time()
        with self.assertRaises(TimeLimit):
            linalg.run_engine(['30'], runtime.Budget(seconds=0.3), deadline=True, engine=shutil.which('sleep'))
        self.assertLess(time.time() - t, 0.3 + linalg.WATCHDOG_S + 2)
        self.assertEqual(runtime._CHILDREN, set())
        rc, out, _ = linalg.run_engine(['0'], runtime.Budget(seconds=60), deadline=True, engine=shutil.which('sleep'))
        self.assertEqual((rc, out), (0, ''))


def chain(n):
    """R_0 = {0}, R_i = {i-1, i} on large columns only."""
    M = linalg.Matrix({'J': []}, [])
    for i in range(n):
        M._add([(10 ** 7 + c, 0) for c in ([0] if i == 0 else [i - 1, i])], 'x')
    return M


def contents(d, names):
    out = {}
    for f in names:
        with open(os.path.join(d, f), 'rb') as h:
            out[f] = h.read()
    return out


class TestPolledDeadline(unittest.TestCase):
    """f2elim built to poll its deadline at every unit of work (-DPOLL_WORK=1): a deadline 10 ms after the start, on
    a purge of 200,000 rows that takes longer, always stops it late (exit 4), promptly and without output files."""
    n, budget = 200000, 0.01

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='rb_polled')
        cls.engine = os.path.join(cls.tmp, 'f2elim_polled')
        subprocess.run(['cc', '-O2', '-Wall', '-Werror', '-pthread', '-DPOLL_WORK=1', '-o', cls.engine,
                        linalg.ENGINE + '.c', '-lm'], check=True)
        cls.M = chain(cls.n)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_purge_only(self):
        with tempfile.TemporaryDirectory(prefix='rb_deadline') as d:
            inp = os.path.join(d, 'matrix.bin')
            self.M.write(inp)
            t = time.time()
            rc, out, _ = linalg.run_engine([inp, '0', '-1', os.path.join(d, 'purge'), '1000', '64', '1', '-1', '1',
                                            '0', '300'], runtime.Budget(seconds=self.budget), deadline=True,
                                           engine=self.engine)
            self.assertEqual(rc, 4, out)
            self.assertLess(time.time() - t, self.budget + 0.2)
            self.assertEqual(sorted(os.listdir(d)), ['matrix.bin'])

    def test_late_solve_keeps_files(self):
        """A late second solve in the folder of a finished one leaves its .ops, .piv, .vs and .json as they were."""
        real = linalg.run_engine
        def engine_late(args, budget=None, *a, **k):        # the deadline counted from the program's start
            return real(args, runtime.Budget(seconds=self.budget), *a, **k)
        with tempfile.TemporaryDirectory(prefix='rb_deadline') as d, \
                mock.patch.object(linalg, 'ENGINE', self.engine):
            first = linalg.solve(chain(3000), d, 0, keep=-1)
            self.assertIsNone(first.late)
            self.assertIsNotNone(first.bound)
            names = ['elim' + e for e in ('.ops', '.piv', '.vs', '.json')]
            before = contents(d, names)
            with mock.patch.object(linalg, 'run_engine', engine_late):
                r = linalg.solve(self.M, d, 0, keep=-1)
            self.assertIsNotNone(r.late)
            self.assertIsNone(r.bound)
            self.assertEqual(contents(d, names), before)

class TestRoots(unittest.TestCase):
    def test_zero_limit(self):
        d = tempfile.mkdtemp(prefix='rb_roots')
        try:
            out = os.path.join(d, 'empty.roots')
            for lim in ('0', '1'):
                r = subprocess.run([SIEVE, 'roots', '-1,-1,0,1', lim, out], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                with open(out) as f:
                    self.assertEqual([l for l in f if not l.startswith('#')], [])
            r = subprocess.run([SIEVE, 'roots', '-1,-1,0,1', '30', out], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            want = ['%d: %s\n' % (p, ','.join(map(str, rs))) for p in range(2, 31)
                    if all(p % q for q in range(2, p)) for rs in [sieve.roots([-1, -1, 0, 1], p)] if rs]
            with open(out) as f:
                self.assertEqual([l for l in f if not l.startswith('#')], want)
            for bad in ('-1', 'x', '', '4294967296'):
                r = subprocess.run([SIEVE, 'roots', '-1,-1,0,1', bad, out], capture_output=True, text=True)
                self.assertNotEqual(r.returncode, 0, bad)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
