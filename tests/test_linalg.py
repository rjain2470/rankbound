"""The F_2 linear algebra against a plain echelon in Python, on thousands of random sparse matrices with adversarial
shapes (empty, repeated and dense rows, empty and dense columns, rank deficiency, repeated pivots, ops with dst = src):
the checker's `check rank` (built also with every word cleared on threads) and the prover's
f2elim, whose witness (its ops and pivots) the checker replays.  `make test-asan` runs these under
AddressSanitizer and UndefinedBehaviorSanitizer."""
import array, os, random, shutil, subprocess, tempfile, unittest
from rankbound import linalg
from rankbound.check import selmer as vsel

HERE = os.path.dirname(os.path.abspath(__file__))


def rank(ints):
    """The rank over F_2 of the rows ints (Python ints, bit c: column c)."""
    piv = {}
    for x in ints:
        while x:
            h = x.bit_length() - 1
            if h not in piv:
                piv[h] = x
                break
            x ^= piv[h]
    return len(piv)


def random_rows(rnd, nrows, ncols):
    """Random rows (sets of columns < ncols) of one of several shapes."""
    shape = rnd.randrange(6)
    rows = []
    for _ in range(nrows):
        k = {0: 3, 1: 8, 2: max(1, ncols // 2), 3: 2, 4: 5, 5: 1}[shape]
        rows.append(set(rnd.sample(range(ncols), min(ncols, rnd.randint(0, k)))) if ncols else set())
    if rows and rnd.random() < 0.3:          # repeated rows, and sums of rows (rank deficiency)
        for _ in range(rnd.randint(1, nrows)):
            rows[rnd.randrange(nrows)] = set(rows[rnd.randrange(nrows)]) ^ set(rows[rnd.randrange(nrows)])
    if rows and ncols and rnd.random() < 0.3:        # a dense column
        c = rnd.randrange(ncols)
        for r in rows:
            if rnd.random() < 0.9:
                r.add(c)
    if rows and rnd.random() < 0.2:          # empty rows
        for _ in range(rnd.randint(1, 3)):
            rows[rnd.randrange(nrows)] = set()
    return rows


def expect(rows, nl, ops, piv):
    """(rank, rank of the columns < nl, dim V) of the pivot rows after the ops, by rank()."""
    R = [set(r) for r in rows]
    for d, s in ops:
        R[d] = R[d] ^ R[s]
    M = [sum(1 << c for c in R[i]) for i in piv]
    rM, rL = rank(M), rank([x & ((1 << nl) - 1) for x in M])
    return rM, rL, rM - rL


class TestF2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix='f2la')
        cls.small = os.path.join(cls.tmp, 'f2rank_small')      # every clearing on threads
        subprocess.run(['cc', '-O1', '-Wall', '-Werror', '-pthread', '-DPAR_MIN=0', '-o', cls.small,
                        vsel.CHECK + '.c'], check=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def f2rank(self, exe, rows, ncols, nl, ops, piv, threads):
        a = array.array('I', [len(rows), ncols, nl, len(ops), len(piv)])
        for r in rows:
            a.append(len(r)); a.extend(sorted(r))
        for o in ops:
            a.extend(o)
        a.extend(piv)
        p = subprocess.run([exe, 'rank', str(threads)], input=a.tobytes(), capture_output=True, timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        t = p.stdout.split()
        self.assertEqual(int(t[7]), int(t[1]) - int(t[3]))
        return int(t[1]), int(t[3]), int(t[7])

    def test_f2rank(self):
        """check rank's rank, rank on L and dim V = those of rank() after the ops, 2,000 random matrices, two builds."""
        rnd = random.Random(17)
        for trial in range(2000):
            big = trial % 10 == 0
            nrows, ncols = rnd.randint(0, 400 if big else 60), rnd.randint(1, 600 if big else 150)
            nl = rnd.randint(0, ncols)
            rows = random_rows(rnd, nrows, ncols)
            ops = [(rnd.randrange(nrows), rnd.randrange(nrows)) for _ in range(rnd.randint(0, 2 * nrows))] if nrows else []
            piv = [rnd.randrange(nrows) for _ in range(rnd.randint(0, nrows + 3))] if nrows else []
            want = expect(rows, nl, ops, piv)
            exe, th = (self.small, rnd.randint(1, 5)) if trial % 2 else (vsel.CHECK, rnd.randint(1, 3))
            self.assertEqual(self.f2rank(exe, rows, ncols, nl, ops, piv, th), want, trial)

    def test_f2rank_peeling(self):
        """Echelon-shaped matrices (a column of each row in no later row), which peeling takes apart, mixed with
        dense rows: both peeling rules and the dense step."""
        rnd = random.Random(19)
        for trial in range(300):
            ncols = rnd.randint(2, 200); nl = rnd.randint(0, ncols)
            cols = rnd.sample(range(ncols), rnd.randint(1, ncols))
            rows = []
            for k, c in enumerate(cols):       # row k: its own column c, and columns of later rows' own columns
                rows.append({c} | set(rnd.sample(cols[k + 1:], min(len(cols) - k - 1, rnd.randint(0, 4)))))
            rows += random_rows(rnd, rnd.randint(0, 30), ncols)
            rnd.shuffle(rows)
            piv = list(range(len(rows)))
            want = expect(rows, nl, [], piv)
            self.assertEqual(self.f2rank(vsel.CHECK if trial % 2 else self.small, rows, ncols, nl, [], piv, 3), want, trial)

    def test_f2rank_input(self):
        """Bad input: exit 2 with a message."""
        def run(words):
            return subprocess.run([vsel.CHECK, 'rank'], input=array.array('I', words).tobytes(), capture_output=True)
        for words in ([1, 2], [1, 2, 3, 0, 0], [1, 2, 1, 0, 1, 1, 2, 0], [1, 2, 1, 0, 1, 2, 1, 1, 0],
                      [1, 2, 1, 1, 0, 0, 0, 1], [1, 2, 1, 0, 1, 0, 1], [1, 3, 1, 0, 1, 4, 0, 1, 2, 0, 0]):
            p = run(words)
            self.assertEqual(p.returncode, 2, words)
            self.assertIn(b'check rank: ', p.stderr)

    def test_rank_V(self):
        """rank_V (labels to columns, small iff p:r with p < GB) = rank() on the pivot rows after the ops; with prefix, also
        the counts of the first pivots alone."""
        rnd = random.Random(23)
        for trial in range(200):
            nrows, ncols = rnd.randint(0, 50), rnd.randint(1, 80)
            names = ['%d:%d' % (rnd.choice([3, 5, 10 ** 6]), c) if rnd.random() < 0.8 else '%d::2' % c for c in range(ncols)]
            small = [z.count(':') == 1 and int(z.split(':')[0]) < 1000 for z in names]
            order = [c for c in range(ncols) if not small[c]] + [c for c in range(ncols) if small[c]]
            rows = random_rows(rnd, nrows, ncols)
            ops = [(rnd.randrange(nrows), rnd.randrange(nrows)) for _ in range(rnd.randint(0, nrows))] if nrows else []
            piv = [rnd.randrange(nrows) for _ in range(rnd.randint(0, nrows))] if nrows else []
            pos = {c: k for k, c in enumerate(order)}
            rM, rL, dimV = expect([{pos[c] for c in r} for r in rows], ncols - sum(small), ops, piv)
            got = vsel.rank_V([set(r) for r in rows], names, ops, piv, 1000, jobs=rnd.randint(1, 3))
            self.assertEqual((got[0], got[2]), (dimV, rM), trial)
            # the first pre pivots alone, from the same run (the checker's G and k in one): as rank_V on them alone
            pre = rnd.randint(0, len(piv))
            both = vsel.rank_V([set(r) for r in rows], names, ops, piv, 1000, jobs=rnd.randint(1, 3), prefix=pre)
            alone = vsel.rank_V([set(r) for r in rows], names, ops, piv[:pre], 1000)
            self.assertEqual((both[0], both[1]), (got, alone), trial)

    def test_f2elim(self):
        """f2elim's bound = ns - dim V, and its witness (ops, pivots) replayed by rank_V gives dim V, on 400
        random matrices (trimmed and not, with 1 and 3 threads, merge to 0, 5 or 1000 dense columns)."""
        rnd = random.Random(29)
        d = tempfile.mkdtemp(prefix='f2elim')
        try:
            for trial in range(400):
                ns, nl = rnd.randint(1, 40), rnd.randint(1, 300)
                M = linalg.Matrix({'J': []}, [(i, 0) for i in range(ns)])
                for r in random_rows(rnd, rnd.randint(0, nl + ns + 30), ns + nl):
                    M._add([(c, 0) if c < ns else ('L', c) for c in r], 'x')
                ints = [sum(1 << c for c in r) for r in M.rows]
                rM, rL = rank(ints), rank([x >> ns for x in ints])
                if not M.rows:
                    continue
                res = linalg.solve(M, d, 0, dense=rnd.choice([0, 5, 1000]), threads=rnd.choice([1, 3]))
                self.assertEqual((res.bound, res.dimV), (ns - rM + rL, rM - rL), trial)
                kept, piv, _ = linalg.cone(res.ops, res.piv)
                names = ['%d:1' % c if c < ns else '%d:1:2' % c for c in range(len(M.col))]
                self.assertEqual(vsel.rank_V([set(r) for r in M.rows], names, kept, piv, 10 ** 9)[0], res.dimV, trial)
                if trial % 4 == 0:      # a trim only removes rows: the bound can only rise
                    self.assertGreaterEqual(linalg.solve(M, d, 0, keep=rnd.randint(0, 10)).bound, res.bound)
        finally:
            shutil.rmtree(d)

    def test_f2elim_dense(self):
        """f2elim's dense phase with hundreds of rows per word (its tables) and threads: the bound and the replay."""
        rnd = random.Random(31)
        d = tempfile.mkdtemp(prefix='f2elim')
        try:
            for trial in range(12):
                ns, nl = rnd.randint(50, 700), rnd.randint(20, 300)
                M = linalg.Matrix({'J': []}, [(i, 0) for i in range(ns)])
                rows = [set(rnd.sample(range(ns + nl), rnd.randint(5, 30))) for _ in range(rnd.randint(300, 1200))]
                for r in rows + rnd.sample(rows, 30):       # weights of 20 or more (few singletons); repeated rows
                    M._add([(c, 0) if c < ns else ('L', c) for c in r], 'x')
                ints = [sum(1 << c for c in r) for r in M.rows]
                rM, rL = rank(ints), rank([x >> ns for x in ints])
                res = linalg.solve(M, d, 0, dense=rnd.choice([10 ** 5, 50]), threads=rnd.choice([1, 4]))
                self.assertEqual((res.bound, res.dimV), (ns - rM + rL, rM - rL), trial)
                kept, piv, _ = linalg.cone(res.ops, res.piv)
                names = ['%d:1' % c if c < ns else '%d:1:2' % c for c in range(len(M.col))]
                self.assertEqual(vsel.rank_V([set(r) for r in M.rows], names, kept, piv, 10 ** 9, jobs=3)[0], res.dimV)
        finally:
            shutil.rmtree(d)

    def test_trim_keeps_light(self):
        """The trim (kernel_rows at keep = T) removes the heavy rows first: with 40 pairs of rows on a large column of
        their own among 400 rows of 10 large columns each, every element returned is a sum of pairs (no heavy row), and
        there are at least T of them
        (the 2-descent's elements stay small; a trim by clique size kept heavy rows, and the 2-descent of 0,-1,1,-929,-10595 lost
        an element)."""
        from rankbound import selmer
        rnd = random.Random(37)
        T = 8
        light = [[T + 1000 + i] for i in range(40) for _ in range(2)]
        heavy = [[T + c for c in rnd.sample(range(200), 10)] for _ in range(400)]
        rk = heavy + light
        rnd.shuffle(rk)
        els, part = selmer.kernel_rows(rk, 0, T, T + 1040)
        self.assertGreaterEqual(len(els), T)
        self.assertTrue(all(len(rk[i]) == 1 for e in els for i in e), els)

    def test_k_of(self):
        """The 2-descent's k from the solve's basis (selmer.k_of, f2elim's OUT.vs and held columns) on 300 random matrices with
        small and large columns above S, large ones above a bad prime outside S and of degree 2 (held, not s_columns):
        it is the checker's k from format 4's ops_s and pivots_s (the solve's ops, its pivots and the unit rows), and
        untrimmed the exact k (rank of all rows with the unit rows); the held columns leave the bound exact."""
        from rankbound import selmer
        rnd = random.Random(47)
        d = tempfile.mkdtemp(prefix='f2elim')
        try:
            for trial in range(300):
                ns, nl = rnd.randint(4, 40), rnd.randint(1, 200)
                labs = [(i, 0) for i in range(ns)]          # small: 2 and 3 lie above S (S = {2, 3, P} or {2, P})
                for c in range(nl):
                    labs.append(rnd.choice([(10 ** 7 + c, 0)] * 6 + [(1000003, c), (1000003, c), (1000033, c),
                                                                         ('J', '1000003:k2:f%d' % c)]))
                M = linalg.Matrix({'J': [], 'bad_primes': [3, 1000003, 1000033]}, labs[:ns])
                rows = random_rows(rnd, rnd.randint(1, nl + ns + 30), ns + nl)
                for r in rows:
                    M._add([labs[c] for c in r], 'x')
                if not M.rows:
                    continue
                S = {2, 1000003} | ({3} if rnd.random() < 0.5 else set())
                lab = {c: z for z, c in M.col.items()}
                scols = [c for c in sorted(lab) if isinstance(lab[c][0], int) and lab[c][0] in S]
                keep = rnd.choice([-1, -1, rnd.randint(0, 10)])
                res = linalg.solve(M, d, 0, dense=rnd.choice([0, 5, 1000]), threads=rnd.choice([1, 3]), keep=keep)
                ints = [sum(1 << c for c in r) for r in M.rows]
                rM, rL = rank(ints), rank([x >> ns for x in ints])
                if keep < 0:
                    self.assertEqual(res.bound, ns - rM + rL, trial)
                k = selmer.k_of(res, ns, scols)
                kept, piv, _ = linalg.cone(res.ops, res.piv)
                names = ['%d:1' % c if c < ns else '%d:1:2' % c for c in range(len(M.col))]
                units = [{c} for c in scols]
                n = len(M.rows)
                dimVs = vsel.rank_V([set(r) for r in M.rows] + units, names, kept, piv + list(range(n, n + len(units))),
                                    10 ** 9)[0]
                self.assertEqual(k, ns - dimVs, trial)
                if keep < 0:
                    u = ints + [1 << c for c in scols]
                    self.assertEqual(k, ns - rank(u) + rank([x >> ns for x in u]), trial)
        finally:
            shutil.rmtree(d)

    def test_deadline(self):
        """f2elim with a deadline it cannot meet (its F2ELIM_DEADLINE from the budget, here one whose wall-clock end
        has passed): late, no bound, no output files; a budget with time changes nothing."""
        import time
        from rankbound import runtime

        class Past(runtime.Budget):
            def wall(self):
                return time.time() - 1
        rnd = random.Random(41)
        M = linalg.Matrix({'J': []}, [(i, 0) for i in range(20)])
        for r in random_rows(rnd, 200, 120):
            M._add([(c, 0) if c < 20 else ('L', c) for c in r], 'x')
        d = tempfile.mkdtemp(prefix='f2elim')
        try:
            r = linalg.solve(M, d, 0, budget=Past(seconds=3600))
            self.assertTrue(r.late is not None and r.bound is None and not os.path.exists(r.ops), r)
            self.assertEqual([f for f in os.listdir(d) if f.startswith('elim')], [])
            r2 = linalg.solve(M, d, 0, budget=runtime.Budget(seconds=3600))
            self.assertIsNone(r2.late)
            self.assertEqual(r2.bound, linalg.solve(M, d, 0, name='x').bound)
        finally:
            shutil.rmtree(d)

    def test_dense_s(self):
        """solve with dense_s (F2ELIM_DENSE_S): an elimination whose dense phase's estimate is above it is too_dense,
        with the estimate; with room, the bound of the solve without it."""
        rnd = random.Random(43)
        M = linalg.Matrix({'J': []}, [(i, 0) for i in range(20)])
        for r in random_rows(rnd, 200, 120):
            M._add([(c, 0) if c < 20 else ('L', c) for c in r], 'x')
        d = tempfile.mkdtemp(prefix='f2elim')
        try:
            r = linalg.solve(M, os.path.join(d, 'a'), 0, keep=-1, dense_s=1e-12)
            self.assertTrue(r.too_dense and r.bound is None and r.dense_s is not None and r.dense_s > 1e-12, r)
            r = linalg.solve(M, os.path.join(d, 'b'), 0, keep=-1, dense_s=3600)
            self.assertEqual(r.bound, linalg.solve(M, os.path.join(d, 'c'), 0, keep=-1).bound)
        finally:
            shutil.rmtree(d)


def _rank(vecs):
    """The rank over F_2 of vecs (Python ints)."""
    piv = {}
    for v in vecs:
        while v:
            h = v.bit_length() - 1
            if h not in piv:
                piv[h] = v
                break
            v ^= piv[h]
    return len(piv)


def _gauss_kernel(rk, nx, T, ncols, keep):
    """The 2-descent's kernel vectors as before f2kernel: f2elim's dense phase with T random tags, then selmer.small_pivots."""
    from rankbound import arith, selmer
    rnd = random.Random(1)
    bits = [rnd.getrandbits(T) | ((1 << (T + t)) if t < nx else 0) for t in range(len(rk))]
    E = linalg.eliminate(rk, ncols, T + nx, 1, bits=bits, keep=keep)
    comb = selmer.small_pivots(lambda i: (frozenset(rk[i]), bits[i]), E.ops, E.pivots, T + nx)
    return selmer.elements_of([[E.pivots[j] for j in arith.bits(cb)] for cb in comb], [x for o in E.ops for x in o])


class TestKernel(unittest.TestCase):
    """f2kernel (block Lanczos, or elimination for a small pool) and linalg.kernel / selmer.kernel_rows."""

    def setUp(self):
        self.env = os.environ.get('F2KERNEL_DENSE_N')

    def tearDown(self):
        if self.env is None:
            os.environ.pop('F2KERNEL_DENSE_N', None)
        else:
            os.environ['F2KERNEL_DENSE_N'] = self.env

    def pool(self, rows, dense_n, seed=1, threads=2):
        """f2kernel on rows as a pool file (row t has id 3 t + 7): its vectors as lists of t, and its result line."""
        d = tempfile.mkdtemp(prefix='f2kernel')
        try:
            a = array.array('I', [len(rows)])
            for t, r in enumerate(rows):
                a.extend([3 * t + 7, len(r)]); a.extend(r)
            with open(os.path.join(d, 'p'), 'wb') as f:
                a.tofile(f)
            r = subprocess.run([linalg.KERNEL, os.path.join(d, 'p'), os.path.join(d, 'o'), str(threads), str(seed), '0'],
                               capture_output=True, text=True, env=dict(os.environ, F2KERNEL_DENSE_N=str(dense_n)))
            self.assertEqual(r.returncode, 0, r.stderr)
            b = array.array('I')
            with open(os.path.join(d, 'o'), 'rb') as f:
                b.frombytes(f.read())
            out, t = [], 1
            for _ in range(b[0]):
                out.append([(x - 7) // 3 for x in b[t + 1:t + 1 + b[t]]]); t += 1 + b[t]
            return out, r.stdout
        finally:
            shutil.rmtree(d)

    def test_f2kernel(self):
        """On random sparse matrices (an empty row among them): every vector vanishes; elimination gives a basis of the
        whole left kernel; Lanczos, from two seeds and on 1 or 3 threads (the same output), independent vectors, all
        of a kernel of dimension up to about 60, else at least 48."""
        if not os.path.exists(linalg.KERNEL):
            self.skipTest('f2kernel not built')
        for n, m, w, seed in [(60, 50, 4, 1), (400, 380, 6, 2), (2000, 1950, 8, 3), (3000, 2800, 10, 5),
                              (5000, 4990, 12, 6)]:
            rnd = random.Random(seed)
            rows = [sorted(rnd.sample(range(m), rnd.randint(1, w))) for _ in range(n)]
            rows[5] = []
            kdim = n - _rank([sum(1 << c for c in r) for r in rows])
            for dense_n in (10 ** 6, 0):
                for sd in (1, 2):
                    vecs, info = self.pool(rows, dense_n, seed=sd)
                    for v in vecs:
                        x = set()
                        for t in v:
                            x ^= set(rows[t])
                        self.assertFalse(x, (n, dense_n))
                    rk = _rank([sum(1 << t for t in v) for v in vecs])
                    self.assertEqual(rk, len(vecs), info)
                    if dense_n:
                        self.assertEqual(rk, kdim, info)
                    else:
                        self.assertIn(' method lanczos ', info)
                        self.assertGreaterEqual(rk, min(kdim, 48), info)
                        self.assertEqual(self.pool(rows, 0, seed=sd, threads=1)[0], vecs)

    def test_kernel_kept(self):
        """linalg.kernel with a work folder keeps its files: the same rows again run neither f2elim nor f2kernel (a
        resumed the 2-descent), another seed only f2kernel, other rows both; the vectors are the same."""
        rnd = random.Random(7)
        rows = [sorted(rnd.sample(range(10, 400), rnd.randint(1, 5))) for _ in range(500)]
        d = tempfile.mkdtemp(prefix='kernel')
        calls, real = [], linalg.run_engine
        def counting(args, *a, **k):
            calls.append(os.path.basename(k.get('engine') or 'f2elim')); return real(args, *a, **k)
        linalg.run_engine = counting
        try:
            k1 = linalg.kernel(rows, 400, 10, work=d)
            k2 = linalg.kernel(rows, 400, 10, work=d)
            k3 = linalg.kernel(rows, 400, 10, seed=2, work=d)
            self.assertEqual(calls, ['f2elim', 'f2kernel', 'f2kernel'])
            self.assertEqual((k1.vectors, list(k1.ops)), (k2.vectors, list(k2.ops)))
            linalg.kernel(rows[1:], 400, 10, work=d)
            self.assertEqual(calls[3:], ['f2elim', 'f2kernel'])
            self.assertTrue(k3.vectors)
        finally:
            linalg.run_engine = real
            shutil.rmtree(d)

    def test_kernel_rows_vs_small_pivots(self):
        """selmer.kernel_rows (f2kernel) against the 2-descent's former kernel (f2elim's dense phase with tags, small_pivots) on
        random kernel matrices with xrel rows: every vector cancels its large columns, and both lie in the same kernel:
        the span of the former's vectors and of Lanczos's lies in the span of the elimination's (the whole kernel of the
        rows the trim leaves)."""
        from rankbound import selmer
        for trial in range(6):
            rnd = random.Random(100 + trial)
            T, nx, nl = 64, rnd.randint(0, 4), rnd.randint(300, 1500)
            n = nl + rnd.randint(80, 300)
            L = lambda c: T + nx + c
            rk = [sorted({L(c) for c in rnd.sample(range(nl), rnd.randint(1, 6))}) for _ in range(n)]
            rk[0] = []
            ncols = L(nl)
            out = {}
            for mode, dense_n in (('dense', 10 ** 6), ('lanczos', 0)):
                os.environ['F2KERNEL_DENSE_N'] = str(dense_n)
                out[mode] = selmer.kernel_rows(rk, nx, T, ncols)[0]
            out['gauss'] = _gauss_kernel(rk, nx, T, ncols, T)
            vec = lambda e: sum(1 << i for i in e)
            for mode, els in out.items():
                for e in els:
                    x = set()
                    for i in e:
                        x ^= set(rk[i])
                    self.assertFalse(x, (trial, mode))
            full = [vec(e) for e in out['dense']]
            for mode in ('lanczos', 'gauss'):
                self.assertEqual(_rank(full + [vec(e) for e in out[mode]]), _rank(full), (trial, mode))
            self.assertGreaterEqual(_rank(full), _rank([vec(e) for e in out['gauss']]))


if __name__ == '__main__':
    unittest.main()
