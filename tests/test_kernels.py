"""The kernels of the linear algebra.  f2elim's matching selection, always on with a target and no characters: on
random matrices with targets at, above and below the full bound, the bound is the full one when that is above the
target, else at most the target; one f2elim run per solve, also when the selection's bound is short (it goes on from
the basis it has); its ops and pivots replayed by the checker's rank_V give its dim V, and its OUT.vs rows give it as
well; with characters no selection, so the crank is that of all the rows.  f2kernel's
threads are capped at KERNEL_THREADS, which never ran slower than more threads.  arith's factoring tries Pollard-Brent
briefly before ECM."""
import json, os, random, shutil, tempfile, time, unittest
from unittest import mock
from rankbound import arith, linalg
from rankbound.check import selmer as vsel


def f2rank(ints):
    """The rank over F_2 of the rows ints (Python ints, bit c: column c)."""
    basis = {}
    for x in ints:
        while x:
            h = x.bit_length() - 1
            if h not in basis:
                basis[h] = x
                break
            x ^= basis[h]
    return len(basis)


def random_matrix(rnd, ns, nl, extra=60, small=.4, least=1, gap=0):
    """A random Matrix of least to nl + ns + extra rows (fewer: repeats) with ns small columns and nl large ones, a
    fraction small of its entries in the small columns, none in the last gap of them (a bound of at least gap)."""
    M = linalg.Matrix({'J': []}, [(i, 0) for i in range(ns)])
    for _ in range(rnd.randint(least, nl + ns + extra)):
        k = rnd.randint(1, 6)
        cols = {rnd.randrange(ns - gap) if rnd.random() < small else ns + rnd.randrange(nl) for _ in range(k)}
        M._add([(c, 0) if c < ns else ('L', c) for c in cols], 'x')
    return M


def match_line(stdout):
    """f2elim's decision on the selection: 'kept', 'all rows' or None (no selection)."""
    l = next((l for l in stdout.splitlines() if l.startswith('match bound ')), None)
    return None if l is None else l.split(' ', 5)[5]


@unittest.skipUnless(os.path.exists(linalg.ENGINE), 'f2elim not built')
class TestMatch(unittest.TestCase):
    def test_match(self):
        rnd = random.Random(40)
        d = tempfile.mkdtemp(prefix='match')
        seen = {'kept': 0, 'all rows': 0, None: 0}
        try:
            for trial in range(300):
                if trial % 2:           # pools of a few rows
                    ns, nl = rnd.randint(2, 40), rnd.randint(1, 200)
                    M = random_matrix(rnd, ns, nl)
                else:                   # pools of more rows than the matching keeps
                    ns, nl = rnd.randint(40, 120), rnd.randint(20, 100)
                    M = random_matrix(rnd, ns, nl, rnd.randint(100, 400), .85, ns + nl + 100, rnd.randint(0, 8))
                ints = [sum(1 << c for c in r) for r in M.rows]
                full = ns - f2rank(ints) + f2rank([x >> ns for x in ints])
                target = max(1, full + rnd.choice([-2, 0, 0, 3]))
                w = os.path.join(d, 't%d' % trial)
                with mock.patch.object(linalg, 'run_engine', wraps=linalg.run_engine) as run:
                    r = linalg.solve(M, w, target, keep=-1, dense=rnd.choice([0, 1000]), threads=rnd.choice([1, 4]))
                self.assertEqual(run.call_count, 1, trial)          # no second elimination
                self.assertGreaterEqual(r.bound, full, trial)
                if full > target:
                    self.assertEqual(r.bound, full, trial)
                else:
                    self.assertLessEqual(r.bound, target, trial)
                kept, piv, _ = linalg.cone(r.ops, r.piv)
                names = ['%d:1' % c if c < ns else '%d:1:2' % c for c in range(len(M.col))]
                self.assertEqual(vsel.rank_V([set(x) for x in M.rows], names, kept, piv, 10 ** 9)[0], r.dimV, trial)
                after = list(ints)
                for dst, src in kept:
                    after[dst] ^= after[src]
                self.assertEqual(f2rank([after[i] for i in piv]), len(piv), trial)     # independent pivot rows
                with open(r.vs, 'rb') as f:
                    vs = f.read()
                rw = 8 * linalg.vs_words(len(r.held), ns)
                self.assertEqual(sum(1 for o in range(0, len(vs), rw) if not any(vs[o:o + 8 * ((len(r.held) + 63) // 64)])),
                                 r.dimV, trial)
                with open(os.path.join(w, 'elim.json')) as f:
                    how = match_line(json.load(f)['stdout'])
                seen[how] += 1
                if how == 'all rows':
                    self.assertEqual(r.bound, full, trial)
            self.assertGreater(seen['kept'], 20)
            self.assertGreater(seen['all rows'], 20)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_no_target_no_selection(self):
        """Without a target (0) the dense phase runs on every row at once."""
        rnd = random.Random(7)
        d = tempfile.mkdtemp(prefix='match')
        try:
            M = random_matrix(rnd, 20, 100)
            r = linalg.solve(M, d, 0, keep=-1)
            with open(os.path.join(d, 'elim.json')) as f:
                out = json.load(f)['stdout']
            self.assertFalse(any(l.startswith('match ') for l in out.splitlines()))
            self.assertIsNotNone(r.bound)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_characters_no_selection(self):
        """With characters (a check's elimination, whose crank is a lower bound) there is no selection, even with a
        target the bound reaches: the crank is that of all the rows, as without a target."""
        rnd = random.Random(11)
        Q = [q for q in range(1001, 2000, 2) if arith.is_prime(q)]
        d = tempfile.mkdtemp(prefix='match')
        try:
            for trial in range(10):
                ns, nl = rnd.randint(20, 60), rnd.randint(10, 60)
                chars = [(q, rnd.randrange(q)) for q in rnd.sample(Q, 16)]
                M = linalg.Matrix({'J': []}, [(i, 0) for i in range(ns)])
                for _ in range(ns + nl + 300):
                    while True:
                        a, b = rnd.randint(-10 ** 6, 10 ** 6), rnd.randint(0, 10 ** 4)
                        if all((a - b * r) % q for q, r in chars):
                            break
                    cols = set(rnd.sample(range(ns), rnd.randint(0, 3))) | {ns + c for c in rnd.sample(range(nl), 2)}
                    M._add([(c, 0) if c < ns else ('L', c) for c in cols], 'x', a, b)
                r0 = linalg.solve(M, os.path.join(d, '%d0' % trial), 0, keep=-1, chars=chars)
                r1 = linalg.solve(M, os.path.join(d, '%d1' % trial), ns, keep=-1, chars=chars)
                with open(os.path.join(d, '%d1' % trial, 'elim.json')) as f:
                    self.assertIsNone(match_line(json.load(f)['stdout']), trial)
                self.assertEqual((r1.bound, r1.dimV, r1.crank), (r0.bound, r0.dimV, r0.crank), trial)
                self.assertIsNotNone(r1.crank, trial)
                r2 = linalg.solve(M, os.path.join(d, '%d2' % trial), ns, keep=-1)     # the same rows without them
                with open(os.path.join(d, '%d2' % trial, 'elim.json')) as f:
                    self.assertEqual(match_line(json.load(f)['stdout']), 'kept', trial)
                self.assertLessEqual(r2.bound, ns, trial)
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(os.path.exists(linalg.ENGINE) and os.path.exists(linalg.KERNEL), 'f2elim or f2kernel not built')
class TestKernelThreads(unittest.TestCase):
    def test_cap(self):
        """linalg.kernel gives f2kernel at most KERNEL_THREADS = 16 threads, and the same vectors for any threads."""
        self.assertEqual(linalg.KERNEL_THREADS, 16)
        rnd = random.Random(3)
        rows = [sorted({rnd.randrange(10)} | {10 + rnd.randrange(400) for _ in range(rnd.randint(1, 4))})
                for _ in range(600)]
        got = {}
        for threads, want in ((1, 1), (8, 8), (16, 16), (64, 16), (126, 16)):
            with mock.patch.object(linalg, 'run_engine', wraps=linalg.run_engine) as run:
                k = linalg.kernel(rows, 410, 10, threads=threads)
            used = [int(c.args[0][2]) for c in run.call_args_list if c.kwargs.get('engine') == linalg.KERNEL]
            self.assertEqual(used, [want], threads)
            got[threads] = k.vectors
        self.assertTrue(got[1])
        self.assertTrue(all(v == got[1] for v in got.values()))


class TestFactorRho(unittest.TestCase):
    def test_one_short_try_before_ecm(self):
        """A composite of 80 bits or more gets one Pollard-Brent try of RHO_FIRST steps before ECM (two tries of 2^22
        took 16 s each on a 60-digit number); products with a factor of 10 to 15 digits still factor, quickly."""
        self.assertEqual(arith.RHO_FIRST, 1 << 16)
        n = 136166388099307833469386435137951813042139249674456383119617       # 16-digit * 44-digit
        with mock.patch.object(arith, '_rho', wraps=arith._rho) as rho:
            g = arith._split(n)
        self.assertTrue(1 < g < n and n % g == 0)
        self.assertEqual([c.args[1:] for c in rho.call_args_list], [(1, arith.RHO_FIRST)])
        rnd = random.Random(5)

        def prime(lo, hi):
            x = rnd.randrange(lo, hi)
            while not arith.is_prime(x):
                x += 1
            return x
        state = random.getstate()
        try:
            random.seed(1)
            for d in range(10, 16):
                p, q = prime(10 ** (d - 1), 10 ** d), prime(10 ** 39, 10 ** 40)
                t = time.monotonic()
                self.assertEqual(arith.factor(p * q), sorted([(p, 1), (q, 1)]), d)
                self.assertLess(time.monotonic() - t, 60, d)
        finally:
            random.setstate(state)


if __name__ == '__main__':
    unittest.main()
