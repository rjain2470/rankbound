"""Fast tests of the sieve: las's aborts and the trial, the dummy side's root, the route to the line siever,
discarded special-q and shifted polynomials (las faked or not run: no CADO-NFS needed)."""
import gzip, io, math, os, random, subprocess, tempfile, unittest
from rankbound import field, runner, sieve, state


class TestLasCrash(unittest.TestCase):
    """CADO's "p = N does not divide" abort (ICARM #184, #682 with two large primes): mfb0 = lpb0 while lim0 = 2^lpb0
    (no second large prime is possible); otherwise the trial lowers lim0 until las runs; a production slice that
    aborts is recorded as empty, another failure is an error, and frequent aborts stop the run."""

    def test_mfb(self):
        for lpb, cap, mfb in ((17, None, 17), (26, None, 26), (27, None, 54), (17, 2 ** 16, 34)):
            prm = sieve.default_params(lpb, lim_cap=cap)
            self.assertEqual(prm['mfb0'], mfb, (lpb, cap))

    def test_trial(self):
        F = {'GB': 9650, 'ns': 1205, 'log2_skew': 2.3, 'poly': [7, -5, 3, 2], 'real_roots': [1]}
        def rate(self, name, prm, nq=10):
            if prm['lim0'] > 2 ** 17:
                raise sieve.LasCrash('Error, p = 138 does not divide', prm['lim0'])
            return {'rate': 100.0 * 2 ** (prm['lpb0'] - 18), 'sec_per_q': 0.3, 'rels_per_q': 30, 'mem_MB': 100}
        real, sieve.Trial.rate = sieve.Trial.rate, rate
        try:
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.tune(d, F, mem_gb=2.0, log=lambda *a: None)
                self.assertLessEqual(prm['lim0'], 2 ** 17); self.assertGreaterEqual(prm['lim0'], F['GB'])
                self.assertEqual(prm['trial']['lim_cap'], prm['lim0'])
            F['GB'] = 2 ** 17 + 1                    # cannot go below GB: an error
            with tempfile.TemporaryDirectory() as d:
                self.assertRaisesRegex(RuntimeError, 'p = 138 does not divide', sieve.tune, d, F, 2.0, lambda *a: None)
        finally:
            sieve.Trial.rate = real

    def test_trial_other_abort(self):
        """In the trial only the two-large-prime bug lowers lim0; another las abort (the dummy side's degree bug) is an
        error at once."""
        F = {'GB': 9650, 'ns': 1205, 'log2_skew': 2.3, 'poly': [7, -5, 3, 2], 'real_roots': [1]}
        calls = []
        def fake(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, -6, '', 'code BUG() : condition fij->deg == cpoly[side]->deg failed')
        real = sieve.subprocess.run, sieve.Las.prepare, sieve.binary
        sieve.subprocess.run, sieve.Las.prepare, sieve.binary = fake, lambda self, state, prm, poly: ('p', 'f0', 'f1'), str
        try:
            with tempfile.TemporaryDirectory() as d:
                self.assertRaisesRegex(RuntimeError, r'^CADO-NFS failed on this curve \(las, exit -6\): code BUG', sieve.tune, d, F, 2.0, lambda *a: None)
        finally:
            sieve.subprocess.run, sieve.Las.prepare, sieve.binary = real
        self.assertEqual(len(calls), 1)

    def test_slice(self):
        class P:
            def __init__(self, rc, err):
                self.returncode, self.stderr = rc, io.BytesIO(err)
        with tempfile.TemporaryDirectory() as d:
            S = state.State(d)
            tally = {'done': 20, 'crash': 0}
            runner.slice_failed(S, P(-6, b'# Error, p = 138 does not divide at (N,x) = (44,1175)'), 'F', 100, 200,
                             os.path.join(S.sdir, 'F_100.las'), tally, lambda *a: None)
            self.assertTrue(S.done('F_100')); self.assertEqual(S.slice_stats(), (0, 0)); self.assertEqual(tally['crash'], 1)
            with self.assertRaisesRegex(RuntimeError, '^The siever failed: some other error$'):
                runner.slice_failed(S, P(1, b'some other error'), 'F', 200, 300, 'x', tally, lambda *a: None)
            with self.assertRaisesRegex(RuntimeError, "^CADO-NFS's siever crashed on 11 of 31 parts"):
                for k in range(10):
                    runner.slice_failed(S, P(-6, b'does not divide'), 'F', 1000 + k, 1001 + k, 'x', tally, lambda *a: None)



class TestDummyRoot(unittest.TestCase):
    def test_root(self):
        """Side 1 is x - m with no special-q dividing F(m, 1); with m = 0 las aborts on ICARM #3911, where q = 17539
        divides c0.  That abort is also a skippable las crash."""
        f = [96370438343, -509099301417, -176287699, 236436000]          # #3911's F (c0..c3)
        self.assertEqual(f[0] % 17539, 0)
        m = sieve.dummy_root(f)
        self.assertIsNotNone(m); self.assertNotEqual(m, 0)
        n = abs(field.norm(f, m, 1))
        for p in range(2, 2 ** 10):
            while n % p == 0:
                n //= p
        self.assertTrue(n == 1 or (n > 2 ** 32 and sieve.probable_prime(n)))
        self.assertTrue(all(n % q for q in (17539, 68281, 33703, 323899)))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'F.poly')
            sieve.write_poly({'coeffs': f, 'skew': 45}, path)
            self.assertEqual(sieve.side1_root(path), m)
            with open(path) as fp:
                self.assertIn('n: %d\n' % abs(field.norm(f, m, 1)), fp.read())
        self.assertEqual(sieve.fb1_name('F', 0), 'F.fb1.gz')               # states made with m = 0 keep theirs
        self.assertEqual(sieve.fb1_name('F_r1', -3), 'F_r1.fb1_m-3.gz')
        self.assertEqual([n for n in range(60) if sieve.probable_prime(n)],
                         [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59])
        self.assertFalse(sieve.probable_prime(3215031751))                 # a strong pseudoprime to bases 2, 3, 5, 7
        self.assertTrue(sieve.probable_prime(2 ** 61 - 1))
        err = ("terminate called after throwing an instance of 'std::runtime_error'\n  what():  code BUG() : "
               "condition fij->deg == cpoly[side]->deg failed at las-norms.cpp:228")
        self.assertTrue(sieve.las_crashed(-6, err))
        self.assertFalse(sieve.las_crashed(0, err))
        self.assertFalse(sieve.las_crashed(1, 'some other error'))



class TestLineBelow(unittest.TestCase):
    def test_route(self):
        """GB 2^11..2^12 (lpb_A 16) goes to las, never below lpb 14; GB up to 2^11 keeps the line siever."""
        F = {'GB': 4059, 'ns': 585, 'log2_skew': 5.0, 'poly': [7, -5, 3, 2], 'real_roots': [1]}
        tried = []
        def rate(self, name, prm, nq=10):
            tried.append(prm['lpb0'])
            return {'rate': 100.0, 'sec_per_q': 0.3, 'rels_per_q': 30, 'mem_MB': 100}
        real, sieve.Trial.rate = sieve.Trial.rate, rate
        try:
            with tempfile.TemporaryDirectory() as d:
                prm = sieve.tune(d, F, mem_gb=2.0, log=lambda *a: None)
            self.assertNotEqual(prm.get('siever'), 'line')
            self.assertEqual(min(tried), 14); self.assertEqual(prm['lpb0'], 14)
            lpbA = lambda GB: math.ceil(math.log2(GB)) + 4
            self.assertTrue(lpbA(2048) < sieve.LINE_BELOW <= lpbA(2049))       # GB 2048: still the line siever
        finally:
            sieve.Trial.rate = real



class TestDiscarded(unittest.TestCase):
    def test_shift_skew(self):
        """Shifted polynomials declare skew min(skew of F, 2^26): at skew 2^41 to 2^44 las discards every special-q."""
        F = {'log2_skew': 52.5, 'poly': [7, -5, 3, 2], 'real_roots': [3]}
        P = sieve.polys(F)
        self.assertEqual([p['skew'] for p in P], [round(2 ** 52.5), 2 ** 26])
        self.assertEqual(sieve.polys(dict(F, log2_skew=20))[1]['skew'], 2 ** 20)
        self.assertEqual(len(sieve.polys(dict(F, log2_skew=15))), 1)

    def test_discarded(self):
        """A field whose skew is far beyond q (ICARM #443, 2^94): las discards every special-q; the trial says so."""
        q = '# Now sieving side-0 q=8405149; rho=2799583\n'
        d = '# Discarding side-0 q=8405149; rho=2799583; a0=8405149; b0=0; a1=2799583; b1=1; raw_J=%u;\n'
        with tempfile.TemporaryDirectory() as t:
            for txt, want in ((q + d + q + d, True), (q + d + q, False), ('', False)):
                with open(os.path.join(t, 'o'), 'w') as f:
                    f.write(txt)
                self.assertEqual(sieve.stats(os.path.join(t, 'o'))['discarded'], want, txt)
        F = {'GB': 687631, 'ns': 55644, 'log2_skew': 94.2, 'poly': [7, -5, 3, 2], 'real_roots': [1]}
        def rate(self, name, prm, nq=10):
            return {'rate': 0.0, 'sec_per_q': 0.0, 'rels_per_q': 0, 'mem_MB': 40, 'discarded': True}
        real, sieve.Trial.rate = sieve.Trial.rate, rate
        try:
            with tempfile.TemporaryDirectory() as t:
                self.assertRaisesRegex(RuntimeError, "cannot handle this curve's cubic field", sieve.tune, t, F, 2.0, lambda *a: None)
        finally:
            sieve.Trial.rate = real



class TestShifted(unittest.TestCase):
    def test_shifted(self):
        """A shifted polynomial F_k(x, 1) = F(x + k, 1) (a lopsided field's, ICARM #356's F_r1 with k = -5725295679):
        F_k(a, b) = F(a + k b, b), and a slice of F_k is written in F's coordinates, (a + k b, b) with the primes of
        F_k(a, b) (sieve.relations in finish_slice).  The end-to-end runs sieve F alone (ICARM #1: log2 skew 15.25)."""
        from rankbound import linalg
        poly, rnd = [-5, 0, 1, 1], random.Random(11)
        for _ in range(200):
            a, b, k = rnd.randrange(-10 ** 6, 10 ** 6), rnd.randrange(1, 10 ** 4), rnd.randrange(-10 ** 10, 10 ** 10)
            self.assertEqual(field.norm(field.shifted(poly, k), a, b), field.norm(poly, a + k * b, b))
        k = -7
        Fk = field.shifted(poly, k)
        lines, want = [], []
        for a, b in ((3, 2), (-5, 3), (1, 1)):         # las's lines: a,b:side 0 (F_k):side 1, in F_k's coordinates
            m, H = abs(field.norm(Fk, a, b)), []
            for p in range(2, m + 1):
                while m % p == 0:
                    m //= p; H.append(p)
                if m == 1:
                    break
            hx = ','.join('%x' % h for h in H)
            lines.append('%d,%d:%s:1f\n' % (a, b, hx))
            want.append('%d,%d:%s' % (a + k * b, b, hx))
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'F_r1_100.las')
            with open(out, 'w') as f:
                f.writelines(lines + ['# Total cpu time 0.50s\n', '# Total 3 reports [0.167s/r, 3.0r/sq]\n'])
            self.assertEqual(sieve.finish_slice(d, 'F_r1_100', {'poly': 'F_r1'}, k, out), (0.5, 3))
            with gzip.open(os.path.join(d, 'F_r1_100.rels.gz'), 'rt') as f:
                got = f.read().splitlines()
        self.assertEqual(got, want)
        for l in got:
            a, b, H = sieve.parse_line(l)
            self.assertEqual(math.prod(H), abs(field.norm(poly, a, b)))


if __name__ == '__main__':
    unittest.main()
