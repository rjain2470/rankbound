"""Regressions of the 2-Selmer route (selmer, ctp, witness) and of check.gp's tdzp: each test reproduces a failure
of an earlier version and checks the repair, through the independent checker where a witness is involved."""
import dataclasses, gzip, hashlib, json, math, os, shutil, tempfile, time, tracemalloc, unittest
from fractions import Fraction as Fr
from rankbound import arith as A, check, ctp, curve as cv, descent, rankbound, selmer, witness, TimeLimit
TESTS = os.path.dirname(os.path.abspath(__file__))
from rankbound import runtime

DATA = os.path.join(TESTS, 'data')
E571 = [0, -1, 1, -929, -10595]             # rank 0, Sha = (Z/2)^2
E697691 = [0, -1, 1, -7117, 233485]         # 697691.a1: rank 2, Sha[2] = (Z/2)^2
QUIET = lambda *a: None
def load(path):
    with open(path) as f:
        return json.load(f)


FULL = bool(os.environ.get('RANKBOUND_TEST_FULL'))


def sel_witness(ai, name, r=None):
    c = cv.from_ainvs(ai, name=name)
    if r is not None:
        c = dataclasses.replace(c, r=r)
    return selmer.prove(c, log=QUIET)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='rb_selmer_regressions')
        self.addCleanup(shutil.rmtree, self.d, True)

    def checked(self, wit, name='w.witness.json.gz'):
        p = os.path.join(self.d, name)
        witness.write(wit, p)
        return p, check.check(p)


class TestCT(Tmp):
    """The pairing gate, the square-leading filter, coherent publication."""

    def test_partial_basis(self):
        """697691.a1 with r = 2 (no points): dim(Sel_2 cap V) = 3, S = 4.  The old gate (3 - min(r, 3) < 2) skipped the Cassels-Tate step
        (bound 4); the pairing on the partial basis has rank 2: bound 2 through the public API, checked."""
        res = rankbound(E697691, r=2, compute=True, workdir=os.path.join(self.d, 'job'))
        self.assertEqual((res.rank_bound, res.checked), (2, True))
        v = check.check(res.witness)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 2))

    def test_square_leading(self):
        """697691.a1: two of its quartics have square leading coefficients (rational points): two quartics are paired,
        one pair instead of six; the same checked bound 2."""
        c = ctp.add(sel_witness(E697691, 'c697691', r=2), log=QUIET)
        self.assertEqual((c['claims']['S'], c['claims']['S_ctp'], c['cassels_tate']['rank']), (4, 2, 2))
        self.assertEqual((len(c['_ctp']['complement']), c['_ctp']['paired']), (4, 2))
        self.assertEqual(len(c['cassels_tate']['pairs']), 1)
        for g in c['cassels_tate']['quartics']:
            a = int(g[0])
            self.assertFalse(a > 0 and math.isqrt(a) ** 2 == a)
        _, v = self.checked(c)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 2))

    def test_retry(self):
        """571a1: after the Cassels-Tate step (S 2 -> 0), a retry past its budget's end (after_selmer) or a second add keeps the section,
        S_ctp and statement together: the witness stays accepted with bound 0."""
        c = ctp.add(sel_witness(E571, 'c571a1'), log=QUIET)
        self.assertEqual(c['claims']['S_ctp'], 0)
        c = ctp.after_selmer(c, runtime.Budget(seconds=0))
        self.assertEqual((c['cassels_tate']['rank'], c['claims']['S_ctp']), (2, 0))
        c = ctp.add(c, seconds=0.01, log=QUIET)
        self.assertEqual((c['cassels_tate']['rank'], c['claims']['S_ctp']), (2, 0))
        _, v = self.checked(c)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 0))

    def test_roots_failure(self):
        """arith.roots's ValueError (roots closer than its precision) inside the Cassels-Tate step: the Cassels-Tate step is skipped, the 2-descent's bound stands
        (571a1: S = 2, no section), accepted; with an earlier section, that section stays."""
        c = sel_witness(E571, 'c571r')
        old = A.roots

        def close(*a, **k):
            raise ValueError('roots: two roots closer than the precision')
        A.roots = close
        try:
            c = ctp.add(c, log=QUIET)
        finally:
            A.roots = old
        self.assertEqual((c['cassels_tate'], c['claims']['S'], 'S_ctp' in c['claims']), (None, 2, False))
        _, v = self.checked(c)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 2))

    def test_decoded(self):
        """A witness read back (no _pairing): add keeps its coherent section; an incoherent one (S_ctp gone) becomes the 2-descent's
        claims (no section, the statement for S), which the checker accepts."""
        c = witness.read(os.path.join(DATA, '571a1.witness.json'))
        c = ctp.add(c, log=QUIET)
        self.assertEqual((c['cassels_tate']['rank'], c['claims']['S_ctp']), (2, 0))
        _, v = self.checked(c, 'a.json')
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 0))
        del c['claims']['S_ctp']
        c = ctp.after_selmer(c, runtime.Budget(seconds=0))
        self.assertIsNone(c['cassels_tate'])
        self.assertNotIn('S_ctp', c['claims'])
        _, v = self.checked(c, 'b.json')
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 2))


class TestClassX(unittest.TestCase):
    """_class_x exactly at negative valuations (a 2/5 example and a deep 2-adic one)."""

    def test_negative_valuation(self):
        # 2 w / W0 + R with w^2 = 1/25: +-2/5, both of class [1, 1] at 5 (the old float gave [0, 0])
        self.assertEqual(descent._class_x(Fr(0), Fr(1, 25), 1, 5), descent.loc(Fr(2, 5), 5))
        self.assertEqual(descent.loc(Fr(2, 5), 5), descent.loc(Fr(-2, 5), 5))
        # 2^-200: N = 150 + v was negative (a float modulus, TypeError); +-2^-99
        self.assertEqual(descent._class_x(Fr(0), Fr(1, 2 ** 200), 1, 2), descent.loc(Fr(2, 2 ** 100), 2))

    def test_against_exact(self):
        """Rational squares wsq = (a/b)^2 (either sign of w is allowed): the class is that of +-2 w / W0 + R."""
        for p in (2, 3, 5, 7):
            for a, b, R, W0 in ((6, 35, Fr(3, 7), 5), (1, p ** 3, Fr(1, p), 3), (p ** 2, 1, Fr(5, 1), 1),
                                (3, 2 * p ** 2, Fr(-1, 4), 7), (5, p, Fr(2, p ** 3), 11)):
                w = Fr(a, b)
                got = descent._class_x(R, w * w, W0, p)
                want = [descent.loc(s * 2 * w / W0 + R, p) for s in (1, -1) if s * 2 * w / W0 + R]
                self.assertIn(got, want, (p, a, b, R, W0))
        # the scaling by a rational square does not change the class
        self.assertEqual(descent._class_x(Fr(0), Fr(1, 25), 1, 5), descent._class_x(Fr(0), Fr(9, 25 * 49), 1, 5))


class TestSelmerSearch(unittest.TestCase):
    """One absolute budget (arith's time limit restored), the T2 Gram matrix, an exact quota."""

    def test_take(self):
        def gen():
            yield 1
            raise AssertionError('asked for a result beyond the quota')
        self.assertEqual(selmer._take(gen(), 1), [1])
        self.assertEqual(selmer._take(gen(), 0), [])
        self.assertEqual(selmer._take(iter([1, 2]), 5), [1, 2])

    def test_restored(self):
        """tri past its time limit raises, and leaves arith's time limit (the current budget) as it was (None,
        or the enclosing one); a later limit inside an earlier one keeps the earlier."""
        for before in (None, 3600):
            with runtime.limited(seconds=before):
                end = runtime.current().end
                with self.assertRaises(A.TimeUp):
                    selmer.tri([1, 0, -2, 1][::-1], [], [2], 2, 10 ** 5, 2, 0)
                self.assertEqual(runtime.current().end, end)
                with runtime.limited(seconds=7200):
                    if end is None:
                        self.assertIsNotNone(runtime.current().end)
                    else:
                        self.assertEqual(runtime.current().end, end)
                self.assertEqual(runtime.current().end, end)
        with runtime.scope(runtime.Budget(seconds=3600, label='run')):
            self.assertEqual(selmer.run_py(lambda: runtime.current().end is not None), True)   # tick's budget
        self.assertIsNone(runtime.current().end)

    def test_one_budget(self):
        """Four ranges of 0.4 s each on one core with a budget of 1 s: TimeLimit by about 1.2 s, the fourth range never
        started (the old per-range budgets let all four finish, 1.6 s)."""
        old = selmer.tri
        calls = []

        def slow(f, P, S, lo, hi, smax, seconds=None):
            calls.append(lo)
            t = time.monotonic()
            time.sleep(0.4)
            if time.monotonic() - t > seconds:
                raise A.TimeUp('The time limit was reached.')
            return 'NS\t%d\t%d\t0\n' % (lo, hi)
        selmer.tri = slow
        try:
            self.assertEqual(len(selmer.tri_ranges(2, 10 ** 5, 4)), 4)
            t = time.time()
            with self.assertRaises(TimeLimit):
                selmer.triangular({'poly': [1, 0, -2, 1]}, [], [2], 2, 10 ** 5, cores=1, seconds=1.0)
            self.assertLessEqual(len(calls), 3)
            self.assertLess(time.time() - t, 3.0)       # loose: a slow runner adds its own time
        finally:
            selmer.tri = old

    def test_t2basis(self):
        """ICARM #711's index form and its degree-1 prime above 82422216381668818301: the rounded Gram matrix of the
        HNF was indefinite (lll_gram failed); the order's form carried exactly is definite, the lattice the same."""
        k = selmer.K([79915280, -167827015, -12326811, 13637294], [])
        p = 82422216381668818301
        n = 0
        for Q in k.dec(p):
            if Q.f != 1:
                continue
            R = k.t2basis(Q)
            self.assertEqual(abs(A.det3(R)), abs(A.det3(Q.hnf)))
            for row in R:           # in the lattice of the HNF rows (upper triangular)
                rest = list(row)
                for i in range(3):
                    q, r = divmod(rest[i], Q.hnf[i][i])
                    self.assertEqual(r, 0)
                    rest = [a - q * b for a, b in zip(rest, Q.hnf[i])]
                self.assertEqual(rest, [0, 0, 0])
            n += 1
        self.assertGreater(n, 0)
        G = selmer.t2gram(k.f, k.F)
        self.assertGreater(A.det3(G), 0)

    @unittest.skipUnless(FULL, 'RANKBOUND_TEST_FULL: the full-S proof of ICARM #711 (~25 s)')
    def test_full_s(self):
        old = selmer.REDUCE_S
        selmer.REDUCE_S = False
        try:
            c = cv.from_ainvs([1, 0, 1, -222469142523859833, 34460003896785987245736268], name='icarm711',
                              bad_primes=[2, 3, 5, 11, 61, 67, 73, 1039, 8677847, 82422216381668818301], r=15)
            w = selmer.prove(c, log=QUIET)
        finally:
            selmer.REDUCE_S = old
        self.assertIn('82422216381668818301', w['s_primes'])
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        p = os.path.join(d, 'w.json.gz')
        witness.write(w, p)
        v = check.check(p)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 15))


class TestCodec(Tmp):
    """A format-4 witness read and written is format 5, labelled so; a second elimination format 5 cannot
    write is written in format 4, with its meaning."""

    def test_legacy(self):
        src = os.path.join(DATA, '37a1.witness.json')
        self.assertEqual(check.check(src)['verdict'], 'accepted')
        c = witness.read(src)
        self.assertEqual(c['format'], 'rankbound-selmer-witness-4')
        p, v = self.checked(c)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 1))
        with gzip.open(p, 'rt') as f:
            self.assertEqual(json.load(f)['format'], 'rankbound-selmer-witness-5')
        self.assertEqual(witness.read(p)['pivots_s'] and sorted(witness.read(p)['pivots_s']), sorted(c['pivots_s']))

    def weaker(self):
        """A conservative 571a1 witness: no Cassels-Tate step, no lower bound, no elements or characters, the class group
        elimination kept, the second one empty (k = ns = 6, U = U - k + 6 = 9, the element dimensions 0)."""
        c = load(os.path.join(DATA, '571a1.witness.json'))
        for k in ('cassels_tate', 'rank_lower_bound'):
            c.pop(k, None)
        cl = c['claims']
        cl.pop('S_ctp', None)
        ns = c['field']['bound']['ns']
        cl.update(U=cl['U'] - cl['k'] + ns, k=ns, v=0, dim_sel_cap_V=0)
        cl['S'] = min(cl['dim_sel_cap_V'] + cl['U'] - cl['v'], cl['S_bk'])
        c['elements'], c['characters'] = [], {'real': [], 'primes': []}
        c['ops_s'], c['pivots_s'] = [], []
        c['statement'] = witness.statement(cl['S'], c['hypothesis'], None)
        return c

    def test_not_superset(self):
        c = self.weaker()
        p0 = os.path.join(self.d, 'f4.json')
        with open(p0, 'w') as f:
            json.dump(c, f)
        v0 = check.check(p0)
        self.assertEqual(v0['verdict'], 'accepted', v0['reason'])
        self.assertIsNotNone(witness.packable(c))
        with self.assertRaises(witness.Unpackable):
            witness.encode(c)
        p, v = self.checked(witness.read(p0))
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', v0['rank_upper_bound']))
        with gzip.open(p, 'rt') as f:
            self.assertEqual(json.load(f)['format'], 'rankbound-selmer-witness-4')

    def test_superset_packs(self):
        """A superset second selection with its own ops_s packs in format 5 and is accepted (571a1)."""
        c = witness.read(os.path.join(DATA, '571a1.witness.json'))
        self.assertIsNone(witness.packable(c))
        p, v = self.checked(c)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 0))
        with gzip.open(p, 'rt') as f:
            self.assertEqual(json.load(f)['format'], 'rankbound-selmer-witness-5')


class TestChecker(Tmp):
    """One snapshot hashed and checked; the CT pair count first."""

    def test_snapshot(self):
        """The file is replaced (atomically, by 571a1's witness) right after check reads it: the verdict is for the
        bytes read (37a1, bound 1) with their hash."""
        p = os.path.join(self.d, 'w.json')
        shutil.copyfile(os.path.join(DATA, '37a1.witness.json'), p)
        with open(p, 'rb') as f:
            want = hashlib.sha256(f.read()).hexdigest()
        other = os.path.join(self.d, 'other.json')
        old = check.snapshot

        def racy(path, max_mb=None):
            data = old(path, max_mb)
            shutil.copyfile(os.path.join(DATA, '571a1.witness.json'), other)
            os.replace(other, path)
            return data
        check.snapshot = racy
        try:
            v = check.check(p)
        finally:
            check.snapshot = old
        self.assertEqual((v['verdict'], v['rank_upper_bound'], v['sha256']), ('accepted', 1, want))

    def test_max_mb(self):
        p = os.path.join(self.d, 'w.json')
        shutil.copyfile(os.path.join(DATA, '37a1.witness.json'), p)
        v = check.check(p, max_mb=0.001)
        self.assertEqual((v['verdict'], v['sha256']), ('limit', None))
        self.assertIn('--max-mb', v['reason'])

    def test_pairs_count(self):
        """1,500 quartics and no pairs: rejected at once, without the quadratic list of expected pairs."""
        c = load(os.path.join(DATA, '571a1.witness.json'))
        ct = c['cassels_tate']
        nv = len(ct['local'][0])
        bad = dict(ct, quartics=[ct['quartics'][0]] * 1500, local=[ct['local'][0]] * 1500, pairs=[])
        tracemalloc.start()
        try:
            with self.assertRaisesRegex(ValueError, 'one pair per'):
                check.ct_form(bad, nv)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, 100000)
        check.ct_form(ct, nv)       # the honest section still passes
        swapped = dict(ct, pairs=ct['pairs'][::-1]) if len(ct['pairs']) > 1 else None
        if swapped:
            with self.assertRaises(ValueError):
                check.ct_form(swapped, nv)
        three = dict(ct, quartics=[ct['quartics'][0]] * 3, local=[ct['local'][0]] * 3,     # 3 quartics: 3 pairs
                     pairs=[dict(ct['pairs'][0], i=i, j=j) for i, j in ((1, 2), (1, 3), (2, 3))])
        check.ct_form(three, nv)
        for pairs in (three['pairs'][::-1], [three['pairs'][k] for k in (0, 2, 1)]):
            with self.assertRaisesRegex(ValueError, 'in order'):
                check.ct_form(dict(three, pairs=pairs), nv)


class TestDeepLocal(unittest.TestCase):
    """check.gp's tdzp (iterative) on the deep separable example (3 x^4 + 2 x^2 + 3^1201):
    no recursion failure, the obstruction (0) as PARI's local test; smaller members of the family as before."""

    def test_deep(self):
        out = check.run_gp('print(tdqp(2, 3^1202, 3, 3), " ", tdqp(2, 3^12, 3, 3), " ", tdqp(2, 3^13, 3, 3));')
        self.assertEqual(out.split(), ['0', '0', '1'])


if __name__ == '__main__':
    unittest.main()
