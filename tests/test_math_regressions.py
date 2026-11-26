"""Arithmetic regressions: each test reproduces a failure of an earlier version of curve.py, arith.py, field.py or
descent.py and checks the repair, without PARI."""
import json, os, random, tempfile, time, unittest
from fractions import Fraction
from math import prod
from helpers import run_route
from rankbound import RankboundError, Stopped, TimeLimit, arith as A, curve as C, descent as D, field as F, runtime

ICARM1501 = ([1, 0, 0, -103261464630444172572903933816814229471272130,
              403780767639707610641424464011353774314390587505020052993877000900],
             [2, 3, 5, 7, 13, 17, 19, 23, 59, 79, 137, 223, 431, 701, 769, 2617])


def sign_at(f, N, M):
    v = sum(c * N ** i << (M * (len(f) - 1 - i)) for i, c in enumerate(f))
    return (v > 0) - (v < 0)


class TestMinimal(unittest.TestCase):
    def test_large_scaling_prime(self):           # scaling by 10007 (beyond the trial division to 10^4)
        for q in (9973, 10007, 10007 * 10009, 10007 ** 2 * 10009, 100003 ** 3):
            a = [0, 0, q ** 3, -q ** 4, 0]
            mm, v = C.minimal(a)
            self.assertEqual((mm, v[0]), ([0, 0, 1, -1, 0], q))
            P = C.from_model((0, 0), v)                     # (0, 0) on the minimal model, transported
            self.assertTrue(C.on_curve(a, P))
            self.assertEqual(C.to_model(P, v), (0, 0))
        c = C.from_ainvs([0, 0, 10007 ** 3, -10007 ** 4, 0])
        self.assertEqual((c.minimal_model, c.bad_primes), (['0', '0', '1', '-1', '0'], ['37']))

    def test_zero_c6(self):                         # j = 1728 scaled by 3 (c6 = 0) terminates, as PARI
        self.assertEqual(C.minimal([0, 0, 0, -81, 0]), ([0, 0, 0, -1, 0], [3, 0, 0, 0]))
        self.assertEqual(C.minimal([0, 0, 0, -3 ** 16, 0]), ([0, 0, 0, -1, 0], [81, 0, 0, 0]))
        a = C.rst([0, 0, 0, -3 ** 4 * 5 ** 4, 0], 7, 1, 2)       # translated and scaled presentation
        mm, v = C.minimal(a)
        self.assertEqual((mm, v[0]), ([0, 0, 0, -1, 0], 15))
        self.assertTrue(C.on_curve(a, C.from_model((1, 0), v)))


class TestTorsion(unittest.TestCase):
    def test_no_usable_prime(self):                 # every prime below 1000 is bad; PARI gives [2, 2]
        primes = A.primes(5, 1000)
        c = prod(primes)
        a = [0, 0, 0, -c * c, 0]
        self.assertEqual(C.torsion(a), [2, 2])
        z = C.from_ainvs(a, bad_primes=[2] + primes)
        self.assertEqual((z.torsion, z.two_torsion()), ([2, 2], True))

    def test_1501(self):                            # the same group and generators, without the huge psi_16
        t = time.time()
        g, gens = C.torsion_group(C.minimal(*ICARM1501)[0])
        self.assertLess(time.time() - t, 3)
        self.assertEqual(g, [8, 2])
        self.assertEqual(gens, [(5612805273203617033660, 31902216633809565296643371091670),
                                (Fraction(23162129063595149354719, 4), Fraction(-23162129063595149354719, 8))])

    def test_int_roots(self):                       # the exact bound still finds every integer root
        rng = random.Random(3)
        for _ in range(200):
            k = rng.randint(1, 30)
            rs = sorted({rng.randint(-10 ** k, 10 ** k) for _ in range(rng.randint(1, 4))})
            f = [rng.choice([1, -1]) * rng.randint(1, 10 ** rng.randint(0, 20))]
            for r in rs:                            # f * (x - r)
                f = [(f[i - 1] if i else 0) - r * (f[i] if i < len(f) else 0) for i in range(len(f) + 1)]
            f = [c * rng.randint(1, 5) for c in f] if rng.random() < .3 else f
            g = A.ptrim([f[0] + 1] + f[1:]) if rng.random() < .2 else f
            out = C.int_roots(g)
            self.assertTrue(all(A.peval(g, r) == 0 for r in out), g)                  # roots, exactly
            self.assertLessEqual({r for r in rs if A.peval(g, r) == 0}, set(out), g)  # and none missed


class TestRealRoots(unittest.TestCase):
    def test_huge_coefficient(self):                # no float overflow; the floor is exact
        f = [-10 ** 400, 1, 0, 1]
        out = A.real_roots_floor(f, 128)
        self.assertEqual(len(out), 1)
        self.assertEqual((sign_at(f, out[0], 128), sign_at(f, out[0] + 1, 128)), (-1, 1))

    def test_close_roots(self):                     # q x (3x - 1)^2 - 1 has three real roots
        for q in (2 ** 300, 2 ** 600):
            f = [-1, q, -6 * q, 9 * q]
            self.assertEqual(len(A.real_roots_floor(f, 128)), 3)
            self.assertEqual(len(A.roots(f, 250)[0]), 3)
        with self.assertRaises(ValueError):         # closer than 2^-124: an explicit precision failure
            A.roots([-1, 2 ** 300, -6 * 2 ** 300, 9 * 2 ** 300], 35)

    def test_complex_pair_large_real_root(self):    # x^3 + B x^2 + 1: the pair is ~ +-i/sqrt(B) (ICARM #568's form)
        from decimal import Decimal, localcontext
        for B in (10 ** 430 + 7, 3 ** 900):
            rr, cc = A.roots([1, 0, B, 1], 60 + 2 * len(str(abs(B))))
            (re, im), = cc
            with localcontext() as ctx:
                ctx.prec = 80
                self.assertLess(abs(im * im * abs(B) - 1), Decimal(10) ** -50)
                self.assertLess(abs(re * B * B * 2 - 1), Decimal(10) ** -50)

    def test_dyadic_and_random(self):               # exact dyadic roots, adjacent at 2^-200, and random cubics
        a = 3 ** 120
        r1, r2 = Fraction(a, 2 ** 200), Fraction(a + 1, 2 ** 200)
        f = [Fraction(1)]
        for r in (r1, r2, Fraction(-3)):
            f = [(f[i - 1] if i else 0) - r * (f[i] if i < len(f) else 0) for i in range(len(f) + 1)]
        f = [int(c * 2 ** 400) for c in f]
        self.assertEqual(A.real_roots_floor(f, 200), [-3 << 200, a, a + 1])
        rng = random.Random(5)
        for _ in range(300):
            f = [rng.randint(-10 ** 6, 10 ** 6) for _ in range(rng.randint(3, 6))]
            if not f[-1]:
                continue
            S = A.sturm(f)
            if len(S[-1]) > 1:
                continue                            # not squarefree
            out = A.real_roots_floor(f, 40)
            B = 1 << A.root_bound_bits(f)
            self.assertEqual(len(out), A._variations(S, -B, 0) - A._variations(S, B, 0))
            for N in out:                           # each floor brackets a root: a sign change or a root on the grid
                self.assertTrue(sign_at(f, N, 40) == 0 or sign_at(f, N, 40) != sign_at(f, N + 1, 40), (f, N))


class TestFactorDeadline(unittest.TestCase):
    def test_expired(self):                         # an expired budget does not split
        n = 549756813973 * 549757813903
        t = time.time()
        with self.assertRaises(TimeoutError):
            A.factor(n, seconds=0)
        with self.assertRaises(TimeoutError):
            A.factor(A.next_prime(10 ** 30) * A.next_prime(10 ** 31), seconds=0.3)
        self.assertLess(time.time() - t, 3)
        self.assertEqual(A.factor(n), [(549756813973, 1), (549757813903, 1)])


class TestForms(unittest.TestCase):
    def test_size(self):                            # finite in log space, within the skew window
        lg, sk = F.size((1, 0, 0, -2 ** 601))
        self.assertEqual((lg, sk), (451.0, 2.0 ** 100))
        n = 10 ** 400
        self.assertLess(F.size((n + 1, n + 2, n + 2, n + 2))[0], 1330)

    def test_reduce_identity(self):                 # reduced = form o matrix exactly
        rng = random.Random(11)
        forms = [(-1, 0, 1, 1)] + [tuple(rng.randint(-50, 50) for _ in range(4)) for _ in range(60)]
        for f in forms:
            if f[0] == 0 or F._apply(f, ((1, 0), (0, 1))) != f:
                continue
            if A.sturm([f[3], f[2], f[1], f[0]])[-1:] and len(A.sturm([f[3], f[2], f[1], f[0]])[-1]) > 1:
                continue
            try:
                G, M = F.reduce_form(f)
            except (ValueError, ZeroDivisionError):
                continue
            self.assertEqual(F._apply(f, M), G, f)
            self.assertGreater(G[0], 0)


class TestLocal(unittest.TestCase):
    def test_deep(self):                            # 3^1202 (PARI: insoluble), no RecursionError
        self.assertFalse(D.qp(2, 3 ** 1202, 3, 3))
        H = [3 ** 1201, 0, 2, 0, 3]
        self.assertFalse(A.zp(H, 3))
        self.assertIsNone(A.wit(H, 3))

    def test_witnesses(self):                       # wit's s gives a nonzero square in Q_p, and then zp holds
        rng = random.Random(7)
        found = 0
        for _ in range(400):
            p = rng.choice([2, 3, 5, 7, 11, 101])
            H = [rng.randint(-60, 60) * p ** rng.randint(0, 3) for _ in range(5)]
            if not H[-1] or len(A.sturm(A.ptrim(H))[-1]) > 1:
                continue
            w = A.wit(H, p)
            if w is not None:
                found += 1
                self.assertTrue(A.peval(H, w) and A.is_square_qp(A.peval(H, w), p), (H, p, w))
                self.assertTrue(A.zp(H, p))
        self.assertGreater(found, 100)


class TestDescent(unittest.TestCase):
    def test_deadline(self):                        # a deadline inside the descent stops it promptly, never a result
        a, P = ICARM1501
        for kw in ({'budget': runtime.Budget(seconds=0.005)}, {'seconds': 0.005}):
            t = time.monotonic()
            with self.assertRaises(TimeLimit):
                D.descent(a, P, r=2, **kw)
            self.assertLess(time.monotonic() - t, 0.05, kw)
        self.assertIsNone(runtime.current().end)          # the descent's limit is gone after it
        c = C.classified(C.from_ainvs(a, bad_primes=P, r=2))
        with tempfile.TemporaryDirectory() as d:
            t = time.monotonic()
            with self.assertRaises((Stopped, TimeLimit)):
                run_route(D.Descent(d), c, budget=runtime.Budget(seconds=0.02))
            self.assertLess(time.monotonic() - t, 0.02 + 0.05)
        self.assertEqual(D.descent(a, P, r=2)['S'], 2)

    def test_conflicting_r(self):                   # r = 2 with one point is refused, not run as r = 1
        cur = {'id': 'bad-r', 'ainvs': [0, 0, 0, -25, 0], 'bad_primes': [2, 5], 'r': 2,
               'proof': {'points': [['-4', '6']]}}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'list.json')
            with open(path, 'w') as f:
                json.dump({'curves': [cur]}, f)
            with self.assertRaises(ValueError):
                C.from_list(path)
            c = C.from_ainvs([0, 0, 0, -25, 0], bad_primes=[2, 5])
            with self.assertRaises(ValueError):
                C.with_points(c, [['-4', '6']], r=2)
            with self.assertRaises(RankboundError):
                D.descent([0, 0, 0, -25, 0], [2, 5], {'points': [['-4', '6']]}, r=2)
            import dataclasses
            bad = dataclasses.replace(C.with_points(c, [['-4', '6']]), r=2)
            with self.assertRaises(RankboundError):
                run_route(D.Descent(os.path.join(d, 'w')), bad)
            self.assertEqual(D.descent([0, 0, 0, -25, 0], [2, 5], {'points': [['-4', '6']]}, r=1)
                             ['rank_lower_bound']['r'], 1)


if __name__ == '__main__':
    unittest.main()
