"""The arithmetic's speedups: each test compares the new code with the old implementation (kept here as the
reference) on fixtures and random inputs, without PARI."""
import json, os, random, unittest
from math import prod
from fractions import Fraction
from rankbound import arith as A, curve as C, descent as D

NEW = {f: getattr(D, f) for f in ('lclass', 'legendre', 'char_primes')}   # legendre: its calls only
REF = json.load(open(os.path.join(os.path.dirname(__file__), 'data', 'reference.json')))


# ---------- the old implementations, as they were before the optimizations

def old_padic_roots(f, p, k):
    out = []

    def go(g, r0, m, depth):
        c = min(A.val(x, p) for x in g if x)
        g = [x // p ** c for x in g]
        if m >= k or depth > 4 * k:
            out.append(r0 % p ** k)
            return
        for y in A.roots_mod(g, p):
            if A.peval(A.pderiv(g), y, p) % p:
                M = p ** (k - m)
                z = y
                for _ in range(max(1, (k - m).bit_length() + 1)):
                    z = (z - A.peval(g, z, M) * pow(A.peval(A.pderiv(g), z, M), -1, M)) % M
                out.append((r0 + p ** m * z) % p ** k)
            else:
                go(A._shift(g, y, p), r0 + p ** m * y, m + 1, depth + 1)
    go(list(f), 0, 0, 0)
    return sorted(set(out))


def old_iroot(n, k):
    x = round(n ** (1.0 / k)) if n < 1 << 1000 else 1 << -(-n.bit_length() // k)
    while True:
        y = ((k - 1) * x + n // max(x, 1) ** (k - 1)) // k if x else 0
        if abs(y - x) <= 1:
            for z in (y - 1, y, y + 1, x):
                if z >= 0 and z ** k == n:
                    return z
            return None
        x = y


def old_power_base(n):
    for k in range(n.bit_length(), 1, -1):
        g = C.root(n, k)
        if g is not None and g > 1:
            return g, k
    return n, 1


def old_sign_at(f, N, s):
    d = len(f) - 1
    v = sum(c * N ** i << (s * (d - i)) for i, c in enumerate(f))
    return (v > 0) - (v < 0)


def old_solve3(M, v):
    M = [[Fraction(c) for c in r] + [Fraction(b)] for r, b in zip(M, v)]
    for c in range(3):
        piv = next(r for r in range(c, 3) if M[r][c])
        M[c], M[piv] = M[piv], M[c]
        for r in range(3):
            if r != c and M[r][c]:
                t = M[r][c] / M[c][c]
                M[r] = [a - t * b for a, b in zip(M[r], M[c])]
    return tuple(M[i][3] / M[i][i] for i in range(3))


def old_rkproof(a0, a, P, r, pts, PR, pmax):
    Q = C.rkmap(a0, a, P, pts)
    if r == 0:
        return Q, []
    D = C.cinv(a)[2]
    rows, keep = [], []
    for p in PR + [q for q in A.primes(3, pmax) if q not in PR]:
        if p < 3 or not A.is_prime(p) or D % p == 0:
            continue
        row = [old_two_sig(a, z, p) for z in Q]
        if row[0] is None:
            continue
        if A.insert(rows, sum(b << i for i, b in enumerate(row))):
            keep.append(p)
        if len(keep) == r:
            break
    return Q, keep


def old_lclass(xi, Ac, B, L, s, t, p):
    q = D.q4(xi, Ac, B, s, t)
    if q == 0:
        return 0
    u = L[0] * s * s + L[1] * t * t
    if p == 0:
        return NEW['lclass'](xi, Ac, B, L, s, t, 0)        # the real place is unchanged
    v = A.val(q, p)
    if v % 2:
        return 0
    w = q / Fraction(p) ** v
    if not A.is_square_qp(w, p):
        return 0
    if L[2] == 0:
        return u
    rel = 100 if p > 2 else 99
    N = A.val(L[2], p) + v // 2 + rel
    m = p ** N
    r = D.sqrt_padic(w, p, rel) * p ** (v // 2)
    l = (u + L[2] * r) % m
    if l == 0:
        return 0
    vl = A.val(l, p)
    k = N - vl
    if k < (3 if p == 2 else 1):
        return 0
    return p ** vl * (l // p ** vl % p ** k)


def old_char_primes(mm, T, R, PR, pmax):
    if len(R) == 0:
        return []
    Dc = C.cinv(mm)[2]
    cols, keep = [], []
    for p in PR + [q for q in A.primes(3, pmax + 1) if q not in PR]:
        if p < 3 or not A.is_prime(p) or Dc % p == 0:
            continue
        new = [[D.chi(T, P, p, e) for P in R] for e in A.roots_mod(T, p)]
        if A.rank2([[c[i] for c in cols + new] for i in range(len(R))]) > A.rank2([[c[i] for c in cols]
                                                                                for i in range(len(R))]):
            cols += new
            keep.append(p)
        if cols and A.rank2([[c[i] for c in cols] for i in range(len(R))]) == len(R):
            return keep
    return None


def same_class(c1, c2, p):
    return (not c1 and not c2) or (c1 and c2 and A.is_square_qp(Fraction(c1) / Fraction(c2), p))


def rfrac(rng, h=50):
    return Fraction(rng.randint(-h, h), rng.randint(1, 12))


class TestArith(unittest.TestCase):
    def test_padic_roots(self):                     # the same roots as the fixed-modulus Newton, simple and multiple
        rng = random.Random(1)
        for _ in range(300):
            p = rng.choice([2, 3, 5, 7, 11, 101])
            k = rng.choice([1, 2, 5, 23, 64, 200])
            rs = [rng.randint(-10 ** 6, 10 ** 6) for _ in range(rng.randint(1, 4))]
            f = [rng.randint(-5, 5) or 1]
            for r in rs + rs[:rng.randint(0, 1)]:          # sometimes a double root
                f = [a - r * b for a, b in zip([0] + f, f + [0])]
            f = [c * rng.choice([1, 1, p, p * p]) for c in f]
            if rng.random() < 0.3:
                f[0] += rng.randint(-3, 3) * p ** rng.randint(0, 4)
            if any(f):
                self.assertEqual(A.padic_roots(f, p, k), old_padic_roots(f, p, k), (f, p, k))
        g = [-2, 0, 1]                                      # sqrt(2) in Z_7 to 300 digits
        r = A.padic_roots(g, 7, 300)
        self.assertEqual(len(r), 2)
        self.assertTrue(all(A.peval(g, x) % 7 ** 300 == 0 for x in r))

    def test_iroot_power_base(self):                # upper-bound Newton, prime exponents, the largest exponent kept
        rng = random.Random(2)
        cases = [0, 1, 2, 3, 4, 8, 9, 2 ** 64, 3 ** 40 * 5 ** 40, (2 ** 61 - 1) ** 6, 10 ** 300]
        for _ in range(300):
            b, e = rng.randint(2, 10 ** rng.randint(1, 30)), rng.randint(1, 12)
            cases += [b ** e, b ** e + rng.choice([-1, 1]), b ** e * rng.randint(2, 50)]
        for n in cases:
            for k in range(1, 14):
                self.assertEqual(A.iroot(n, k), old_iroot(n, k), (n, k))
            if n.bit_length() < 400:
                self.assertEqual(C.power_base(n), old_power_base(n), n)
        self.assertEqual(C.power_base(2 ** 360), (2, 360))
        self.assertEqual(C.power_base(6 ** 35 * 10 ** 0), (6, 35))
        for _ in range(30):                             # _split's power branch: the same proper factor
            b, e = rng.randint(3, 10 ** 12) | 1, rng.randint(2, 9)
            n = b ** e
            r0 = next((old_iroot(n, k) for k in range(2, n.bit_length()) if old_iroot(n, k)), None)
            self.assertEqual(A._split(n), r0)

    def test_ecm_table(self):                       # seeded ECM finds the same factors with the table to B1 only
        rng = random.Random(3)
        for _ in range(12):
            n = A.next_prime(rng.randint(10 ** 9, 10 ** 11)) * A.next_prime(rng.randint(10 ** 11, 10 ** 13))
            A.random.seed(n)
            g = A._ecm(n, 2000, 25)
            self.assertTrue(g is None or (1 < g < n and n % g == 0))
        self.assertEqual(len(A.primes(2, 200001)), 17984)

    def test_sign_at(self):                         # Horner gives the same exact sign; the roots unchanged
        rng = random.Random(4)
        for _ in range(500):
            f = [rng.randint(-10 ** 6, 10 ** 6) for _ in range(rng.randint(1, 9))]
            N, s = rng.randint(-10 ** 9, 10 ** 9), rng.randint(0, 80)
            self.assertEqual(A._sign_at(f, N, s), old_sign_at(f, N, s))
        f = [-2, 0, 0, 1]
        x, = A.real_roots_floor(f, 100)                 # floor(2^(1/3) 2^100)
        self.assertTrue(x ** 3 <= 2 ** 301 < (x + 1) ** 3)

    def test_solve3_inverse(self):                  # the same exact Fractions as Gaussian elimination
        rng = random.Random(5)
        for _ in range(500):
            M = [[rfrac(rng) for _ in range(3)] for _ in range(3)]
            v = [rfrac(rng) for _ in range(3)]
            if A.det3(M) == 0:
                with self.assertRaises(StopIteration):
                    A.solve3(M, v)
                continue
            x = A.solve3(M, v)
            self.assertEqual(x, old_solve3(M, v))
            self.assertTrue(all(isinstance(c, Fraction) for c in x))
            I = A._inverse(M)
            old = [[old_solve3(M, e)[i] for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])] for i in range(3)]
            self.assertEqual(I, old)
        self.assertEqual(A.solve3([[0, 1, 0], [1, 0, 0], [0, 0, 2]], [1, 2, 3]), (2, 1, Fraction(3, 2)))


class TestRankProof(unittest.TestCase):
    def test_rank22(self):                          # ICARM #8 (r = 22): the same points and primes, any PR
        c = REF['icarm8']
        rl = c['rank_lower_bound']
        a, P = C._ainvs5(c['curve']['ainvs']), [int(p) for p in c['curve']['bad_primes']]
        pts = [[Fraction(x) for x in p] for p in rl['points']]
        PR = [int(p) for p in rl['primes']]
        rng = random.Random(6)
        for prs in (PR, PR + PR, PR[:15], [], [2, 3, 4, 9] + PR[::-1], rng.sample(PR, 10) * 2):
            self.assertEqual(C.rkproof(a, a, P, 22, pts, prs, 10 ** 6), old_rkproof(a, a, P, 22, pts, prs, 10 ** 6))
        self.assertEqual(C.rkproof(a, a, P, 22, pts, PR[:5], 200), old_rkproof(a, a, P, 22, pts, PR[:5], 200))
        self.assertEqual(C.rkproof(a, a, P, 22, pts, PR, 10 ** 6)[1], [int(p) for p in rl['primes']])

    def test_small(self):                           # 37a1 (and its short model), points with denominators
        mm, short = [0, 0, 1, -1, 0], [0, 0, 0, -1296, 11664]
        P0 = (Fraction(0), Fraction(0))
        pts = [C.mul(mm, P0, k) for k in (1, 2, 3, 5, 7, 12)]
        for a0, a in ((mm, mm), (mm, short)):
            for prs in ([], [5, 7, 5], list(range(2, 60))):
                for Q in pts:                   # the old result where it proved rank 1; 2P0, 12P0 now halved
                    got, want = C.rkproof(a0, a, [37], 1, [Q], prs, 3000), old_rkproof(a0, a, [37], 1, [Q], prs, 3000)
                    if want[1]:
                        self.assertEqual(got, want)
                    else:
                        self.assertEqual((len(got[1]), C.on_curve(a, got[0][0])), (1, True))
        for p in A.primes(3, 400):
            for Q in pts:
                e = C._sig_root(C._sig_cubic(mm), p)
                self.assertEqual(None if e is None else C._sig_value(Q, p, e), old_two_sig(mm, Q, p))


def old_two_sig(a, P, p):
    b2, b4, b6, _ = C.binv(a)
    T = [16 * b6, 8 * b4, b2, 1]
    rt = A.roots_mod(T, p)
    if len(rt) != 1 or A.kronecker(A.peval(A.pderiv(T), rt[0], p), p) != -1:
        return None
    x, y = P
    if x.denominator % p == 0 or y.denominator % p == 0:
        return 0
    X = 4 * x.numerator * pow(x.denominator, -1, p) % p
    return 1 if X == rt[0] else int(A.kronecker(X - rt[0], p) == -1)


def curves(n=14):
    """y^2 = x^3 + a x^2 + b x for random small a, b, with their bad primes; and 34^2's congruent curve with points."""
    rng = random.Random(7)
    out = [([0, 0, 0, -34 * 34, 0], {'points': [['-16', '120'], ['-2', '48']]}),
           ([0, 0, 0, -36, 0], {'points': [['-3', '9']]}), ([0, 0, 0, -25, 0], None)]
    while len(out) < n:
        a, b = rng.randint(-20, 20), rng.randint(-60, 60)
        if b and a * a - 4 * b:
            out.append(([0, a, 0, b, 0], None))
    return [(e, [p for p, _ in A.factor(abs(C.cinv(e)[2]))], pr) for e, pr in out]


class TestDescent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):                            # the calls of real descents, and the descents themselves
        cls.calls = {f: [] for f in NEW}
        cls.runs = []
        for f in NEW:
            setattr(D, f, (lambda *a, _f=f: cls.calls[_f].append(a) or NEW[_f](*a)))
        try:
            for e, P, pr in curves():
                random.seed(1)
                cls.runs.append(D.descent(e, P, proof=pr))
        finally:
            for f in NEW:
                setattr(D, f, NEW[f])

    def test_whole(self):                           # the two together: the same descents as the old functions
        old = {'lclass': old_lclass, 'char_primes': old_char_primes}
        try:
            for f in old:
                setattr(D, f, old[f])
            for (e, P, pr), new in zip(curves(), self.runs):
                random.seed(1)
                self.assertEqual(D.descent(e, P, proof=pr), new, e)
        finally:
            for f in NEW:
                setattr(D, f, NEW[f])
        self.assertTrue(any(r.get('rank_lower_bound') for r in self.runs))

    def test_lclass(self):                          # the same square class (or both none) on every captured call
        finite = [a for a in self.calls['lclass'] if a[-1]]
        self.assertGreater(len(finite), 300)
        for a in finite:
            self.assertTrue(same_class(D.lclass(*a), old_lclass(*a), a[-1]), a)

    def test_lclass_cancellation(self):             # u + L3 r cancelled to p^e: short precision cannot see the
        rng = random.Random(8)                      # class beyond its digits, and the fallback gives the old one
        n = fallback = 0
        while n < 300:
            p = rng.choice([2, 3, 5, 7, 11])
            xi, Ac, B, t = rng.choice([1, -1, 2, -3, 5, 6]), rng.randint(-30, 30), rng.randint(-30, 30) or 1, rng.randint(0, 50)
            q = D.q4(xi, Ac, B, 1, t)
            if q == 0:
                continue
            v = A.val(q, p)
            w = q / Fraction(p) ** v
            if v % 2 or not A.is_square_qp(w, p):
                continue
            ex, c = rng.randint(1, 60), rng.choice([1, 2, 3, 5, 6, 7])
            r = D.sqrt_padic(w, p, 150) * p ** (v // 2)         # u + L3 r = p^ex c + O(p^150)
            for L in ([(-r + p ** ex * c) % p ** 160, 0, 1], [(-r + p ** ex * c) % p ** 160 * p * p, 0, p * p],
                      [(r + p ** ex * c) % p ** 160, 0, -1]):
                new, old = D.lclass(xi, Ac, B, L, 1, t, p), old_lclass(xi, Ac, B, L, 1, t, p)
                self.assertTrue(same_class(new, old, p), (xi, Ac, B, L, t, p))
                self.assertTrue(old)
                n += 1
                if ex > 8:                                  # beyond the short digits: uncertain there, so the full value
                    fallback += 1
                    self.assertEqual(D._lclass_l(L[0], L[2], w, v, p, 4 if p > 2 else 6), 0)
                    self.assertEqual(new, old)
        self.assertGreater(fallback, 100)

    def test_legendre(self):                        # a solution of every soluble conic (the captured ones too)
        rng = random.Random(9)
        cases = list(self.calls['legendre'])
        soluble = len(cases)
        ps = [2, 3, 5, 7, 11, 13]
        for _ in range(500):                        # soluble ones: x^2 a + y^2 b + z^2 c = 0 from a point
            x, y, z = [rng.randint(1, 40) for _ in range(3)]
            a = rng.choice([1, -1]) * prod(rng.choice(ps) ** rng.randint(0, 9) for _ in range(3))
            b = rng.choice([1, -1]) * prod(rng.choice(ps) ** rng.randint(0, 9) for _ in range(3))
            num = -(a * x * x + b * y * y)
            if num == 0 or num % (z * z):
                continue
            c = num // (z * z)
            cases.append((a, b, c, ps + [p for p, _ in A.factor(abs(c))]))
            soluble += 1
        for _ in range(200):                        # random ones, soluble or not
            co = [rng.choice([1, -1]) * prod(rng.choice(ps) ** rng.randint(0, 30) for _ in range(3)) for _ in '123']
            cases.append((*co, ps))
        self.assertGreater(len(cases), 300)
        for t, (a, b, c, P) in enumerate(cases):
            got = D.legendre(a, b, c, P)
            if t < soluble:
                self.assertIsInstance(got, list, (a, b, c))
            if got is not None:
                self.assertNotEqual(got, [0, 0, 0])
                self.assertEqual(a * got[0] ** 2 + b * got[1] ** 2 + c * got[2] ** 2, 0)

    def test_char_primes(self):                     # the same primes as the old search, its two stages included
        cases = list(self.calls['char_primes'])
        self.assertTrue(cases)
        rng = random.Random(10)
        for mm, T, R, PR, pmax in list(cases):
            for prs in ([], [3, 5, 7, 5, 3], rng.sample(range(2, 200), 20), PR + PR[::-1]):
                cases.append((mm, T, R, prs, pmax))
            cases.append((mm, T, R + R[-1:], PR, 2000))                 # dependent: None
        for mm, T, R, PR, pmax in cases:
            two = old_char_primes(mm, T, R, PR, min(pmax, 10 ** 4)) or old_char_primes(mm, T, R, PR, pmax)
            self.assertEqual(D.char_primes(mm, T, R, PR, pmax), two)


if __name__ == '__main__':
    unittest.main()
