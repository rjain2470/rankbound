"""The 2-isogeny descent (rankbound/verify/two_descent.gp): known ranks, small curves against PARI's ellrank, local
duality, and the local solubility test against a brute-force oracle; its certificates (round trip, tampering), the
class group certificates of formats 2 and 3, and the command line on 14a1 and y^2 = x^3 - 25x."""
import json, os, shutil, subprocess, sys, tempfile, unittest
from rankbound import verify as verifier
from rankbound.verify import run_gp
from helpers import TESTS, cert0

ROOT = os.path.dirname(TESTS)
CLI = [sys.executable, '-m', 'rankbound']
ENV = dict({k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}, PYTHONPATH=ROOT)

LIBS = ('verify.gp', 'two_descent.gp')


def gp(body, seconds=600):
    """Return the output lines of body (GP source; a multi-line block in braces starting a line), run after
    verify.gp and two_descent.gp."""
    return run_gp(body, libs=LIBS, seconds=seconds).splitlines()


# td1(ai) = [B, S, t, 1 if every isogeny passes the duality and torsion checks], by the 2-isogeny descent alone
DESCENT = r'''
td1(ai) =
{
  my(E = ellinit(ai), P = factor(abs(E.disc))[, 1]~, M = tdmodel(ai, P), S = Set(concat([2], P)), D);
  D = vector(#M[3], i, tdisogeny(M[2], M[3][i], S));
  concat(tdbound(apply(d -> d[8], D), ellrootno(M[1])), [#M[3] \ 2 + 1, vecmin(apply(d -> d[9] && d[10], D))]);
}
'''

# td2(ai) = [B0, B1, the pairing ranks [[rk Theta_1, rk Theta_1'], ...] per isogeny, [r2 + s of ellrank on E and on
# each isogenous curve E'], [r2 of ellrank on E and on each E']]; tdpairing asserts that Theta_1 is alternating and
# has the 2-torsion image in its kernel
PAIRING = r'''
td2(ai) =
{
  my(E = ellinit(ai), P = factor(abs(E.disc))[, 1]~, M = tdmodel(ai, P), S = Set(concat([2], P)), D, Th, R, R2);
  D = vector(#M[3], i, tdisogeny(M[2], M[3][i], S));
  Th = vector(#D, i, my(d = D[i], ab2 = tdisog(d[2..3])); [tdpairing(d[2], d[3], S, d[4])[4], tdpairing(ab2[1], ab2[2], S, d[5])[4]]);
  R = ellrank(M[1]);
  R2 = vector(#D, i, my(ab = D[i][2..3]); ellrank(ellminimalmodel(ellinit([0, -2 * ab[1], 0, ab[1]^2 - 4 * ab[2], 0]))));
  [vecmin(apply(d -> d[8], D)), vecmin(vector(#D, i, D[i][8] - Th[i][1] - Th[i][2])), Th,
   concat([R[2] + R[3]], apply(z -> z[2] + z[3], R2)), concat([R[2]], apply(z -> z[2], R2))];
}
'''


class TestKnown(unittest.TestCase):
    """y^2 = x^3 - n^2 x (rank 0 for n = 1, 2, 3; rank 1 for n = 5, 6, 7), 14a1 (Z/6, rank 0) and n = 113, where
    ellrank gives [0, 2, 0]: 2-descent cannot see Sha[2] there, so S = 2."""

    def test_bounds(self):
        cases = [([0, 0, 0, -n * n, 0], S) for n, S in ((1, 0), (2, 0), (3, 0), (5, 1), (6, 1), (7, 1), (113, 2))]
        cases.append(([1, 0, 1, 4, -6], 0))
        out = gp(DESCENT + ''.join('print(td1(%s));\n' % a for a, _ in cases))
        for (a, S), l in zip(cases, out):
            B, S1, t, ok = eval(l)
            self.assertEqual((S1, t, ok), (S, 1 if a[0] else 2, 1), a)

    def test_points(self):
        """The characters of (-4, 6), (-3, 9), (25, 120) and the torsion generators prove rank >= 1 at the primes
        tdprimes picks; 2P + T adds nothing to P; a torsion point is not independent; a bad prime or a non-point
        fails tdvrank."""
        out = gp(r'''
{
  foreach([[-25, [-4, 6]], [-36, [-3, 9]], [-49, [25, 120]]], c,
    my(ai = [0, 0, 0, c[1], 0], PR = tdprimes(ai, [c[2]], [], 1000)); tdvrank(ai, 1, [c[2]], PR));
  E = ellinit([0, 0, 0, -25, 0]); P = [-4, 6]; Q = elladd(E, ellmul(E, P, 2), [5, 0]);
  print(tdrank(E, x^3 - 400*x, [P, Q], primes([3, 200]))[2], " ", tdrank(E, x^3 - 400*x, [P], primes([3, 200]))[2]);
  tdvrank([0, 0, 0, -25, 0], 1, [[-4, 6]], [5]);
  tdvrank([0, 0, 0, -25, 0], 1, [[-4, 7]], [3, 7]);
}
''')
        self.assertEqual([l.split()[2] for l in out[:3]], ['1', '1', '1'])
        self.assertEqual(out[3], '3 3')                     # rank t + 1 = 3 either way
        self.assertEqual([l.split()[2] for l in out[4:]], ['0', '0'])
        with self.assertRaisesRegex(RuntimeError, 'do not prove rank >= 2'):
            gp('tdprimes([0,0,0,-25,0], [[-4,6], [5,0]], [], 300);')


class TestSmallCurves(unittest.TestCase):
    """y^2 = x(x^2 + a x + b), |a|, |b| <= 9, and y^2 = x(x - e1)(x - e2), 0 < |e1| < |e2| <= 12: W_v and W'_v are
    orthogonal complements and b, b' lie in the Selmer groups; against ellrank [r1, r2, s]: B >= r2 + s (=
    dim Sel_2 - t) and S >= r2 >= r1, and S = B whenever B = r2 + s (then B = dim Sel_2 - t, of parity w).  B can
    exceed r2 by Sha[2]: y^2 = x(x^2 - 8x + 1) has [r1, r2, s] = [0, 0, 2] and B = S = 2."""

    def test_ellrank(self):
        out = gp(DESCENT + r'''
tdcmp(ai) =
{
  my(R = ellrank(ellminimalmodel(ellinit(ai))), d = td1(ai));
  if(d[4] != 1 || d[1] < R[2] + R[3] || d[2] < R[2] || (d[1] == R[2] + R[3] && d[2] != d[1]),
    print("BAD ", ai, " ", d, " ", R[1..3]));
  1;
}
{
  n = 0;
  for(a = -9, 9, for(b = -9, 9, if(b && a^2 != 4*b, n += tdcmp([0, a, 0, b, 0]))));
  for(e1 = -12, 12, for(e2 = -12, 12, if(e1 && e2 && abs(e1) < abs(e2), n += tdcmp([0, -e1 - e2, 0, e1 * e2, 0]))));
  print("N ", n);
}
''')
        self.assertEqual([l for l in out if l.startswith('BAD')], [])
        self.assertGreater(int(out[-1].split()[1]), 500)


class TestPairing(unittest.TestCase):
    """The isogeny pairings Theta_1: y^2 = x^3 - 113^2 x (both zero, B1 = 2); y^2 = x(x^2 - 8x + 1) (rank 2 on
    Sel(a', b'), B1 = 0 = r2(E')); curves y^2 = x(x^2 + a x + b) with ellrank's s > 0 on E or E': B1 >= the r2 of
    every curve in the isogeny class (a theorem: F17, Lemma 2.2), and B1 = the least of them on most; the matrix does
    not depend on the conic point or the local points."""

    def test_examples(self):
        out = gp(PAIRING + 'print(td2([0, 0, 0, -113^2, 0]));\nprint(td2([0, -8, 0, 1, 0]));')
        B0, B1, th, _, _ = eval(out[0])
        self.assertEqual((B0, B1, th), (2, 2, [[0, 0]] * 3))
        B0, B1, th, _, r2 = eval(out[1])
        self.assertEqual((B0, B1, th, min(r2)), (2, 0, [[0, 2]], 0))

    def test_small(self):
        out = gp(PAIRING + r'''
{
  my(n = 0, eq = 0, k = 0);
  for(a = -12, 12, for(b = -150, 150, if(!b || a^2 == 4*b, next);
    my(E = ellinit([0, a, 0, b, 0]), E2 = ellinit([0, -2*a, 0, a^2 - 4*b, 0]), R, R2, d);
    R = ellrank(E); R2 = ellrank(E2);
    if(R[3] == 0 && R2[3] == 0, next);
    k++; if(k % 4, next);
    d = td2([0, a, 0, b, 0]); n++;
    if(d[2] < vecmax(d[5]), print("BAD ", [a, b], " ", d));
    eq += d[2] == vecmin(d[5])));
  print("N ", n, " ", eq);
}
''')
        self.assertEqual([l for l in out if l.startswith('BAD')], [])
        n, eq = map(int, out[-1].split()[1:])
        self.assertGreater(n, 100)
        self.assertGreater(eq, n * 3 // 4)

    def test_selmer_identity(self):
        """dim Sel_2(E) - t = s' + (s - rank Theta_1) - 2 [F17, Lemma 2.2: the 2-descent bound is dim S_1 + dim S_2'
        - 2, S_2' the kernel of Theta_1 on Sel(a, b)], and the same for E' with Theta_1': against ellrank's r2 + s on
        both curves, for |a| <= 6, |b| <= 30."""
        out = gp(r'''
{
  my(n = 0);
  for(a = -6, 6, for(b = -30, 30, if(!b || a^2 == 4*b, next);
    my(ai = [0, a, 0, b, 0], E = ellinit(ai), P = factor(abs(E.disc))[, 1]~, S = Set(concat([2], P)), M, d, ab2, t1, t2,
       R, R2);
    M = tdmodel(ai, P); d = tdisogeny(M[2], M[3][1], S); ab2 = tdisog(d[2..3]);
    t1 = tdpairing(d[2], d[3], S, d[4])[4]; t2 = tdpairing(ab2[1], ab2[2], S, d[5])[4];
    R = ellrank(ellinit([0, d[2], 0, d[3], 0])); R2 = ellrank(ellinit([0, ab2[1], 0, ab2[2], 0])); n++;
    if(d[6] + d[7] - 2 - t1 != R[2] + R[3] || d[6] + d[7] - 2 - t2 != R2[2] + R2[3], print("BAD ", [a, b]))));
  print("N ", n);
}
''')
        self.assertEqual([l for l in out if l.startswith('BAD')], [])
        self.assertGreater(int(out[-1].split()[1]), 700)

    def test_choices(self):
        """Theta_1 with another conic point (Q(v) P0 - 2 <P0, v> v) and other local points (moved within their p-adic
        classes; other real points) is the same matrix."""
        out = gp(r'''
\\ another point than z in its class at p, of known L-class (p = 0: tdlocal's real point for this L)
tdmove(xi, A, B, L, z, p) =
{
  my(K, w);
  if(p == 0, return(tdlocal(xi, A, B, L, 0)));
  K = valuation(xi^2 * tdq(xi, A, B, z[1], z[2]), p) + 4;
  for(m = 1, 50, w = if(z[2] % p, [z[1] + m * p^K, z[2]], [z[1], z[2] + m * p^K]);
    if(tdlclass(xi, A, B, L, w[1], w[2], p), return(w)));
  error("no second point");
}
tdalt2(A, B, S, Xi) =
{
  my(V = tdplaces(S), C, C2, L1, L2, P1, P2, M1, M2);
  C = apply(xi -> tdconic(xi, A, B, S), Xi);
  C2 = vector(#Xi, i, my(P0 = C[i], xi = Xi[i], G = [xi, A / 2, 0; A / 2, B / xi, 0; 0, 0, -1], v = [1, 2, 3]~, Q);
    Q = (v~ * G * v) * P0~ - 2 * (P0 * G * v) * v; Q = Q / content(Q); Q~);
  L1 = vector(#Xi, i, tdtangent(Xi[i], A, B, C[i])); L2 = vector(#Xi, i, tdtangent(Xi[i], A, B, C2[i]));
  P1 = vector(#Xi, i, vector(#V, j, tdlocal(Xi[i], A, B, L1[i], V[j])));
  P2 = vector(#Xi, i, vector(#V, j, tdmove(Xi[i], A, B, L2[i], P1[i][j], V[j])));
  M1 = tdtheta(A, B, V, Xi, L1, P1); M2 = tdtheta(A, B, V, Xi, L2, P2);
  [M1 == M2, matrank(Mod(M1, 2))];
}
{
  my(cnt = 0, nz = 0);
  foreach([[-8, 1], [-11, -57], [3, -106], [0, -113^2], [-12, -109], [5, -84], [-6, -91]], ab,
    my(E = ellinit([0, ab[1], 0, ab[2], 0]), P = factor(abs(E.disc))[, 1]~, M = tdmodel(E[1..5], P),
       S = Set(concat([2], P)), d = tdisogeny(M[2], M[3][1], S), ab2 = tdisog(d[2..3]), r);
    foreach([[d[2], d[3], d[4]], [ab2[1], ab2[2], d[5]]], g, r = tdalt2(g[1], g[2], S, g[3]);
      cnt++; nz += r[2] > 0; if(!r[1], print("BAD ", ab, " ", g[1..2]))));
  print("N ", cnt, " ", nz);
}
''')
        self.assertEqual([l for l in out if l.startswith('BAD')], [])
        self.assertGreater(int(out[-1].split()[2]), 0)        # some nonzero pairings among them


class TestCasselsTate(unittest.TestCase):
    """The Cassels-Tate pairing on Sel_2 by Fisher's formula [F22, Thm. 3.1]: F22's Example 3.4 (571a1: m, gamma1 and
    <g1, g2> = 1 as in the paper; rank 2 on Sel_2 from ell2cover); its rank equals ellrank's s on curves
    y^2 = x(x^2 + a x + b) with s > 0 (j = 1728 included); the certificate of ICARM #1737 (Z/8 x Z/2, rank 0, where the
    isogeny pairings vanish and ellrank's s = 2) proves rank 0, and changed data fail."""

    def test_example_3_4(self):
        out = gp(r'''
{
  my(g1 = [-11, 68, -52, -164, -64], g2 = [-4, -60, -232, -52, -3], g3 = [-31, -78, 32, 102, -53], I = 44608,
     J = 18842960, F = tdcubic(I, J), m = [936032, -8656, 20] / 9, gam, S = [2, 571], X, Pts);
  print([tdI(g1), tdJ(g1), tdI(g2), tdJ(g2), tdI(g3), tdJ(g3)] == [I, J, I, J, I, J]);
  print(tdL(m, F)^2 == tdz(g1, F) * tdz(g2, F) * tdz(g3, F));
  gam = tdgamma(g1, g2, g3, m, I, F); print(gam == 4/9 * [5, -16, -12]);
  X = setminus(Set(concat(factor(abs(g2[1]))[, 1]~, factor(denominator(content(gam)))[, 1]~)), Set(tdplaces(S)));
  Pts = apply(p -> tdqpoint(g1, p, gam), concat(tdplaces(S), X));
  print(tdctpair(g1, g2, g3, m, I, F, S, X, Pts));
  addprimes(S); print(tdct(ellminimalmodel(ellinit([0, -1, 1, -929, -10595]))[1..5], S)[5]);
}
''')
        self.assertEqual(out, ['1', '1', '1', '1', '2'])

    def test_ellrank(self):
        out = gp(r'''
{
  my(n = 0);
  for(a = -12, 12, for(b = -150, 150, if(!b || a^2 == 4*b || (a + b) % 5, next);
    my(E = ellminimalmodel(ellinit([0, a, 0, b, 0])), R = ellrank(E), P, S, r);
    if(R[3] == 0, next);
    P = factor(abs(E.disc))[, 1]~; S = Set(concat([2], P)); removeprimes(addprimes()); addprimes(P);
    r = tdct(E[1..5], S)[5]; n++;
    if(r != R[3], print("BAD ", [a, b], " ", r, " ", R[1..3]))));
  print("N ", n);
}
''', seconds=900)
        self.assertEqual([l for l in out if l.startswith('BAD')], [])
        self.assertGreater(int(out[-1].split()[1]), 20)

    def test_fisher_10_1(self):
        """[F17, Example 10.1] (Z/12): the Selmer groups are F17's S_1' = <-272196179> and S_1 = <15, 73, 87, 231, 28619>,
        both isogeny pairings vanish (B = 4); the Cassels-Tate pairing is 0 on Sel_2(E) (Sha(E)[2^inf] = (Z/4)^4) and has
        rank 4 on Sel_2(E'): rank E(Q) = 0, as F17 proves with Theta_3."""
        from rankbound import rankbound
        with tempfile.TemporaryDirectory() as d:
            res = rankbound([0, 91502230365284038, 0, 489792722057841784540058275212361, 0], points=[],
                            bad_primes=[2, 3, 5, 7, 11, 29, 71, 73, 127, 28619, 30187], workdir=os.path.join(d, 'w'))
            with open(res.certificate) as f:
                c = json.load(f)
        (i,) = c['descent']
        self.assertEqual((i['selmer'], sorted(map(int, i['selmer_isogenous']))), (['-272196179'], [15, 73, 87, 231, 28619]))
        self.assertEqual((i['pairing']['rank'], i['pairing']['rank_isogenous'], c['cassels_tate']['rank'],
                          i['cassels_tate']['rank'], i['B']), (0, 0, 0, 4, 0))
        self.assertEqual((res.rank_bound, res.verified, res.torsion), (0, True, [12]))

    def test_certificate(self):
        from rankbound import rankbound
        with tempfile.TemporaryDirectory() as d:
            res = rankbound([1, 0, 0, -1319539461660, -159402536950172400], bad_primes=[2, 3, 5, 7, 11, 73, 103],
                            points=[], workdir=os.path.join(d, 'w'))
            self.assertEqual((res.rank_bound, res.proved_equal, res.verified), (0, True, True))
            with open(res.certificate) as f:
                raw = f.read()
            c0 = json.loads(raw)
            self.assertEqual(c0['cassels_tate']['rank'], 2)
            self.assertEqual([(i['pairing']['rank'], i['pairing']['rank_isogenous'], i['B']) for i in c0['descent']],
                             [(0, 0, 0)] * 3)
            def changed(f):
                c = json.loads(raw)
                f(c['cassels_tate'])
                return c
            def m(ct):
                ct['pairs'][-1]['m'][0] = str(int(ct['pairs'][-1]['m'][0].split('/')[0]) + 1)
            cases = {'sum quartic': (lambda ct: ct['pairs'][-1]['sum'].__setitem__(4, '7'), 'cassels_tate'),
                     'm': (m, 'cassels_tate'),
                     'point': (lambda ct: ct['pairs'][-1]['points'][0].__setitem__(0, '0'), 'cassels_tate'),
                     'rank': (lambda ct: ct.update(rank=4), 'cassels_tate'),
                     'quartic': (lambda ct: ct['quartics'][0].__setitem__(0, '2'), 'cassels_tate')}
            for what, (f, check) in cases.items():
                self.assertIn(check, fails(changed(f), d), what)
            gone = json.loads(raw)
            gone['cassels_tate'] = None             # without it the bound is 2
            self.assertIn('bound_1', fails(gone, d))


class TestLocal(unittest.TestCase):
    """tdquartic against a brute-force oracle: a primitive (u, v) mod p^k at which g is a square of known valuation
    and unit class proves solubility (Hensel); if every primitive (u, v) mod p^k gives a non-square of known
    valuation and unit class, g is insoluble.  Also tdsquare's Weil branch against enumeration."""

    ORACLE = r'''
\\ 1 soluble, 0 insoluble, -1 undecided, from the primitive pairs (u, v) mod p^k
orac(g, p, k) =
{
  my(q = p^k, need = if(p == 2, 3, 1), und = 0);
  for(u = 0, q - 1, for(v = 0, q - 1, my(f, l, uu);
    if(u % p == 0 && v % p == 0, next);
    f = sum(i = 0, 4, g[i + 1] * u^(4 - i) * v^i) % q;
    if(f == 0, und = 1; next);
    l = valuation(f, p);
    if(l % 2, next);
    if(k - l < need, und = 1; next);
    uu = (f / p^l) % p^(k - l);
    if(if(p == 2, uu % 8 == 1, kronecker(uu, p) == 1), return(1))));
  if(und, -1, 0);
}
\\ a random separable quartic [g4, ..., g0], |g_i| <= m
rq(m) =
{
  my(g = [0, 0, 0, 0, 0]);
  until(poldegree(Pol(g)) >= 3 && poldisc(Pol(g)) && poldisc(Polrev(g)), g = vector(5, i, random(2 * m + 1) - m));
  g;
}
\\ a random quartic of the descent, d^2 (d u^4 + A u^2 v^2 + (B/d) v^4)
rd() =
{
  my(d = 0, A = 0, B = 0);
  while(!d || !B || A^2 == 4*B, d = random(41) - 20; A = random(41) - 20; B = random(41) - 20);
  [d^3, 0, A*d^2, 0, B*d];
}
{
  setrand(1); bad = 0; dec = 0;
  foreach([[2, 6], [3, 4], [5, 3], [7, 2]], pk,
    for(j = 1, 120, my(g = if(j % 2, rq(6), rd()), o = orac(g, pk[1], pk[2]), t = tdquartic(g, pk[1]));
      if(o >= 0, dec++; if(o != t, bad++; print("BAD ", g, " ", pk[1], " oracle ", o, " test ", t)))));
  print("DEC ", dec, " ", bad);
}
'''

    def test_oracle(self):
        out = gp(self.ORACLE)
        self.assertEqual([l for l in out if l.startswith('BAD')], [])
        self.assertGreater(int(out[-1].split()[1]), 300)

    def test_weil(self):
        """For 17 <= p < 64 the Weil branch (TDENUM = 17) agrees with enumeration (TDENUM = 64), also on c h^2."""
        out = gp(r'''
{
  setrand(2); bad = 0;
  forprime(p = 17, 61, for(j = 1, 60, my(H, a, b);
    H = if(j % 3, Pol(vector(5, i, random(p))), random(p) * (x^2 + random(p) * x + random(p))^2);
    if(H * Mod(1, p) == 0, next);
    TDENUM = 64; a = tdsquare(H, p); TDENUM = 17; b = tdsquare(H, p);
    if(a != b, bad++; print("BAD ", H, " ", p))));
  print("N ", bad);
}
''')
        self.assertEqual(out[-1], 'N 0')

    def test_duality_large(self):
        """Local duality at large primes (the Weil branch) and at 2: y^2 = x(x - e1)(x - e2) with e1 < 10^6 and
        e2 < 10^12."""
        out = gp(DESCENT + r'''
{
  setrand(3); bad = 0;
  for(j = 1, 30, my(e1 = random(10^6) + 1, e2 = -random(10^12) - 1, ai, d);
    ai = [0, -e1 - e2, 0, e1 * e2, 0]; d = td1(ai);
    if(d[4] != 1, bad++; print("BAD ", ai)));
  print("N ", bad);
}
''')
        self.assertEqual(out[-1], 'N 0')



def fails(c, d, name='t'):
    """Return the names of the failed checks of the certificate dict c (written to the folder d)."""
    path = os.path.join(d, name + '.cert.json')
    with open(path, 'w') as f:
        json.dump(c, f)
    v = verifier.check(path)
    assert v['verdict'] in ('accepted', 'rejected'), v['reason']
    return [x['check'] for x in v['checks'] if not x['ok']]


class TestCertificate(unittest.TestCase):
    """rankbound() on y^2 = x^3 - 25x with the point (-4, 6): rank = 1 proved, unconditionally, by a certificate the
    verifier accepts; a changed Selmer basis element, S, torsion, GRH, point or a missing isogeny fails."""

    @classmethod
    def setUpClass(cls):
        from rankbound import rankbound
        cls.tmp = tempfile.mkdtemp()
        cls.res = rankbound([0, 0, 0, -25, 0], points=[(-4, 6)], workdir=os.path.join(cls.tmp, 'n5'))
        with open(cls.res.certificate) as f:
            cls.raw = f.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_result(self):
        r = self.res
        self.assertEqual((r.rank_bound, r.rank_lower_bound, r.r_proved, r.proved_equal, r.verified, r.grh, r.method,
                          r.torsion, r.w, r.selmer_bound, r.outcome),
                         (1, 1, True, True, True, False, '2-isogeny descent', [2, 2], -1, None, 'descent'))
        self.assertEqual(r.statement, 'Unconditionally, rank E(Q) <= 1. As E(Q) has 1 independent rational point, we '
                                      'therefore have rank E(Q) = 1.')
        for f in ('curve.json', 'verify.log', 'verify.json', 'result.json'):
            self.assertTrue(os.path.exists(os.path.join(r.workdir, f)), f)
        with open(os.path.join(r.workdir, 'curve.json')) as f:
            self.assertEqual(json.load(f)['torsion'], [2, 2])

    def test_round_trip(self):
        c = json.loads(self.raw)
        self.assertEqual(list(c), ['name', 'curve', 'statement', 'torsion', 'GRH', 'rank_lower_bound', 'descent',
                                   'cassels_tate', 'local', 'claims', 'created', 'format'])
        self.assertIsNone(c['cassels_tate'])            # S = r after Theta_1: no Cassels-Tate step
        self.assertEqual([d['e'] for d in c['descent']], ['-20', '0', '20'])
        self.assertEqual(list(c['descent'][0]), ['e', 'a', 'b', 'selmer', 'selmer_isogenous', 's', 's_isogenous',
                                                 'pairing', 'cassels_tate', 'B'])
        self.assertEqual(c['descent'][0]['pairing']['places'], ['R', '2', '3', '5', '7'])
        v = verifier.check(self.res.certificate, curve=[0, 0, 0, -25, 0])
        self.assertEqual((v['verdict'], v['rank'], v['GRH'], v['method'], v['torsion'], v['G']),
                         ('accepted', 1, 'no', '2-isogeny descent', [2, 2], None))
        self.assertIn('no GRH', v['hypothesis'])
        self.assertEqual(verifier.check(self.res.certificate, curve=[0, 0, 0, -36, 0])['verdict'], 'rejected')

    def test_tampered(self):
        def changed(f):
            c = json.loads(self.raw)
            f(c)
            return c
        def basis(c):
            c['descent'][0]['selmer'][0] = str(3 * int(c['descent'][0]['selmer'][0]))
        def S(c):
            c['claims']['S'] += 2
            c['statement'] = c['statement'].replace('<= 1.', '<= 3.').split(' As')[0]
        cases = {'selmer basis': (basis, 'descent_1'), 'S + 2': (S, 'claims'),
                 'torsion': (lambda c: c.update(torsion=[2]), 'torsion'),
                 'GRH yes': (lambda c: c.update(GRH='yes'), 'GRH'),
                 'point': (lambda c: c['rank_lower_bound']['points'][0].__setitem__(1, '7'), 'rank_lower_bound'),
                 'isogenies left out': (lambda c: c.update(descent=c['descent'][:1]), 'two_torsion'),
                 'B_e': (lambda c: c['descent'][1].update(B=0), 'bound_2'),
                 'conic point': (lambda c: c['descent'][0]['pairing']['conic'][0].__setitem__(0, '12345'), 'pairing_1'),
                 'local point': (lambda c: c['descent'][0]['pairing']['local'][0][1].__setitem__(0, '3'), 'pairing_1'),
                 'pairing rank': (lambda c: c['descent'][2]['pairing'].update(rank=2), 'pairing_3')}
        for what, (f, check) in cases.items():
            self.assertIn(check, fails(changed(f), self.tmp), what)
        self.assertEqual(fails(changed(lambda c: c['descent'][0].update(extra=[])), self.tmp), ['format'])
        self.assertEqual(fails(changed(lambda c: c['descent'][0]['pairing'].update(places=['R', '2'])), self.tmp),
                         ['format'])


class TestForgery(unittest.TestCase):
    """Certificates forged from honest ones (the 2.0 code review): ICARM #2209's Cassels-Tate data with an extra prime
    listed twice (and its point twice), its rank raised and the claims lowered to rank <= -1, which the verifier
    accepted before the extra primes had to be strictly increasing; a negative bound; a singular model."""
    FORGED = os.path.join(TESTS, 'data', 'forged2209.cert.json')

    def setUp(self):
        with open(self.FORGED) as f:
            self.c = json.load(f)

    def test_repeated_prime(self):
        """Rejected as malformed by the form check, and by gp's own check of the pair without it."""
        from rankbound.verify import two_descent as td
        self.assertIn(['2786783', '2786783'], [p['primes'] for p in self.c['cassels_tate']['pairs']])
        v = verifier.check(self.FORGED)
        self.assertEqual(v['verdict'], 'rejected')
        self.assertEqual([x['check'] for x in v['checks'] if not x['ok']], ['format'])
        out = run_gp(td.gp_body(self.c), libs=('verify.gp', 'two_descent.gp'))     # gp alone, past the form check
        chk = {t[1]: t[2] for t in (l.split(' ', 3) for l in out.splitlines()) if t[0] == 'CHK'}
        self.assertEqual(chk.get('cassels_tate'), '0')
        self.assertNotIn('TD -1 -1', out)

    def test_negative_bound(self):
        from rankbound.verify import two_descent as td
        got = []
        td.claims(self.c, 'TD -1 -1 %d' % self.c['local']['w'], lambda n, passed, info='': got.append((n, passed)))
        self.assertIn(('claims', False), got)

    def test_singular(self):
        """A singular model is a malformed certificate (FAIL, exit 1), not an error of the verifier (exit 4)."""
        with tempfile.TemporaryDirectory() as d:
            for k in ('ainvs', 'minimal_model'):
                c = json.loads(json.dumps(self.c))
                c['curve'][k] = ['0', '0', '0', '0', '0']
                self.assertEqual(fails(c, d), ['format'], k)


class TestFormats(unittest.TestCase):
    """Class group certificates: format 3 records torsion and GRH "yes" (checked); format 2 (without them) is still
    accepted."""

    def test_formats(self):
        with open(cert0()) as f:
            c3 = json.load(f)
        self.assertEqual((c3['format'], c3['torsion'], c3['GRH']), ('rankbound-selmer-certificate-3', [5], 'yes'))
        with tempfile.TemporaryDirectory() as d:
            c2 = {k: v for k, v in c3.items() if k not in ('torsion', 'GRH')}
            c2['format'] = 'rankbound-selmer-certificate-2'
            self.assertEqual(fails(c2, d), [])
            path = os.path.join(d, 't.cert.json')
            v = verifier.check(path)
            self.assertEqual((v['GRH'], v['torsion'], v['method']), ('yes', None, 'class group'))
            self.assertEqual(fails(dict(c3, torsion=[3]), d), ['torsion'])
            self.assertEqual(fails(dict(c3, GRH='no'), d), ['GRH'])
            self.assertEqual(fails(dict(c3, torsion=[2]), d), ['format'])        # E(Q)[2] = 0 here
            self.assertEqual(fails(dict(c2, torsion=[5], GRH='yes'), d), ['format'])

    def test_known(self):
        """certificates.tsv: the columns torsion and GRH; a row with GRH "no" is stated unconditionally."""
        from rankbound import known
        row = {known.UB: '1', known.LB: '1', known.EQ: 'yes', known.GRH: 'no', known.TOR: '[2, 2]'}
        self.assertEqual(known.statement(row), 'rank E(Q) = 1 unconditionally')
        self.assertEqual(known.statement(dict(row, GRH='yes')), 'rank E(Q) = 1 under GRH')
        self.assertEqual(known.statement(dict(row, **{known.EQ: 'no', known.UB: '3'})), '1 <= rank E(Q) <= 3 unconditionally')
        with open(known.INDEX) as f:
            head = f.readline().rstrip('\n').split('\t')
            rows = [dict(zip(head, l.rstrip('\n').split('\t'))) for l in f]
        self.assertEqual(head[-3:], ['torsion', 'GRH', 'minimal model'])
        self.assertEqual({r['GRH'] for r in rows if r['file']}, {'yes', 'no'})
        two = lambda t: any(int(x) % 2 == 0 for x in t.strip('[]').split(',') if x.strip())
        self.assertTrue(all((r['GRH'] == 'no') == two(r['torsion']) for r in rows if r['file']))
        self.assertEqual(known.lookup(['1', '0', '1', '4', '-6'])['GRH'], 'no')            # 14a1 = ICARM #865
        self.assertEqual(known.lookup(['0', '-1', '1', '0', '0'])['torsion'], '[5]')       # 11a3


class TestCommandLine(unittest.TestCase):
    """`rankbound bound` on 14a1 (Z/6, rank 0) and on y^2 = x^3 - 25x with --points (-4, 6) (rank = 1), `status`,
    `verify`, and `bound --relations` refused on a curve with 2-torsion."""

    def run_cli(self, *args):
        return subprocess.run(CLI + list(args), env=ENV, capture_output=True, text=True, timeout=300)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'w')
            r = self.run_cli('bound', '1,0,1,4,-6', '--dir', w)                       # ICARM #865: looked up
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('icarm865 is already certified: rank E(Q) = 0 unconditionally.', r.stdout)
            r = self.run_cli('bound', '1,0,1,4,-6', '--dir', w, '--compute')
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('E(Q)_tors = Z/6: E(Q)[2] != 0', r.stdout)
            self.assertIn('RESULT 1_0_1_4_-6: rank = 0 unconditionally, certificate verified (0.00 core-h)', r.stdout)
            pts = os.path.join(d, 'points.txt')
            with open(pts, 'w') as f:
                f.write('-4 6\n')
            r = self.run_cli('bound', '0,0,0,-25,0', '--points', pts, '--dir', w)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('RESULT 0_0_0_-25_0: rank = 1 unconditionally, certificate verified (0.00 core-h)', r.stdout)
            r = self.run_cli('bound', '0,0,0,-25,0', '--points', pts, '--dir', w)          # done: read back
            self.assertIn('done earlier', r.stdout)
            r = self.run_cli('status', w)
            self.assertEqual(r.stdout.splitlines()[1:], ['0_0_0_-25_0\t\t0.0\t0.0\t1\tfinished, verified (2-descent)',
                                                         '1_0_1_4_-6\t\t0.0\t0.0\t0\tfinished, verified (2-descent)'])
            cert = os.path.join(w, '0_0_0_-25_0_r1p', '0_0_0_-25_0.cert.json')
            r = self.run_cli('verify', cert)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('PASS: 0_0_0_-25_0', r.stdout)
            with open(os.path.join(w, '1_0_1_4_-6', 'verify.json')) as f:
                self.assertEqual(json.load(f)['verdict'], 'accepted')
            r = self.run_cli('estimate', '0,0,0,-1,0', '--dir', w)
            self.assertIn('E(Q)[2] != 0, so a 2-isogeny descent bounds the rank', r.stdout)
            rels = os.path.join(d, 'rels.txt')
            with open(rels, 'w') as f:
                f.write('1,1\n')
            r = self.run_cli('bound', '0,0,0,-1,0', '--relations', rels, '--dir', w)
            self.assertEqual(r.returncode, 1)
            self.assertIn('2-isogeny descent, not from relations', r.stderr)


if __name__ == '__main__':
    unittest.main()
