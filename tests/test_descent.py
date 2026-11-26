"""The 2-isogeny descent's prover (rankbound/descent.py): known ranks, small curves against PARI's ellrank, local
duality, the isogeny and Cassels-Tate pairings (Fisher's examples), and the command line on 14a1 and y^2 = x^3 - 25x.
The checker's tests of the 2-isogeny format are test_check_isogeny.py's."""
import json, os, subprocess, sys, tempfile, unittest
from rankbound.check import run_gp
from helpers import DESCENT, SRC, TESTS, fails, run_route

ROOT = os.path.dirname(TESTS)
CLI = [sys.executable, '-m', 'rankbound']
FULL = bool(os.environ.get('RANKBOUND_TEST_FULL'))        # RANKBOUND_TEST_FULL=1: every case
ENV = dict({k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}, PYTHONPATH=SRC)

LIBS = (os.path.join(ROOT, 'tests', 'data', 'oracle', 'ctp.gp'), os.path.join(ROOT, 'tests', 'data', 'oracle', 'tower.gp'))


def gp(body, seconds=600):
    """Return the output lines of body (GP source; a multi-line block in braces starting a line), run after
    core.gp, ctp.gp, isogeny.gp and the prover's ctp.gp and tower.gp."""
    return run_gp(body, libs=LIBS, seconds=seconds).splitlines()

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
        fails vrank."""
        out = gp(r'''
{
  foreach([[-25, [-4, 6]], [-36, [-3, 9]], [-49, [25, 120]]], c,
    my(ai = [0, 0, 0, c[1], 0], PR = tdprimes(ai, [c[2]], [], 1000)); vrank(ai, 1, [c[2]], PR));
  E = ellinit([0, 0, 0, -25, 0]); P = [-4, 6]; Q = elladd(E, ellmul(E, P, 2), [5, 0]);
  print(tdrank(E, x^3 - 400*x, [P, Q], primes([3, 200]))[2], " ", tdrank(E, x^3 - 400*x, [P], primes([3, 200]))[2]);
  vrank([0, 0, 0, -25, 0], 1, [[-4, 6]], [5]);
  vrank([0, 0, 0, -25, 0], 1, [[-4, 7]], [3, 7]);
}
''')
        self.assertEqual([l.split()[2] for l in out[:3]], ['1', '1', '1'])
        self.assertEqual(out[3], '3 3')                     # rank t + 1 = 3 either way
        self.assertEqual([l.split()[2] for l in out[4:]], ['0', '0'])
        with self.assertRaisesRegex(RuntimeError, 'do not prove rank >= 2'):
            gp('tdprimes([0,0,0,-25,0], [[-4,6], [5,0]], [], 300);')


class TestSaturation(unittest.TestCase):
    def test_doubled_point(self):
        """y^2 = x^3 - 25x (rank 1, torsion Z/2 x Z/2) with 2P, P = (-4, 6): with the torsion, 2P's characters vanish;
        2P is halved over Q and the checked witness proves rank 1 (ICARM #1085, #1133, ...: 407 curves)."""
        from fractions import Fraction
        from rankbound import curve as cv, rankbound
        a = [0, 0, 0, -25, 0]
        Q = cv.mul(a, (Fraction(-4), Fraction(6)), 2)
        with tempfile.TemporaryDirectory() as d:
            res = rankbound(a, points=[[str(Q[0]), str(Q[1])]], workdir=os.path.join(d, 'w'), compute=True)
        self.assertEqual((res.rank_bound, res.rank_lower_bound, res.r_proved, res.checked), (1, 1, True, True))


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

    @unittest.skipUnless(FULL, 'RANKBOUND_TEST_FULL: ellrank on 7,000 curves (38 s); test_examples and the quick '
                               'set cover the bound')
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
    y^2 = x(x^2 + a x + b) with s > 0 (j = 1728 included); the witness of ICARM #1737 (Z/8 x Z/2, rank 0, where the
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

    @unittest.skipUnless(FULL, 'RANKBOUND_TEST_FULL: the quick set runs this curve too (hard:fisher17_ex10.1)')
    def test_fisher_10_1(self):
        """[F17, Example 10.1] (Z/12): the Selmer groups are F17's S_1' = <-272196179> and S_1 = <15, 73, 87, 231, 28619>,
        both isogeny pairings vanish (B = 4); the Cassels-Tate pairing has rank 4 on Sel_2(E'): rank E(Q) = 0, as F17
        proves with Theta_3.  (It is 0 on Sel_2(E), Sha(E)[2^inf] = (Z/4)^4: the pairing on E', with 5 lifted quartics
        to E's 1, comes first and gives the bound 0, so the one on E is not computed.)"""
        import itertools, math
        from rankbound import descent as isog, rankbound
        with tempfile.TemporaryDirectory() as d:
            res = rankbound([0, 91502230365284038, 0, 489792722057841784540058275212361, 0], points=[],
                            bad_primes=[2, 3, 5, 7, 11, 29, 71, 73, 127, 28619, 30187], workdir=os.path.join(d, 'w'))
            with open(res.witness) as f:
                c = json.load(f)
        (i,) = c['descent']
        S = [2, 3, 5, 7, 11, 29, 71, 73, 127, 28619, 30187]
        span = lambda V: {tuple(isog.tdclass(math.prod(v for v, k in zip(V, c) if k), S))
                          for c in itertools.product((0, 1), repeat=len(V))}
        self.assertEqual((i['selmer'], span([int(x) for x in i['selmer_isogenous']])), (['-272196179'], span([15, 73, 87, 231, 28619])))
        self.assertEqual((i['pairing']['rank'], i['pairing']['rank_isogenous'], c['cassels_tate'],
                          i['cassels_tate']['rank'], i['B']), (0, 0, None, 4, 0))
        self.assertEqual((res.rank_bound, res.checked, res.torsion), (0, True, [12]))

    def test_witness(self):
        from rankbound import rankbound
        with tempfile.TemporaryDirectory() as d:
            res = rankbound([1, 0, 0, -1319539461660, -159402536950172400], bad_primes=[2, 3, 5, 7, 11, 73, 103],
                            points=[], workdir=os.path.join(d, 'w'))
            self.assertEqual((res.rank_bound, res.proved_equal, res.checked), (0, True, True))
            with open(res.witness) as f:
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


class TestCTTimeLimit(unittest.TestCase):
    """The Cassels-Tate pairings of the 2-isogeny route run under alarm() within ctp_seconds (and before a
    deadline); a pairing cut short is skipped (CTSKIP) and the bound already computed stands, instead of the whole gp
    run timing out."""
    AI, P = [1, 0, 0, -1319539461660, -159402536950172400], [2, 3, 5, 7, 11, 73, 103]   # test_witness's curve

    def test_alarm(self):
        import time
        from rankbound import descent as tower
        logs = []
        with tempfile.TemporaryDirectory():
            from rankbound import arith
            def forever(mm, S, left=None):                                  # a pairing that never ends
                while True:
                    arith.tick()
            real_ct, real_log = tower.twotor_ct, tower.LOG
            tower.twotor_ct, tower.LOG = forever, logs.append
            try:
                t = time.time()
                D = tower.descent(self.AI, self.P, ctp_seconds=2, seconds=600)
                self.assertLess(time.time() - t, 60)
                self.assertEqual((D['cassels_tate'], D['S']), (None, 2))          # without the pairing: rank <= 2
                self.assertTrue(any('its time limit was reached' in l for l in logs), logs)
                # a budget with no time left for the pairings: skipped at once
                from rankbound import runtime
                logs.clear(); t = time.time()
                D = tower.descent(self.AI, self.P, budget=runtime.Budget(seconds=tower.CT_MARGIN + 0.5))
                self.assertEqual((D['cassels_tate'], D['S']), (None, 2))
                self.assertLess(time.time() - t, 30)
            finally:
                tower.twotor_ct, tower.LOG = real_ct, real_log
        D = tower.descent(self.AI, self.P, ctp_seconds=600)               # the real pairing: rank 2, so rank <= 0
        self.assertEqual((D['cassels_tate']['rank'], D['S']), (2, 0))

    def test_stopped(self):
        from rankbound import Stopped, curve as cv, descent as tower, runtime
        c = cv.classified(cv.from_ainvs(self.AI, bad_primes=self.P))
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'w')
            self.assertRaises(Stopped, run_route, tower.Descent(w), c,
                              budget=runtime.Budget.for_run(seconds=0, workdir=w))


class TestCommandLine(unittest.TestCase):
    """`rankbound bound` on 14a1 (Z/6, rank 0) and on y^2 = x^3 - 25x with --points (-4, 6) (rank = 1), `status`,
    `check`, and `bound --relations` refused on a curve with 2-torsion."""

    def run_cli(self, *args):
        return subprocess.run(CLI + list(args), env=ENV, capture_output=True, text=True, timeout=300)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'w')
            r = self.run_cli('bound', '1,0,1,4,-6', '--dir', w)                       # ICARM #865: looked up
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('This curve is already in the table of witnesses. Rank 0, unconditionally. Add '
                          '--compute to compute it again.', r.stdout)
            r = self.run_cli('bound', '1,0,1,4,-6', '--dir', w, '--compute')
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            progress = [l[9:] for l in r.stderr.splitlines() if not l[9:].startswith('Work folder ')]
            self.assertEqual(progress, ['Conductor 14, two bad primes, root number +1, torsion Z/6.',
                                        'Rank at most 0. Verification method: 2-isogeny descent (unconditional).'])
            self.assertEqual([l[9:] for l in r.stdout.splitlines()],       # the result alone on stdout
                             ['Rank 0, unconditionally. Witness: %s (checked)' % os.path.join(w, '1_0_1_4_-6', '1_0_1_4_-6.witness.json')])
            pts = os.path.join(d, 'points.txt')
            with open(pts, 'w') as f:
                f.write('-4 6\n')
            r = self.run_cli('bound', '0,0,0,-25,0', '--points', pts, '--dir', w)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('Rank at least 1.', r.stderr)
            self.assertIn('Rank 1, unconditionally. Witness: ', r.stdout)
            r = self.run_cli('bound', '0,0,0,-25,0', '--points', pts, '--dir', w)          # done: read back
            self.assertIn('Already done earlier. Rank 1, unconditionally.', r.stdout)
            r = self.run_cli('status', w)
            self.assertEqual([l.split() for l in r.stdout.splitlines()[1:]],
                             [['0_0_0_-25_0', '0', '0.00', '1', 'finished', 'and', 'checked'],
                              ['1_0_1_4_-6', '0', '0.00', '0', 'finished', 'and', 'checked']])
            wit = os.path.join(w, '0_0_0_-25_0', '0_0_0_-25_0.witness.json')
            r = self.run_cli('check', wit)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('0_0_0_-25_0 passes in', r.stdout)
            with open(os.path.join(w, '1_0_1_4_-6', 'check.json')) as f:
                self.assertEqual(json.load(f)['verdict'], 'accepted')
            r = self.run_cli('bound', '--estimate', '0,0,0,-1,0', '--dir', w)
            self.assertIn('has rational 2-torsion, so a 2-isogeny descent bounds its rank.\n', r.stdout)
            rels = os.path.join(d, 'rels.txt')
            with open(rels, 'w') as f:
                f.write('1,1\n')
            r = self.run_cli('bound', '0,0,0,-1,0', '--relations', rels, '--dir', w)
            self.assertEqual(r.returncode, 2)
            self.assertIn('so its bound comes from a 2-isogeny descent and not from relations', r.stderr)


if __name__ == '__main__':
    unittest.main()
