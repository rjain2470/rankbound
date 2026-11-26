"""The Cassels-Tate step's prover (rankbound/ctp.py, ctp.gp): the Cassels-Tate pairing on the 2-descent's elements for E(Q)[2] = 0.  571a1
(Fisher 2022, Example 3.4: S 2 -> 0) and a rank-2 curve with Sha[2] = (Z/2)^2 whose points are located in the 2-descent's basis
(S 4 -> 2), both against PARI's ellrank; a curve with Sha[4] = Z/4 x Z/4 (the pairing vanishes on Sel_2: the
4-descent limit); no Cassels-Tate step when S is the rank lower bound or past the time limit; the reduction modulo squares and the
fake-prime qfsolve; the route (rankbound()).  The verifier's checks of the section are test_verify_selmer.py's."""
import dataclasses, json, os, shutil, tempfile, unittest
from rankbound import ctp, curve as cv, rankbound, selmer, witness
from rankbound.check import run_gp
from helpers import TESTS

ROOT = os.path.dirname(TESTS)
SPLIT_GP = os.path.join(ROOT, 'tests', 'data', 'oracle', 'split.gp')
CTP_GP = os.path.join(ROOT, 'tests', 'data', 'oracle', 'ctp.gp')
LIBS = (SPLIT_GP, CTP_GP)
E571 = [0, -1, 1, -929, -10595]             # rank 0, Sha = (Z/2)^2: ellrank [0, 0, 2]
E697691 = [0, -1, 1, -7117, 233485]         # 697691.a1: rank 2, Sha[2] = (Z/2)^2: ellrank [2, 2, 2]
E20258 = [1, 1, 1, -21563569, -38550526127]  # 20258.d1: rank 0, #Sha = 64: ellrank [0, 2, 0]
QUIET = lambda *a: None


def sel_witness(ai, name, points=None):
    c = cv.from_ainvs(ai, name=name)
    if points:
        c = dataclasses.replace(c, r=len(points), proof={'points': points}, source=None)
    return selmer.prove(c, log=QUIET)


def ellrank(ai):
    out = run_gp('my(z = ellrank(ellinit([%s]))); print("ER\\t", z[1..3], "\\t", '
                 'strjoin(apply(P -> Str(P[1], ":", P[2]), z[4]), ","));' % ','.join(map(str, ai)))
    t = next(l.split('\t') for l in out.splitlines() if l.startswith('ER'))
    return json.loads(t[1]), [p.split(':') for p in t[2].strip().split(',') if p]


class TestTargets(unittest.TestCase):
    """S_ctp = ellrank's upper bound (z[2]), the Cassels-Tate drop its z[3]."""

    def test_571a1(self):
        c = ctp.add(sel_witness(E571, 'c571a1'), log=QUIET)
        self.assertEqual((c['claims']['S'], c['claims']['S_ctp'], c['cassels_tate']['rank']), (2, 0, 2))
        self.assertEqual(ellrank(E571)[0], [0, 0, 2])
        self.assertEqual(c['statement'], 'Assuming the GRH, we have that rank E(Q) <= 0.')      # GRH: the default
        self.assertEqual((len(c['cassels_tate']['quartics']), len(c['cassels_tate']['pairs'])), (2, 1))
        self.assertEqual([k for k in c if not k.startswith('_')][-2:], ['cassels_tate', 'format'])

    def test_points(self):
        """The two points of 697691.a1 are located in the 2-descent's basis (4 classes): one pair on the complement."""
        er, pts = ellrank(E697691)
        self.assertEqual(er, [2, 2, 2])
        c = ctp.add(sel_witness(E697691, 'c697691', pts), log=QUIET)
        info = c['_ctp']
        self.assertEqual((info['basis'], info['located'], info['points_rank'], len(info['complement'])), (4, 2, 2, 2))
        self.assertEqual((c['claims']['S'], c['claims']['S_ctp']), (4, 2))
        self.assertTrue(c['statement'].endswith('we therefore have rank E(Q) = 2.'), c['statement'])

    def test_order_four(self):
        """#Sha = 64: the pairing is 0 on Sel_2, no section; S = 2 = ellrank's bound."""
        c = ctp.add(sel_witness(E20258, 'c20258'), log=QUIET)
        self.assertEqual((c['cassels_tate'], c['claims']['S'], c['_ctp']['rank']), (None, 2, 0))
        self.assertNotIn('S_ctp', c['claims'])
        self.assertEqual(ellrank(E20258)[0], [0, 2, 0])

    def test_not_run(self):
        """No the Cassels-Tate step when S = the rank lower bound (37a1, r = 1), nor past the time limit (no section, the 2-descent's bound)."""
        c = ctp.add(sel_witness([0, 0, 1, -1, 0], 'c37a1', [['0', '0']]), log=QUIET)
        self.assertEqual((c['cassels_tate'], c['claims']['S'], '_ctp' in c), (None, 1, False))
        m = sel_witness(E571, 'c571b')
        c = ctp.add(m, seconds=0.01, log=QUIET)
        self.assertEqual((c['cassels_tate'], c['claims']['S'], 'S_ctp' in c['claims']), (None, 2, False))
        c = ctp.add(m, log=QUIET)                   # the Cassels-Tate step returns the completed witness; the 2-descent's is unchanged
        self.assertEqual((c['claims']['S_ctp'], 'S_ctp' in m['claims'], 'cassels_tate' in m), (0, False, False))


class TestGP(unittest.TestCase):
    """ctp.gp's helpers: the tree reduction modulo squares, and qfsolve with fake primes."""

    def test_reduction(self):
        """The reduced product of 571a1's relations of a Selmer element is that product times a square in K, with
        (y) = A D^2 and A above S; small (coefficients below 10^6)."""
        c = sel_witness(E571, 'c571c')
        J, _ = ctp.selmer_basis(c)[1]
        rec = {j: (r, H) for j, r, H in c['_pairing']['rels']}         # the 2-descent's relations as records (no text round trip)
        rels = [rec[j] for j in J]
        f = '+'.join('(%s)*x^%d' % (a, i) for i, a in enumerate(c['field']['poly']))
        body = '''{
  my(K = m1K(%s, [571]), nf, L, y, x0, q);
  nf = K[1];
  L = [%s]; L = vector(#L, k, ctleaf(K, L[k][1], L[k][2]));
  y = ctprod(K, L, 1, #L);
  x0 = prod(k = 1, #L, nfbasistoalg(nf, L[k][1]));
  q = x0 / nfbasistoalg(nf, y[1]);
  print("SQ ", #nfroots(subst(nf.pol, x, 'ss), x^2 - subst(lift(q), x, 'ss)) > 0, " ", vecmax(abs(y[1])) < 10^6, " ",
        if(#y[2], idealfactorback(nf, y[2]), idealhnf(nf, 1)) == idealdiv(nf, y[1], idealpow(nf, y[3], 2)), " ", #select(pr -> pr.p != 2 && pr.p != 571, y[2]));
}''' % (f, ','.join('[%s,%s]' % ('[%s]' % ','.join(map(str, r)), '[%s]' % ','.join(map(str, H))) for r, H in rels))
        out = run_gp(body, libs=LIBS)
        self.assertIn('SQ 1 1 1 0', out)

    def test_reduction_py(self):
        """ct's (the prover's) product of the same relations, and that reduced (ct.red): the product times a square in
        K, (y) E^2 = A (squarefree, its norm the product of A's), A above S; red makes y smaller."""
        from rankbound import ctp as ct, field as fld
        c = sel_witness(E571, 'c571e')
        J, _ = ctp.selmer_basis(c)[1]
        rec = {j: (r, H) for j, r, H in c['_pairing']['rels']}
        rels = [rec[j] for j in J]
        f = [int(a) for a in c['field']['poly']]
        k = fld.form_field(f)
        t = ct.cprod(k, [ct.leaf(k, r, h) for r, h in rels], 0, len(rels) - 1)
        el = lambda v, d=1: '(%s)/%d' % ('+'.join('(%d)*al^%d' % (int(x), i) for i, x in enumerate(k.power(v))), d)
        for y, A_, (H, den) in (t, ct.red(k, t)):
            nA = 1
            for p, lab in A_:
                nA *= k.prime(p, lab).q
            body = '''{
  my(K = m1K(%s, [571]), nf = K[1], al, x0, y, E, I, fa);
  al = m1el(K, [0, 0, -1]);
  x0 = prod(k = 1, %d, m1el(K, [%s][k]));
  y = %s; E = idealhnf(nf, Mat(apply(z -> nfalgtobasis(nf, z), [%s])));
  I = idealmul(nf, y, idealpow(nf, E, 2)); fa = idealfactor(nf, I);
  print("SQ ", #nfroots(subst(nf.pol, x, 'ss), x^2 - subst(lift(x0 / y), x, 'ss)) > 0, " ", idealnorm(nf, I) == %d, " ",
        #fa~ == 0 || vecmax(fa[, 2]) == 1);
}''' % ('+'.join('(%s)*x^%d' % (a, i) for i, a in enumerate(f)), len(rels),
                 ','.join('[%s]' % ','.join(map(str, r)) for r, _ in rels), el(y), ','.join(el(r, den) for r in H), nA)
            self.assertIn('SQ 1 1 1', run_gp(body, libs=LIBS))
            self.assertTrue(all(p in (2, 571) for p, _ in A_))
        self.assertLess(max(map(abs, ct.red(k, t)[0])), max(map(abs, t[0])))

    def test_fake_primes(self):
        """x^2 + y^2 = p q r z^2 (soluble: p, q, r = 1 mod 4) with p q r given as one 'prime': solved, checked."""
        body = '''{
  my(P = select(t -> t % 4 == 1, primes([10^15, 10^15 + 10^4]))[1..3], N = vecprod(P), G, s);
  G = [1, 0, 0; 0, 1, 0; 0, 0, -N];
  s = tdqfsolve(G, [2, 3], [N]);
  print("FS ", type(s) == "t_COL" && s~ * G * s == 0 && s != 0);
  G = [3, 1, 0; 1, -7, 0; 0, 0, -13 * (P[1] * P[2])^2];
  s = tdqfsolve(G / 5, [2, 3, 13], [P[1] * P[2]]);
  print("FS ", type(s) == type(qfsolve(G)) && (type(s) != "t_COL" || s~ * G * s == 0));
}'''
        out = [l for l in run_gp(body, libs=LIBS).splitlines() if l.startswith('FS')]
        self.assertEqual(out, ['FS 1', 'FS 1'])


class TestRoute(unittest.TestCase):
    """rankbound() on 571a1 runs the 2-descent then the Cassels-Tate step (ctp.after_selmer): rank 0, checked."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_route(self):
        res = rankbound(E571, r=0, compute=True, workdir=os.path.join(self.tmp, 'a'))
        self.assertEqual((res.method, res.rank_bound, res.checked, res.proved_equal), (selmer.METHOD, 0, True, True))
        c = witness.read(res.witness)
        self.assertEqual((c['claims']['S'], c['claims']['S_ctp'], c['cassels_tate']['rank']), (2, 0, 2))
        from rankbound import runtime
        wit = ctp.after_selmer(sel_witness(E571, 'c571d'), runtime.Budget(seconds=0))     # past the budget's end: no Cassels-Tate step
        self.assertIsNone(wit['cassels_tate'])



class TestGate(unittest.TestCase):
    """The Cassels-Tate step runs only when S is at least 2 above the effective lower bound r' (r, or r + 1 when the root number rules out
    r's parity): 571a1 (S = 2, w = +1) with r = 1 has r' = 2, so no pairing.  Its time limit is max(ctp.SECONDS,
    the 2-descent's time on the curve) unless given."""

    def test_parity_gate(self):
        c = cv.from_ainvs(E571, name='g571')
        wit = selmer.prove(dataclasses.replace(c, r=1, source='test'), log=QUIET)
        self.assertEqual((wit['claims']['S'], wit['claims']['w']), (2, 1))
        out = ctp.add(wit, log=QUIET)
        self.assertIsNone(out['cassels_tate'])
        self.assertNotIn('ctp', out.get('_timing', {}))          # never started

    def test_time_limit(self):
        seen = []
        real = ctp.add
        try:
            ctp.add = lambda wit, seconds, **kw: seen.append(round(seconds)) or wit
            for kw in ({'selmer_seconds': 5}, {'selmer_seconds': 100}, {'selmer_seconds': 100, 'seconds': 7}):
                ctp.after_selmer({'claims': {}}, **kw)
        finally:
            ctp.add = real
        self.assertEqual(seen, [ctp.SECONDS, 100, 7])

if __name__ == '__main__':
    unittest.main()
