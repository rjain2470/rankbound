"""The checker of the 2-isogeny format (rankbound/check/__init__.py, check.gp part 3): the local solubility test
against a brute-force oracle; witnesses (round trip, tampering) and forged ones."""
import json, os, shutil, tempfile, unittest
from rankbound import check as checker
from rankbound.check import run_gp
from helpers import DESCENT, TESTS
from test_check import fails


def gp(body, seconds=600):
    """Return the output lines of body (GP source; a multi-line block in braces starting a line), run after
    check.gp."""
    return run_gp(body, seconds=seconds).splitlines()


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


class TestWitness(unittest.TestCase):
    """rankbound() on y^2 = x^3 - 25x with the point (-4, 6): rank = 1 proved, unconditionally, by a witness the
    checker accepts; a changed Selmer basis element, S, torsion, GRH, point or a missing isogeny fails."""

    @classmethod
    def setUpClass(cls):
        from rankbound import rankbound
        cls.tmp = tempfile.mkdtemp()
        cls.res = rankbound([0, 0, 0, -25, 0], points=[(-4, 6)], workdir=os.path.join(cls.tmp, 'n5'))
        with open(cls.res.witness) as f:
            cls.raw = f.read()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_result(self):
        r = self.res
        self.assertEqual((r.rank_bound, r.rank_lower_bound, r.r_proved, r.proved_equal, r.checked, r.grh, r.method,
                          r.torsion, r.details['w'], r.details['outcome']),
                         (1, 1, True, True, True, False, '2-isogeny descent', [2, 2], -1, 'descent'))
        self.assertEqual(r.statement, 'Unconditionally, rank E(Q) <= 1. As E(Q) has 1 independent rational point, we '
                                      'therefore have rank E(Q) = 1.')
        for f in ('curve.json', 'check.log', 'check.json', 'run.json'):
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
        v = checker.check(self.res.witness, curve=[0, 0, 0, -25, 0])
        self.assertEqual((v['verdict'], v['rank'], v['GRH'], v['method'], v['torsion'], v['G']),
                         ('accepted', 1, 'no', '2-isogeny descent', [2, 2], None))
        self.assertIn('no GRH', v['hypothesis'])
        self.assertEqual(checker.check(self.res.witness, curve=[0, 0, 0, -36, 0])['verdict'], 'rejected')

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
    """Witnesses forged from honest ones (the 2.0 code review): ICARM #2209's Cassels-Tate data with an extra prime
    listed twice (and its point twice), its rank raised and the claims lowered to rank <= -1, which the checker
    accepted before the extra primes had to be strictly increasing; a negative bound; a singular model."""
    FORGED = os.path.join(TESTS, 'data', 'forged2209.cert.json')

    def setUp(self):
        with open(self.FORGED) as f:
            self.c = json.load(f)

    def test_repeated_prime(self):
        """Rejected as malformed by the form check, and by gp's own check of the pair without it."""
        self.assertIn(['2786783', '2786783'], [p['primes'] for p in self.c['cassels_tate']['pairs']])
        v = checker.check(self.FORGED)
        self.assertEqual(v['verdict'], 'rejected')
        self.assertEqual([x['check'] for x in v['checks'] if not x['ok']], ['format'])
        out = run_gp(checker.gp_body(self.c))     # gp alone, past the form check
        chk = {t[1]: t[2] for t in (l.split(' ', 3) for l in out.splitlines()) if t[0] == 'CHK'}
        self.assertEqual(chk.get('cassels_tate'), '0')
        self.assertNotIn('TD -1 -1', out)

    def test_negative_bound(self):
        got = []
        checker.claims(self.c, 'TD -1 -1 %d' % self.c['local']['w'], lambda n, passed, info='': got.append((n, passed)))
        self.assertIn(('claims', False), got)

    def test_singular(self):
        """A singular model is a malformed witness (rejected, exit 1), not an error of the checker (exit 4)."""
        with tempfile.TemporaryDirectory() as d:
            for k in ('ainvs', 'minimal_model'):
                c = json.loads(json.dumps(self.c))
                c['curve'][k] = ['0', '0', '0', '0', '0']
                self.assertEqual(fails(c, d), ['format'], k)


if __name__ == '__main__':
    unittest.main()
