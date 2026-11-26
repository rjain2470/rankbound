"""The prover without PARI/GP (arith.py and the parts of curve, field, descent, selmer and ctp ported from PARI):
`rankbound bound` with no gp on the PATH (the witness is written, not checked), and, where gp is installed, its
results against PARI's on random curves (the minimal model, Tate's algorithm, the root number, the torsion, the field
data, the 2-isogeny descent) and its witnesses through the checker; the HNF and the prime ideals."""
import json, os, random, shutil, subprocess, sys, tempfile, unittest
from fractions import Fraction
from rankbound import arith, curve as ec, field
from rankbound.check import run_gp
from helpers import SRC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAVE_GP = shutil.which('gp') is not None
ORACLE = os.path.join(ROOT, 'tests', 'data', 'oracle', '')


def kubert(n, t):
    """A curve with a rational point of order n (Kubert's Tate normal forms; 24, 26: Z/2 x Z/4, Z/2 x Z/6)."""
    t = Fraction(t)
    b, c = {4: (t, 0), 5: (t, t), 6: (t + t * t, t), 7: (t ** 3 - t * t, t * t - t), 24: (t * t - Fraction(1, 16), 0)}.get(
        n, (None, None))
    if n == 8:
        b = (2 * t - 1) * (t - 1); c = b / t
    if n == 9:
        c = t * t * (t - 1); b = c * (t * t - t + 1)
    if n == 26:
        c = (10 - 2 * t) / (t * t - 9); b = c + c * c
    b, c = Fraction(b), Fraction(c)
    d = b.denominator * c.denominator
    return [int(x * d ** i) for x, i in zip([1 - c, -b, -b, 0, 0], [1, 2, 3, 4, 6])]


def curves(n, seed):
    rnd, out = random.Random(seed), []
    while len(out) < n:
        k = rnd.random()
        if k < 0.4:
            a = [rnd.randrange(2), rnd.randrange(-1, 2), rnd.randrange(2), rnd.randrange(-10 ** 4, 10 ** 4),
                 rnd.randrange(-10 ** 6, 10 ** 6)]
        elif k < 0.7:
            e2, e3 = rnd.randrange(5), rnd.randrange(4)
            a = [0, 0, 0, rnd.randrange(-200, 200) * 2 ** e2 * 3 ** e3, rnd.randrange(-200, 200) * 2 ** e2 * 3 ** e3]
        else:
            try:
                a = kubert(rnd.choice([4, 5, 6, 7, 8, 9, 24, 26]), Fraction(rnd.randrange(-30, 30), rnd.randrange(1, 12)))
            except ZeroDivisionError:
                continue
        if ec.cinv(a)[2]:
            out.append(a)
    return out


class TestNoGP(unittest.TestCase):
    """`rankbound bound` with no gp: the usual lines, the last without "(checked)", and (with gp) the checker accepts
    the witness."""

    def test_bound(self):
        d = tempfile.mkdtemp()
        try:
            env = dict(os.environ, PATH='/usr/bin:/bin', RANKBOUND_PROVER='py', PYTHONPATH=SRC)
            for ai in ('0,0,0,-25,0', '0,-1,1,-929,-10595'):
                r = subprocess.run([sys.executable, '-m', 'rankbound', 'bound', ai, '--compute', '--dir', d, '--json'],
                                   capture_output=True, text=True, cwd=d, env=env)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                self.assertIsNone(json.loads(r.stdout)['checked'])
                end = r.stderr.splitlines()[-1]
                self.assertRegex(end, r'^\S+ Rank (at most )?\d+, (assuming the GRH|unconditionally)\. Witness: \S+\.witness\.json(\.gz)?$')
            wits = [os.path.join(w, f) for w, _, fs in os.walk(d) for f in fs if f.endswith(('.witness.json', '.witness.json.gz'))]
            self.assertEqual(len(wits), 2)
            if HAVE_GP:
                from rankbound import check as verify
                for c in wits:
                    self.assertEqual(verify.check(c)['verdict'], 'accepted', c)
        finally:
            shutil.rmtree(d)


@unittest.skipUnless(HAVE_GP, 'needs PARI/GP as the oracle')
class TestAgainstPARI(unittest.TestCase):

    def test_curves(self):
        """The minimal model and its change of variables, elllocalred, ellrootno and elltors, as PARI's."""
        cs = curves(150, 1)
        out = run_gp('{ foreach([%s], a, my(M = ellminimalmodel(ellinit(a), &v), P = factor(abs(M.disc))[, 1]~);\n'
                     '  print(M[1..5], ";", v, ";", apply(p -> my(r = elllocalred(M, p)); [r[1], r[2], r[4]], P), ";",'
                     ' ellrootno(M), ";", elltors(M)[2])); }' % ','.join(map(str, cs))).splitlines()
        for a, l in zip(cs, out):
            mm, v = ec.minimal(a)
            P = arith.prime_divisors(ec.cinv(mm)[2])
            mine = '%s;%s;%s;%d;%s' % (mm, v, [list(ec.localred(mm, p)) for p in P], ec.rootno(mm, P), ec.torsion_group(mm)[0])
            self.assertEqual(mine.replace(' ', ''), l.replace(' ', ''), a)

    def test_field(self):
        """field.json from sfield, as the PARI prover's field.gp gave it (tests/data/oracle), but for the reduced
        form, which may be another."""
        for a in [c for c in curves(60, 2) if not ec.int_roots(_T(ec.minimal(c)[0]))][:15]:
            P = arith.prime_divisors(ec.cinv(ec.minimal(a)[0])[2])
            F, _ = field.field_data(P, a)
            out = run_gp('sfield(%s, %s);' % (a, P), libs=(ORACLE + 'field.gp',)).splitlines()
            sf = dict(l.split('\t')[1:3] for l in out if l.startswith('SF\t'))
            self.assertEqual((F['dK'], F['GB'], F['ns'], F['u'], F['n'], F['w']),
                             (sf['dK'], int(sf['GB']), int(sf['ns']), int(sf['u']), int(sf['n']), int(sf['w'])), a)

    def test_descent(self):
        """The 2-isogeny descent (with the Cassels-Tate pairings) gives the PARI prover's bound (tower.gp's tdprove),
        and the verifier accepts its witness."""
        from rankbound import descent as tower, check as verify
        d = tempfile.mkdtemp()
        try:
            rnd = random.Random(3)
            for _ in range(8):
                e1, e2 = rnd.randrange(-300, 300), rnd.randrange(-300, 300)
                ai = [0, rnd.randrange(-200, 200), 0, rnd.randrange(-5000, 5000), 0] if rnd.random() < .5 else \
                     [0, -(e1 + e2), 0, e1 * e2, 0]
                if not ec.cinv(ai)[2]:
                    continue
                mm = ec.minimal(ai)[0]
                P = arith.prime_divisors(ec.cinv(mm)[2])
                D = tower.descent(ai, [str(p) for p in P])
                p = os.path.join(d, 'c.witness.json')
                from rankbound import witness
                witness.write(tower.witness('c', [str(p) for p in P], D, None), p)
                self.assertEqual(verify.check(p)['verdict'], 'accepted', ai)
                out = run_gp('tdprove(%s, %s, [], [], 100, 0);' % (mm, P), libs=(
                    ORACLE + 'field.gp', ORACLE + 'ctp.gp', ORACLE + 'tower.gp'))
                bs = next(l.split('\t')[1:] for l in out.splitlines() if l.startswith('BS'))
                self.assertEqual([D['B'], D['S']], [int(x) for x in bs], ai)
        finally:
            shutil.rmtree(d)


class TestHNF(unittest.TestCase):
    def test_modular(self):
        """arith.hnf modulo a multiple of the determinant (Cohen 2.4.8) is the HNF."""
        nf = arith
        rnd = random.Random(5)
        for _ in range(3000):
            rows = [[rnd.randint(-10 ** rnd.randint(1, 25), 10 ** rnd.randint(1, 25)) for _ in range(3)]
                    for _ in range(rnd.randint(3, 10))]
            try:
                H = nf.hnf(rows)
            except StopIteration:               # rank < 3
                continue
            self.assertEqual(nf.hnf(rows, H[0][0] * H[1][1] * H[2][2] * rnd.choice([1, 2, 12])), H, rows)

    def test_prime(self):
        """Field.prime's degree-1 primes without the decomposition have dec's P^-1 (ct.pinv_ideal)."""
        from rankbound import ctp as ct, field as nf
        for f in ([13888888889083333333334, 0, 0, 3], [-929 * 4, 5, 7, 2], [101, -3, 0, 1]):
            K, n = nf.Field(f), 0
            for p in arith.primes(5, 400):
                for P in K.dec(p):
                    K2 = nf.Field(f)
                    Q = K2.prime(p, P.label)
                    n += not isinstance(Q, nf.Prime)
                    self.assertEqual(ct.pinv_ideal(K2, Q), ct.pinv_ideal(K, P), (f, P.label))
            self.assertGreater(n, 20)


def _T(mm):
    b2, b4, b6, _ = ec.binv(mm)
    return [16 * b6, 8 * b4, b2, 1]


if __name__ == '__main__':
    unittest.main()
