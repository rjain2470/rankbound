"""Tests against PARI and independent computations: the GRH bound, the prover's row rule, the elimination, and G on
fields whose class groups bnfinit knows."""
import gzip, json, math, os, random, shutil, subprocess, sys, tempfile, unittest
from unittest import mock
from rankbound import check as vf
from rankbound.check import selmer as vsel
from rankbound import arith, curve as cv, field, linalg
from rankbound import sieve
from helpers import REFROWS, SRC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# a folder of the published witnesses, if there is one (RANKBOUND_WITNESSES; default witnesses/ in this checkout)
WITS = os.environ.get('RANKBOUND_WITNESSES') or os.path.join(ROOT, 'witnesses')
WIT728 = os.path.join(WITS, 'icarm728.cert.json')
WIT8 = WIT728.replace('icarm728', 'icarm8')       # not in git: the tests read REF
# tests/data/reference.json: the curve and rank_lower_bound of ICARM #728 and #8, and 1,000 of #728's relations
# (every len // 1000-th, free ones left out); test_fixture checks it against the witnesses
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'reference.json')) as _f:
    REF = json.load(_f)
FIELD_GP = os.path.join(ROOT, 'tests', 'data', 'oracle', 'field.gp')

SMALL = [  # a-invariants, dim Cl(K)[2] (bnfinit)
    ([0, 0, 0, 421, 141594], 0), ([0, 0, 0, -1623, -148259], 2), ([0, 0, 0, 1527, -177941], 1),
    ([1, -1, 0, 31, -412], 0), ([0, 0, 0, 474, -13363], 1), ([0, 0, 0, 259, -92764], 1), ([0, 0, 0, 1933, 102706], 2),
    ([0, 0, 0, 134, -2569], 2), ([0, 0, 0, -953, 74673], 3), ([0, 0, 0, -1930, -32937], 3),
    ([0, 0, 1, -1, 0], 0), ([0, -1, 1, -929, -10595], 1)]


FIELDS = {}


def field_of(ai):
    """Return (field dict, small labels) of the curve AI, computed once per test run."""
    if tuple(ai) not in FIELDS:
        cur = cv.from_ainvs(ai)
        F, small = field.compute(cur.ainvs, cur.bad_primes)
        F['bad_primes'] = cur.bad_primes
        FIELDS[tuple(ai)] = F, small
    return FIELDS[tuple(ai)]


def write_roots(f, pmax, out):
    """Write the roots of F mod p for p < pmax in makefb's format (by trial: sieve.roots), for the free relations."""
    with gzip.open(out, 'wt') as fo:
        for p in (lambda n: arith.primes(2, n))(pmax):
            rs = sieve.roots(f, p)
            if rs:
                fo.write('%d: %s\n' % (p, ','.join(map(str, rs))))


def line_relations(F, lpb, b1, A=2 ** 9):
    """Return the relations (a, b, H) with b in [1, b1), a in [-A, A) and F(a, b) 2^lpb-smooth: a line sieve by logs,
    then trial division (a reference, independent of lsieve)."""
    f, fb = F['poly'], [(p, sieve.roots(F['poly'], p), math.log2(p)) for p in (lambda n: arith.primes(2, n))(2 ** lpb)]
    div, out = [p for p, rs, _ in fb if rs], []
    for b in range(1, b1):
        logs = [0.0] * (2 * A)
        for p, rs, lp in fb:
            for r in rs:
                if r == p:
                    if b % p == 0:
                        logs = [x + lp for x in logs]
                    continue
                for i in range((r * b + A) % p, 2 * A, p):
                    logs[i] += lp
        for i in range(2 * A):
            a = i - A
            v = abs(field.norm(f, a, b))
            if math.gcd(a, b) != 1 or v == 0 or logs[i] < math.log2(v) - 2 * lpb:
                continue
            m, H = v, []
            for p in div:
                while m % p == 0:
                    m //= p; H.append(p)
                if m == 1:
                    break
            if m == 1:
                out.append((a, b, H))
    return out


def bound_of(F, small, rels, lpb):
    M = linalg.Matrix(F, small)
    d = tempfile.mkdtemp()
    try:
        fb = os.path.join(d, 'roots.gz')
        write_roots(F['poly'], 2 ** lpb, fb)
        for p, labs in linalg.free_rows(F, fb, 2 ** lpb):
            M.add_free(p, labs)
        for a, b, H in rels:
            M.add_relation('%d,%d:%s' % (a, b, ','.join('%x' % h for h in H)))
        return linalg.solve(M, d, 0).bound
    finally:
        shutil.rmtree(d)


def pari_labels(F, rels):
    """Return the odd-valuation labels of the relations (a, b, H) by PARI's idealval (REFROWS: "p:r" is the label
    (p, r))."""
    f = ' + '.join('(%d)*x^%d' % (c, i) for i, c in enumerate(F['poly']))
    body = REFROWS + 'f = %s; K = ckinit(f, [%s]);\nrefrows(K, [%s]);\n' % (f, ','.join(F['bad_primes']), ','.join(
        '[%d,%d,[%s]]' % (a, b, ','.join(map(str, H))) for a, b, H in rels))
    cols = {}
    for l in vf.run_gp(body, libs=(), seconds=600).splitlines():
        t = l.split()
        if t and t[0] == 'R':
            cols[int(t[1])] = {tuple(map(int, z.split(':'))) for z in t[2:]}
    return [cols[i] for i in range(len(rels))]


def f2rank(vecs):
    basis = {}
    for v in vecs:
        while v:
            h = v.bit_length() - 1
            if h not in basis:
                basis[h] = v; break
            v ^= basis[h]
    return len(basis)


class TestReference(unittest.TestCase):
    def test_bdf(self):
        """The BDF bound (field.gp) = PARI's internal LIMC2 on 4 fields, and 200,271 for the field of Elkies's rank-28 curve."""
        body = r'''
f=64023127168000*x^3 + 10309553525987840512490787747*x^2 - 3858878002265332645698861066081585182608*x - 69043295714402138353376748510210837676894689434302674;
addprimes([376018840263193489397987439236873583997122096511452343225772113000611087671413,315574902691581877528345013999136728634663121]);
print("B ", bdf(nfinit(f), 10^6));
foreach([x^3-x-1, x^3-1000003*x+7, 5*x^3+123456*x^2-9876543*x+1234567891, x^3+x^2-2*x-1+3*10^9], g, print("B ", bdf(nfinit(g), 10^6)));
'''
        got = [int(l.split()[1]) for l in vf.run_gp(body, libs=(FIELD_GP,)).splitlines() if l.startswith('B ')]
        self.assertEqual(got, [200271, 7, 794, 2772, 2399])
        from rankbound import field as nf                    # the prover's (field.bdf), on the maximal orders
        big = [376018840263193489397987439236873583997122096511452343225772113000611087671413,
               315574902691581877528345013999136728634663121]
        polys = [[-69043295714402138353376748510210837676894689434302674, -3858878002265332645698861066081585182608,
                  10309553525987840512490787747, 64023127168000], [-1, -1, 0, 1], [7, -1000003, 0, 1],
                 [1234567891, -9876543, 123456, 5], [3 * 10 ** 9 - 1, -2, 1, 1]]
        mine = []
        for g in polys:
            F = (g[3], g[2], g[1], g[0])
            ps = [p for p, _ in arith.factor(nf.disc(F), 10 ** 6, big)]
            F, _ = nf.maximal_form(g, ps)
            f = [F[3], F[2], F[1], F[0]]
            mine.append(field.bdf(f, nf.disc(F), len(arith.real_roots_floor(f, 128)), 10 ** 6))
        self.assertEqual(mine, got)

    def test_bdfcert(self):
        """The exact check of (10) agrees with the floating-point one at the BDF bound T and at T - 1 (it fails there:
        T is the smallest), and its constants enclose PARI's at 80 digits."""
        body = r'''{
foreach([x^3-x-1, x^3-1000003*x+7, 5*x^3+123456*x^2-9876543*x+1234567891, x^3+x^2-2*x-1+3*10^9], g,
  my(nf = nfinit(g), T = bdf(nf, 10^6), D = bdfdata(nf, T)); print("C ", bdfcert(nf, D, T), bdfcert(nf, D, T - 1)));
default(realprecision, 80);
my(C = bdfconst(), c = [Pi, Euler, Catalan, log(Pi)]); print("K ", vector(4, i, C[i] <= c[i] && c[i] <= C[i] + 10^-40));
my(l2 = bdfat(1/3)); print("K ", vector(4, i, my(n = [2, 3, 10^9 + 7, 2^200 + 1][i], L = bdflog(n, l2));
  L[1] <= log(n) && log(n) <= L[2] && L[2] - L[1] < 10^-35));
}'''
        out = vf.run_gp(body, libs=(FIELD_GP,)).splitlines()
        self.assertEqual([l for l in out if l.startswith('C ')], ['C 10'] * 4)
        self.assertEqual([l for l in out if l.startswith('K ')], ['K [1, 1, 1, 1]'] * 2)

    @unittest.skipUnless(os.path.exists(WIT728) and os.path.exists(WIT8), 'needs the published witnesses of ICARM #728 '
                         'and #8 (from the Zenodo record; set RANKBOUND_WITNESSES to their folder)')
    def test_fixture(self):
        """REF is what the witnesses of ICARM #728 and #8 give (where they are)."""
        for name, path in (('icarm728', WIT728), ('icarm8', WIT8)):
            c = vf.load(path)
            want = {'name': c['name'], 'curve': c['curve'], 'rank_lower_bound': c['rank_lower_bound']}
            if name == 'icarm728':
                rels = [l for l in c['relations'] if not l.startswith('f:')]
                want['relations'] = rels[::len(rels) // 1000][:1000]
            self.assertEqual(REF[name], want, name)

    def test_rows_728(self):
        """The prover's row rule = PARI's idealval on ICARM #728 (non-monic index form: J correction at 2, 3, 11, 13,
        23, 1021, 1223, 2756959289691073), on 1,000 of its witness's relations (REF)."""
        c = REF['icarm728']['curve']
        F, small = field.compute(c['minimal_model'], c['bad_primes'])
        F['bad_primes'] = c['bad_primes']
        self.assertEqual((F['GB'], F['ns'], F['u'], F['n'], F['w']), (92085, 8888, 1, 11, -1))
        rels = [linalg.parse_line(l) for l in REF['icarm728']['relations']]
        self.assertEqual(len(rels), 1000)
        F['poly'] = REF['icarm728']['poly']         # PARI's index form, which the relations are for (field.compute's
        J = [(p, p) for p, e in arith.factor(abs(F['poly'][3])) if e % 2]      # may be another): J at (1:0)
        self.assertEqual([linalg.relation_labels(a, b, H, J) for a, b, H in rels], pari_labels(F, rels))

    def test_rows_small(self):
        for ai, _ in SMALL:
            F, small = field_of(ai)
            rels = line_relations(F, 10, 3)[:150]
            J = [tuple(l) for l in F['J']]
            self.assertEqual([linalg.relation_labels(a, b, H, J) for a, b, H in rels], pari_labels(F, rels), ai)

    def test_class_groups(self):
        """G (line siever + f2elim) = dim Cl(K)[2] from bnfinit on 12 small fields (2-ranks 0..3)."""
        for ai, g in SMALL:
            F, small = field_of(ai)
            lpb = max(10, F['GB'].bit_length() + 3)
            self.assertEqual(bound_of(F, small, line_relations(F, lpb, 40), lpb), g, (ai, F['GB'], F['ns']))

    def test_brumer_kramer(self):
        """S = the parity bound of g + u + n (u, n from the verifier's vbk, g from bnfinit) is never below ellrank's
        upper bound on the rank (<= dim Sel_2) on 200 random curves, half of them with a1, a2, a3 != 0 (p = 2, 3); it
        equals it on many (a run on 600 curves: 341 equal, none below)."""
        body = '''{ my(cnt = 0, eq = 0); setrand(20261014);
  while(cnt < 200, my(ai = [0, 0, 0, random(4001) - 2000, random(40001) - 20000], E, T, nf, g, bk, S, R);
    if(cnt % 2, ai = concat([random(3) - 1, random(3) - 1, random(2)], ai[4..5]));
    E = iferr(ellinit(ai), e, []); if(#E == 0, next);
    E = ellminimalmodel(E); T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6; if(!polisirreducible(T), next);
    nf = nfinit(T); g = #select(c -> c % 2 == 0, bnfinit(T, 1).cyc);
    bk = vbk(E, nf, factor(abs(E.disc))[, 1]~); S = g + bk[1] + bk[2]; if((-1)^S != ellrootno(E), S--);
    R = ellrank(E)[2]; cnt++; eq += (S == R);
    if(S < R, print("BELOW ", E[1..5], " S ", S, " R ", R)));
  print("EQ ", eq); }'''
        out = vf.run_gp(body, libs=(), seconds=600)
        self.assertNotIn('BELOW', out)
        self.assertGreater(int(out.split('EQ ')[1]), 50)

    def test_vrank_rank0(self):
        """The rank proof of r = 0 needs no point (571a1 has rank 0); r = 1 with no point, or a point off the curve,
        fails."""
        def chk(r, pts, prs):
            out = vf.run_gp('vrank([0,-1,1,-929,-10595], %d, [%s], [%s]);' % (r, pts, prs), libs=())
            return 'CHK rank_lower_bound 1' in out
        self.assertTrue(chk(0, '', ''))
        self.assertFalse(chk(1, '', ''))
        self.assertFalse(chk(1, '[0,0]', '3'))

    def test_rank_primes(self):
        """The witness keeps the first r primes whose 2-signatures have rank r (ICARM #8, r = 22: its primes, each
        listed twice, give them back once; the points stay as they are on their own model)."""
        c = REF['icarm8']
        rl = c['rank_lower_bound']
        got = cv.rank_proof(c['curve']['ainvs'], c['curve']['bad_primes'], 22,
                                {'points': rl['points'], 'primes': [p for p in rl['primes'] for _ in '12']})
        self.assertEqual(got, rl)

    def test_rank_proof_models(self):
        """Points of one model are written on another: mapped exactly, primes chosen on the target model or completed
        by a search, and vrank passes.  A point off its model, another curve, too few points or dependent points are
        refused; 2P is halved.  37a1, P = (0, 0); its short model y^2 = x^3 - 1296x + 11664."""
        mm, short, P = ['0', '0', '1', '-1', '0'], ['0', '0', '0', '-1296', '11664'], [['0', '0']]
        prs = [p for p in range(2, 100) if all(p % q for q in range(2, p))]
        rl = cv.rank_proof(short, ['37'], 1, {'points': P, 'primes': prs, 'ainvs': mm})
        self.assertIn(rl['points'], ([['0', '108']], [['0', '-108']]))
        self.assertTrue(set(rl['primes']) <= set(prs) - {2, 3})
        body = 'vrank([%s], 1, [[%s,%s]], [%s]);' % (','.join(short), *rl['points'][0], ','.join(map(str, rl['primes'])))
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp(body, libs=()))
        self.assertEqual(cv.rank_proof(mm, ['37'], 1, {'points': P, 'primes': prs})['points'], P)   # same model
        rl = cv.rank_proof(short, ['37'], 1, {'points': P, 'primes': [3], 'ainvs': mm})    # 3 | the short disc
        self.assertEqual(len(rl['primes']), 1)
        self.assertNotEqual(rl['primes'], [3])
        body = 'vrank([%s], 1, [[%s,%s]], [%s]);' % (','.join(short), *rl['points'][0], ','.join(map(str, rl['primes'])))
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp(body, libs=()))
        for ai, proof, err in ((short, {'points': [['0', '1']], 'primes': prs, 'ainvs': mm}, 'Point 1 is not on the curve'),
                               (['0', '0', '1', '-1', '1'], {'points': P, 'primes': prs, 'ainvs': mm}, 'The points are given on \\[0,0,1,-1,0\\], which is not a model of this curve'),
                               (short, {'points': [], 'primes': prs, 'ainvs': mm}, '^r = 1 but 0 points were given')):
            with self.assertRaisesRegex(RuntimeError, err), mock.patch.object(cv, 'PRIME_SEARCH', 10 ** 4):
                cv.rank_proof(ai, ['37'], 1, proof)
        with mock.patch.object(cv, 'PRIME_SEARCH', 10 ** 4):     # 2P, halved on the short model
            rl = cv.rank_proof(short, ['37'], 1, {'points': [['1', '0']], 'primes': [2, 3, 5, 7], 'ainvs': mm})
        self.assertIn(rl['points'], ([['0', '108']], [['0', '-108']]))

    def test_rank_proof_search(self):
        """Points given without primes (rankbound(points=...)): the primes are searched, and vrank passes with them;
        37a1's 2P = (1, 0) is in 2E(Q): halved to +-P, it proves rank >= 1; P and -P are dependent: refused.  Two
        points of rank 2 (389a1: (-1, 1), (0, 0)) need two primes."""
        rl = cv.rank_proof(['0', '0', '1', '-1', '0'], ['37'], 1, {'points': [['0', '0']]})
        self.assertEqual(len(rl['primes']), 1)
        body = 'vrank([0,0,1,-1,0], 1, [[0,0]], [%d]);' % rl['primes'][0]
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp(body, libs=()))
        with mock.patch.object(cv, 'PRIME_SEARCH', 10 ** 4):
            rl = cv.rank_proof(['0', '0', '1', '-1', '0'], ['37'], 1, {'points': [['1', '0']]})
        self.assertIn(rl['points'], ([['0', '0']], [['0', '-1']]))
        body = 'vrank([0,0,1,-1,0], 1, [[%s,%s]], [%d]);' % (*rl['points'][0], rl['primes'][0])
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp(body, libs=()))
        with self.assertRaisesRegex(RuntimeError, 'look dependent; check them'), \
                mock.patch.object(cv, 'PRIME_SEARCH', 10 ** 4):
            cv.rank_proof(['0', '0', '1', '-1', '0'], ['37'], 2, {'points': [['0', '0'], ['0', '-1']]})
        rl = cv.rank_proof(['0', '1', '1', '-2', '0'], ['389'], 2, {'points': [['-1', '1'], ['0', '0']]})
        self.assertEqual(len(rl['primes']), 2)
        body = 'vrank([0,1,1,-2,0], 2, [[-1,1],[0,0]], [%s]);' % ','.join(map(str, rl['primes']))
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp(body, libs=()))

    def test_rank_one_cyclic(self):
        """r = 1 without primes: vrank accepts a point of infinite order and refuses a torsion point.  ICARM #3246,
        y^2 = x^3 - 9x + 9 (a cyclic cubic field, torsion Z/3), P = (1, 1); rank_proof gives it without primes, as
        whenever the 2-signatures fail for r = 1."""
        rl = cv.rank_proof(['0', '0', '0', '-9', '9'], ['2', '3'], 1, {'points': [['1', '1']]})
        self.assertEqual((rl['points'], rl['primes']), ([['1', '1']], []))
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp('vrank([0,0,0,-9,9], 1, [[1,1]], []);', libs=()))
        out = vf.run_gp('T = elltors(ellinit([0,0,0,-9,9]))[3][1]; vrank([0,0,0,-9,9], 1, [T], []);', libs=())
        self.assertIn('CHK rank_lower_bound 0', out)
        # ICARM #1791 (an S3 field): its saturated point lies in 2E(F_p) at every usable prime, so no 2-signature
        # proves it; the same point of infinite order does
        with mock.patch.object(cv, 'PRIME_SEARCH', 10 ** 4):
            rl = cv.rank_proof(['1', '-1', '0', '-6', '8'], ['2', '3'], 1, {'points': [['-2', '4']]})
        self.assertEqual((rl['points'], rl['primes']), ([['-2', '4']], []))
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp('vrank([1,-1,0,-6,8], 1, [[-2,4]], []);', libs=()))

    def test_split_primes(self):
        """The free relations of a run on given relations (field.write_split, PARI) are those of the roots by trial
        (sieve.roots) on the fields of SMALL: below a bound, and at the extra primes given above it (the large primes of
        the relations)."""
        d = tempfile.mkdtemp()
        try:
            big = [p for p in range(500, 2000) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
            for ai, _ in SMALL[:6]:
                F, _ = field_of(ai)
                a, b, c = os.path.join(d, 'a.gz'), os.path.join(d, 'b.gz'), os.path.join(d, 'c.gz')
                field.write_split(F['poly'], 2000, a)
                write_roots(F['poly'], 2000, b)
                field.write_split(F['poly'], 500, c, extra=big + [7, 1999])     # below 500 or repeated: ignored
                self.assertEqual(list(linalg.free_rows(F, a, 2000)), list(linalg.free_rows(F, b, 2000)), ai)
                self.assertEqual(list(linalg.free_rows(F, c, 2000)), list(linalg.free_rows(F, b, 2000)), ai)
                os.remove(a); os.remove(b); os.remove(c)
        finally:
            shutil.rmtree(d)

    def test_bad_primes(self):
        """field.gp refuses a composite, repeated, extra or missing bad prime with a ValueError carrying gp's message;
        the command line prints it as one line."""
        E571 = [0, -1, 1, -929, -10595]
        with self.assertRaisesRegex(ValueError, '^The bad prime 3997 is not prime.$'):
            field.compute(E571, [571 * 7])
        with self.assertRaisesRegex(ValueError, '^The bad prime 571 is listed twice.$'):
            field.compute(E571, [571, 571])
        with self.assertRaisesRegex(ValueError, '^The bad prime 7 does not divide the minimal discriminant.$'):
            field.compute(E571, [571, 7])
        with self.assertRaisesRegex(ValueError, '^The bad primes given do not account for the whole discriminant.$'):
            field.compute(E571, [])
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.run([sys.executable, '-m', 'rankbound', 'bound', '--estimate', '--dir', d, '0,-1,1,-929,-10595',
                                '--bad-primes', '7'], cwd=d, env=dict(os.environ, PYTHONPATH=SRC), capture_output=True,
                               text=True, timeout=600)
            self.assertEqual((p.returncode, p.stderr.strip()), (1, 'The bad prime 7 does not divide the minimal discriminant.'))

    def test_f2elim(self):
        """f2elim's bound = ns - rank(M) + rank(M_L) on random sparse matrices, and the verifier's replay of its ops and
        pivots gives the same dim V."""
        rnd = random.Random(5)
        for trial in range(25):
            ns, nl = rnd.randint(5, 60), rnd.randint(20, 400)
            nrows = rnd.randint(nl // 2, nl + ns + 40)
            M = linalg.Matrix({'J': []}, [(i, 0) for i in range(ns)])
            for _ in range(nrows):
                cols = set(rnd.sample(range(ns), rnd.randint(0, min(4, ns)))) | {ns + c for c in rnd.sample(range(nl), rnd.randint(1, 3))}
                M._add([(c, 0) if c < ns else ('L', c) for c in cols], 'x')
            ids = [set(r) for r in M.rows]
            rM = f2rank([sum(1 << c for c in r) for r in ids])
            rL = f2rank([sum(1 << c for c in r if c >= ns) for r in ids])
            d = tempfile.mkdtemp()
            try:
                res = linalg.solve(M, d, 0, dense=rnd.choice([0, 5, 50]))
                self.assertEqual(res.bound, ns - rM + rL)
                kept, piv, need = linalg.cone(res.ops, res.piv)
                names = ['%d:1' % c if c < ns else '%d:1:2' % c for c in range(len(M.col))]
                self.assertEqual(vsel.rank_V(ids, names, kept, piv, 10 ** 9)[0], res.dimV)
            finally:
                shutil.rmtree(d)

    def test_characters(self):
        """f2elim with characters: the bound is ns - rank(M) + rank(M_L) and crank = rank([M | C]) - rank(M), C the
        rows' character bits ((a - b r / q) = -1), on random sparse matrices, untrimmed; no crank with an element beyond
        2^62, or when a character vanishes on a row."""
        rnd = random.Random(9)
        Q = [q for q in range(1001, 2000, 2) if arith.is_prime(q)]     # f2elim's tables of squares
        Q2 = [q for q in range(2 ** 22 + 1, 2 ** 22 + 3000, 2) if arith.is_prime(q)]     # its Jacobi symbols
        with tempfile.TemporaryDirectory() as d:
            for trial in range(30):
                ns, nl, K = rnd.randint(3, 50), rnd.randint(10, 300), rnd.randint(1, 64)
                chars = [(q, rnd.randrange(q)) for q in rnd.sample(Q if trial % 2 else Q2, K)]
                M = linalg.Matrix({'J': []}, [(i, 0) for i in range(ns)])
                ints, cints = [], []
                for _ in range(rnd.randint(1, nl + 2 * ns)):
                    while True:
                        a, b = rnd.randint(-10 ** 6, 10 ** 6), rnd.randint(0, 10 ** 4)
                        if all((a - b * r) % q for q, r in chars):
                            break
                    cols = set(rnd.sample(range(ns), rnd.randint(0, min(3, ns)))) | \
                        {ns + c for c in rnd.sample(range(nl), rnd.randint(0, 3))}
                    M._add([(c, 0) if c < ns else ('L', c) for c in cols], 'x', a, b)
                    r = M.rows[-1]
                    ints.append(sum(1 << c for c in r))
                    cints.append(sum(1 << j for j, (q, rt) in enumerate(chars)
                                     if pow((a - b * rt) % q, (q - 1) // 2, q) != 1))
                res = linalg.solve(M, os.path.join(d, str(trial)), 0, keep=-1, chars=chars, threads=rnd.choice([1, 3]))
                nc = len(M.col)
                rM, rL = f2rank(ints), f2rank([x >> ns for x in ints])
                self.assertEqual((res.bound, res.dimV), (ns - rM + rL, rM - rL), trial)
                self.assertEqual(res.crank, f2rank([x | (c << nc) for x, c in zip(ints, cints)]) - rM, trial)
            # an element beyond 2^62, then a character that vanishes on a row: no crank, the same bound
            M = linalg.Matrix({'J': []}, [(i, 0) for i in range(3)])
            for t in range(6):
                M._add([(t % 3, 0), ('L', t // 2)], 'x', 5 + t, 1)
            for k, row in enumerate(((2 ** 62, 1), (500 * 3 + 1009, 3))):
                self.assertIsNotNone(linalg.solve(M, os.path.join(d, 'a%d' % k), 0, keep=-1, chars=[(1009, 500)]).crank)
                M._add([(0, 0)], 'x', *row)
                self.assertIsNone(linalg.solve(M, os.path.join(d, 'b%d' % k), 0, keep=-1, chars=[(1009, 500)]).crank)
                M.rows.pop(); M.src.pop(); M.elt.pop(); M.elt.pop(); M.big = False

    def test_cubic_root(self):
        """cubic_root finds a root mod q when F(x, 1) has one (0, 1, 2 or 3 roots: brute force over F_q), and
        characters gives primes above qmin not dividing lc(F)."""
        rnd = random.Random(3)
        for q in [3, 5, 7, 11, 101, 1009, 65537]:
            for _ in range(30):
                F = [rnd.randrange(q) for _ in range(3)] + [rnd.randrange(1, q)]
                roots = [x for x in range(q) if sum(c * x ** i for i, c in enumerate(F)) % q == 0]
                r = linalg.cubic_root(F, q)
                self.assertEqual(r is None, not roots, (F, q))
                self.assertTrue(r is None or r in roots)
        F = [-5, 0, 0, 7 * 11 * 13]
        ch = linalg.characters(F, 100, 10)
        self.assertEqual(len(ch), 10)
        self.assertTrue(all(q > 100 and arith.is_prime(q) and F[3] % q and (F[0] + F[3] * r ** 3) % q == 0
                            for q, r in ch))

    def test_rank_V_triangular(self):
        """The verifier's rank_V (triangular rows taken off, f2rank on the rest) = dim V = rank(M) - rank(M_L) by a
        plain echelon, on random matrices with a triangular part (pure small rows with distinct and repeated tops,
        repeated pivots, rows with large columns) and on the relation rows of witnesses of the 2-descent of 37a1 and 389a1 (the
        prover's SPLIT: a triangular relation per degree-1 prime), with and without the s_columns' unit rows."""
        rnd = random.Random(13)
        for trial in range(60):
            ns, nl = rnd.randint(3, 60), rnd.randint(0, 40)
            names = ['%d:%d' % (p, 0) for p in range(2, ns + 2)] + ['%d:1' % (10 ** 6 + c) for c in range(nl)]
            rows = []
            for c in rnd.sample(range(ns), rnd.randint(0, ns)):        # triangular: top c, columns below it
                rows.append({c} | set(rnd.sample(range(c), min(c, rnd.randint(0, 4)))))
            for _ in range(rnd.randint(0, ns + nl)):
                k = rnd.randint(0, 6)
                rows.append(set(rnd.sample(range(ns + nl), min(k, ns + nl))))
            piv = list(range(len(rows))) + [rnd.randrange(len(rows)) for _ in range(rnd.randint(0, 3))] if rows else []
            rnd.shuffle(piv)
            ints = [sum(1 << c for c in rows[i]) for i in piv]
            rM, rL = f2rank(ints), f2rank([x >> ns for x in ints])
            dimV, nlc, rank = vsel.rank_V([set(r) for r in rows], names, [], piv, 10 ** 5)
            self.assertEqual((dimV, rank), (rM - rL, rM), trial)
        from rankbound import curve as cv, selmer
        for ai in ([0, 0, 1, -1, 0], [0, 1, 1, -2, 0]):
            c = selmer.prove(cv.from_ainvs(ai), log=lambda *a: None)
            rels = [vsel._gpvec(l, True) for l in c['relations']]
            rows, names = vsel.gp_rows(c['field']['poly'], c['curve']['bad_primes'], rels, 1)
            B = c['field']['bound']['B']
            for ops, piv in ((c['ops'], c['pivots']), (c.get('ops_s', []), c.get('pivots_s'))):
                if piv is None:
                    continue
                R = [set(r) for r in rows]
                if piv is c.get('pivots_s'):
                    names2 = list(names)
                    for p, r in c['s_columns']:          # the verifier's unit rows, by label (ckcol's names)
                        z = '%d:%d' % (p, r)
                        if z not in names2:
                            names2.append(z)
                        R.append({names2.index(z)})
                else:
                    names2 = names
                ref = [set(r) for r in R]
                for d, s_ in ops:
                    ref[d] ^= ref[s_]
                sm2 = {i for i, z in enumerate(names2) if z.count(':') == 1 and int(z.split(':')[0]) < B}
                lg = [i for i in range(len(names2)) if i not in sm2]
                ints = [sum(1 << (lg.index(x) if x in lg else len(lg) + sorted(sm2).index(x)) for x in ref[i]) for i in piv]
                rM, rL = f2rank(ints), f2rank([x & ((1 << len(lg)) - 1) for x in ints])
                self.assertEqual(vsel.rank_V(R, names2, [tuple(o) for o in ops], piv, B)[0], rM - rL, ai)

    def test_realroots(self):
        """field.gp's real roots are precise for 170-digit coefficients, beyond gp's default 38 digits."""
        a, b = 10 ** 85 + 7, -(10 ** 84 + 3)
        out = vf.run_gp('print(Vec(apply(round, realroots((x - %d) * (x + %d) * (x - 5)))));' % (a, -b),
                     libs=(os.path.join(ROOT, 'tests', 'data', 'oracle', 'field.gp'),), seconds=60)
        self.assertEqual(out.strip(), '[%d, 5, %d]' % (b, a))

    def test_trim(self):
        """The trim (excess cut to KEEP, heaviest rows first) keeps the rows of scarce small columns: columns 18 and 19
        lie only in two rows of three large columns each."""
        rnd = random.Random(7)
        ns = 20
        M = linalg.Matrix({'J': []}, [(i, 0) for i in range(ns)])
        for _ in range(300):
            M._add([(c, 0) for c in rnd.sample(range(ns - 2), rnd.randint(1, 3))] + [('L', rnd.randrange(30))], 'x')
        for c in (ns - 2, ns - 1):
            M._add([(c, 0), ('L', 100), ('L', 101), ('L', 102)], 'x')
        d = tempfile.mkdtemp()
        try:
            full = linalg.solve(M, d, 0, keep=-1).bound
            self.assertEqual(linalg.solve(M, d, 0, keep=5).bound, full)
        finally:
            shutil.rmtree(d)

    def test_threads(self):
        """f2elim and f2rank give the same bound, pivots and dim V with 1 and 4 threads.  Threads start only when rows
        x words left >= 10^6, hence 6,000 rows of 50 entries over 17,000 columns.  f2rank is rebuilt here from the same
        source with pthread_create counted."""
        rnd = random.Random(11)
        ns, nl = 14000, 3000
        M = linalg.Matrix({'J': []}, [(i, 0) for i in range(ns)])
        for _ in range(6000):
            M._add([(c, 0) for c in rnd.sample(range(ns), 40)] + [('L', c) for c in rnd.sample(range(nl), 10)], 'x')
        d = tempfile.mkdtemp()
        try:
            out = []
            for th in (1, 4):
                r = linalg.solve(M, os.path.join(d, str(th)), 0, dense=10 ** 5, threads=th)
                with open(r.piv, 'rb') as f:
                    out.append((r.bound, r.dimV, r.rank, f.read()))
                self.assertEqual(r.threaded > 0, th > 1, 'f2elim: threaded panels %s with %d threads' % (r.threaded, th))
            self.assertEqual(out[0], out[1])
            kept, piv, need = linalg.cone(r.ops, r.piv)
            ids = [set(x) for x in M.rows]
            names = ['%d:1' % c if c < ns else '%d:1:2' % c for c in range(len(M.col))]
            self.assertEqual([vsel.rank_V([set(x) for x in ids], names, kept, piv, 10 ** 9, jobs=j)[0] for j in (1, 4)],
                             [r.dimV] * 2)
            # f2rank's threaded path: the same source with a counting pthread_create
            shim, exe = os.path.join(d, 'count.h'), os.path.join(d, 'f2rank_count')
            with open(shim, 'w') as f:
                f.write('#include <pthread.h>\n#include <stdio.h>\n#include <stdlib.h>\nstatic int made;\n'
                        'static void report(void) { fprintf(stderr, "threads %d\\n", made); }\n'
                        'static int counted(pthread_t *t, const pthread_attr_t *a, void *(*f)(void *), void *x) {\n'
                        '  if (!made++) atexit(report); return pthread_create(t, a, f, x); }\n'
                        '#define pthread_create counted\n')
            subprocess.run(['cc', '-O3', '-pthread', '-include', shim, '-o', exe, vsel.CHECK + '.c'], check=True)
            real, runs = (vsel.CHECK, vsel.subprocess.run), []
            def run(*a, **k):
                runs.append(real[1](*a, **k)); return runs[-1]
            vsel.CHECK, vsel.subprocess.run = exe, run
            try:
                self.assertEqual(vsel.rank_V([set(x) for x in ids], names, kept, piv, 10 ** 9, jobs=4)[0], r.dimV)
            finally:
                vsel.CHECK, vsel.subprocess.run = real
            self.assertRegex(''.join(x.stderr.decode() for x in runs), r'threads [1-9]', 'f2rank made no thread')
        finally:
            shutil.rmtree(d)

if __name__ == '__main__':
    unittest.main()
