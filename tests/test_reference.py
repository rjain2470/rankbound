"""Tests against PARI and independent computations: the GRH bound, the prover's row rule, the elimination, and G on
fields whose class groups bnfinit knows."""
import json, os, random, shutil, subprocess, sys, tempfile, unittest
from rankbound import verify as vf
from rankbound import field, linalg, sieve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# a folder of the published certificates, if there is one (RANKBOUND_CERTS; default certificates/ in this checkout)
CERTS = os.environ.get('RANKBOUND_CERTS') or os.path.join(ROOT, 'certificates')
CERT728 = os.path.join(CERTS, 'icarm728.cert.json')
CERT8 = CERT728.replace('icarm728', 'icarm8')       # not in git: the tests read REF
# tests/data/reference.json: the curve and rank_lower_bound of ICARM #728 and #8, and 1,000 of #728's relations
# (every len // 1000-th, free ones left out); test_fixture checks it against the certificates
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'reference.json')) as _f:
    REF = json.load(_f)
FIELD_GP = os.path.join(os.path.dirname(field.__file__), 'field.gp')

SMALL = [  # a-invariants, dim Cl(K)[2] (bnfinit)
    ([0, 0, 0, 421, 141594], 0), ([0, 0, 0, -1623, -148259], 2), ([0, 0, 0, 1527, -177941], 1),
    ([1, -1, 0, 31, -412], 0), ([0, 0, 0, 474, -13363], 1), ([0, 0, 0, 259, -92764], 1), ([0, 0, 0, 1933, 102706], 2),
    ([0, 0, 0, 134, -2569], 2), ([0, 0, 0, -953, 74673], 3), ([0, 0, 0, -1930, -32937], 3),
    ([0, 0, 1, -1, 0], 0), ([0, -1, 1, -929, -10595], 1)]


FIELDS = {}


def field_of(ai):
    """Return (field dict, small labels) of the curve AI, computed once per test run."""
    if tuple(ai) not in FIELDS:
        cur = field.from_ainvs(ai)
        F, small = field.compute(cur.ainvs, cur.bad_primes)
        F['bad_primes'] = cur.bad_primes
        FIELDS[tuple(ai)] = F, small
    return FIELDS[tuple(ai)]


def line_relations(F, lpb, b1, A=2 ** 9):
    d = tempfile.mkdtemp()
    try:
        out = os.path.join(d, 'l.out')
        sieve.line_sieve(F['poly'], lpb, A, 1, b1, out)
        return list(sieve.relations(out, 0))
    finally:
        shutil.rmtree(d)


def bound_of(F, small, rels, lpb):
    M = linalg.Matrix(F, small)
    d = tempfile.mkdtemp()
    try:
        fb = os.path.join(d, 'roots.gz')
        sieve.write_roots(F['poly'], 2 ** lpb, fb)
        for p, labs in linalg.free_rows(F, fb, 2 ** lpb):
            M.add_free(p, labs)
        for a, b, H in rels:
            M.add_relation('%d,%d:%s' % (a, b, ','.join('%x' % h for h in H)))
        return linalg.solve(M, d, 0).bound
    finally:
        shutil.rmtree(d)


def pari_labels(F, rels):
    """Return the odd-valuation labels of the relations (a, b, H), by the verifier's ckrels and field.gp."""
    f = ' + '.join('(%d)*x^%d' % (c, i) for i, c in enumerate(F['poly']))
    head = 'f = %s; K = ckinit(f, [%s]); Jinv = idealhnf(K[1], 1, K[2]);\n' % (f, ','.join(F['bad_primes']))
    body = head + 'R = [%s];\nckrels(K, f, R, 0);\n' % ','.join('[%d,%d,[%s]]' % (a, b, ','.join(map(str, H))) for a, b, H in rels)
    cols = {}
    for l in vf.run_gp(body, libs=('verify.gp', FIELD_GP), seconds=600).splitlines():
        t = l.split()
        if t and t[0] == 'R':
            cols[int(t[1])] = [tuple(map(int, z.split(':'))) for z in t[2:]]
    pk = sorted({c for v in cols.values() for c in v if len(c) == 2})
    body = head + 'PK = [%s];\n' % ','.join('[%d,%d]' % c for c in pk)
    body += 'foreach(PK, z, my(t = select(w -> w[2] == z[2], sflabels(K[1], K[2], Jinv, f, z[1]))); print("L ", z[1], " ", z[2], " ", t[1][1]));\n'
    lab = {}
    for l in vf.run_gp(body, libs=('verify.gp', FIELD_GP), seconds=600).splitlines():
        t = l.split()
        if t and t[0] == 'L':
            lab[(int(t[1]), int(t[2]))] = int(t[3])
    return [{(c[0], lab[c]) if len(c) == 2 else c for c in cols[i]} for i in range(len(rels))]


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
        """The BDF bound (verify.gp) = PARI's internal LIMC2 on 4 fields, and 200,271 for the field of Elkies's rank-28 curve."""
        body = r'''
f=64023127168000*x^3 + 10309553525987840512490787747*x^2 - 3858878002265332645698861066081585182608*x - 69043295714402138353376748510210837676894689434302674;
addprimes([376018840263193489397987439236873583997122096511452343225772113000611087671413,315574902691581877528345013999136728634663121]);
print("B ", bdf(nfinit(f), 10^6));
foreach([x^3-x-1, x^3-1000003*x+7, 5*x^3+123456*x^2-9876543*x+1234567891, x^3+x^2-2*x-1+3*10^9], g, print("B ", bdf(nfinit(g), 10^6)));
'''
        got = [int(l.split()[1]) for l in vf.run_gp(body, libs=('verify.gp',)).splitlines() if l.startswith('B ')]
        self.assertEqual(got, [200271, 7, 794, 2772, 2399])

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
        out = vf.run_gp(body, libs=('verify.gp',)).splitlines()
        self.assertEqual([l for l in out if l.startswith('C ')], ['C 10'] * 4)
        self.assertEqual([l for l in out if l.startswith('K ')], ['K [1, 1, 1, 1]'] * 2)

    @unittest.skipUnless(os.path.exists(CERT728) and os.path.exists(CERT8), 'the certificates of ICARM #728 and #8 '
                         'are not in RANKBOUND_CERTS (or certificates/: not in git)')
    def test_fixture(self):
        """REF is what the certificates of ICARM #728 and #8 give (where they are)."""
        for name, path in (('icarm728', CERT728), ('icarm8', CERT8)):
            c = vf.load(path)
            want = {'name': c['name'], 'curve': c['curve'], 'rank_lower_bound': c['rank_lower_bound']}
            if name == 'icarm728':
                rels = [l for l in c['relations'] if not l.startswith('f:')]
                want['relations'] = rels[::len(rels) // 1000][:1000]
            self.assertEqual(REF[name], want, name)

    def test_rows_728(self):
        """The prover's row rule = PARI's idealval on ICARM #728 (non-monic index form: J correction at 2, 3, 11, 13,
        23, 1021, 1223, 2756959289691073), on 1,000 of its certificate's relations (REF)."""
        c = REF['icarm728']['curve']
        F, small = field.compute(c['minimal_model'], c['bad_primes'])
        F['bad_primes'] = c['bad_primes']
        self.assertEqual((F['GB'], F['ns'], F['u'], F['n'], F['w']), (92085, 8888, 1, 11, -1))
        rels = [sieve.parse_line(l) for l in REF['icarm728']['relations']]
        self.assertEqual(len(rels), 1000)
        J = [tuple(l) for l in F['J']]
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
        out = vf.run_gp(body, libs=('verify.gp',), seconds=600)
        self.assertNotIn('BELOW', out)
        self.assertGreater(int(out.split('EQ ')[1]), 50)

    def test_vrank_rank0(self):
        """The rank proof of r = 0 needs no point (571a1 has rank 0); r = 1 with no point, or a point off the curve,
        fails."""
        def chk(r, pts, prs):
            out = vf.run_gp('vrank([0,-1,1,-929,-10595], %d, [%s], [%s]);' % (r, pts, prs), libs=('verify.gp',))
            return 'CHK rank_lower_bound 1' in out
        self.assertTrue(chk(0, '', ''))
        self.assertFalse(chk(1, '', ''))
        self.assertFalse(chk(1, '[0,0]', '3'))

    def test_rank_primes(self):
        """The certificate keeps the first r primes whose 2-signatures have rank r (ICARM #8, r = 22: its primes, each
        listed twice, give them back once; the points stay as they are on their own model)."""
        c = REF['icarm8']
        rl = c['rank_lower_bound']
        got = linalg.rank_proof(c['curve']['ainvs'], c['curve']['bad_primes'], 22,
                                {'points': rl['points'], 'primes': [p for p in rl['primes'] for _ in '12']})
        self.assertEqual(got, rl)

    def test_rank_proof_models(self):
        """Points of one model are written on another: mapped exactly, primes chosen on the target model or completed
        by a search, and vrank passes.  A point off its model, another curve, too few points or dependent points are
        refused.  37a1, P = (0, 0); its short model y^2 = x^3 - 1296x + 11664."""
        mm, short, P = ['0', '0', '1', '-1', '0'], ['0', '0', '0', '-1296', '11664'], [['0', '0']]
        prs = [p for p in range(2, 100) if all(p % q for q in range(2, p))]
        rl = linalg.rank_proof(short, ['37'], 1, {'points': P, 'primes': prs, 'ainvs': mm})
        self.assertIn(rl['points'], ([['0', '108']], [['0', '-108']]))
        self.assertTrue(set(rl['primes']) <= set(prs) - {2, 3})
        body = 'vrank([%s], 1, [[%s,%s]], [%s]);' % (','.join(short), *rl['points'][0], ','.join(map(str, rl['primes'])))
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp(body, libs=('verify.gp',)))
        self.assertEqual(linalg.rank_proof(mm, ['37'], 1, {'points': P, 'primes': prs})['points'], P)   # same model
        rl = linalg.rank_proof(short, ['37'], 1, {'points': P, 'primes': [3], 'ainvs': mm})    # 3 | the short disc
        self.assertEqual(len(rl['primes']), 1)
        self.assertNotEqual(rl['primes'], [3])
        body = 'vrank([%s], 1, [[%s,%s]], [%s]);' % (','.join(short), *rl['points'][0], ','.join(map(str, rl['primes'])))
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp(body, libs=('verify.gp',)))
        for ai, proof, err in ((short, {'points': [['0', '1']], 'primes': prs, 'ainvs': mm}, 'Point 1 is not on the curve'),
                               (['0', '0', '1', '-1', '1'], {'points': P, 'primes': prs, 'ainvs': mm}, 'The points are not on this curve'),
                               (short, {'points': [], 'primes': prs, 'ainvs': mm}, '^r = 1 but 0 points were given'),
                               (short, {'points': [['1', '0']], 'primes': [2, 3, 5, 7], 'ainvs': mm},
                                '^The points do not prove rank >= 1. At every usable prime below 10\\^6')):
            with self.assertRaisesRegex(RuntimeError, err):
                linalg.rank_proof(ai, ['37'], 1, proof)

    def test_rank_proof_search(self):
        """Points given without primes (rankbound(points=...)): the primes are searched, and vrank passes with them;
        37a1's 2P = (1, 0) is in 2E(Q), so no prime proves rank >= 1 with it: refused.  Two points of rank 2 (389a1:
        (-1, 1), (0, 0)) need two primes."""
        rl = linalg.rank_proof(['0', '0', '1', '-1', '0'], ['37'], 1, {'points': [['0', '0']]})
        self.assertEqual(len(rl['primes']), 1)
        body = 'vrank([0,0,1,-1,0], 1, [[0,0]], [%d]);' % rl['primes'][0]
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp(body, libs=('verify.gp',)))
        with self.assertRaisesRegex(RuntimeError, 'look dependent; check them'):
            linalg.rank_proof(['0', '0', '1', '-1', '0'], ['37'], 1, {'points': [['1', '0']]})
        rl = linalg.rank_proof(['0', '1', '1', '-2', '0'], ['389'], 2, {'points': [['-1', '1'], ['0', '0']]})
        self.assertEqual(len(rl['primes']), 2)
        body = 'vrank([0,1,1,-2,0], 2, [[-1,1],[0,0]], [%s]);' % ','.join(map(str, rl['primes']))
        self.assertIn('CHK rank_lower_bound 1', vf.run_gp(body, libs=('verify.gp',)))

    def test_split_primes(self):
        """The free relations of a run on given relations (field.write_split, PARI) are the line siever's (sieve.roots,
        by trial) on the fields of SMALL: below a bound, and at the extra primes given above it (the large primes of
        the relations)."""
        d = tempfile.mkdtemp()
        try:
            big = [p for p in range(500, 2000) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
            for ai, _ in SMALL[:6]:
                F, _ = field_of(ai)
                a, b, c = os.path.join(d, 'a.gz'), os.path.join(d, 'b.gz'), os.path.join(d, 'c.gz')
                field.write_split(F['poly'], 2000, a)
                sieve.write_roots(F['poly'], 2000, b)
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
        with self.assertRaisesRegex(ValueError, '^The bad prime 7 does not divide the discriminant.$'):
            field.compute(E571, [571, 7])
        with self.assertRaisesRegex(ValueError, '^The bad primes given do not account for the whole discriminant.$'):
            field.compute(E571, [])
        with tempfile.TemporaryDirectory() as d:
            p = subprocess.run([sys.executable, '-m', 'rankbound', 'estimate', '--dir', d, '0,-1,1,-929,-10595',
                                '--bad-primes', '7'], cwd=d, env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True,
                               text=True, timeout=600)
            self.assertEqual((p.returncode, p.stderr.strip()), (1, 'The bad prime 7 does not divide the discriminant.'))
            p = subprocess.run([sys.executable, '-m', 'rankbound', 'sieve', os.path.join(d, 'none')], cwd=d,
                               env=dict(os.environ, PYTHONPATH=ROOT), capture_output=True, text=True, timeout=600)
            self.assertEqual(p.returncode, 1)
            self.assertIn('is not a rankbound work folder. Start it with rankbound bound or rankbound estimate.',
                          p.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, 'none')))

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
                self.assertEqual(vf.rank_V(ids, names, kept, piv, 10 ** 9)[0], res.dimV)
            finally:
                shutil.rmtree(d)

    def test_realroots(self):
        """field.gp's real roots are precise for 170-digit coefficients, beyond gp's default 38 digits."""
        a, b = 10 ** 85 + 7, -(10 ** 84 + 3)
        out = vf.run_gp('print(Vec(apply(round, realroots((x - %d) * (x + %d) * (x - 5)))));' % (a, -b),
                     libs=('verify.gp', os.path.join(os.path.dirname(field.__file__), 'field.gp')), seconds=60)
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
            self.assertEqual([vf.rank_V([set(x) for x in ids], names, kept, piv, 10 ** 9, jobs=j)[0] for j in (1, 4)],
                             [r.dimV] * 2)
            # f2rank's threaded path: the same source with a counting pthread_create
            shim, exe = os.path.join(d, 'count.h'), os.path.join(d, 'f2rank_count')
            with open(shim, 'w') as f:
                f.write('#include <pthread.h>\n#include <stdio.h>\n#include <stdlib.h>\nstatic int made;\n'
                        'static void report(void) { fprintf(stderr, "threads %d\\n", made); }\n'
                        'static int counted(pthread_t *t, const pthread_attr_t *a, void *(*f)(void *), void *x) {\n'
                        '  if (!made++) atexit(report); return pthread_create(t, a, f, x); }\n'
                        '#define pthread_create counted\n')
            subprocess.run(['cc', '-O3', '-pthread', '-include', shim, '-o', exe, vf.F2RANK + '.c'], check=True)
            real, runs = (vf.F2RANK, vf.subprocess.run), []
            def run(*a, **k):
                runs.append(real[1](*a, **k)); return runs[-1]
            vf.F2RANK, vf.subprocess.run = exe, run
            try:
                self.assertEqual(vf.rank_V([set(x) for x in ids], names, kept, piv, 10 ** 9, jobs=4)[0], r.dimV)
            finally:
                vf.F2RANK, vf.subprocess.run = real
            self.assertRegex(runs[-1].stderr.decode(), r'threads [1-9]', 'f2rank made no thread')
        finally:
            shutil.rmtree(d)

if __name__ == '__main__':
    unittest.main()
