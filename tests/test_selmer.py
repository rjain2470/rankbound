"""The 2-descent (selmer.py): the exact 2-Selmer bound for E(Q)[2] = 0, against PARI's ellrank on small curves with known
Sel_2, on target curves of ICARM (tests/data/selmer_targets.tsv), its witnesses through the verifier (rankbound.verify;
the corruptions of each check are tests/test_verify_selmer.py's), the Minkowski bound and the route of rankbound()."""
import json, os, random, shutil, tempfile, unittest
from fractions import Fraction
import math
FULL = bool(os.environ.get('RANKBOUND_TEST_FULL'))        # RANKBOUND_TEST_FULL=1: every case
from rankbound import arith, curve as cv, field as fld, linalg, rankbound, selmer, witness
from rankbound import selmer as split
from rankbound import check as verify
from rankbound.check import run_gp
from helpers import run_route

TESTS = os.path.dirname(os.path.abspath(__file__))
# Cremona label: (a-invariants, rank, dim Sel_2)
SMALL = {'11a1': ([0, -1, 1, -10, -20], 0, 0), '37a1': ([0, 0, 1, -1, 0], 1, 1), '389a1': ([0, 1, 1, -2, 0], 2, 2),
         '5077a1': ([0, 0, 1, -7, 6], 3, 3), '571a1': ([0, -1, 1, -929, -10595], 0, 2)}
# ICARM targets of rankbound 2.0 that were 2 above the rank (Brumer-Kramer); the 2-descent must give the rank
TARGETS = ('925', '926', '954', '1024', '1461', '3397', '1094')


def targets():
    with open(os.path.join(TESTS, 'data', 'selmer_targets.tsv')) as f:
        rows = [l.rstrip('\n').split('\t') for l in f][1:]
    return {r[0]: r for r in rows}


def curve(ai, name, r=None, bad=None):
    c = cv.from_ainvs(ai, bad_primes=bad, r=r, name=name)
    return c if r is None else c


def verdict(wit):
    """Return the verifier's verdict on wit (a dict), written to a temporary file."""
    d = tempfile.mkdtemp()
    try:
        p = os.path.join(d, 'c.cert.json')
        witness.write(wit, p)
        return verify.check(p)['verdict']
    finally:
        shutil.rmtree(d)


def ellrank_sel(ai):
    """Return dim Sel_2 from PARI's ellrank: [r, R, s, L] with R = dim Sel_2 - s - dim E(Q)[2] and s the
    Cassels-Tate drop (ellrank.c, ell2selmer), so dim Sel_2 = R + s when E(Q)[2] = 0."""
    out = run_gp('my(z = ellrank(ellinit([%s]))); print("ER\\t", z[2] + z[3]);' % ','.join(map(str, ai)))
    return int(next(l.split('\t')[1] for l in out.splitlines() if l.startswith('ER')))


class TestSmall(unittest.TestCase):
    """dim Sel_2 of small curves, unconditionally (Minkowski) and under GRH, equals ellrank's."""

    @classmethod
    def setUpClass(cls):
        cls.wits = {}
        for lab, (ai, r, s) in SMALL.items():
            for hyp in ('none', 'GRH'):
                cls.wits[lab, hyp] = selmer.prove(curve(ai, 'c' + lab), hypothesis=hyp, log=lambda *a: None)

    def test_selmer(self):
        for lab, (ai, r, s) in SMALL.items():
            self.assertEqual(ellrank_sel(ai), s, lab)
            for hyp in ('none', 'GRH'):
                c = self.wits[lab, hyp]['claims']
                self.assertEqual(c['v'], c['U'], (lab, hyp))
                self.assertEqual(c['dim_sel_cap_V'], s, (lab, hyp))
                self.assertEqual(c['S'], s, (lab, hyp))
                self.assertEqual(self.wits[lab, hyp]['hypothesis'], hyp)

    def test_verifier(self):
        for key, wit in self.wits.items():
            self.assertEqual(verdict(wit), 'accepted', key)

    def test_split_check(self):
        """The checker's classes of the relations of the elements in several gp processes (check.selmer.PAR_RELS, here
        lowered to 1) give the same checks as one; selmer.par_map in forked processes gives the same list."""
        from rankbound.check import selmer as vsel
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, 'c.witness.json.gz')
            witness.write(self.wits['571a1', 'GRH'], p)
            one = verify.check(p, cores=1)
            old = vsel.PAR_RELS
            vsel.PAR_RELS = 1
            try:
                three = verify.check(p, cores=3)
            finally:
                vsel.PAR_RELS = old
            self.assertEqual(one['verdict'], 'accepted')
            self.assertEqual([(x['check'], x['ok']) for x in one['checks']], [(x['check'], x['ok']) for x in three['checks']])
        finally:
            shutil.rmtree(d)
        f = lambda xs: [x * x + 1 for x in xs]
        self.assertEqual(selmer.par_map(f, list(range(100)), 3, least=1), f(list(range(100))))

    def test_format(self):
        """The witness's format key."""
        c = self.wits['571a1', 'none']
        self.assertEqual(c['format'], 'rankbound-selmer-witness-6')
        self.assertEqual(set(c['field']), {'poly', 'dK', 'bound'})
        self.assertEqual(c['field']['bound']['type'], 'minkowski')
        self.assertEqual(sorted(c['local']['p']), sorted(c['s_primes']))
        a1, a2, a3, a4, a6 = (int(x) for x in c['curve']['minimal_model'])
        b2, b4, b6 = a1 * a1 + 4 * a2, 2 * a4 + a1 * a3, a3 * a3 + 4 * a6      # W = x^3 + b2 x^2 + 8 b4 x + 16 b6
        self.assertEqual(c['weierstrass'], [str(16 * b6), str(8 * b4), str(b2), '1'])
        self.assertTrue(all(e and len(set(e)) == len(e) for e in c['elements']))
        self.assertEqual(c['statement'], 'We have that rank E(Q) <= 2.')
        g = self.wits['571a1', 'GRH']
        self.assertEqual(g['field']['bound']['type'], 'BDF')
        self.assertTrue(g['statement'].startswith('Assuming the GRH'))
        json.dumps(c)


class TestTargets(unittest.TestCase):
    """ICARM curves where v2.0 (Brumer-Kramer) was 2 above the rank: the 2-descent gives the rank."""

    def test_targets(self):
        T = targets()
        for i in TARGETS:
            row = T[i]
            c = cv.Curve(name='icarm' + i, ainvs=row[1].split(','), bad_primes=row[4].split(','), r=int(row[2]),
                            source='ICARM #' + i, minimal_model=row[1].split(','))
            wit = selmer.prove(c, hypothesis='none' if i in ('925', '926', '954', '1024') else 'GRH', log=lambda *a: None)
            cl = wit['claims']
            self.assertEqual(cl['S'], int(row[2]), i)        # sharp (v = U is not needed for it: #1094, v = U - 1)
            self.assertLess(cl['S'], int(row[3]), i)
            self.assertEqual(verdict(wit), 'accepted', i)
            if i in ('925', '926', '954', '1024'):      # Cl[2] = 0 and B-K = u + n = 2, but Sel_2 = 0
                self.assertEqual((cl['k'], cl['S_bk'], wit['hypothesis']), (0, 2, 'none'), i)


def ref_small(rows, nsmall):
    """dim(rowspace cap the vectors without bits >= nsmall), rows as ints: an independent echelon (higher bits first)."""
    piv, cnt = {}, 0
    for r in rows:
        while r:
            h = r.bit_length() - 1
            if h not in piv:
                piv[h] = r
                cnt += h < nsmall
                break
            r ^= piv[h]
    return cnt


class TestAlgebra(unittest.TestCase):
    """The 2-descent's algebra by f2elim (linalg.eliminate) against a plain echelon on Python ints."""

    def test_eliminate(self):
        """eliminate's dim V on random sparse matrices (heavy columns, so that the merge leaves a dense phase), and
        the pivots after its ops: independent, with the same dim V."""
        rnd = random.Random(3)
        for trial in range(30):
            ns, nl = rnd.randint(1, 40), rnd.randint(1, 300)
            rows = []
            for _ in range(rnd.randint(1, nl + ns + 60)):
                r = set(rnd.sample(range(ns), rnd.randint(0, min(5, ns))))
                r |= {ns + c for c in rnd.sample(range(nl), rnd.randint(0, min(3, nl)))}
                if rnd.random() < 0.3:
                    r ^= {ns + c for c in range(min(4, nl))}             # heavy large columns
                rows.append(sorted(r))
            dense = rnd.choice([0, 3, 1000])
            e = linalg.eliminate(rows, ns + nl, ns, dense=dense)
            ints = [sum(1 << c for c in r) for r in rows]
            # the same with the columns below 70 given as bits (f2elim's WORDS)
            low = [sum(1 << c for c in r if c < 70) for r in rows]
            self.assertEqual(linalg.eliminate([[c for c in r if c >= 70] for r in rows], ns + nl, ns, dense=dense,
                                              bits=low), e, trial)
            self.assertEqual(e.dimV, ref_small(ints, ns), trial)
            self.assertEqual(e.rank, len(e.pivots), trial)
            for dst, src in e.ops:
                ints[dst] ^= ints[src]
            p = [ints[i] for i in e.pivots]
            self.assertEqual((ref_small(p, ns + nl), ref_small(p, ns)), (len(p), e.dimV), trial)
        self.assertEqual(linalg.eliminate([], 5, 2), (0, 0, [], []))

    def test_kernel_keeps_xrel(self):
        """Sieve_witness's kernel (selmer.kernel_rows) is trimmed to an excess of T, and f2elim's trim
        drops the rows with the most large columns first; an xrel row (here row 0, the only row that can complete a
        kernel vector with rows 1-3) gets a small column of its own, which keeps it.  Without it (nx = 0) the vector
        is lost."""
        T = 64
        L = lambda k: T + 1 + k
        rk = [[L(0), L(1), L(2)], [L(0)], [L(1)], [L(2)]]
        for g in range(150):                    # excess: 150 groups of 3 equal rows with 2 large columns each
            rk += [[L(10 + 2 * g), L(11 + 2 * g)]] * 3
        nc = L(11 + 2 * 150)
        with0 = lambda nx: [v for v in selmer.kernel_rows(rk, nx, T, nc)[0] if 0 in v]
        self.assertEqual(with0(1), [[0, 1, 2, 3]])
        self.assertEqual(with0(0), [])
        for v in selmer.kernel_rows(rk, 1, T, nc)[0]:       # every vector cancels its large columns
            out = set()
            for i in v:
                out ^= set(rk[i])
            self.assertEqual(out, set())

    def test_xrel_columns(self):
        """An xrel element's degree-1 prime takes M's column (p, r); a prime M does not have (one of degree
        > 1, (q, k, f)) gets a column of its own, shared between xrel rows; primes above S are left out."""
        col = {(5, 2): 0, (7, 1): 1, (3, 1): 3}
        kid = {0: 100, 1: 101}                  # M's columns outside S = {3}
        a = selmer.xrel_row([(5, 2), (2, 2, 2), (11, 1, 2), (3, 1)], col, {3}, kid, 100)
        b = selmer.xrel_row([(11, 1, 2), (7, 1)], col, {3}, kid, 100)
        self.assertEqual((a, b), ([100, 102, 103], [101, 103]))
        self.assertEqual(kid[(11, 1, 2)], 103)

    def test_claims(self):
        """G, k and v of the targets (1461 with triangular relations, 711 with ops) against the echelon on all rows
        and on the kernel's characters; the elements vanish outside S; the pivots after the ops give dim V."""
        T = targets()
        for i in ('1461', '711'):
            row = T[i]
            c = cv.Curve(name='icarm' + i, ainvs=row[1].split(','), bad_primes=row[4].split(','), r=int(row[2]),
                            source='ICARM #' + i, minimal_model=row[1].split(','))
            seen = []
            alg = selmer.algebra
            selmer.algebra = lambda *a, **kw: seen.append((a, alg(*a))) or seen[-1][1]    # with elements
            try:
                wit = selmer.prove(c, log=lambda *a: None)
            finally:
                selmer.algebra = alg
            (d, tri, free, ns, B, S, F, _), a = seen[-1]
            cols, rels, nfb, cl = a['cols'], a['rels'], a['nfb'], a['claims']
            nsmall = sum(1 for z in cols if selmer.is_deg1(z) and z[0] < B)
            ints = [sum(1 << cols[z] for z in lab) for _, _, lab in rels]
            unit = [1 << cols[z] for z in a['s_columns']]
            self.assertEqual(cl['dimV'], ref_small(ints, nsmall), i)
            self.assertEqual(ns - cl['k'], ref_small(ints + unit, nsmall), i)
            for ops, piv, rows, dim in ((a['ops'], a['pivots'], ints, cl['dimV']),
                                        (a['ops_s'], a['pivots_s'], ints + unit, ns - cl['k'])):
                rows = list(rows)
                for dst, src in ops:
                    rows[dst] ^= rows[src]
                self.assertEqual(ref_small([rows[j] for j in piv], nsmall), dim, i)
            # v: the rank of the characters on the kernel of d's relations outside S, as (outside S | characters)
            nch = len(d['real']) + len(d['chars'])
            nonS = {z: k for k, z in enumerate(z for z in cols if z[0] not in set(S))}
            kr = [(sum(1 << nonS[z] for z in lab if z in nonS) << nch) | d['ch'][j]
                  for j, (_, _, lab) in enumerate(rels[:nfb])]
            self.assertEqual(cl['v'], ref_small(kr, nch), i)
            for e in a['elements']:
                odd = set()
                for j in e:
                    odd ^= set(rels[j][2])
                self.assertTrue(all(z[0] in S for z in odd), i)
            self.assertEqual(wit['elements'], a['elements'])
            self.assertEqual(verdict(wit), 'accepted', i)
        self.assertTrue(seen[-1][1]['ops'], 'no ops for 711')



class TestClasses(unittest.TestCase):
    """The characters and local classes of relations that selmer.py computes in Python (residue symbols, valuation and
    unit at unramified degree-1 places, the tables mod 8 above 2) equal selmer.gp's (m1batch: m1sgn, m1chi, m1cls1) on
    random a - b alpha and (p), at every place of S, on fields with 2 or 11 dividing lc(F) and with ramified primes
    of S; a changed table entry or residue symbol is caught."""

    def test_python_against_gp(self):
        T = targets()
        cases = [([0, -1, 1, -929, -10595], None), ([0, 0, 1, -7, 6], None)] + [
            (T[i][1].split(','), T[i][4].split(',')) for i in ('711', '1461', '203', '785')]
        rnd = random.Random(5)
        for ai, bad in cases:
            c = cv.from_ainvs(ai, bad_primes=bad, name='t')
            F, _ = fld.compute(c.ainvs, c.bad_primes)
            S0 = sorted({2} | {int(p) for p in F['bad_primes']})
            d = selmer.base(F, [str(p) for p in F['bad_primes']], S0, 10 ** 12, 5, 10 ** 30, 2)
            pairs = set()
            while len(pairs) < 120:
                a, b = rnd.randint(-300, 300), rnd.randint(1, 60)
                if math.gcd(a, b) == 1 and fld.norm(F['poly'], a, b):
                    pairs.add((a, b))
            pairs = sorted(pairs)
            out = run_gp('{ my(V = [%s]); for(i = 1, #V, my(fa = factor(abs(V[i])), L = List()); for(j = 1, #fa~, '
                         'for(t = 1, fa[j, 2], listput(L, Str(fa[j, 1])))); print("F ", strjoin(Vec(L), ","))); }' %
                         ','.join(str(fld.norm(F['poly'], a, b)) for a, b in pairs))
            Hs = [[int(x) for x in l[2:].split(',') if x] for l in out.splitlines() if l.startswith('F ')]
            rels = [([0, a, b], H, None) for (a, b), H in zip(pairs, Hs)]
            rels += [([2, p], [p], None) for p in sorted({2, 3, 5, 7, 11, 13, 17} | set(d['S'][:5]))]
            pl = selmer.Places(d, F)
            n = len(pl.pld)
            ref = {(j, i): x for j, i, x in selmer.batch(
                F, S0, [r for r, _, _ in rels], [[j, j, list(range(n))] for j in range(len(rels))], d['chars'])}
            fast = [(j, i) for j, (r, H, _) in enumerate(rels) for i in range(n) if pl.one(i, r, H) is not None]
            self.assertGreater(len(fast), len(rels) * n // 2, ai)
            for j, i in fast:
                r, H, _ = rels[j]
                self.assertEqual(pl.one(i, r, H), ref[j, i], (ai, r, i))
            got = pl.classes(rels, range(len(rels)), F, d['chars'])
            for j in range(len(rels)):
                self.assertEqual(got[j], sum(ref[j, i] << pl.off[i] for i in range(n)), (ai, j))
            ch = selmer.characters(d, rels, F)
            gch = selmer.batch(F, S0, [r for r, _, _ in rels], [[j, j, -1] for j in range(len(rels))], d['chars'])
            self.assertEqual([ch[j] for j, _, _ in gch], [x for _, _, x in gch], ai)
            # a changed table entry above 2, or a changed residue symbol, gives other classes
            for j, i in fast:
                r, H, _ = rels[j]
                if pl.pld[i]['p'] == 2 and r[0] == 0 and 2 not in H:
                    t = pl.pld[i]['t']
                    a, b = (r[1], r[2]) if t is None else (r[2], t * r[2] - r[1])     # al' = 1/(t - al)
                    tab, k = pl.pld[i]['tab'], 8 * (b % 8) + a % 8
                    pl.pld[i]['tab'] = tab[:k] + [tab[k] ^ 2] + tab[k + 1:]
                    self.assertNotEqual(pl.one(i, r, H), ref[j, i], ai)
                    pl.pld[i]['tab'] = tab
                    break
            j, i = next((j, i) for j, i in fast if pl.pld[i]['p'] > 2 and rels[j][0][0] == 0)
            real = selmer.nonresidue
            try:
                selmer.nonresidue = lambda n, p: not real(n, p)
                self.assertNotEqual(pl.one(i, *rels[j][:2]), ref[j, i], ai)
            finally:
                selmer.nonresidue = real



class TestReducedS(unittest.TestCase):
    """REDUCE_S (the default): the odd bad primes with odd c_p are left out of S and listed with their
    c_p (s_dropped); the bound is the same as with every bad prime in S, with a smaller U."""

    def test_same_bound(self):
        T = targets()
        cases = [('571a1', [0, -1, 1, -929, -10595], None, None)] + [
            (i, T[i][1].split(','), T[i][4].split(','), int(T[i][2])) for i in ('1461', '3397')]
        for name, ai, bad, r in cases:
            c = cv.from_ainvs(ai, bad_primes=bad, r=r, name='t')
            red = selmer.prove(c, log=lambda *a: None)
            selmer.REDUCE_S = False
            try:
                full = selmer.prove(c, log=lambda *a: None)
            finally:
                selmer.REDUCE_S = True
            self.assertNotIn('s_dropped', full)
            self.assertEqual(red['claims']['S'], full['claims']['S'], name)
            self.assertTrue(red['s_dropped'] and all(p % 2 and cp % 2 for p, cp in red['s_dropped']), name)
            self.assertEqual(sorted(int(p) for p in red['s_primes']) + sorted(p for p, _ in red['s_dropped']),
                             sorted(int(p) for p in full['s_primes'] if int(p) not in {q for q, _ in red['s_dropped']})
                             + sorted(p for p, _ in red['s_dropped']), name)
            self.assertEqual(set(int(p) for p in red['s_primes']) | {p for p, _ in red['s_dropped']},
                             set(int(p) for p in full['s_primes']), name)
            self.assertLess(red['claims']['U'], full['claims']['U'], name)

    def test_uncond(self):
        """GRH (BDF) by default; uncond=True: the Minkowski bound (37a1), accepted by the verifier with hypothesis
        "none"; refused, with its estimated cost, when out of reach (a field of |d_K| ~1e22), also by --estimate --uncond."""
        d = tempfile.mkdtemp()
        try:
            g = rankbound([0, 0, 1, -1, 0], r=1, compute=True, workdir=os.path.join(d, 'g'))
            u = rankbound([0, 0, 1, -1, 0], r=1, compute=True, workdir=os.path.join(d, 'u'), uncond=True)
            self.assertEqual((g.grh, u.grh, g.rank_bound, u.rank_bound, u.checked), (True, False, 1, 1, True))
            self.assertEqual(witness.read(u.witness)['hypothesis'], 'none')
            from rankbound import RankboundError
            with self.assertRaisesRegex(RankboundError, r"^An unconditional bound needs Minkowski's bound [0-9.e]+, about "
                                        r"[0-9.e+]+ (hours|days|years) on 1 core, beyond the 3-hour limit. Run without "
                                        r"--uncond or give more cores.$"):
                rankbound([0, 0, 1, -16359067, 26274178986], bad_primes=['18031737725935636520843'], compute=True,
                          workdir=os.path.join(d, 'x'), uncond=True)
            # `bound --estimate --uncond` prints the Minkowski estimate, and says when bound --uncond would refuse it
            from rankbound.cli import main
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(main(['bound', '--estimate', '--uncond', '0,0,1,-1,0', '--dir', os.path.join(d, 'e')]), 0)
                self.assertEqual(main(['bound', '--estimate', '--uncond', '0,0,1,-16359067,26274178986', '--bad-primes',
                                       '18031737725935636520843', '--dir', os.path.join(d, 'e')]), 0)
            est = [l for l in out.output if 'An unconditional bound' in l]      # (under Slurm a time line too)
            self.assertEqual(est[0], "INFO:rankbound:An unconditional bound for 0_0_1_-1_0 needs Minkowski's bound 3. "
                                     "This should take about 1 second on 1 core.")
            self.assertRegex(est[1], r"^INFO:rankbound:An unconditional bound for 0_0_1_-16359067_26274178986 needs "
                             r"Minkowski's bound [0-9.]+e\d+. This should take about [0-9.e+]+ (hours|days|years) on 1 "
                             r"core, beyond the 3-hour limit, so --uncond would refuse it\.$")
        finally:
            shutil.rmtree(d)

class TestFixes(unittest.TestCase):
    """Failures of the 2-descent on the benchmark.  ICARM #1664: a prime of lc(F) in [B0, B) outside S (the J part of every a - b alpha lies
    above it) is a factor base prime, not the top of a triangular relation; G and k equal bnfinit's dim Cl/2 and
    dim Cl_S[2] (test-only), and v = U.  LMFDB 365874.d1 (nonsplit I_40 at 3) and 76784.g1: the local images need
    points near the singular point of the reduction (m1points' centers)."""

    @unittest.skipUnless(FULL, 'RANKBOUND_TEST_FULL: the quick set runs icarm:1664/uncond too')
    def test_lc_prime(self):
        with open(os.path.join(TESTS, 'data', 'quick.tsv')) as f:
            row = next(l.rstrip('\n').split('\t') for l in f if l.startswith('icarm:1664\t'))
        c = cv.Curve(name='icarm1664', ainvs=row[2].split(','), bad_primes=row[3].split(','), r=int(row[4]),
                     source='ICARM #1664', minimal_model=None)
        seen = []
        alg = selmer.algebra
        selmer.algebra = lambda *a, **kw: seen.append(a) or alg(*a, **kw)
        try:
            wit = selmer.prove(c, hypothesis='none', log=lambda *a: None)       # Minkowski: the lc prime in [B0, B)
        finally:
            selmer.algebra = alg
        d, tri, free, ns, B, S, F, _ = seen[-1]
        B0 = max(z[0] for z in d['fb']) + 1
        lc = abs(int(F['poly'][3]))
        if not any(B0 <= p < B and p not in S for p in selmer.prime_factors(lc)):
            self.skipTest('the index form (field.sfield) has no prime of lc(F) in [B0, B) for #1664 (PARI\'s had one)')
        cl = wit['claims']
        out = run_gp('{ my(f = %s, P = [%s], SP = [%s], K, B); addprimes(P); K = nfinit([f, concat([2, 3], P)], 3)[1]; '
                     'B = bnfinit(K, 1); print("CL ", #select(c -> c %% 2 == 0, B.cyc), " ", #select(c -> c %% 2 == 0, '
                     'bnfsunit(B, concat(apply(p -> idealprimedec(K, p), SP)))[5][2])); }' % (
                         '+'.join('(%d)*x^%d' % (int(c), i) for i, c in enumerate(F['poly'])), ','.join(map(str, F['bad_primes'])),
                         ','.join(map(str, S))))
        g, ks = map(int, next(l for l in out.splitlines() if l.startswith('CL')).split()[1:])
        self.assertEqual((cl['G'], cl['k'], cl['v'], cl['S']), (g, ks, cl['U'], 5))

    def test_points_near_the_node(self):
        for ai, s in (([1, 1, 0, -1532913423, -48303013033179], 3), ([0, 1, 0, -6659840, -6617436748], 3)):
            wit = selmer.prove(curve(ai, 't'), log=lambda *a: None)
            self.assertEqual(wit['claims']['S'], s, ai)
            self.assertEqual(verdict(wit), 'accepted', ai)

class TestBound(unittest.TestCase):
    def test_minkowski(self):
        """B > (4/pi40)^r2 (2/9) sqrt|d| >= the Minkowski bound, B - 1 not."""
        for dK, r2 in ((-23, 1), (49, 0), (81 * 4, 0), (-44, 1), (10 ** 15 + 37, 0), (-(10 ** 20) - 3, 1)):
            B = selmer.minkowski(dK, r2)
            m2 = (16 / selmer.PI40 ** 2) ** r2 * Fraction(4, 81) * abs(dK)
            self.assertTrue(B * B > m2 and ((B - 1) ** 2 <= m2 or B == 1), dK)
        self.assertEqual(selmer.minkowski(81 * 4, 0), 5)      # (2/9) sqrt(324) = 4 exactly: B = 5 > M

    def test_triangular_ranges(self):
        rg = split.tri_ranges(100, 10 ** 6, 8)
        self.assertEqual(rg[0][0], 100)
        self.assertEqual(rg[-1][1], 10 ** 6)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(rg, rg[1:])))


class TestDeadline(unittest.TestCase):
    """The 2-descent by our SPLIT stops at a run's deadline (Stopped, not TimeLimit or a run past it), and the
    triangular relations found before it are kept in the work folder (STATE/tri) and not searched again."""

    def test_triangular_resume(self):
        import threading
        from rankbound import Stopped
        c = cv.classified(cv.from_ainvs([1, -1, 1, 2308, -273], factor_seconds=600))
        F, small = fld.compute(c.ainvs, c.bad_primes)
        S, _ = selmer.s_primes(F)
        P = [str(p) for p in F['bad_primes']]
        B = selmer.minkowski(F['dK'], (3 - F['r1']) // 2)
        B0 = max(F['GB'], selmer.FB_MIN)
        self.assertGreater(B - B0, 10 ** 4)                 # several ranges
        from unittest import mock
        from rankbound import selmer as sm
        fresh = split.triangular(F, P, S, B0, B, cores=1)
        calls, lock, tri = [], threading.Lock(), sm.tri
        def failing(*a):                                    # the deadline comes after two ranges
            with lock:
                calls.append(1)
                n = len(calls)
            if n > 2:
                raise Stopped('w')
            return tri(*a)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sm, 'tri', failing):
            self.assertRaises(Stopped, split.triangular, F, P, S, B0, B, 1, state=d)
            done = [n for n in os.listdir(os.path.join(d, 'tri')) if n.startswith('range_')]       # their manifests
            with open(os.path.join(d, 'tri', 'plan.json')) as f:
                nr = len(json.load(f)['ranges'])
            self.assertEqual((len(done), len(calls)), (2, nr))
            counted = []
            def counting(*a):
                counted.append(1)
                return tri(*a)
            with mock.patch.object(sm, 'tri', counting):
                again = split.triangular(F, P, S, B0, B, 1, state=d)     # resumed: the same plan
            self.assertEqual(len(counted), nr - 2)
            self.assertEqual(sorted(map(str, again[0])), sorted(map(str, fresh[0])))
            self.assertEqual(again[1:], fresh[1:])

    def test_stopped(self):
        import time
        from rankbound import Stopped
        with tempfile.TemporaryDirectory() as d:
            t = time.time()
            def forever():
                while True:
                    arith.tick()
            with selmer.runtime.scope(selmer.runtime.Budget.for_run(seconds=1, workdir=d)):
                self.assertRaises(Stopped, selmer.run_py, forever, seconds=600)
            self.assertLess(time.time() - t, 30)
            self.assertIsNone(selmer.runtime.current().end)          # the budget is gone after it
            c = cv.classified(cv.from_ainvs([0, 0, 1, -1, 0], factor_seconds=600))
            self.assertRaises(Stopped, selmer.prove, c,
                              budget=selmer.runtime.Budget.for_run(seconds=0, workdir=d))
            self.assertRaises(Stopped, run_route, selmer.Selmer(os.path.join(d, 'w')), c,
                              budget=selmer.runtime.Budget.for_run(seconds=0, workdir=os.path.join(d, 'w')))


class TestRoute(unittest.TestCase):
    """rankbound(): E(Q)[2] = 0 goes to the 2-descent (relations from the sieve, or for an unconditional bound from our SPLIT),
    E(Q)[2] != 0 to the 2-descent; a finished folder answers at once."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_route(self):
        res = rankbound([0, 0, 1, -1, 0], r=1, compute=True, workdir=os.path.join(self.tmp, 'a'))
        self.assertEqual((res.method, res.rank_bound, res.grh), ('2-Selmer group', 1, True))
        self.assertTrue(os.path.exists(res.witness))
        again = rankbound([0, 0, 1, -1, 0], r=1, compute=True, workdir=os.path.join(self.tmp, 'a'))   # resumes
        self.assertEqual((again.witness, again.details.get('earlier')), (res.witness, True))
        d = rankbound([0, 0, 0, -1, 0], compute=True, workdir=os.path.join(self.tmp, 'b'))
        self.assertEqual(d.method, '2-isogeny descent')
        c37 = cv.classified(cv.from_ainvs([0, 0, 1, -1, 0]))
        from rankbound import runner
        self.assertEqual((runner.route(c37), runner.route(c37, uncond=True)), (runner.SIEVE, runner.SPLIT))
        self.assertEqual(runner.route(cv.classified(cv.from_ainvs([0, 0, 0, -1, 0]))), runner.DESCENT)


if __name__ == '__main__':
    unittest.main()
