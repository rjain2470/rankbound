"""The routes' proofs: they return the unwritten witness (core.ProofData.data; the runner writes it once), no
GP-text round trips inside the prover, one Budget (no wall clock limits), the checkpoints as runtime artifacts, the
norm-kernel target (selmer.norm_target; --uncond's element-free bound and its stop at slack 0), the T2 Gram matrix
in integers, and the Cassels-Tate allowance of the 2-isogeny route by measured cost."""
import dataclasses, inspect, json, os, shutil, tempfile, time, unittest
from fractions import Fraction
from unittest import mock
from rankbound import (arith as A, check, ctp, curve as cv, descent, field as fld, messages, rankbound, runtime, selmer,
                       witness)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.abspath(runtime.__file__))     # the package's sources
FILES = ('descent.py', 'selmer.py', 'ctp.py', 'field.py', 'curve.py', 'witness.py')
QUIET = lambda *a: None
E571 = [0, -1, 1, -929, -10595]
E5077 = [0, 0, 1, -7, 6]
E37 = [0, 0, 1, -1, 0]
I1664 = ([1, 0, 0, -37989785679112630, 2411227172016348395335076], [2, 3, 7, 11, 19, 29, 37, 41, 238529])
I1737 = ([1, 0, 0, -1319539461660, -159402536950172400], [2, 3, 5, 7, 11, 73, 103])
FISHER17 = ([0, 91502230365284038, 0, 489792722057841784540058275212361, 0],
            [2, 3, 5, 7, 11, 29, 71, 73, 127, 28619, 30187])


def source(name):
    with open(os.path.join(SRC, name)) as f:
        return f.read()


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)


class TestProofData(Tmp):
    """The routes return the witness unwritten; it is written once (by the runner)."""

    def test_written_once(self):
        real, calls = witness.write, []

        def counting(wit, path):
            calls.append(os.path.basename(path))
            return real(wit, path)
        with mock.patch.object(witness, 'write', counting):
            res = rankbound([0, 0, 0, -25, 0], compute=True, workdir=os.path.join(self.d, 'd'))      # descent
            self.assertEqual((res.rank_bound, res.checked, len(calls)), (1, True, 1))
            res = rankbound(E571, uncond=True, compute=True, workdir=os.path.join(self.d, 's'))     # the 2-descent, then the Cassels-Tate step
            self.assertEqual((res.rank_bound, res.checked, len(calls)), (0, True, 2))
        self.assertEqual(calls[1], os.path.basename(res.witness))

    def test_no_route_writes(self):
        """No witness writer in the routes' modules, and no path among the arguments of what makes a witness."""
        for name in ('descent.py', 'selmer.py', 'ctp.py'):
            s = source(name)
            self.assertNotRegex(s, r'witness(_file)?\.write\(|\bwrite\(wit|write_json\(', name)
        for f in (selmer.prove, selmer.sieve_witness, selmer.bk_witness, ctp.after_selmer, ctp.add, descent.witness):
            self.assertNotIn('path', inspect.signature(f).parameters, f.__name__)

    def test_add_returns_new(self):
        """The Cassels-Tate step returns the completed witness; the 2-descent's dict is left as it was (no change in place)."""
        m = selmer.prove(cv.from_ainvs(E571, name='n571'), log=QUIET)
        before = json.dumps({k: v for k, v in m.items() if k != '_pairing'}, sort_keys=True, default=str)
        c = ctp.after_selmer(m)
        self.assertEqual((c['claims']['S_ctp'], c['cassels_tate']['rank']), (0, 2))
        self.assertEqual(json.dumps({k: v for k, v in m.items() if k != '_pairing'}, sort_keys=True, default=str), before)


class TestRecords(unittest.TestCase):
    """Deviation 6: no GP-shaped text inside one process (field data, the Cassels-Tate step's relations, points, CT skip reasons)."""

    def test_gone(self):
        for mod, name in ((fld, 'sfield'), (fld, 'field_dict'), (ctp, 'parse_rel'), (selmer, 'deadline'),
                          (selmer, 'arith_until'), (selmer, 'seconds_left')):
            self.assertFalse(hasattr(mod, name), name)

    def test_field_record(self):
        F, small = fld.field_data([5077], [int(a) for a in E5077])
        self.assertEqual((type(F['dK']), type(F['r1']), type(F['GB']), type(F['w']), type(F['u'])), (str, int, int, int, int))
        self.assertTrue(all(isinstance(x, int) for x in F['real_roots'] + F['poly']))
        self.assertEqual(F['tamagawa'], {'5077': 1})
        self.assertEqual(F['bk_primes'][0]['p'], '5077')
        self.assertEqual(F['ns'], len(small))

    def test_m2_relations(self):
        """The Cassels-Tate step reads the 2-descent's relations as records (_pairing's rels), the same ones as the witness's lines."""
        w = selmer.prove(cv.from_ainvs(E571, name='r571'), log=QUIET)
        self.assertTrue(w['_pairing']['rels'])
        for j, r, H in w['_pairing']['rels']:
            self.assertEqual(selmer.rel_text(r, H), w['relations'][j])
        pts = {'rank_lower_bound': {'points': [['1/4', '3'], ['-2', '5']]}}
        self.assertEqual(ctp.points_x(pts), [Fraction(1), Fraction(-8)])

    def test_ct_reasons(self):
        """The 2-isogeny route's skipped pairings are (who, why, detail) records, why in time, cost, fail."""
        mm = cv.minimal(I1737[0])[0]

        def fail(E, S, left=None):
            raise RuntimeError('no quartic')
        out = descent.prove(mm, I1737[1], [], [], 1000, 0, fail)
        self.assertEqual(out['skipped'][0][1:], ('fail', 'no quartic'))
        out = descent.prove(mm, I1737[1], [], [], 1000, 0, descent.twotor_ct, budget=runtime.Budget(seconds=0))
        self.assertEqual(out['skipped'][0][1], 'time')


class TestBudget(unittest.TestCase):
    """Proof's files stop on a Budget, not on wall-clock limits."""

    def test_sources(self):
        """No budget.wall(), no time.time() compared with a limit (time.time() for costs and logs stays)."""
        for name in FILES:
            s = source(name)
            self.assertNotIn('.wall()', s, name)
            self.assertNotRegex(s, r'time\.time\(\)\s*[<>]|[<>]=?\s*time\.time\(\)|until\s*-\s*time|\bend\s*-\s*time\.time',
                                name)


class TestArtifacts(Tmp):
    """field.json, curve.json, m1 and the triangular state are artifacts with full fingerprints."""

    def test_field(self):
        st = os.path.join(self.d, 'f')
        F = fld.prepare(st, E5077, [5077], r=3, name='f5077')
        self.assertEqual(runtime.inspect(st, 'field', fld.FORMAT, fld.fingerprint(E5077, [5077], 3, 'f5077'))[0], 'valid')
        real, calls = fld.compute, []
        with mock.patch.object(fld, 'compute', lambda *a: calls.append(a) or real(*a)):
            self.assertEqual(fld.prepare(st, E5077, [5077], r=3, name='f5077'), F)      # reused
            self.assertEqual(len(calls), 0)
            G = fld.prepare(st, E5077, [5077], r=2, name='f5077')                       # another r: made again
            self.assertEqual((len(calls), G['rank_lower_bound']), (1, 2))
            with open(os.path.join(st, 'small.txt'), 'a') as f:                         # damaged: made again
                f.write('2 1\n')
            fld.prepare(st, E5077, [5077], r=2, name='f5077')
            self.assertEqual(len(calls), 2)

    def test_curve(self):
        c = cv.from_ainvs(E5077, name='c5077', r=3)
        cv.save(self.d, c)
        self.assertEqual(cv.saved(self.d, c), c)
        self.assertIsNone(cv.saved(self.d, cv.from_ainvs(E37, name='c5077')))          # another curve
        with open(os.path.join(self.d, 'curve.json'), 'a') as f:
            f.write(' ')
        self.assertIsNone(cv.saved(self.d, c))                                          # damaged

    def test_selmer_state(self):
        w = selmer.prove(cv.from_ainvs(E571, name='m571'), log=QUIET)
        F = fld.field_data([571], [int(a) for a in E571])[0]
        fp = selmer.selmer_fingerprint(F, None, 'cited', {'outcome': 'goal', 'relations': 10})
        self.assertTrue(ctp.wanted(w))
        selmer.selmer_save(self.d, w, fp)
        back = selmer.selmer_load(self.d, fp)
        self.assertEqual(json.dumps(back, sort_keys=True), json.dumps(w, sort_keys=True))
        self.assertEqual(ctp.after_selmer(back)['claims']['S_ctp'], 0)                     # the Cassels-Tate step resumes from it
        self.assertIsNone(selmer.selmer_load(self.d, selmer.selmer_fingerprint(F, None, 'cited', {'outcome': 'goal',
                                                                                         'relations': 11})))


class TestNormTarget(unittest.TestCase):
    """norm_target(F, r), against its definition: the largest G with norm_bound(0, r1 + r2 + #S_K + G,
    #S, 0, w) <= r (r + 1 when w rules out r's parity), #S_K from the 2-descent's places (sel_head), not from norm_target's."""

    def test_definition(self):
        for ai, P in ((E5077, [5077]), (E571, [571]), (E37, [37]), ([0, 1, 1, -2, 0], [389]), I1664):
            F = fld.field_data(P, [int(a) for a in ai])[0]
            S = selmer.s_primes(F)[0]
            d = selmer.head(F, [str(p) for p in F['bad_primes']], S)
            self.assertIsNone(selmer.norm_target(F, None))
            for r in range(0, 9):
                lo = r if F['w'] == (-1) ** r else r + 1
                ok = [G for G in range(0, 40) if selmer.norm_bound(0, d['r1'] + d['r2'] + d['SK'] + G, len(S), 0,
                                                                     F['w']) <= lo]
                self.assertEqual(selmer.norm_target(F, r), max(ok) if ok else -1, (ai, r))

    def test_k_at_most_G(self):
        """The sieve stops at G <= norm_target; its witness then needs no elements because k <= G (k adds the unit
        rows of S to G's relations): checked on the algebra of 5077a1 and 571a1."""
        seen = []
        real = selmer.algebra
        with mock.patch.object(selmer, 'algebra', lambda *a, **k: seen.append(real(*a, **k)) or seen[-1]):
            for ai, n in ((E5077, 'k5077'), (E571, 'k571')):
                selmer.prove(cv.from_ainvs(ai, name=n), log=QUIET)
        self.assertTrue(all(a['claims']['k'] <= a['claims']['G'] for a in seen))


class TestUncond(Tmp):
    """The norm-kernel bound on --uncond: the element-free bound where it already reaches the least bound, and no round past slack 0."""

    def test_element_free(self):
        c = dataclasses.replace(cv.from_ainvs(E5077, name='u5077'), r=3, source='test')
        w = selmer.prove(c, hypothesis='none', log=QUIET)
        self.assertEqual((w['claims']['v'], w['elements'], w['claims']['S'], '_pairing' in w), (0, [], 3, False))
        p = os.path.join(self.d, 'u.witness.json.gz')
        witness.write(w, p)
        v = check.check(p)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 3))

    def test_slack_stop(self):
        """Every round but the last has slack > 0 and v < U; with r None the norm bound still counts (S <= S_norm)."""
        rounds, real = [], selmer.algebra
        with mock.patch.object(selmer, 'algebra', lambda *a, **k: rounds.append(real(*a, **k)) or rounds[-1]), \
                mock.patch.object(selmer, 'SHELLS', 2):
            w = selmer.prove(cv.from_ainvs(E5077, name='s5077'), hypothesis='none', log=QUIET)
        slack = lambda c: c['U'] - 1 - len(w['s_primes']) - c['dim_V_cap_ker_N']
        for a in rounds[:-1]:
            self.assertTrue(slack(a['claims']) > 0 and a['claims']['v'] < a['claims']['U'])
        last = rounds[-1]['claims']
        self.assertTrue(slack(last) == 0 or last['v'] == last['U'])
        self.assertEqual(w['claims']['S'], min(w['claims']['S_norm'], w['claims']['S_bk']))


class TestT2(unittest.TestCase):
    """The T2 form in integers (t2int) through LLL, not ~300-bit rationals; the same bases."""

    def test_integer_gram(self):
        k = selmer.K([79915280, -167827015, -12326811, 13637294], [])
        p = 82422216381668818301
        seen, real = [], A.lll_gram
        with mock.patch.object(A, 'lll_gram', lambda G, **kw: seen.append(G) or real(G, **kw)):
            for Q in k.dec(p):
                if Q.f == 1:
                    k.t2basis(Q)
        self.assertTrue(seen)
        g0 = max(abs(int(x)).bit_length() for r in selmer.t2gram(k.f, k.F) for x in r)        # the form's own size
        for G in seen:          # H G0 H^t, H of entries < p: 2 log2 p + T2_BITS + log2 |G0| bits, and a few more
            self.assertTrue(all(type(x) is int for r in G for x in r))
            self.assertLess(max(abs(x).bit_length() for r in G for x in r), 2 * p.bit_length() + selmer.T2_BITS + g0 + 8)

    def test_same_bases(self):
        """On 300 degree-1 primes of #1664's field: the bases of the exact rational form and of t2int's."""
        F = fld.field_data(I1664[1], I1664[0])[0]
        k = selmer.kfield(F['poly'], I1664[1])
        G0 = selmer.t2gram(k.f, k.F)
        n = 0
        for p in A.primes(1000, 20000):
            for Q in k.dec(p):
                if Q.f != 1:
                    continue
                H = Q.hnf
                G = [[sum(H[i][a] * G0[a][b] * H[j][b] for a in range(3) for b in range(3)) for j in range(3)]
                     for i in range(3)]
                U = A.lll_gram(G, eta=Fraction(51, 100))
                exact = [[sum(U[t][c] * H[t][i] for t in range(3)) for i in range(3)] for c in range(3)]
                self.assertEqual(k.t2basis(Q), exact, p)
                n += 1
            if n >= 300:
                break


class TestCTRule(unittest.TestCase):
    """The 2-isogeny route's Cassels-Tate allowance: a cap, each pairing stopped once its projected cost does not fit
    (Costly), the pairings with the most lifted quartics first."""

    def test_allowance(self):
        self.assertEqual(descent.ct_allowance(0.5, None), descent.CT_SECONDS)
        self.assertEqual(descent.ct_allowance(500, None), 500)
        self.assertEqual(descent.ct_allowance(0.5, descent.CT_MARGIN + 10), 10)
        self.assertEqual(descent.ct_allowance(0.5, None, 7), 7)

    def test_costly(self):
        mm = cv.minimal(I1737[0])[0]
        S = sorted({2} | set(I1737[1]))
        t = time.monotonic()
        with self.assertRaises(ctp.Costly):
            descent.twotor_ct(mm, S, left=0.0)
        self.assertLess(time.monotonic() - t, 10)
        self.assertEqual(descent.twotor_ct(mm, S)[4], 2)                 # without a limit: its rank

    def test_order(self):
        """fisher17: the pairing on E' (5 lifted quartics) before the one on E (1): the bound 0 from E' alone."""
        mm = cv.minimal(FISHER17[0])[0]
        order = []

        def ct(E, S, left=None):
            order.append(list(E) == list(mm))
            raise RuntimeError('not here')
        out = descent.prove(mm, FISHER17[1], [], [], 1000, 0, ct)
        self.assertEqual(order, [False, True])
        self.assertEqual([x[0] for x in out['skipped']], ['1', 'E'])


class TestMessages(unittest.TestCase):
    def test_keys(self):
        for k in ('ct_skip_time', 'ct_skip_cost', 'ct_skip_fail'):
            self.assertTrue(messages.text('ct_skip', curve='E', why=messages.text(k, detail='3 s')))


if __name__ == '__main__':
    unittest.main()
