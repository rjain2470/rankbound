"""The norm-kernel bound in format 6 (rankbound-selmer-witness-6).  N: K(S, 2) -> Q(S, 2)
is onto and Sel_2 lies in ker N, so dim Sel_2 <= dim(Sel_2 cap V) + U - 1 - #S - dim(V cap ker N).  Honest witnesses
(the prover's, and format 6 of the format-4 fixtures, with and without elements) are accepted; every change of the new
claims is rejected; the attack corpus's single changes (test_check_attack) run on format 6; formats 4 and 5 keep
their meaning and reject the new claims; a witness format 5's packing cannot write falls back to format 4 with format
4's bound."""
import copy, gzip, json, os, shutil, tempfile, unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from rankbound import check, curve as cv, rankbound, selmer, witness
from rankbound.core import parity_floor
from helpers import TESTS
import test_check_attack as attack

DATA = os.path.join(TESTS, 'data')
FULL = bool(os.environ.get('RANKBOUND_TEST_FULL'))
QUIET = lambda *a: None
F6, F5, F4 = 'rankbound-selmer-witness-6', 'rankbound-selmer-witness-5', 'rankbound-selmer-witness-4'
NEW = ('dim_V_cap_ker_N', 'S_norm')


def load(name):
    with open(os.path.join(DATA, name + '.witness.json')) as f:
        return json.load(f)


def rel_vec(line):
    """The witness line of a relation as selmer.gp's vector (as the prover holds it)."""
    if line.startswith('f:'):
        return [2, int(line[2:])]
    if line.startswith('e:'):
        return [1] + [int(x) for x in line[2:].split(':')[0].split(',')]
    a, b = line.split(':')[0].split(',')
    return [0, int(a), int(b)]


def to6(c, elements=True):
    """The format-4 witness c (unpacked) as format 6, unpacked, its claims from the prover's functions (norm_class,
    norm_bound): with its elements, or without them (v = 0: no elements, characters, local points or pairing)."""
    c = copy.deepcopy(c)
    cl = c['claims']
    S = sorted(int(p) for p in c['s_primes'])
    if not elements:
        c['elements'], c['characters'] = [], {'real': [], 'primes': []}
        c['local']['p'] = {}
        c.pop('cassels_tate', None)
        cl.pop('S_ctp', None)
        cl.update(v=0, dim_sel_cap_V=0)
    rels = [rel_vec(l) for l in c['relations']]
    nv = [selmer.xor({j: selmer.norm_class(rels[j], c['field']['poly'], S) for j in e}, e) for e in c['elements']]
    vkn = len(nv) - selmer.rank(nv)
    Sn = selmer.norm_bound(cl['dim_sel_cap_V'], cl['U'], len(S), vkn, cl['w'])
    ctr = cl['S'] - cl['S_ctp'] if 'S_ctp' in cl else None
    new = {}
    for x, y in cl.items():
        if x == 'S_bk':
            new.update(dim_V_cap_ker_N=vkn, S_norm=Sn)
        new[x] = y
    new['S'] = min(Sn, cl['S_bk'])
    if ctr is not None:
        new['S_ctp'] = new['S'] - ctr
    c['claims'] = new
    rl = c.get('rank_lower_bound') or {}
    c['statement'] = witness.statement(new.get('S_ctp', new['S']), c['hypothesis'], rl.get('r'),
                                      None if 'points' in rl else rl.get('source'))
    c['format'] = F6
    return c


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='rb_norm_kernel')
        self.addCleanup(shutil.rmtree, self.d, True)

    def written(self, wit, name='w'):
        """Write wit (unpacked) with the prover's writer; return (the path, the packed JSON written)."""
        p = os.path.join(self.d, name + '.witness.json.gz')
        witness.write(wit, p)
        with gzip.open(p, 'rt') as f:
            return p, json.load(f)

    def raw(self, c, name='r'):
        """Check the JSON dict c as it is (no writer)."""
        p = os.path.join(self.d, name + '.witness.json')
        with open(p, 'w') as f:
            json.dump(c, f)
        return check.check(p)


class TestMath(unittest.TestCase):
    def test_norm_class(self):
        """x^3 - 2: N(alpha) = 2, N(1 + alpha) = 3 (1 + alpha^3), N((1 + alpha)/2) = 3/8, N(-1) = -1, N(3) = 27,
        N(3 - alpha) = F(3, 1)/1 = 25 (F(x, y) = x^3 - 2 y^3); bit 0 the sign, bit t + 1 the valuation mod 2 at S[t]."""
        f, S = ['-2', '0', '0', '1'], [2, 3, 5]
        self.assertEqual(selmer.norm_class([1, 0, 1, 0, 1], f, S), 0b0010)
        self.assertEqual(selmer.norm_class([1, 1, 1, 0, 1], f, S), 0b0100)
        self.assertEqual(selmer.norm_class([1, 1, 1, 0, 2], f, S), 0b0110)
        self.assertEqual(selmer.norm_class([1, -1, 0, 0, 1], f, S), 0b0001)
        self.assertEqual(selmer.norm_class([2, 3], f, S), 0b0100)
        self.assertEqual(selmer.norm_class([0, 3, 1], f, S), 0b0000)
        self.assertEqual(selmer.norm_class([0, 1, 1], f, S), 0b0001)          # 1 - 2 = -1
        # a non-monic index form, 2x^3 + x - 1 (lc 2): N(a - b alpha) = F(a, b)/2
        g = ['-1', '1', '0', '2']
        self.assertEqual(selmer.norm_class([0, 1, 1], g, S), 0)               # F(1, 1) = 2: N(1 - alpha) = 1
        self.assertEqual(selmer.norm_class([0, 0, 1], g, S), 0b0011)          # F(0, 1) = -1: N(alpha) = -1/2
        self.assertEqual(selmer.norm_class([0, 1, 1], g, S),
                         selmer.norm_class([1, 1, -1, 0, 1], g, S))           # the same element as e:

    def test_norm_bound(self):
        """slack = U - 1 - #S - dim(V cap ker N): 0 gives dim(Sel_2 cap V) itself (its parity must be w); > 0 the
        parity floor; < 0 is a contradiction."""
        self.assertEqual(selmer.norm_bound(1, 4, 1, 2, -1), 1)
        with self.assertRaises(RuntimeError):
            selmer.norm_bound(2, 4, 1, 2, -1)
        with self.assertRaises(RuntimeError):
            selmer.norm_bound(0, 4, 1, 3, 1)
        self.assertEqual(selmer.norm_bound(0, 39, 6, 0, 1), 32)                 # ICARM #6782 without elements
        self.assertEqual(selmer.norm_bound(0, 39, 6, 0, -1), 31)
        for dsel, U, v, nS, rkN, w in [(28, 39, 35, 6, 7, 1), (3, 12, 6, 3, 2, -1), (0, 9, 0, 2, 0, 1)]:
            vkn = v - rkN
            old = dsel + U - v if v == U else parity_floor(dsel + U - v, w)
            self.assertLessEqual(selmer.norm_bound(dsel, U, nS, vkn, w), old)


class TestHonest(Tmp):
    """The prover's format-6 witnesses (prove) and format 6 of the fixtures are accepted, with the true bound."""

    def test_prover(self):
        for ai, name, S in (([0, 0, 1, -1, 0], 'c37a1', 1), ([0, -1, 1, -929, -10595], 'c571a1', 2),
                            ([0, 0, 1, -7, 6], 'c5077a1', 3)):
            with self.subTest(name):
                w = selmer.prove(cv.from_ainvs(ai, name=name), log=QUIET)
                self.assertEqual(w['format'], F6)
                cl = w['claims']
                self.assertEqual(cl['S'], min(cl['S_norm'], cl['S_bk']))
                p, c = self.written(w)
                self.assertEqual(c['format'], F6)
                v = check.check(p)
                self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', S), v['reason'])
                self.assertEqual(witness.read(p)['claims'], cl)

    def test_fixtures(self):
        """37a1 (rank 1), 571a1 (rank 0, dim Sel_2 = 2; its pairing gives 0) and 37a1_cited with their elements; and
        without elements, where the norm kernel alone gives parity_floor(U - 1 - #S, w) = 1, 2, 1, below format 5's
        parity_floor(U, w) = 3, 4, 3 (so S_bk decided before)."""
        for name, with_el, S in (('37a1', True, 1), ('571a1', True, 0), ('37a1_cited', True, 1),
                                 ('37a1', False, 1), ('571a1', False, 2), ('37a1_cited', False, 1)):
            with self.subTest(name, elements=with_el):
                c = to6(load(name), with_el)
                if not with_el:
                    cl = c['claims']
                    self.assertLess(cl['S_norm'], parity_floor(cl['U'], cl['w']))
                p, _ = self.written(c)
                v = check.check(p)
                self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', S), v['reason'])


def honest6(test):
    """The packed format-6 witnesses of the fixtures, with and without elements, as JSON dicts, with their bounds."""
    out = {}
    for name, with_el, S in (('37a1', True, 1), ('571a1', True, 0), ('571a1', False, 2), ('37a1_cited', False, 1)):
        _, c = test.written(to6(load(name), with_el), '%s%d' % (name, with_el))
        out['%s/%s' % (name, 'el' if with_el else 'none')] = (S, c)
    return out


class TestSieve(Tmp):
    """sieve_witness's norm-only path: with r given and parity_floor(U - 1 - #S, w) already r (or r + 1), no
    elements are built; the witness has k's second elimination and no elements, and is accepted.  37a1 with r = 1:
    its S_bk is 1 as well, which takes the Brumer-Kramer path first, so sieve_witness's first parity_bound (that
    test only) is made to say 99."""

    def test_norm_only(self):
        real, calls = selmer.parity_bound, []

        def pb(*a):
            calls.append(a)
            return 99 if len(calls) == 1 else real(*a)
        with mock.patch.object(selmer, 'parity_bound', pb):
            res = rankbound([0, 0, 1, -1, 0], r=1, compute=True, workdir=os.path.join(self.d, 'job'))
        self.assertEqual((res.rank_bound, res.checked), (1, True))
        with gzip.open(res.witness, 'rt') as f:
            c = json.load(f)
        cl = c['claims']
        self.assertEqual((c['format'], c['elements'], cl['v'], cl['dim_V_cap_ker_N']), (F6, [], 0, 0))
        self.assertIn('pivots_s', c)
        self.assertEqual(cl['S_norm'], parity_floor(cl['U'] - 1 - len(c['s_primes']), cl['w']))
        self.assertEqual(cl['S'], 1)
        v = check.check(res.witness)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 1), v['reason'])

    def test_curve_only(self):
        """Without r the least bound is least(0, w): 37a1 (w = -1) curve-only has S_bk = 1 = least(0, -1), so its
        witness has no elements, with the bound the elements give (SKIP off), and is accepted."""
        res = rankbound([0, 0, 1, -1, 0], compute=True, workdir=os.path.join(self.d, 'co'))
        with gzip.open(res.witness, 'rt') as f:
            c = json.load(f)
        self.assertEqual((res.rank_bound, res.checked, c['elements'], c['claims']['v']), (1, True, [], 0))
        v = check.check(res.witness)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 1), v['reason'])
        with mock.patch.object(selmer, 'SKIP', False):
            full = rankbound([0, 0, 1, -1, 0], compute=True, workdir=os.path.join(self.d, 'co_el'))
        with gzip.open(full.witness, 'rt') as f:
            self.assertNotEqual(json.load(f)['elements'], [])
        self.assertEqual((full.rank_bound, full.checked), (res.rank_bound, True))


class TestMutations(Tmp):
    def test_new_claims(self):
        """Every change of dim_V_cap_ker_N, S_norm and S (+-1, 0, doubled) is rejected, the statement changed
        with S (so that only the claim is wrong) or not; so is a format-6 witness without them, a format-5 or 4
        label on one with them, and a format-5 witness given them."""
        bad = []
        for key, (S, c) in honest6(self).items():
            for x in NEW + ('S',):
                n = c['claims'][x]
                for m in sorted({n + 1, n - 1, 0, 2 * n} - {n}):
                    if m < 0:
                        continue
                    for restate in (False, True):
                        d = copy.deepcopy(c)
                        d['claims'][x] = m
                        if restate and x == 'S':
                            rl = d.get('rank_lower_bound') or {}
                            if 'S_ctp' in d['claims']:
                                d['claims']['S_ctp'] += m - n
                            d['statement'] = witness.statement(d['claims'].get('S_ctp', m), d['hypothesis'],
                                                              rl.get('r'), None if 'points' in rl else rl.get('source'))
                        v = self.raw(d)
                        if v['verdict'] != 'rejected':
                            bad.append((key, x, m, restate, v['verdict']))
            for x in NEW:
                d = copy.deepcopy(c)
                del d['claims'][x]
                if self.raw(d)['verdict'] != 'rejected':
                    bad.append((key, 'without', x))
            for lab in (F5, F4):
                d = copy.deepcopy(c)
                d['format'] = lab
                if self.raw(d)['verdict'] != 'rejected':
                    bad.append((key, 'label', lab))
        self.assertEqual(bad, [])
        f5 = attack.to5(load('37a1'))
        self.assertEqual(self.raw(f5)['verdict'], 'accepted')
        f5['claims'].update(dim_V_cap_ker_N=2, S_norm=1)
        v = self.raw(f5)
        self.assertEqual(v['verdict'], 'rejected')
        self.assertIn('claims must be an object', v['reason'])

    def test_corpus_changes(self):
        """test_check_attack's single changes on the format-6 witnesses: never 'error'; a change of a claim never
        accepted; an accepted change never below the true rank (all with RANKBOUND_TEST_FULL, else one in 20)."""
        todo = [(key, S) + x for key, (S, c) in honest6(self).items() for x in attack.changes(c)]
        todo = todo if FULL else todo[::20]

        def one(k):
            key, S, path, what, c = todo[k]
            v = self.raw(c, 'c%d' % k)
            bad = v['verdict'] == 'error' or (v['verdict'] == 'accepted' and (
                attack.claimed(path) or v['rank_upper_bound'] < S))
            return '%s %s %s: %s %s' % (key, path, what, v['verdict'], v['reason'] or v['rank_upper_bound']) if bad else None
        with ThreadPoolExecutor(int(os.environ.get('NCORES', 2))) as ex:
            bad = [x for x in ex.map(one, range(len(todo))) if x]
        self.assertEqual(bad, [])
        self.assertGreater(len(todo), 1000 if FULL else 50)


class TestOldFormats(Tmp):
    def test_fixtures_unchanged(self):
        """The format-4 fixtures and their format 5 keep their bounds (no new claims), and a format-4 or 5 witness
        rewritten by the prover stays format 5, without the new claims."""
        for name, S in (('37a1', 1), ('571a1', 0), ('37a1_cited', 1)):
            with self.subTest(name):
                c = load(name)
                self.assertEqual(c['format'], F4)
                for d in (c, attack.to5(c)):
                    v = self.raw(d)
                    self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', S), v['reason'])
                p, w = self.written(witness.read(os.path.join(DATA, name + '.witness.json')))
                self.assertEqual(w['format'], F5)
                self.assertFalse(set(NEW) & set(w['claims']))
                self.assertEqual(check.check(p)['rank_upper_bound'], S)

    def test_unpackable_falls_back(self):
        """A format-6 witness whose second elimination is not on a superset of pivots (format 5's packing cannot write
        it) is written in format 4, with format 4's claims, S and statement, and accepted: 571a1 without
        elements, ops_s and pivots_s empty (k = ns)."""
        c = load('571a1')
        for k in ('cassels_tate', 'rank_lower_bound'):
            c.pop(k, None)
        cl = c['claims']
        cl.pop('S_ctp', None)
        ns = c['field']['bound']['ns']
        cl.update(U=cl['U'] - cl['k'] + ns, k=ns)
        c['ops_s'], c['pivots_s'] = [], []
        c6 = to6(c, False)
        self.assertIsNotNone(witness.packable(c6))
        p, w = self.written(c6)
        self.assertEqual(w['format'], F4)
        self.assertFalse(set(NEW) & set(w['claims']))
        v = check.check(p)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', min(
            parity_floor(cl['U'], cl['w']), cl['S_bk'])), v['reason'])


    def test_unpackable_keeps_pairing(self):
        """The same with 571a1's elements and Cassels-Tate pairing kept: format 4's S = min(parity_floor(dim Sel_2 cap V
        + U - v, w), S_bk) = 2 (U = 9, v = 4), S_ctp = S less the pairing's rank 2 = 0, the bound the checker accepts."""
        c = load('571a1')
        cl = c['claims']
        cl.update(U=cl['U'] - cl['k'] + c['field']['bound']['ns'], k=c['field']['bound']['ns'])
        c['ops_s'], c['pivots_s'] = [], []
        c6 = to6(c, True)
        self.assertIn('cassels_tate', c6)
        self.assertIsNotNone(witness.packable(c6))
        p, w = self.written(c6)
        self.assertEqual(w['format'], F4)
        self.assertFalse(set(NEW) & set(w['claims']))
        S = min(parity_floor(cl['dim_sel_cap_V'] + cl['U'] - cl['v'], cl['w']), cl['S_bk'])
        self.assertEqual((S, w['claims']['S'], w['claims']['S_ctp']), (2, 2, 0))
        v = check.check(p)
        self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 0), v['reason'])


if __name__ == '__main__':
    unittest.main()
