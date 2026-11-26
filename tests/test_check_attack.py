"""The regression corpus of attacks on the checker, regenerated from the
honest witnesses of tests/data/corpus.json (helpers.corpus; every bound there is the true rank) and the new-name
witnesses of tests/data (*.witness.json), with no prover:

  - every honest witness passes, with the true rank as its bound;
  - every forgery fails: the hand-made attacks (FORGERIES), and every change of a claimed value (claims, the
    statement, torsion, w, u, n, each B_e, s and pairing rank, the rank lower bound's r);
  - every other single change (the fuzz: each number +-1, 0, negated, doubled or halved; each list entry dropped,
    repeated or swapped with the next) gets a verdict ('accepted', 'rejected' or 'limit', never 'error'), and when
    accepted, a bound no lower than the true rank: such a change rewrites a proof the bound does not depend on.

RANKBOUND_TEST_FULL=1 checks every change (some 12,000), else one in 40, on NCORES threads (default 2).
Also the checker's fixes: fail closed (TestFailClosed) and the checker's hash over every source (TestProvenance)."""
import copy, glob, json, os, random, re, shutil, tempfile, unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from rankbound import check as checker
from rankbound.check import selmer
from helpers import TESTS, corpus, get, paths, put

FULL = bool(os.environ.get('RANKBOUND_TEST_FULL'))
C = corpus()
NEW = {'37a1': 1, '571a1': 0, 'n113': 0, '37a1_cited': 1}       # tests/data/NAME.witness.json: rankbound 3's names
for _k, _r in NEW.items():
    with open(os.path.join(TESTS, 'data', _k + '.witness.json')) as _f:
        C[_k + '.w'] = (_r, json.load(_f))


def to5(c):
    """The format-4 witness c in format 5 (docs/verification.md): ops flat, pivots delta-coded, pivots_s the rows
    beyond pivots (delta-coded), ops_s left out when it is ops, each element delta-coded."""
    c = copy.deepcopy(c)
    dl = lambda xs: [x - y for x, y in zip(xs, [0] + xs[:-1])]
    pv = sorted(c['pivots'])
    if 'pivots_s' in c:
        if c['ops_s'] == c['ops']:
            del c['ops_s']
        else:
            c['ops_s'] = [x for o in c['ops_s'] for x in o]
        c['pivots_s'] = dl(sorted(set(c['pivots_s']) - set(pv)))
    c['ops'] = [x for o in c['ops'] for x in o]
    c['pivots'] = dl(pv)
    c['elements'] = [dl(sorted(e)) for e in c['elements']]
    c['format'] = 'rankbound-selmer-witness-5'
    return c


for _k in list(C):                      # format 5 of every format-4 witness of rankbound 3
    if C[_k][1].get('format') == 'rankbound-selmer-witness-4':
        C[_k + '5'] = (C[_k][0], to5(C[_k][1]))
CLAIMED = {'w', 'u', 'n', 'B', 's', 's_isogenous', 'rank', 'rank_isogenous', 'r'}    # a claim wherever it is


def claimed(path):
    """Whether the value at path is a claim the checker recomputes (and so cannot change in a valid proof)."""
    return path[0] in ('claims', 'statement', 'torsion') or (path[0] != 'field' and path[-1] in CLAIMED)


def changes(c):
    """Yield (path, what, the changed witness) for every single change of c: each integer (JSON, or a string in
    decimal) to n +- 1, 0, -n, 2n, n // 2; each list entry dropped, repeated, or swapped with the next."""
    for path in paths(c):
        v = get(c, path)
        if type(v) is int or (isinstance(v, str) and re.fullmatch(r'-?[0-9]+', v)):
            n = int(v)
            for m in sorted({n + 1, n - 1, 0, -n, 2 * n, n // 2} - {n}):
                yield path, '= %d' % m, put(c, path, m if type(v) is int else str(m))
        if isinstance(v, list):
            for i in range(len(v)):
                for what, new in (('drop', v[:i] + v[i + 1:]), ('repeat', v[:i + 1] + v[i:]),
                                  ('swap', v[:i] + v[i + 1:i + 2] + v[i:i + 1] + v[i + 2:])):
                    if new != v:
                        yield path + (i,), what, put(c, path, new)


def set_(path, value):
    return lambda c: get(c, path[:-1]).__setitem__(path[-1], value)


def claims(**kw):
    """Claims changed, the statement rewritten to match (so that only the check under attack can stop it)."""
    def f(c):
        c['claims'].update(kw)
        S = kw.get('S_ctp', kw.get('S'))
        if S is not None:
            c['statement'] = re.sub(r'<= \d+\..*', '<= %d.' % S, c['statement'])
    return f


def ct571_on_37(c):
    c['cassels_tate'] = copy.deepcopy(C['571a1'][1]['cassels_tate'])
    claims(S_ctp=0)(c)


def times121(c):
    d = c['descent'][0]
    d['selmer'][0] = str(int(d['selmer'][0]) * 121)


def two_points(c):
    c['rank_lower_bound'] = {'r': 2, 'points': [['0', '0'], ['0', '-1']], 'primes': [17, 19, 23]}


# the attacks: (the witness, what changes); each must be rejected
FORGERIES = {
    'S = 0': ('37a1', claims(S=0, dim_sel_cap_V=0)),
    'B = 5, ns = 1': ('37a1', lambda c: c['field']['bound'].update(B=5, ns=1)),
    'none with the GRH bound': ('571a1', lambda c: (c.update(hypothesis='none'), c['field']['bound'].update(
        type='minkowski', B=10), claims(S_ctp=0)(c), c.update(statement=c['statement'].replace('Assuming the GRH, w', 'W')))),
    'none with type BDF': ('37a1', set_(('hypothesis',), 'none')),
    '2 dropped from S': ('37a1', lambda c: (c.update(s_primes=[]), c['s_dropped'].append([2, 1]))),
    'a prime in S and dropped': ('37a1', lambda c: c['s_primes'].append('37')),
    'a prime in neither': ('37a1', set_(('s_dropped',), [])),
    'unit row at the dropped prime': ('37a1', lambda c: c['s_columns'].append([37, 0])),
    'element odd outside S': ('37a1', lambda c: c['elements'][0].append(8)),
    'format 5: element odd outside S': ('37a1.w5', lambda c: c['elements'][0].append(8)),
    'format 5: a repeated pivot (delta 0)': ('37a1.w5', lambda c: c['pivots'].append(0)),
    'format 5: odd ops': ('37a1.w5', lambda c: c['ops'].append(0)),
    'format 5: ops_s without pivots_s': ('37a1.w5', lambda c: (c.pop('pivots_s', None), c.update(ops_s=[]))),
    'format 5: a negative delta': ('571a1.w5', lambda c: c['elements'][0].append(-1)),
    'element (37)': ('37a1', lambda c: (c['relations'].append('f:37'), c['elements'].append([41]))),
    'character at the dropped prime': ('37a1', set_(('characters', 'primes', 0), [37, 0])),
    'S as 1.0': ('37a1', set_(('claims', 'S'), 1.0)), 'S as true': ('37a1', set_(('claims', 'S'), True)),
    'S as "1"': ('37a1', set_(('claims', 'S'), '1')),
    'another curve': ('37a1', lambda c: c['curve'].update(ainvs=['0', '-1', '1', '-10', '-20'],
                                                          minimal_model=['0', '-1', '1', '-10', '-20'])),
    '571a1 quartics on 37a1': ('37a1', ct571_on_37),
    'CT rank 4, S_ctp = -2': ('571a1', lambda c: (c['cassels_tate'].update(rank=4), claims(S_ctp=-2)(c))),
    'CT rank 3, S_ctp = 0': ('571a1', lambda c: (c['cassels_tate'].update(rank=3), claims(S_ctp=0)(c))),
    'invalid conic point, rank -1': ('n5', lambda c: (set_(('descent', 0, 'pairing', 'conic', 0), ['1', '1', '1'])(c),
                                                           c['descent'][0]['pairing'].update(rank=-1))),
    'invalid conic point, B_e + 1': ('n5', lambda c: (set_(('descent', 0, 'pairing', 'conic', 0), ['1', '1', '1'])(c),
                                                      c['descent'][0].update(B=c['descent'][0]['B'] + 1))),
    'Selmer basis times 121': ('n5', times121),
    '2-isogeny S = 0': ('n5', lambda c: (claims(B=0, S=0)(c), c.pop('rank_lower_bound'))),
    '2-isogeny S = 0 with points': ('n5', claims(B=0, S=0)),
    '2-isogeny empty CT section': ('n5', set_(('cassels_tate',), {'quartics': [], 'local': [], 'pairs': [], 'rank': 0})),
    '2-isogeny CT rank + 1': ('n113', lambda c: c['cassels_tate'].update(rank=c['cassels_tate']['rank'] + 1)),
    'r = 2 from dependent points': ('37a1', two_points),
    'a prime with #E(F_p) odd': ('37a1', set_(('rank_lower_bound', 'primes'), [3])),
    's_primes [0]': ('37a1', set_(('s_primes',), ['0'])), 's_primes [4]': ('37a1', set_(('s_primes',), ['4'])),
    's_column at 10': ('37a1', set_(('s_columns',), [[10, 1]])),
    'F reducible': ('37a1', set_(('field', 'poly'), ['0', '-3', '1', '1'])),
    'W reducible (2-torsion)': ('37a1', lambda c: c['curve'].update(ainvs=['0', '0', '0', '-1', '0'])),
    'singular': ('37a1', lambda c: c['curve'].update(ainvs=['0', '0', '0', '0', '0'])),
    'format 3: a composite bad prime': ('37a1.f3', set_(('curve', 'bad_primes'), ['74'])),
    'u, n claimed otherwise': ('37a1', lambda c: c['local'].update(n=c['local']['n'] + 1)),
    'format 3: Brumer-Kramer n + 1': ('37a1.f3', lambda c: c['local'].update(n=c['local']['n'] + 1)),
    'format 3: S - 1': ('37a1.f3', claims(S=0)), 'format 3: G - 1': ('icarm0.f3', claims(G=-1)),
    'format 2: GB / 2': ('icarm0_f2', lambda c: c['field'].update(GB=c['field']['GB'] // 2)),
    # rankbound 3's names: the same attacks, and a statement in the other format's word
    'new names: S = 0': ('37a1.w', claims(S=0, dim_sel_cap_V=0)),
    'new names: CT rank 3, S_ctp = 0': ('571a1.w', lambda c: (c['cassels_tate'].update(rank=3), claims(S_ctp=0)(c))),
    'new names: 2-isogeny CT rank + 1': ('n113.w', lambda c: c['cassels_tate'].update(rank=c['cassels_tate']['rank'] + 1)),
    'new names: cited r, S = 0': ('37a1_cited.w', claims(S=0, dim_sel_cap_V=0)),
    'new names: "certificate" in the statement': ('37a1_cited.w', lambda c: c.update(statement=c['statement'].replace(
        'this witness', 'this certificate'))),
    'old names: "witness" in the statement': ('37a1_cited.w', lambda c: c.update(format='rankbound-selmer-certificate-4')),
    'format 3 under a new name': ('37a1.f3', set_(('format',), 'rankbound-selmer-witness-3')),
    '2-isogeny under the format-4 name': ('n113.w', set_(('format',), 'rankbound-selmer-witness-4')),
}


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def check(self, c, name='t'):
        path = os.path.join(self.d, name + '.cert.json')
        with open(path, 'w') as f:
            json.dump(c, f)
        return checker.check(path)


class TestCorpus(Base):
    def test_honest(self):
        for name, (rank, c) in C.items():
            with self.subTest(name):
                v = self.check(c)
                self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', rank), v['reason'])

    def test_forgeries(self):
        for what, (name, f) in FORGERIES.items():
            with self.subTest(what):
                c = copy.deepcopy(C[name][1])
                f(c)
                v = self.check(c)
                self.assertEqual(v['verdict'], 'rejected', v['reason'])

    def test_changes(self):
        todo = [(name, rank) + x for name, (rank, c) in C.items() for x in changes(c)]
        todo = todo if FULL else todo[::40]

        def one(k):
            name, rank, path, what, c = todo[k]
            v = self.check(c, 'c%d' % k)
            bad = v['verdict'] == 'error' or (v['verdict'] == 'accepted' and (
                claimed(path) or v['rank_upper_bound'] < rank))
            return '%s %s %s: %s %s' % (name, path, what, v['verdict'], v['reason'] or v['rank_upper_bound']) if bad else None
        with ThreadPoolExecutor(int(os.environ.get('NCORES', 2))) as ex:
            bad = [x for x in ex.map(one, range(len(todo))) if x]
        self.assertEqual(bad, [])
        self.assertGreater(len(todo), 10000 if FULL else 250)


class TestFailClosed(Base):
    """A format-4 path that stopped before the claims without recording a failure (a point class of the wrong length
    from gp: `good = False` and return) accepted the claims unchecked: 37a1 (rank 1) with rank <= 0.  Now the
    mismatch is a failed check, and no witness is accepted unless its claims and statement were checked."""

    def forged(self):
        c = copy.deepcopy(C['37a1'][1])
        FORGERIES['S = 0'][1](c)
        return c

    def test_point_class_length(self):
        real = checker.run_gp

        def longer(body, **kw):     # one more bit on every point class at 2
            return '\n'.join(l + '0' if l.startswith('PI 2 ') else l for l in real(body, **kw).splitlines())
        with mock.patch.object(selmer, 'run_gp', longer):
            v = self.check(self.forged())
        self.assertEqual(v['verdict'], 'rejected')
        self.assertIn('local_data', [x['check'] for x in v['checks'] if not x['ok']])

    def test_claims_unchecked(self):
        """Any path that returns before the claims (here: the checks stubbed out) is rejected ('complete')."""
        with mock.patch.object(selmer, 'verify', lambda c, rels, res, ok, *a: ('', False)):
            v = self.check(self.forged())
        self.assertEqual(v['verdict'], 'rejected')
        self.assertEqual([x['check'] for x in v['checks'] if not x['ok']], ['complete'])


def gp_number(expr):
    """The integer that gp prints for expr (setrand(3) first)."""
    return int(checker.run_gp('setrand(3); print(%s);' % expr).split()[0])


class TestPrompt(Base):
    """Witnesses that crashed the checker (verdict 'error') or made it work without bound before a cheap check that
    rejects them: each gets 'rejected' or 'limit', in seconds."""

    def verdict(self, name, f, want='rejected', first=None, seconds=30, **kw):
        c = copy.deepcopy(C[name][1])
        f(c)
        path = os.path.join(self.d, 't.witness.json')
        with open(path, 'w') as fo:
            json.dump(c, fo)
        v = checker.check(path, **kw)
        failed = [x['check'] for x in v['checks'] if not x['ok']]
        self.assertEqual(v['verdict'], want, v['reason'])
        if first:
            self.assertEqual(failed[0], first, v['reason'])
        self.assertLess(v['seconds'], seconds, v['reason'])
        return v

    def test_minkowski_huge_dK(self):
        """A claimed |d_K| of 10^700 overflowed math.exp in selmer.limits ('error'); the cap is now in logs, and
        gp stops at maximal_order."""
        self.verdict('37a1_unc', lambda c: c['field'].update(dK='-1' + '0' * 700), first='maximal_order')
        c = {'field': {'dK': '1' + '0' * 3999, 'bound': {'type': 'minkowski', 'B': 10 ** 1999}},
             'rank_lower_bound': None, 'characters': {'primes': []}}
        selmer.limits(c)                                    # under the cap (about 10^1999.3)
        c['field']['bound']['B'] = 10 ** 2001
        with self.assertRaisesRegex(checker.LimitExceeded, r"above twice Minkowski's bound about 10\^1999"):
            selmer.limits(c)

    def test_small_dK(self):
        """|d_K| < 23 is no cubic field's: rejected at the format check (dK = 0 was 'limit', from Bach's bound 0)."""
        for name in ('37a1', '37a1.f3'):
            for dK in ('0', '-22', '1'):
                with self.subTest(name=name, dK=dK):
                    self.verdict(name, lambda c: c['field'].update(dK=dK), first='format')

    def test_index_form_not_primitive(self):
        """F times 2 (formats 4 and 3) reached check rel, whose exit 2 was 'error'; now rejected at index_form."""
        double = lambda c: c['field'].update(poly=[str(2 * int(x)) for x in c['field']['poly']])
        for name in ('37a1', '37a1.f3'):
            with self.subTest(name):
                self.verdict(name, double, first='index_form')

    def test_unconfirmed_dK(self):
        """B under the cap of a claimed d_K the field does not have: gp worked up to B (forprime to 10^290, BDF's
        data to 10^9) after maximal_order failed; it stops there now."""
        def big(dK, B):
            return lambda c: (c['field'].update(dK=dK), c['field']['bound'].update(B=B))
        self.verdict('37a1_unc', big('-1' + '0' * 600, 10 ** 290), first='maximal_order')
        self.verdict('37a1', big('-1' + '0' * 3990, 10 ** 9), first='maximal_order')

    def test_cheap_checks_first(self):
        """A bad prime of 1,000 digits that does not divide the discriminant was proved prime (APRCL, minutes)
        before disc_factored; a Cassels-Tate pair's extra prime before its points."""
        p = gp_number('nextprime(10^999)')
        self.verdict('37a1', lambda c: c['curve']['bad_primes'].append(str(p)), seconds=10)
        p = gp_number('nextprime(10^399 + random(10^398))')
        self.verdict('571a1', lambda c: (c['cassels_tate']['pairs'][0]['primes'].append(str(p)),
                                         c['cassels_tate']['pairs'][0]['points'].append(['1', '1'])), seconds=3)

    def test_relations_time_limit(self):
        """Without --max-seconds the relation check (check rel, ckrels) had no time limit: an e: relation with d
        a 100-digit semiprime factored d for ever.  It has NO_LIMIT, as every program of a check (here 3 s)."""
        n = gp_number('nextprime(10^49 + random(10^48)) * nextprime(10^50 + random(10^49))')
        with mock.patch.object(checker, 'NO_LIMIT', 3), mock.patch.object(selmer, 'NO_LIMIT', 3):
            v = self.verdict('37a1', lambda c: c['relations'].append('e:1,0,0,%d:' % n), 'limit')
        self.assertIn('within 3 seconds', v['reason'])


class TestRewrites(Base):
    """Valid rewrites of a proof, accepted with the same bound, and the forgeries next to them."""

    def test_twin_relation(self):
        """A relation a - b alpha and its e: twin (c0, c1, c2, d) = (a, -b, 0, 1), or (2a, -2b, 0, 2) with 2 added to
        its primes, as one more pivot: the same element, so check rel's and ckgen's rows must have the same column
        names, dim V is unchanged and the bound stays.  With G one lower claimed: rejected."""
        rel = lambda c: next(r for r in c['relations'] if not r.startswith(('f', 'e')) and r.split(':')[1])
        for d in (1, 2):
            def twin(c):
                (a, b), H = rel(c).split(':')[0].split(','), rel(c).split(':')[1]
                c['relations'].append('e:%d,%d,0,%d:%s' % (d * int(a), -d * int(b), d, H + (',2' if d == 2 else '')))
                c['pivots'].append(len(c['relations']) - 1)
            with self.subTest(d=d):
                v = self.check(self._changed('37a1', twin))
                self.assertEqual((v['verdict'], v['rank_upper_bound']), ('accepted', 1), v['reason'])

        def twins(c):
            for r in list(c['relations']):
                if not r.startswith(('f', 'e')) and r.split(':')[1]:
                    a, b = r.split(':')[0].split(',')
                    c['relations'].append('e:%s,%d,0,1:%s' % (a, -int(b), r.split(':')[1]))
                    c['pivots'].append(len(c['relations']) - 1)
            c['claims'].update(G=c['claims']['G'] - 1, dimV=c['claims']['dimV'] + 1)
        v = self.check(self._changed('571a1', twins))
        self.assertEqual(v['verdict'], 'rejected', v['reason'])
        self.assertIn('claims', [x['check'] for x in v['checks'] if not x['ok']])

    def test_relabel_minkowski(self):
        """A GRH witness whose B (9) is above Minkowski's bound for d_K = 148 (3.4), relabelled 'none'/'minkowski':
        a valid unconditional proof, accepted with the same bound."""
        def relabel(c):
            c.update(hypothesis='none', statement=c['statement'].replace('Assuming the GRH, w', 'W'))
            c['field']['bound']['type'] = 'minkowski'
        v = self.check(self._changed('37a1', relabel))
        self.assertEqual((v['verdict'], v['rank_upper_bound'], v['GRH']), ('accepted', 1, 'no'), v['reason'])

    def _changed(self, name, f):
        c = copy.deepcopy(C[name][1])
        f(c)
        return c


NUMS = re.compile(r'([A-Za-z_]+) = (-?\d+)')


def info(v, name):
    return next((x['info'] for x in v['checks'] if x['check'] == name), '') or ''


def fix_claims(c, v):
    """Rewrite the claims of the 2-Selmer witness c to the values the checker recomputed (the infos of the verdict v)
    and its statement to match; return whether c changed."""
    before = json.dumps(c, sort_keys=True)
    m = re.search(r'degree-1 primes of norm < B: (\d+)', info(v, 'ns'))
    if m:
        c['field']['bound']['ns'] = int(m[1])
    for k, x in NUMS.findall(info(v, 'claims').split(';')[0]):
        if k in c['claims'] and int(x) >= 0:
            c['claims'][k] = int(x)
    for k in ('u', 'n'):
        m = re.search(r'%s = (\d+)' % k, info(v, 'claim_' + k))
        for d in (c['claims'], c['local']):
            if m and k in d:
                d[k] = int(m[1])
    ct = c.get('cassels_tate')
    if ct:
        m = re.search(r': (-?\d+)$', info(v, 'cassels_tate'))
        if m and int(m[1]) >= 0:
            ct['rank'] = int(m[1])
        if c['claims']['S'] >= ct['rank']:
            c['claims']['S_ctp'] = c['claims']['S'] - ct['rank']
    S, rl = c['claims'].get('S_ctp', c['claims']['S']), c.get('rank_lower_bound') or {}
    c['statement'] = re.sub(r'<= \d+\..*', '<= %d.' % S, c['statement']) + checker._with_r(
        S, rl.get('r'), None if 'points' in rl else rl.get('source'), checker.word(c))
    return json.dumps(c, sort_keys=True) != before


def mutate(c, rnd):
    """One to three random changes of the data (not the claims) of the 2-Selmer witness c, in place."""
    n, E = len(c['relations']), c['elements']
    for _ in range(rnd.randint(1, 3)):
        m = rnd.randrange(10)
        if m == 0 and E:
            E.pop(rnd.randrange(len(E)))
        elif m == 1:
            E.append(sorted(rnd.sample(range(n), rnd.randint(1, min(3, n)))))
        elif m == 2 and len(E) >= 2:
            i, j = rnd.sample(range(len(E)), 2)
            E[i] = sorted(set(E[i]) ^ set(E[j])) or E[i]
        elif m == 3 and c['pivots']:
            c['pivots'].pop(rnd.randrange(len(c['pivots'])))
        elif m == 4:
            c['pivots'].append(rnd.randrange(n))
        elif m == 5:
            c['ops'].append([rnd.randrange(n), rnd.randrange(n)])
        elif m == 6 and c['characters']['primes']:
            c['characters']['primes'].pop(rnd.randrange(len(c['characters']['primes'])))
        elif m == 7:
            pts = c['local']['p'][rnd.choice(sorted(c['local']['p']))]['points']
            if pts:
                pts.pop(rnd.randrange(len(pts)))
        elif m == 8:
            c['field']['bound']['B'] = max(2, c['field']['bound']['B'] + rnd.choice((-1, 1)))
        elif m == 9 and 'points' in (c.get('rank_lower_bound') or {}) and c['rank_lower_bound']['r']:
            c['rank_lower_bound']['points'].pop()
            c['rank_lower_bound']['r'] -= 1


@unittest.skipUnless(FULL, 'RANKBOUND_TEST_FULL=1 (about 2 minutes on 2 threads)')
class TestClaimsOracle(Base):
    """Stronger than test_changes, which accepts a change only with the claims as they were: after random changes of
    a 2-Selmer witness's data, every claim is rewritten to the checker's own values (up to five times), so that only
    the proof stands between the change and acceptance.  An accepted witness must still bound the true rank, and
    prove no r above it; no verdict is 'error'."""

    def test_oracle(self):
        names = [k for k, (_, c) in C.items() if c['format'] in ('rankbound-selmer-certificate-4', 'rankbound-selmer-witness-4')]

        def one(k):
            rnd = random.Random(k)
            name = rnd.choice(names)
            rank, c = C[name][0], copy.deepcopy(C[name][1])
            mutate(c, rnd)
            for _ in range(5):
                v = self.check(c, 'o%d' % k)
                if v['verdict'] != 'rejected' or not fix_claims(c, v):
                    break
            bad = v['verdict'] == 'error' or (v['verdict'] == 'accepted' and (
                v['rank_upper_bound'] < rank or (v['r_proved'] and v['rank_lower_bound'] > rank)))
            return v['verdict'], '%d %s: %s %s' % (k, name, v['verdict'], v['reason'] or v['rank_upper_bound']) if bad else None
        with ThreadPoolExecutor(int(os.environ.get('NCORES', 2))) as ex:
            out = list(ex.map(one, range(int(os.environ.get('ORACLE_CASES', 1000)))))
        self.assertEqual([x for _, x in out if x], [])
        verdicts = [v for v, _ in out]
        self.assertGreater(min(verdicts.count('accepted'), verdicts.count('rejected')), len(out) // 10, verdicts)


class TestProvenance(unittest.TestCase):
    """The checker's hash covered every source but relcheck.c (the relation check of formats 3 and 4): two checkers
    with different relation checks reported the same hash.  It covers every source, and changes with each."""

    def test_sources(self):
        here = os.path.dirname(checker.__file__)
        mine = {os.path.basename(f) for e in ('*.py', '*.gp', '*.c') for f in glob.glob(os.path.join(here, e))}
        self.assertEqual(set(checker.SOURCES), mine)
        h = checker.checker_hash()
        real = checker.sha256
        for name in checker.SOURCES:
            with mock.patch.object(checker, 'sha256', lambda p: 'x' if p.endswith(os.sep + name) else real(p)):
                self.assertNotEqual(checker.checker_hash(), h, name)


if __name__ == '__main__':
    unittest.main()
