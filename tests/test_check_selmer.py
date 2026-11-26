"""The checker of format-4 witnesses (rankbound/check/selmer.py) on witnesses built by test-only means
(tests/selmer_mk.gp: bnfinit, bnfsunit and ell2cover; neither the prover nor the checker): the right S on small
curves, and one corruption per check, each rejected.  The Cassels-Tate section on 571a1: its quartics from ell2cover
(not the 2-descent's elements), its pairs from the oracle's tdct (tests/data/oracle/ctp.gp, as the 2-isogeny route's); one
corruption per check of the section, each rejected."""
import copy, json, os, re, shutil, subprocess, tempfile, unittest
from unittest import mock
from rankbound import check as checker
from rankbound.check import selmer
from rankbound.check import selmer_statement
from helpers import TESTS

ROOT = os.path.dirname(TESTS)
FAST = bool(os.environ.get('RANKBOUND_TEST_FAST'))

# name: (minimal model, the expected S = dim Sel_2(E), the rank lower bound and its source, or None)
CURVES = {
    '11a1': ([0, -1, 1, -10, -20], 0, None),
    '37a1': ([0, 0, 1, -1, 0], 1, (1, 'LMFDB 37.a1')),
    '389a1': ([0, 1, 1, -2, 0], 2, (2, 'LMFDB 389.a1')),
    '5077a1': ([0, 0, 1, -7, 6], 3, None),
    '571a1': ([0, -1, 1, -929, -10595], 2, None),          # rank 0, Sha[2] = (Z/2)^2
    'icarm925': ([0, 1, 1, -23, -50], 0, None),            # rank 0, Brumer-Kramer bound 2
}


def build(ai, mink=False, name='t', rl=None):
    """Return a format-4 witness (a dict) for the curve ai, from selmer_mk.gp."""
    src = ''
    for f in (os.path.join(ROOT, 'tests', 'data', 'oracle', 'field.gp'), os.path.join(os.path.dirname(checker.__file__), 'check.gp'),
              os.path.join(TESTS, 'selmer_mk.gp')):
        with open(f) as fi:
            src += fi.read() + '\n'
    src += '\nmk4([%s], %d);\nquit;\n' % (','.join(map(str, ai)), 1 if mink else 0)
    fd, path = tempfile.mkstemp(suffix='.gp')
    with os.fdopen(fd, 'w') as f:
        f.write(src)
    try:
        r = subprocess.run(['gp', '-q', '-s', '256M', '-f', path], stdin=subprocess.DEVNULL, capture_output=True,
                           text=True, timeout=600)
    finally:
        os.unlink(path)
    if '***' in r.stdout + r.stderr:
        raise RuntimeError('selmer_mk.gp: ' + (r.stdout + r.stderr)[-2000:])
    L = {}
    for line in r.stdout.splitlines():
        t = line.split(' ', 1)
        L.setdefault(t[0], []).append(t[1] if len(t) > 1 else '')
    one = lambda k: L[k][0]
    lst = lambda k: [x for x in one(k).split(',') if x]
    claims = {t.split()[0]: int(t.split()[1]) for t in L['CL']}
    rels = L['REL']
    scol = [[int(x) for x in t.split()] for t in L.get('SCOL', [])]
    hyp = 'none' if mink else 'GRH'
    S = claims['S']
    st = ('We have that rank E(Q) <= %d.' if mink else 'Assuming the GRH, we have that rank E(Q) <= %d.') % S
    c = {'name': name,
         'curve': {'ainvs': lst('MM'), 'minimal_model': lst('MM'), 'bad_primes': lst('BAD')},
         'statement': st, 'torsion': [int(x) for x in lst('TORS')], 'hypothesis': hyp,
         'field': {'poly': lst('POLY'), 'dK': one('DK'),
                   'bound': {'type': 'minkowski' if mink else 'BDF', 'B': int(one('B')), 'ns': int(one('NS'))}},
         'weierstrass': lst('W'), 'theta': lst('TH'), 's_primes': lst('SPR'), 's_columns': scol,
         'local': {'inf': lst('INF'), 'p': {}},
         'claims': {k: claims[k] for k in selmer.CLAIMS},
         'created': '2026-11-14', 'relations': rels, 'ops': [], 'pivots': list(range(len(rels))),
         'ops_s': [], 'pivots_s': list(range(len(rels) + len(scol))),
         'elements': [[int(x) for x in t.split(',')] for t in L.get('EL', [])],
         'characters': {'real': [int(x) for x in lst('CHR')], 'primes': [[int(x) for x in t.split()] for t in L.get('CHQ', [])]},
         'format': selmer.FORMAT}
    for t in L['LOC']:
        p, _, xs = t.partition(' ')
        c['local']['p'][p] = {'points': [x for x in xs.split(',') if x], 'prec': 20}
    if rl:
        c['rank_lower_bound'] = {'r': rl[0], 'source': rl[1]}
        if S == rl[0]:
            c['statement'] += (' As E(Q) has %d independent rational point%s (%s; not proved in this witness), '
                               'we therefore have rank E(Q) = %d.' % (rl[0], 's' if rl[0] > 1 else '', rl[1], rl[0]))
    return c


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def run_witness(self, c, name='t'):
        path = os.path.join(self.d, name + '.witness.json')
        with open(path, 'w') as f:
            json.dump(c, f)
        return checker.verify(path)[:3]

    def failed(self, c, name='t', why=None):
        ok, res, _ = self.run_witness(c, name)
        self.assertFalse(ok, name)
        if why:             # the reason of a gp error()
            self.assertIn(why, ' '.join(x[2] for x in res if not x[1]), name)
        return [x[0] for x in res if not x[1]]


class TestCurves(Base):
    """Every witness built for the small curves passes, with S = dim Sel_2(E) (ell2cover) and v = U."""

    def test_curves(self):
        names = ['37a1', '571a1'] if FAST else list(CURVES)
        for name in names:
            ai, S, rl = CURVES[name]
            with self.subTest(curve=name):
                c = build(ai, name=name, rl=rl)
                self.assertEqual(c['claims']['S'], S)
                self.assertEqual(c['claims']['v'], c['claims']['U'])
                ok, res, st = self.run_witness(c, name)
                self.assertTrue(ok, [x for x in res if not x[1]])
                self.assertEqual(st, c['statement'])
                self.assertIn('parity', {x[0] for x in res})

    def test_minkowski(self):
        """The unconditional bound: Minkowski's B, hypothesis none."""
        for name in ('37a1', 'icarm925'):
            ai, S, _ = CURVES[name]
            with self.subTest(curve=name):
                c = build(ai, mink=True, name=name)
                self.assertEqual(c['claims']['S'], S)
                ok, res, st = self.run_witness(c, name)
                self.assertTrue(ok, [x for x in res if not x[1]])
                self.assertTrue(st.startswith('We have that rank E(Q) <= %d.' % S))
                if name == '37a1':          # B below Minkowski's bound (2.70 for d_K = 148)
                    c['field']['bound']['B'] = 2
                    self.assertIn('bound', self.failed(c, 'mink'))

    def test_bk_wins(self):
        """#925: S = dim Sel_2 = 0 below the Brumer-Kramer bound 2."""
        c = build(CURVES['icarm925'][0], name='icarm925')
        self.assertEqual((c['claims']['S'], c['claims']['S_bk']), (0, 2))

    def test_check_verdict(self):
        c = build(CURVES['37a1'][0], name='e37a1', rl=CURVES['37a1'][2])
        path = os.path.join(self.d, 'e37a1.witness.json')
        with open(path, 'w') as f:
            json.dump(c, f)
        v = checker.check(path)
        self.assertEqual((v['verdict'], v['rank_upper_bound'], v['GRH'], v['method']), ('accepted', 1, 'yes', '2-descent'))


class TestClasses(unittest.TestCase):
    def test_odd_places(self):
        """v4cls's square class above odd p (Euler's criterion in O/P) = the discrete logarithm's (ideallog in
        (O/P)*, the earlier rule), on random elements at every prime above 3, 5, 7, 11, 13, 10007 and 10^9 + 7 of four
        fields (split, ramified, inert and degree-2 primes)."""
        body = r'''
oldcls(nf, y, pr, p, bid) =
{
  my(ev = select(d -> d % 2 == 0, bid.cyc, 1), z, v = nfeltval(nf, y, pr, &z),
     dec = idealprimedec(nf, p), m = 0, s, lg);
  nfeltval(nf, p, pr, &s);
  foreach(dec, Q, if(Q != pr, my(w = idealval(nf, z, Q)); if(w < 0, m = max(m, ceil(-w / (2 * (Q.e)))))));
  if(m, z = nfeltmul(nf, z, nfeltpow(nf, s, 2 * m)));
  lg = ideallog(nf, z, bid);
  concat([v % 2], vector(#ev, i, lg[ev[i]] % 2));
}
{
setrand(5); bad = 0; n = 0;
foreach([x^3 - x - 1, x^3 - 5, x^3 + x^2 - 2*x - 1, x^3 - 3*x - 1], f, my(nf = nfinit(f));
  foreach([3, 5, 7, 11, 13, 10007, 10^9 + 7], p, foreach(idealprimedec(nf, p), pr, my(pl = v4place(nf, p, pr), bid = idealstar(nf, pr, 1));
    for(t = 1, 40, my(y = Mod(random([-50, 50]) + random([-50, 50]) * x + random([-50, 50]) * x^2, f) * p^random([-2, 2]) / random([1, 9]));
      if(y == 0, next); n++; if(v4cls(nf, y, pl) != oldcls(nf, y, pr, p, bid), bad++)))));
print("CLS ", n, " ", bad);
}
'''
        out = checker.run_gp(body, seconds=600)
        n, bad = map(int, next(l for l in out.splitlines() if l.startswith('CLS ')).split()[1:])
        self.assertGreater(n, 1000)
        self.assertEqual(bad, 0)


class TestDropped(Base):
    """S = {2} and the odd bad primes with c_p even, s_dropped the others ([p, c_p], c_p odd).  The
    witness is the prover's (selmer.prove at REDUCE_S = True) for y^2 + y = x^3 - 20x + 5 (bad primes 11, c = 2,
    and 4133, c = 1): S = {2, 11}, s_dropped [[4133, 1]].  Accepted; each corruption of the partition or of a c_p
    rejected."""

    @classmethod
    def setUpClass(cls):
        from rankbound import curve as cv, selmer as prover, witness
        real, prover.REDUCE_S = prover.REDUCE_S, True
        try:
            c = cv.Curve(name='sdrop', ainvs=['0', '0', '1', '-20', '5'], bad_primes=['11', '4133'],
                         minimal_model=['0', '0', '1', '-20', '5'])
            cls.c0 = witness.encode({k: v for k, v in prover.prove(c, log=lambda *a: None).items() if not k.startswith('_')})
        finally:
            prover.REDUCE_S = real

    def changed(self, f):
        c = copy.deepcopy(self.c0)
        f(c)
        return c

    def test_accepted(self):
        self.assertEqual((sorted(map(int, self.c0['s_primes'])), self.c0['s_dropped']), ([2, 11], [[4133, 1]]))
        self.assertEqual(sorted(self.c0['local']['p']), ['11', '2'])
        ok, res, _ = self.run_witness(self.c0)
        self.assertTrue(ok, [x for x in res if not x[1]])

    def test_corruptions(self):
        def move11(c):              # a kept prime (c_p = 2, even) dropped
            c['s_primes'] = [p for p in c['s_primes'] if int(p) != 11]
            c['s_dropped'] = c['s_dropped'] + [[11, 2]]
        def drop2(c):
            c['s_primes'] = [p for p in c['s_primes'] if int(p) != 2]
            c['s_dropped'] = c['s_dropped'] + [[2, 1]]
        cases = {
            'dropped p with c_p even': (move11, 's_dropped'),
            'wrong c_p': (lambda c: c.__setitem__('s_dropped', [[4133, 3]]), 's_dropped'),
            'dropped p = 2': (drop2, 's_primes'),
            'a bad prime in neither': (lambda c: c.__setitem__('s_dropped', []), 's_primes'),
            'a bad prime in both': (lambda c: c['s_primes'].append(4133), 's_primes'),
        }
        for what, (f, check) in cases.items():
            with self.subTest(what):
                self.assertIn(check, self.failed(self.changed(f), what), what)


WHY = {'ab_relation': 'does not factor as claimed', 'gen_relation': 'a prime of its norm is not given',
       'character_in_S': 'q must be a prime outside S', 'character_root': 'q must be a prime outside S',
       's_column_label': 's_column'}


class TestCorrupted(Base):
    """One corruption per check, on 389a1 (rank 2, lc(F) != 1 or not; three real places: points at infinity)."""

    @classmethod
    def setUpClass(cls):
        cls.c0 = build(CURVES['389a1'][0], name='e389a1')

    def changed(self, f):
        c = copy.deepcopy(self.c0)
        f(c)
        return c

    def test_good(self):
        ok, res, _ = self.run_witness(self.c0)
        self.assertTrue(ok, [x for x in res if not x[1]])

    def test_corruptions(self):
        c0 = self.c0
        ab = next(i for i, l in enumerate(c0['relations']) if l[0] not in 'ef')       # an a,b relation
        gen = next(i for i, l in enumerate(c0['relations']) if l.startswith('e:') and l.split(':')[-1])
        nonS = next(i for i, l in enumerate(c0['relations']) if l[0] not in 'ef' and
                    any(int(h, 16) not in {int(p) for p in c0['s_primes']} for h in l.split(':')[1].split(',') if h))
        p0 = c0['s_primes'][-1]

        def theta(c):
            c['theta'][0] = str(int(c['theta'][0].split('/')[0]) + 1)

        def nonsquare(c):           # a point where W(x) is not a square in Q_p
            T = [int(x) for x in c['weierstrass']]
            W = lambda x: sum(a * x ** i for i, a in enumerate(T))
            p = int(p0)
            x = next(x for x in range(1, 10 ** 4) if W(x) % p and pow(W(x) % p, (p - 1) // 2, p) == p - 1)
            c['local']['p'][p0]['points'].insert(0, str(x))

        def gen_rel(c):             # a general relation loses a prime of its norm
            head, H = c['relations'][gen].rsplit(':', 1)
            c['relations'][gen] = head + ':' + ','.join(H.split(',')[1:])

        def dep_chars(c):           # the character matrix loses rank: keep one character only
            c['characters'] = {'real': [], 'primes': c['characters']['primes'][:1]}

        def dup_element(c):
            c['elements'].append(list(c['elements'][0]))

        def bad_char(c):            # a character at a prime of S
            c['characters']['primes'][0] = [int(p0), 0]

        def bad_root(c):            # c not a root mod q
            q, r = c['characters']['primes'][0]
            c['characters']['primes'][0] = [q, (r + 1) % q]

        cases = {
            'bad_prime': (lambda c: c['curve']['bad_primes'].append('3'), 'disc_factored'),
            'minimal_model': (lambda c: c['curve'].__setitem__('minimal_model', ['0', '1', '1', '-2', '1']), 'minimal_model'),
            'theta': (theta, 'theta'),
            'weierstrass': (lambda c: c['weierstrass'].__setitem__(0, str(int(c['weierstrass'][0]) + 16)), 'weierstrass'),
            'nonsquare_point': (nonsquare, 'points_' + p0),
            'nonsquare_inf': (lambda c: c['local']['inf'].insert(0, '-1000'), 'points_inf'),
            'few_points': (lambda c: c['local']['p'][p0].__setitem__('points', []), 'image_' + p0),
            'few_points_inf': (lambda c: c['local'].__setitem__('inf', []), 'image_inf'),
            'dependent_characters': (dep_chars, 'characters'),
            'dependent_elements': (dup_element, 'characters'),
            'element_outside_S': (lambda c: c['elements'][0].append(nonS), 'elements_in_KS2'),
            'claim_dim_sel': (lambda c: c['claims'].__setitem__('dim_sel_cap_V', c['claims']['dim_sel_cap_V'] + 1), 'claims'),
            'claim_k': (lambda c: c['claims'].__setitem__('k', c['claims']['k'] + 1), 'claims'),
            'claim_S': (lambda c: c['claims'].__setitem__('S', c['claims']['S'] - 1), 'claims'),
            'root_number': (lambda c: c['claims'].__setitem__('w', -c['claims']['w']), 'root_number'),
            'bound_B': (lambda c: c['field']['bound'].__setitem__('B', 3), 'bound'),
            'ns': (lambda c: c['field']['bound'].__setitem__('ns', c['field']['bound']['ns'] + 1), 'ns'),
            'hypothesis': (lambda c: c.__setitem__('hypothesis', 'none'), 'hypothesis'),
            'statement': (lambda c: c.__setitem__('statement', c['statement'].replace('<= 2', '<= 1')), 'statement'),
            's_primes': (lambda c: c['s_primes'].append('3'), 's_primes'),
            'ops_range': (lambda c: c['ops_s'].append([0, 10 ** 6]), 'ops_in_range'),
            'k_pivots': (lambda c: c.__setitem__('pivots_s', []), 'claims'),
            'element_range': (lambda c: c['elements'][0].append(10 ** 6), 'elements_in_range'),
            'ab_relation': (lambda c: c['relations'].__setitem__(ab, '%d,%s' % (int(c['relations'][ab].split(',')[0]) + 1,
                                                                                 c['relations'][ab].split(',', 1)[1])), 'rejected'),
            'gen_relation': (gen_rel, 'rejected'),
            'character_in_S': (bad_char, 'rejected'),
            'character_root': (bad_root, 'rejected'),
            'real_index': (lambda c: c['characters']['real'].append(7), 'format'),
            's_column_label': (lambda c: c['s_columns'].append([int(p0), int(p0) - 1]) if [int(p0), int(p0) - 1] not in c['s_columns']
                               else c['s_columns'].append([2, 2]), 'rejected'),
            'format': (lambda c: c.__setitem__('hypothesis', 'maybe'), 'format'),
            'gp_injection': (lambda c: c['theta'].__setitem__(0, '0*system("touch MARK")'), 'format'),
        }
        for what, (f, check) in cases.items():
            with self.subTest(corruption=what):
                failed = self.failed(self.changed(f), what, WHY.get(what))
                if os.environ.get('SHOW'):
                    print(what, failed)
                self.assertIn(check, failed, (what, failed))
        self.assertFalse(os.path.exists('MARK'))

    def test_no_elements(self):
        """No elements (v = 0): accepted with dim_sel_cap_V = 0 and S = min(U, S_bk), characters and local points empty;
        rejected if a claim says otherwise (v, S, dim_sel_cap_V), the statement another S, or local data off S."""
        def bare(c, S=None):
            c['elements'], c['characters'] = [], {'real': [], 'primes': []}
            c['local'] = dict(c['local'], inf=[], p={})
            cl = c['claims']
            S = min(cl['U'], cl['S_bk']) if S is None else S
            cl.update(v=0, dim_sel_cap_V=0, S=S)
            c['statement'] = selmer_statement(S, c['hypothesis'], (c.get('rank_lower_bound') or {}).get('r'),
                                              (c.get('rank_lower_bound') or {}).get('source'))
        c = self.changed(bare)
        ok, res, _ = self.run_witness(c)
        self.assertTrue(ok, [x for x in res if not x[1]])
        for what, f, check in (
                ('v claimed 1', lambda c: c['claims'].__setitem__('v', 1), 'claims'),
                ('S below the bound', lambda c: bare(c, min(c['claims']['U'], c['claims']['S_bk']) - 1), 'claims'),
                ('dim_sel_cap_V 1', lambda c: c['claims'].__setitem__('dim_sel_cap_V', 1), 'claims'),
                ('statement', lambda c: c.__setitem__('statement', selmer_statement(c['claims']['S'] + 1, c['hypothesis'],
                                                                                 None)), 'statement'),
                ('local data off S', lambda c: c['local']['p'].__setitem__('3', {'points': [], 'prec': 1}), 'local_primes')):
            with self.subTest(what):
                c2 = copy.deepcopy(c)
                f(c2)
                self.assertIn(check, self.failed(c2, what), what)

    def test_parity(self):
        """A wrong w from gp, with every other check passing, is a parity mismatch (v = U)."""
        real = checker.run_gp

        def flip(body, **kw):
            out = real(body, **kw)
            return '\n'.join('NUM w %d' % -int(l.split()[2]) if l.startswith('NUM w ') else l for l in out.splitlines())
        with mock.patch.object(selmer, 'run_gp', flip):
            failed = self.failed(self.c0, 'parity')
        self.assertIn('parity', failed)

    def test_wrong_image(self):
        """Points of another place's image: the local check fails (the images' rank), never a wrong S."""
        c = self.changed(lambda c: c['local']['p'].__setitem__(c['s_primes'][-1], c['local']['p']['2']))
        failed = self.failed(c, 'img')
        self.assertTrue({'points_' + c['s_primes'][-1], 'image_' + c['s_primes'][-1]} & set(failed), failed)


def ct_section(ai, bad):
    """Return the Cassels-Tate section of the curve ai (minimal) from ell2cover's quartics (tdct), as the
    witness's dict."""
    body = 'addprimes(%s); my(c = tdct(%s, Set(concat([2], %s)))); print("CTE\\t", c[5], "\\t", c[1..3]);' % (
        bad, ai, bad)
    out = checker.run_gp(body, libs=(os.path.join(ROOT, 'tests', 'data', 'oracle', 'ctp.gp'),))
    rk, text = next(l.split('\t')[1:] for l in out.splitlines() if l.startswith('CTE'))
    Q, loc, pairs = json.loads(re.sub(r'-?[0-9]+(?:/[0-9]+)?', lambda m: '"%s"' % m.group(), text))
    return {'quartics': Q, 'local': loc, 'rank': int(rk),
            'pairs': [{'i': int(p[0]), 'j': int(p[1]), 'sum': p[2], 'm': p[3], 'sum_local': p[4], 'primes': p[5],
                       'points': p[6]} for p in pairs]}


class TestCasselsTate(Base):
    """571a1 (S = 2, Sha[2] = (Z/2)^2): with the section, rank <= S_ctp = 0; one corruption per check."""

    @classmethod
    def setUpClass(cls):
        c = build(CURVES['571a1'][0], mink=True, name='c571a1')
        c['cassels_tate'] = ct_section('[0,-1,1,-929,-10595]', '[571]')
        c['claims']['S_ctp'] = c['claims']['S'] - c['cassels_tate']['rank']
        c['statement'] = 'We have that rank E(Q) <= 0.'
        c['format'] = c.pop('format')
        cls.c0 = c

    def changed(self, f):
        c = copy.deepcopy(self.c0)
        f(c)
        return c

    def test_good(self):
        self.assertEqual((self.c0['claims']['S'], self.c0['cassels_tate']['rank']), (2, 2))
        ok, res, st = self.run_witness(self.c0)
        self.assertTrue(ok, [x for x in res if not x[1]])
        self.assertEqual(st, 'We have that rank E(Q) <= 0.')
        path = os.path.join(self.d, 'g.witness.json')
        with open(path, 'w') as f:
            json.dump(self.c0, f)
        v = checker.check(path)
        self.assertEqual((v['verdict'], v['rank_upper_bound'], v['rank']), ('accepted', 0, 0))
        nul = self.changed(lambda c: (c.__setitem__('cassels_tate', None), c['claims'].pop('S_ctp'),
                                      c.__setitem__('statement', 'We have that rank E(Q) <= 2.')))
        self.assertTrue(self.run_witness(nul)[0])          # null: the 2-descent's bound

    def test_corruptions(self):
        pr = next(p for p in self.c0['cassels_tate']['pairs'] if p['primes'])     # a pair with an extra prime

        def ct(f):
            return lambda c: f(c['cassels_tate'])

        def num(v, k, d=1):
            a = v[k].split('/')
            v[k] = '/'.join([str(int(a[0]) + d)] + a[1:])

        def drop_prime(t):          # an extra prime (and its point) left out: the symbol there is missing
            p = next(p for p in t['pairs'] if p['primes'])
            p['points'].pop(len(p['points']) - len(p['primes']))
            p['primes'].pop(0)

        def composite(t):           # an extra "prime" that is not prime
            p = t['pairs'][0]
            p['primes'].append('1000000000000000000000000000057' if not p['primes'] else str(10 ** 30 + 57))
            p['points'].append(list(p['points'][0]))

        def repeated(t):            # a prime listed twice would count its symbol twice
            p = next(p for p in t['pairs'] if p['primes'])
            p['primes'].append(p['primes'][-1])
            p['points'].append(list(p['points'][-1]))

        cases = {
            'quartic_coefficient': (ct(lambda t: num(t['quartics'][0], 2)), 'cassels_tate'),
            'quartic_fraction': (ct(lambda t: t['quartics'][0].__setitem__(4, '1/2')), 'format'),
            'quartic_other_curve': (ct(lambda t: t['quartics'].__setitem__(0, ['1', '0', '0', '0', '-1'])), 'cassels_tate'),
            'local_point': (ct(lambda t: t['local'][0].__setitem__(1, ['0', '0'])), 'cassels_tate'),
            'local_points_missing': (ct(lambda t: t['local'][0].pop()), 'format'),
            'sum_quartic': (ct(lambda t: num(t['pairs'][0]['sum'], 4, 7)), 'cassels_tate'),
            'sum_local_point': (ct(lambda t: t['pairs'][0]['sum_local'].__setitem__(0, ['0', '0'])), 'cassels_tate'),
            'm': (ct(lambda t: num(t['pairs'][0]['m'], 0)), 'cassels_tate'),
            'm_zero': (ct(lambda t: t['pairs'][0].__setitem__('m', ['0', '0', '0'])), 'cassels_tate'),
            'pair_point': (ct(lambda t: t['pairs'][0]['points'].__setitem__(0, ['0', '0'])), 'cassels_tate'),
            'prime_dropped': (ct(drop_prime), 'cassels_tate'),
            'prime_composite': (ct(composite), 'cassels_tate'),
            'prime_repeated': (ct(repeated), 'format'),
            'pair_index': (ct(lambda t: t['pairs'][0].__setitem__('i', t['pairs'][0]['j'])), 'format'),
            'pair_index_range': (ct(lambda t: t['pairs'][0].__setitem__('j', 9)), 'format'),
            'pairs_missing': (ct(lambda t: t.__setitem__('pairs', [])), 'format'),
            'rank': (ct(lambda t: t.__setitem__('rank', 0)), 'cassels_tate'),
            'S_ctp': (lambda c: c['claims'].__setitem__('S_ctp', c['claims']['S_ctp'] + 1), 'S_ctp'),
            'S_ctp_negative': (lambda c: c['claims'].__setitem__('S_ctp', -1), 'format'),
            'S_ctp_missing': (lambda c: c['claims'].pop('S_ctp'), 'format'),
            'S_ctp_without_section': (lambda c: c.__setitem__('cassels_tate', None), 'format'),
            'statement': (lambda c: c.__setitem__('statement', 'We have that rank E(Q) <= 2.'), 'statement'),
            'gp_injection': (ct(lambda t: t['pairs'][0]['m'].__setitem__(0, '0*system("touch MARK")')), 'format'),
        }
        self.assertTrue(pr['primes'])
        for what, (f, check) in cases.items():
            with self.subTest(corruption=what):
                failed = self.failed(self.changed(f), what)
                if os.environ.get('SHOW'):
                    print(what, failed)
                self.assertIn(check, failed, (what, failed))
        self.assertFalse(os.path.exists('MARK'))


if __name__ == '__main__':
    unittest.main()
