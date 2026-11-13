"""The verifier's checks of a 2-descent certificate (format FORMAT): curves E/Q with a rational point of order 2.

What such a certificate proves, unconditionally (two_descent.gp has the mathematics and its references).  For each
rational root e of T(X) = X^3 + b2 X^2 + 8 b4 X + 16 b6, the two 2-isogeny Selmer groups Sel(a, b) and Sel(a', b'),
of dimensions s and s', and the ranks of the isogeny pairings Theta_1 and Theta_1' on them [Fisher, Math. Comp. 86
(2017), Thm. 2.1] give rank E(Q) <= r_inf <= s + s' - 2 - rank Theta_1 - rank Theta_1', with (-1)^r_inf = w
(2-parity); with the Cassels-Tate pairing on Sel_2(E) [Fisher, Res. Number Theory 8 (2022), Thm. 3.1], also
r_inf <= s + s' - 2 - rank Theta_1 - rank CT_E, and on Sel_2(E'), r_inf <= s + s' - 2 - rank Theta_1' - rank CT_E'.
B_e is the least of these; rank E(Q) <= S, the largest integer <= B = min_e B_e with (-1)^S = w.  With r points whose
characters (with the torsion generators) have rank t + r, rank E(Q) >= r.

The certificate is one JSON file with these keys, in this order:
  name              e.g. icarm13
  curve             ainvs [a1,a2,a3,a4,a6] (the reduced minimal model), minimal_model, bad_primes
  statement         the theorem proved, in words; the verifier rebuilds it and compares
  torsion           the structure of E(Q)_tors as from PARI's elltors, e.g. [2], [2, 2], [6, 2]
  GRH               "no": the bound does not depend on GRH
  rank_lower_bound  optional: r with points on ainvs and odd good primes p at whose characters (every root of T
                    mod p) the torsion generators and the points have rank t + r; or r with a source
  descent           one entry per root e of T, in increasing order: e, a, b (T(X + e) = X(X^2 + aX + b)),
                    selmer and selmer_isogenous (bases of Sel(a, b) and Sel(a', b') as signed integers), s,
                    s_isogenous (their dimensions), pairing and B (= s + s' - 2 - rank - rank_isogenous).
                    pairing: places (R, then the primes of 2 * the bad primes and 3, 5, 7), and for each group:
                    conic (per basis element xi, a rational point [X0, Y0, W0] of W^2 = xi X^2 + A XY + (B/xi) Y^2),
                    local (per basis element and place, [s, t] with xi s^4 + A s^2 t^2 + (B/xi) t^4 a nonzero
                    square there) and rank (of Theta_1); the same with _isogenous for Sel(a', b').
                    cassels_tate: null, or the Cassels-Tate data of E' = y^2 = x(x^2 + a'x + b') (below)
  cassels_tate      null, or the Cassels-Tate data of E: quartics (binary quartics [a, b, c, d, e] with invariants
                    I = c4, J = 2 c6 of the minimal model), local (per quartic, a point at each place of the pairing
                    where it is a nonzero square), pairs (for every i < j: sum, a quartic of the class of the sum; m,
                    [m0, m1, m2] with m^2 = z(g_i) z(g_j) z(sum) in L; sum_local; primes, the primes of g_j(1, 0) and of
                    the content of gamma1 outside the places; points, on g_i at the places and those primes, where
                    gamma1 != 0) and rank
  local             w, the root number
  claims            B (the least B_e) and S (rank E(Q) <= S)
  created           the date
  format            "rankbound-2descent-certificate-1"

What is checked, in PARI/GP (two_descent.gp's tdverify and tdvrank): the bad primes are prime (APRCL), the minimal
model, the discriminant is the product of powers of the bad primes, the torsion structure (elltors), the roots of T,
and for each root a, b, every local image W_v (local solubility of the quartics), both Selmer groups (equal to the
spans of the stored bases), s, s' and the local duality of the W_v; the conic points, the tangent lines, every
local point and the class of the tangent line there, the Hilbert symbols, the pairing matrices (alternating) and
their ranks; for the Cassels-Tate data the invariants, local solubility, m^2 = z z z (and m invertible), gamma1, the
primality of the extra primes (strictly increasing) and that they, S and 3, 5, 7 cover g_j(1, 0) and content(gamma1),
the Hilbert symbols, the matrix and its rank; B_e; then w, B, S (0 <= S <= B), the statement and the lower bound.
Independence: this module uses nothing of the prover's Python, but the GP checks of two_descent.gp share their local
solubility, Selmer group and pairing code with the prover (tdprove runs from the same file), so a bug there would be
repeated, not caught; the database was also checked against PARI's ellrank, an independent implementation."""
from . import NAME, RAT, LimitExceeded, _int, _invariants, _keys, _list, _str, parity_bound

FORMAT = 'rankbound-2descent-certificate-1'
KEYS = ('name', 'curve', 'statement', 'torsion', 'GRH', 'rank_lower_bound', 'descent', 'cassels_tate', 'local',
        'claims', 'created', 'format')          # rank_lower_bound may be absent
PARTS = {'curve': ('ainvs', 'minimal_model', 'bad_primes'), 'local': ('w',), 'claims': ('B', 'S')}
ISOGENY = ('e', 'a', 'b', 'selmer', 'selmer_isogenous', 's', 's_isogenous', 'pairing', 'cassels_tate', 'B')
CT = ('quartics', 'local', 'pairs', 'rank')
CT_PAIR = ('i', 'j', 'sum', 'm', 'sum_local', 'primes', 'points')
PAIRING = ('places', 'conic', 'local', 'rank', 'conic_isogenous', 'local_isogenous', 'rank_isogenous')
CHECKS = ('bad_primes_prime', 'minimal_model', 'disc_factored', 'torsion', 'two_torsion')   # then per isogeny
HYPOTHESIS = ('rank E(Q) <= rank_upper_bound unconditionally (2-isogeny descent and the 2-parity theorem; no GRH, '
              'no BSD); rank E(Q) >= rank_lower_bound unconditionally when r_proved')


def statement(B, w, r, source=None):
    """Return (S, the statement) for the bound B on dim Sel_2(E) - t, the root number w and the rank lower bound r,
    or None.  source says where r is from when the certificate does not prove it."""
    S = parity_bound(B, 0, 0, w)
    s = 'Unconditionally, rank E(Q) <= %d.' % S
    if r is not None and S == r and r > 0:
        s += ' As E(Q) has %d independent rational point%s%s, we therefore have rank E(Q) = %d.' % (
            r, 's' if r > 1 else '', ' (%s; not proved in this certificate)' % source if source else '', r)
    elif r is not None and S < r:       # fails the check bound_ge_r
        s += ' CONTRADICTION: rank E(Q) >= %d.' % r
    return S, s


def lower_bound_form(rl):
    """Check the form of a rank_lower_bound entry (as the cubic path's); raise ValueError if it is malformed."""
    if isinstance(rl, dict) and 'points' in rl:
        _keys(rl, ('r', 'points', 'primes'), 'rank_lower_bound')
        for pt in _list(rl['points']):
            for x in _list(pt, 2):
                _str(x, RAT)
        for p in _list(rl['primes']):
            _int(p)
    else:
        _keys(rl, ('r', 'source'), 'rank_lower_bound')
        src = rl['source']
        if not (isinstance(src, str) and src.strip() and len(src) <= 200 and src.isprintable()):
            raise ValueError('the source of r must be one line of at most 200 characters')
    if _int(rl['r']) < 0:
        raise ValueError('r is negative')


def clean(c):
    """Check the form of the certificate c; raise ValueError if it is malformed."""
    _keys(c, [k for k in KEYS if k != 'rank_lower_bound' or k in c], 'certificate')
    for k, ks in PARTS.items():
        _keys(c[k], ks, k)
    for k in ('name', 'statement', 'created', 'GRH'):
        if not isinstance(c[k], str):
            raise ValueError('"%s" must be a string' % k)
    if not NAME.fullmatch(c['name']):
        raise ValueError('the name %.80r is not allowed' % c['name'])
    cur = c['curve']
    for v, n in ((cur['ainvs'], 5), (cur['minimal_model'], 5), (cur['bad_primes'], None)):
        for x in _list(v, n):
            _str(x)
    for v in (cur['ainvs'], cur['minimal_model']):
        c4, c6 = _invariants([int(x) for x in v])
        if c4 ** 3 == c6 ** 2:
            raise ValueError('the curve %s is singular' % ','.join(v))
    if len(_list(c['torsion'])) > 2:
        raise ValueError('"torsion" has at most two invariants')
    for x in c['torsion']:
        if _int(x) < 1:
            raise ValueError('a torsion invariant must be positive')
    _int(c['local']['w'])
    for k in ('B', 'S'):
        _int(c['claims'][k])
    if len(_list(c['descent'])) not in (1, 3):
        raise ValueError('"descent" has one entry per rational root of T: 1 or 3')
    for d in c['descent']:
        _keys(d, ISOGENY, 'descent')
        for k in ('e', 'a', 'b'):
            _str(d[k])
        for k in ('selmer', 'selmer_isogenous'):
            if len(_list(d[k])) > len(cur['bad_primes']) + 2:
                raise ValueError('a Selmer basis is longer than the dimension of Q(S, 2)')
            for x in d[k]:
                _str(x)
        for k in ('s', 's_isogenous', 'B'):
            _int(d[k])
        pr = _keys(d['pairing'], PAIRING, 'pairing')
        if pr['places'] != places(cur['bad_primes']):
            raise ValueError('the pairing places must be %s' % ', '.join(places(cur['bad_primes'])))
        for g, sel in (('', 'selmer'), ('_isogenous', 'selmer_isogenous')):
            _int(pr['rank' + g])
            for pt in _list(pr['conic' + g], len(d[sel])):
                for x in _list(pt, 3):
                    _str(x)
            for pts in _list(pr['local' + g], len(d[sel])):
                for pt in _list(pts, len(pr['places'])):
                    for x in _list(pt, 2):
                        _str(x)
    ct_form(c['cassels_tate'], len(places(cur['bad_primes'])))
    for d in c['descent']:
        ct_form(d['cassels_tate'], len(places(cur['bad_primes'])))
    if 'rank_lower_bound' in c:
        lower_bound_form(c['rank_lower_bound'])


def ct_form(ct, nv):
    """Check the form of a Cassels-Tate entry (None, or quartics, their local points at the nv places, the pairs and
    the rank); raise ValueError if it is malformed."""
    if ct is None:
        return
    _keys(ct, CT, 'cassels_tate')
    n = len(_list(ct['quartics']))
    _int(ct['rank'])
    pts = lambda P, k: [_str(x, RAT) for pt in _list(P, k) for x in _list(pt, 2)]
    for g, loc in zip(ct['quartics'], _list(ct['local'], n)):
        for x in _list(g, 5):
            _str(x)
        pts(loc, nv)
    if len(_list(ct['pairs'])) != n * (n - 1) // 2:
        raise ValueError('the Cassels-Tate entry needs one pair per i < j')
    for pr in ct['pairs']:
        _keys(pr, CT_PAIR, 'pairs')
        for k in ('i', 'j'):
            _int(pr[k])
        for x in _list(pr['sum'], 5) + _list(pr['m'], 3):
            _str(x, RAT)
        pts(pr['sum_local'], nv)
        ps = [int(_str(p)) for p in _list(pr['primes'])]
        if any(a >= b for a, b in zip(ps, ps[1:])):
            raise ValueError('the extra primes of a Cassels-Tate pair must be strictly increasing')
        pts(pr['points'], nv + len(pr['primes']))


def places(bad_primes):
    """Return the places of the pairings for the bad primes (strings): R, then the primes of 2 * bad primes and 3, 5, 7
    in increasing order."""
    return ['R'] + [str(p) for p in sorted({2, 3, 5, 7} | {int(p) for p in bad_primes})]


def limits(c):
    """Raise LimitExceeded if c asks for unbounded work: a prime of the points of 2^64 or more."""
    if [p for p in (c.get('rank_lower_bound') or {}).get('primes', []) if p >= 2 ** 64]:
        raise LimitExceeded('a prime of the points in {cert} is 2^64 or more')


def gp_vec(v):
    return '[%s]' % ','.join(v)


def gp_body(c):
    """Return the GP calls that check c: tdverify, and tdvrank if the points are given."""
    cur, rl = c['curve'], c.get('rank_lower_bound') or {}
    pts = lambda P: '[%s]' % ','.join('[%s]' % ','.join(gp_vec(z) for z in row) for row in P)
    D = ','.join('[%s,%s,%s,%s,%s,%d,%d,%d,%s,%s,%d,%s,%s,%d,%s]' % (
        d['e'], d['a'], d['b'], gp_vec(d['selmer']), gp_vec(d['selmer_isogenous']), d['s'], d['s_isogenous'], d['B'],
        '[%s]' % ','.join(gp_vec(z) for z in d['pairing']['conic']), pts(d['pairing']['local']), d['pairing']['rank'],
        '[%s]' % ','.join(gp_vec(z) for z in d['pairing']['conic_isogenous']), pts(d['pairing']['local_isogenous']),
        d['pairing']['rank_isogenous'], gp_ct(d['cassels_tate'])) for d in c['descent'])
    body = 'tdverify(%s, %s, %s, [%s], [%s], %s);\n' % (
        gp_vec(cur['ainvs']), gp_vec(cur['bad_primes']), gp_vec(cur['minimal_model']), ','.join(map(str, c['torsion'])),
        D, gp_ct(c['cassels_tate']))
    if 'points' in rl:
        body += 'tdvrank(%s, %d, [%s], [%s]);\n' % (gp_vec(cur['ainvs']), rl['r'],
                                                     ','.join('[%s,%s]' % tuple(p) for p in rl['points']),
                                                     ','.join(map(str, rl['primes'])))
    return body


def gp_ct(ct):
    """Return a Cassels-Tate entry as tdvct's GP vector [quartics, local points, pairs, rank], or 0 for None."""
    if ct is None:
        return '0'
    nest = lambda v: '[%s]' % ','.join(nest(z) for z in v) if isinstance(v, list) else str(v)
    pairs = [[pr['i'], pr['j'], pr['sum'], pr['m'], pr['sum_local'], pr['primes'], pr['points']] for pr in ct['pairs']]
    return '[%s,%s,%s,%d]' % (nest(ct['quartics']), nest(ct['local']), nest(pairs), ct['rank'])


def check_names(c):
    """Return the names of the checks tdverify (and tdvrank) must print, and pass."""
    return CHECKS + (('cassels_tate',) if c['cassels_tate'] else ()) + tuple(
        '%s_%d' % (k, i) for i, d in enumerate(c['descent'], 1)
        for k in ('descent', 'local_duality', 'pairing') + (('cassels_tate',) if d['cassels_tate'] else ()) + ('bound',)) + \
        (('rank_lower_bound',) if 'points' in (c.get('rank_lower_bound') or {}) else ())


def claims(c, out, ok):
    """Record with ok(name, passed, info) the checks of the claims against the gp output out (its "TD B S w" line);
    return the statement rebuilt."""
    rl = c.get('rank_lower_bound') or {}
    r = rl.get('r')
    td = [l.split() for l in out.splitlines() if l.startswith('TD ')]
    if not td:
        ok('claims', False, 'no bound recomputed (a check above failed)')
        return ''
    B, S_gp, w = map(int, td[0][1:])
    ok('root_number', w == c['local']['w'], 'w = %d' % w)
    S, st = statement(B, w, r, None if 'points' in rl else rl.get('source'))
    ok('claims', B == c['claims']['B'] and S == c['claims']['S'] == S_gp and 0 <= S <= B,
       'B = %d, rank <= %d' % (B, S))
    ok('GRH', c['GRH'] == 'no', 'the bound is unconditional')
    ok('statement', c['statement'] == st)
    if r is not None:
        ok('bound_ge_r', S >= r, 'rank <= %d, rank >= %d' % (S, r))
    return st
