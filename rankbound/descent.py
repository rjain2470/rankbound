"""The unconditional bound by 2-isogeny descent for curves with a rational point of order 2, and its witness.

The second half of the file is the arithmetic, ported from the checker's GP functions (named td*).
"""
import copy, dataclasses, datetime, itertools, math, time
from fractions import Fraction
from decimal import Decimal, localcontext
from .core import FORMAT2 as FORMAT, FactorisationNeeded, ProofData, RankboundError, Rejected, TimeLimit, parity_bound, parity_floor
from .witness import isogeny_statement as statement
from . import arith as A, ctp, curve as cv, field
from .arith import basis, isog_places, kernel2, loc, rank2, rep, soluble, solve2, wit
from . import runtime
from .messages import text
from .runtime import DETAIL, Folder, say, say_bound, say_curve, say_lower


LOG = DETAIL.info
PRIME_SEARCH = 10 ** 5      # primes searched for the characters of the points
METHOD = '2-isogeny descent'


CT_MARGIN = 60          # seconds kept after the Cassels-Tate pairings for the points and the output


def descent(ainvs, bad_primes, proof=None, r=None, ctp_seconds=None, budget=None, seconds=7200):
    """Return the 2-isogeny descent of ainvs (any model; bad_primes all the primes of its discriminant).

    A Cassels-Tate pairing that is cut short, too costly or fails is skipped, and the bound without it stands. Raises
    TimeLimit, RankboundError if the points do not prove rank >= r, and RuntimeError if an internal check fails (a bug).
    """
    a = [cv.canon(x, 'a-invariant') for x in cv._ainvs5(ainvs)]
    P = [cv.canon(p, 'bad prime') for p in bad_primes]
    pts = (proof or {}).get('points') or []
    if pts and r is not None and cv.r_value(r) != len(pts):     # never silently replace an explicit r
        raise RankboundError('r = %s but the proof has %d points.' % (r, len(pts)))
    src = cv._ainvs5(proof.get('ainvs') or ainvs) if pts else a
    # The loops poll a budget ending at the earlier of the two limits.
    try:
        with runtime.scope(budget or runtime.current()), runtime.limited(seconds=seconds or None, error=A.time_up) as b:
            out = prove_py(a, P, src, pts, proof, r, ctp_seconds, b)
            runtime.check()
    except Rejected as e:           # from rkmap, on the points
        raise RankboundError(str(e))
    except A.TimeUp as e:
        raise TimeLimit('The 2-isogeny descent did not finish within its time limit (%s).' % e, seconds or 0)
    V = places(P)
    D = {'isogenies': [], 'points': [], 'cassels_tate': None, 'minimal_model': [str(x) for x in out['mm']],
         'torsion': list(out['torsion']), 'w': out['w'], 'conductor': str(out['conductor'])}
    for who, why, detail in out['skipped']:
        LOG(text('ct_skip', curve='E' if who == 'E' else "E' (isogeny %s)" % who, why=text('ct_skip_' + why,
                                                                                          detail=detail)))
    for g in out['isogenies']:
        if not g['dual'] or not g['in_selmer']:
            raise RuntimeError('The 2-isogeny descent: at the root %d the internal checks fail (local duality %d, '
                               'b and b\' in the Selmer groups %d). This is a bug: please report it.' % (
                                   g['e'], g['dual'], g['in_selmer']))
        pr = {'places': V, 'conic': [], 'local': [], 'rank': g['rank'], 'conic_isogenous': [], 'local_isogenous': [],
              'rank_isogenous': g['rank_isogenous']}
        for side in ('', '_isogenous'):        # each Selmer basis element's conic point and local points
            pr['conic' + side] = [[str(x) for x in C] for C in g['conic' + side]]
            pr['local' + side] = [[['%d' % z[0], '%d' % z[1]] for z in Pts] for Pts in g['local' + side]]
        D['isogenies'].append({'e': '%d' % g['e'], 'a': '%d' % g['a'], 'b': '%d' % g['b'],
                               'selmer': [str(x) for x in g['selmer']],
                               'selmer_isogenous': [str(x) for x in g['selmer_isogenous']], 's': g['s'],
                               's_isogenous': g['s_isogenous'], 'pairing': pr,
                               'cassels_tate': None if g['cassels_tate'] is None else ctp.ct_block(*g['cassels_tate']),
                               'B': g['B']})
    if out['cassels_tate'] is not None:          # on Sel_2(E)
        D['cassels_tate'] = ctp.ct_block(*out['cassels_tate'])
    D['B'], D['S'] = out['B'], out['S']
    for x, y in out['points']:
        D['points'].append([cv.canon(str(x), 'mapped point', cv.RAT), cv.canon(str(y), 'mapped point', cv.RAT)])
    if out['primes'] is not None:
        D['primes'] = list(out['primes'])
    if not D['isogenies'] or D['B'] != min(i['B'] for i in D['isogenies']) or D['S'] != parity_bound(D['B'], 0, 0, D['w']):
        raise RuntimeError('The 2-isogeny descent: the bound does not add up. This is a bug: please report it.')
    if proof is not None:
        D['rank_lower_bound'] = {'r': len(pts), 'points': D.pop('points'), 'primes': D.pop('primes', [])}
    return D


def prove_py(a, P, src, pts, proof, r, ctp_seconds=None, budget=None):
    """Return prove()'s record, with the points mapped onto the minimal model."""
    from fractions import Fraction
    a, P = [int(x) for x in a], [int(p) for p in P]
    mm = field.check_bad_primes(a, P)[0]
    Q = cv.rkmap(src, mm, P, [[Fraction(x) for x in p] for p in pts]) if pts else []
    try:
        return prove(mm, P, Q, [int(p) for p in (proof or {}).get('primes') or []], PRIME_SEARCH,
                     r if r is not None else len(pts), twotor_ct, ctp_seconds, budget)
    except ValueError as e:
        raise RankboundError(str(e))


def places(bad_primes):
    """Return the places of the pairings as the witness lists them: R, then 2, 3, 5, 7 and the bad primes."""
    return ['R'] + [str(p) for p in sorted({2, 3, 5, 7} | {int(p) for p in bad_primes})]


def witness(name, bad_primes, D, rl):
    """Return the witness dict of the descent D with the rank lower bound rl; raise RankboundError if S < r."""
    r = rl['r'] if rl else None
    S, st = statement(D['B'], D['w'], r, None if rl is None or 'points' in rl else rl['source'])
    if r is not None and S < r:
            raise RankboundError(text('points_high', S=S, n=r) if 'points' in rl else text('r_high', S=S, r=r))
    mm = D['minimal_model']
    c = {'name': name, 'curve': {'ainvs': mm, 'minimal_model': mm, 'bad_primes': [str(p) for p in bad_primes]},
         'statement': st, 'torsion': D['torsion'], 'GRH': 'no'}
    if rl:
        c['rank_lower_bound'] = rl
    c.update(descent=D['isogenies'], cassels_tate=D['cassels_tate'], local={'w': D['w']}, claims={'B': D['B'], 'S': S},
             created=datetime.date.today().isoformat(), format=FORMAT)
    return c


class Descent(Folder):
    """The work folder of a curve with rational 2-torsion."""
    method = 'descent'
    route = 'descent'

    def estimate(self, curve, cores=1, ecm_hours=1.0, quiet=False, given=None):
        """Return None: the Cassels-Tate pairings take seconds to hours, so there is no honest estimate."""
        if not quiet:
            say('est_descent_unknown', name=curve.name)
        return None

    def precheck(self, curve):
        """Refuse the curve before any work if r is not the number of points."""
        cv.checked(curve)
        n = len((curve.proof or {}).get('points') or [])
        if n and curve.r is not None and curve.r != n:
            raise RankboundError('r = %d but the proof of %s has %d points.' % (curve.r, curve.name, n))

    def prove(self, curve, req, resources, budget, **_):
        """Bound the rank and return the unchecked witness as ProofData; raise Stopped at the budget's end."""
        cores, ecm_hours = resources.cores, resources.ecm_hours
        budget.check()
        if curve.composites:
            if not ecm_hours:
                raise FactorisationNeeded('The discriminant is not fully factored. Give its prime factors with '
                                          '--bad-primes.', [{'n': c, 'digits': len(c)} for c in curve.composites])
            curve = cv.factor_in(self.dir, curve, ecm_hours, cores, 1.0, budget)
        t = time.time()
        proof = dict(curve.proof, ainvs=curve.proof.get('ainvs') or curve.ainvs) if curve.proof else None
        self.note(stage='solving')
        try:
            D = descent(curve.ainvs, curve.bad_primes, proof, curve.r, budget=budget)
        except TimeLimit:
            runtime.raise_if_stopped(budget)
            raise
        say_curve(D['conductor'], len(curve.bad_primes), D['w'], D['torsion'])
        rl = D.get('rank_lower_bound') or ({'r': curve.r, 'source': curve.source} if curve.r is not None else None)
        say_lower(rl and rl['r'], bool(rl and 'points' in rl))
        if curve.minimal_model is not None and [str(x) for x in curve.minimal_model] != D['minimal_model']:
            raise ValueError('The minimal_model given for %s is wrong; it should be [%s]. Correct it or leave it '
                             'out.' % (curve.name, ','.join(D['minimal_model'])))
        cv.save(self.dir, dataclasses.replace(curve, torsion=D['torsion']))
        path = self.path(curve.name + '.witness.json')
        c = witness(curve.name, curve.bad_primes, D, rl)
        S = c['claims']['S']
        S0 = parity_bound(min(i['s'] + i['s_isogenous'] - 2 - i['pairing']['rank'] - i['pairing']['rank_isogenous']
                              for i in D['isogenies']), 0, 0, D['w'])
        say_bound(max(S0, S), 'descent', False)
        if S0 > S:
            say_bound(S, 'ct', False)
        tc = time.time() - t
        r = rl['r'] if rl else None
        res = {'name': curve.name, 'outcome': 'descent', 'GRH': False, 'B': D['B'], 'rank_bound': S,
               'witness': path, 'rank_lower_bound': r, 'r_proved': r is not None and (r == 0 or 'points' in rl),
               'statement': c['statement'], 'torsion': D['torsion'], 'w': D['w'], 'minimal_model': D['minimal_model'],
               'conductor': D['conductor'],
               'isogenies': [{'e': i['e'], 's': i['s'], 's_isogenous': i['s_isogenous'],
                              'rank_theta': i['pairing']['rank'], 'rank_theta_isogenous': i['pairing']['rank_isogenous'],
                              'rank_ct_isogenous': (i['cassels_tate'] or {}).get('rank'), 'B': i['B']}
                             for i in D['isogenies']],
               'rank_ct': (D['cassels_tate'] or {}).get('rank'), 'core_hours': round(tc * cores / 3600, 4),
               'cost': {'descent': round(tc, 1)}}
        return ProofData(witness=path, data=c, rank_bound=S, hypothesis='none', statement=c['statement'],
                         method=self.method, rank_lower_bound=r, r_proved=res['r_proved'], record=res, cost=res['cost'],
                         outcome='descent')


# ========== the 2-isogeny descent without PARI, ported from the checker's GP functions (td*)
# Polynomials are lists, low degree first; a binary quartic g is [a, b, c, d, e] (a x^4 + b x^3 z + ... + e z^4),
# as in GP.


def qp(Ac, B, d, p):
    """Return whether w^2 = d u^4 + A u^2 v^2 + (B/d) v^4 has a point over Q_p (p = 0: R) (tdqp)."""
    if p == 0:
        return d > 0 or B < 0 or (Ac > 0 and Ac * Ac >= 4 * B)
    return soluble([d ** 3, 0, Ac * d * d, 0, B * d], p)


# ---------- the local images and the Selmer groups

# twotor and lift ask again for the descent's images, pairings and conic points, so one request keeps them.

_REUSE = None                   # {(kind, inputs): value} inside reuse(), else None
REUSE_COUNTS = {}               # {kind: [requests, computations]} of the last request, for tests


class reuse:
    """Context in which images, pairing and conic remember their results; nested blocks share the memory."""

    def __enter__(self):
        global _REUSE
        self.outer = _REUSE is not None
        if not self.outer:
            _REUSE = {}
            REUSE_COUNTS.clear()
        return self

    def __exit__(self, *a):
        global _REUSE
        if not self.outer:
            _REUSE = None


def _reused(kind, key, f):
    """Return a copy of f(), computed once per (kind, key) within reuse()."""
    if _REUSE is None:
        return f()
    n = REUSE_COUNTS.setdefault(kind, [0, 0])
    n[0] += 1
    k = (kind,) + key
    if k not in _REUSE:
        n[1] += 1
        _REUSE[k] = f()
    return copy.deepcopy(_REUSE[k])


def images(Ac, B, S):
    """Return [(p, the local image at p as bit lists)] for p = 0 (R) and S (tdimages)."""
    return _reused('images', (Ac, B, tuple(S)), lambda: _images(Ac, B, S))


def conic(xi, Ac, B, S):
    """Return conic_point(xi, Ac, B, S), computed once within reuse()."""
    return _reused('conic', (xi, Ac, B, tuple(S)), lambda: conic_point(xi, Ac, B, S))


def _images(Ac, B, S):
    out = []
    for p in [0] + list(S):
        A.tick()
        n = len(basis(p))
        out.append((p, [c for c in ([(k >> i) & 1 for i in range(n)][::-1] for k in range(2 ** n))
                         if qp(Ac, B, rep(c, p), p)]))
    return out


def selmer(S, Ws):
    """Return a basis of the Selmer group from the local images (tdselmer)."""
    G = [-1] + list(S)
    cols = []
    for p, W in Ws:
        n = len(basis(p))
        L = [loc(g, p) for g in G]
        for f in kernel2(W, n):
            cols.append([sum(L[i][j] * f[j] for j in range(n)) % 2 for i in range(len(G))])
    K = kernel2(cols, len(G)) if cols else [[int(i == j) for j in range(len(G))] for i in range(len(G))]
    return [math.prod(g ** k for g, k in zip(G, v)) for v in K]


def tdclass(d, S):
    c = [int(d < 0)]
    m = abs(d)
    for p in S:
        v = A.val(m, p)
        c.append(v % 2)
        m //= p ** v
    return c if m == 1 else None


def inspan(d, V, S):
    c = tdclass(d, S)
    if c is None:
        return False
    M = [tdclass(v, S) for v in V]
    return rank2(M + [c]) == rank2(M)


def dual(Ws, Ws2):
    for (p, W), (_, W2) in zip(Ws, Ws2):
        if len(W) * len(W2) != 2 ** len(basis(p)):
            return False
        if any(A.hilbert(rep(u, p), rep(v, p), p) != 1 for u in W for v in W2):
            return False
    return True


def ab(T, e):
    """Return (a, b) with T(X + e) = X (X^2 + a X + b)."""
    V = A._shift(list(T), e, 1)
    if V[0]:
        raise RuntimeError('tdab: %d is not a root of T' % e)
    return V[2], V[1]


def isogeny(T, e, S):
    a, b = ab(T, e)
    a2, b2 = -2 * a, a * a - 4 * b
    W, W2 = images(a, b, S), images(a2, b2, S)
    s1, s2 = selmer(S, W), selmer(S, W2)
    return [e, a, b, s1, s2, len(s1), len(s2), len(s1) + len(s2) - 2, dual(W, W2),
            inspan(b, s1, S) and inspan(b2, s2, S)]


# ---------- the isogeny pairings

def q4(xi, Ac, B, s, t):
    return xi * s ** 4 + Ac * s * s * t * t + Fraction(B, xi) * t ** 4


def tangent(xi, Ac, B, P0):
    l = [2 * xi * P0[0] + Ac * P0[1], Ac * P0[0] + Fraction(2 * B, xi) * P0[1], -2 * P0[2]]
    return A.primitive(l)


def sqrt_padic(w, p, N):
    """Return the square root mod p^N of the unit square w that PARI's sqrt gives, to match the checker."""
    m = p ** (N + 2)
    w = w.numerator * pow(w.denominator, -1, m) % m
    if p == 2:
        r = 1
        for k in range(3, N + 2):
            if (r * r - w) % 2 ** (k + 1):
                r += 2 ** (k - 1)
        return r % 2 ** N if r % 8 in (1, 3) else -r % 2 ** N
    r, k = A.sqrt_mod(w, p), 1
    r = min(r, p - r)
    while k < N:
        k = min(2 * k, N)
        mk = p ** k
        r = (r + w * pow(r, -1, mk)) * pow(2, -1, mk) % mk
    return r


def lclass(xi, Ac, B, L, s, t, p):
    """Return the class of L(s^2, t^2, r) at (s, t, r) on C_xi over Q_p as a rational, or 0 (tdlclass)."""
    q = q4(xi, Ac, B, s, t)
    if q == 0:
        return 0
    u = L[0] * s * s + L[1] * t * t
    if p == 0:
        if q < 0:
            return 0
        if u == 0 or L[2] == 0 or (u > 0) == (L[2] > 0):
            return (u > 0) - (u < 0) if u else (L[2] > 0) - (L[2] < 0)
        k = u * u - L[2] * L[2] * q
        return ((u > 0) - (u < 0)) if k > 0 else ((L[2] > 0) - (L[2] < 0)) if k < 0 else 0
    v = A.val(q, p)
    if v % 2:
        return 0
    w = q / Fraction(p) ** v
    if not A.is_square_qp(w, p):
        return 0
    if L[2] == 0:
        return u
    # A class certified at short precision is the true one; cancellation returns 0 and is redone at full precision.
    return _lclass_l(u, L[2], w, v, p, 4 if p > 2 else 6) or _lclass_l(u, L[2], w, v, p, 100 if p > 2 else 99)


def _lclass_l(u, L3, w, v, p, rel):
    """Return the class of l = u + L3 p^(v/2) sqrt(w) from rel digits, or 0 if too few remain."""
    N = A.val(L3, p) + v // 2 + rel                       # absolute precision of l
    m = p ** N
    r = sqrt_padic(w, p, rel) * p ** (v // 2)
    l = (u + L3 * r) % m
    if l == 0:
        return 0
    vl = A.val(l, p)
    k = N - vl
    if k < (3 if p == 2 else 1):
        return 0
    return p ** vl * (l // p ** vl % p ** k)


def theta(V, Xi, Ls, Pts, Ac, B):
    """Return the matrix of Theta_1, or None if a local point is not one (tdtheta)."""
    n = len(Xi)
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        for k, p in enumerate(V):
            c = lclass(Xi[i], Ac, B, Ls[i], Pts[i][k][0], Pts[i][k][1], p)
            if not c:
                return None
            for j in range(n):
                M[i][j] ^= A.hilbert(c, Xi[j], p) == -1
    return M


def real_point(xi, Ac, B, L):
    for n in range(1, 41):
        for s in range(n + 1):
            t = n - s
            if math.gcd(s, t) == 1 and lclass(xi, Ac, B, L, s, t, 0):
                return [s, t]
    X1 = Fraction(-Ac, 2 * xi)
    if X1 > 0:
        with localcontext() as ctx:
            ctx.prec = 60 + 2 * len(str(X1.numerator)) + 2 * len(str(X1.denominator))
            r = (Decimal(X1.numerator) / X1.denominator).sqrt()
            for z in A.convergents(r, 2 ** 400):
                if lclass(xi, Ac, B, L, z.numerator, z.denominator, 0):
                    return [z.numerator, z.denominator]
    raise RuntimeError('tdreal: no real point found for %d' % xi)


def local_point(xi, Ac, B, L, p):
    """Return a local point of C_xi at p (0: R) with L != 0 (tdlocal)."""
    if p == 0:
        return real_point(xi, Ac, B, L)
    g = [xi ** 3, 0, Ac * xi * xi, 0, B * xi]
    w = wit(g[::-1], p)
    if w is not None:
        s, t = w, 1
    else:
        w = wit([c * p ** i for i, c in enumerate(g)], p)
        if w is None:
            raise RuntimeError('tdlocal: %d is not in the Selmer group at %d' % (xi, p))
        s, t = 1, p * w
    K = A.val(xi * xi * q4(xi, Ac, B, s, t), p) + 4
    for j in range(201):
        A.tick()
        s1, t1 = (s + j * p ** K, t) if t == 1 else (s, t + j * p ** K)
        if lclass(xi, Ac, B, L, s1, t1, p):
            return [s1, t1]
    raise RuntimeError('tdlocal: no point with L != 0 for %d at %d' % (xi, p))


def conic_point(xi, Ac, B, S):
    """Return a primitive rational point of W^2 = xi X^2 + A X Y + (B/xi) Y^2 (tdconic)."""
    d = Ac * Ac - 4 * B
    sol = legendre(1, -d, -4 * xi, S)
    if sol is None:
        raise RuntimeError('tdconic: the conic of %d has no rational point' % xi)
    u, Y, W = sol
    return A.primitive([Fraction(u - Ac * Y, 2 * xi), Y, W])


def legendre(a, b, c, primes):
    """Return a nonzero integer solution of a x^2 + b y^2 + c z^2 = 0, or None if there is none.

    All primes of a, b, c must be in primes. [Cremona and Rusin, Math. Comp. 72 (2003)]
    """
    co, mul = [a, b, c], [1, 1, 1]
    ps = sorted(set(primes) | {2})
    # Reduce to squarefree, pairwise coprime coefficients, tracking the coordinates' scale in mul.
    changed = True
    while changed:
        changed = False
        for p in ps:
            for i in range(3):
                if co[i] % (p * p) == 0:
                    co[i] //= p * p
                    for j in range(3):
                        if j != i:
                            mul[j] *= p
                    changed = True
            for i in range(3):
                j, k = (i + 1) % 3, (i + 2) % 3
                if co[i] % p == 0 and co[j] % p == 0:
                    co[i], co[j], co[k] = co[i] // p, co[j] // p, co[k] * p
                    mul[k] *= p
                    changed = True
    a, b, c = co
    if (a > 0) == (b > 0) == (c > 0):
        return None
    for n in co:
        m = abs(n)
        for p in ps:
            m //= p ** A.val(m, p)
        if m != 1:
            raise RuntimeError('legendre: a coefficient has a prime outside the list')
    # the square roots: l^2 = -c/b mod a, m^2 = -a/c mod b, n^2 = -b/a mod c
    def root(num, den, mod):
        r, M = 0, 1
        for p in ps:
            if mod % p:
                continue
            x = num * pow(den, -1, p) % p
            if p > 2 and A.kronecker(x, p) == -1:
                return None
            r, M = A.crt(r, M, A.sqrt_mod(x, p), p), M * p
        return r
    l, m, n = root(-c, b, abs(a)), root(-a, c, abs(b)), root(-b, a, abs(c))
    if None in (l, m, n):
        return None
    A_, B_, C_ = abs(a), abs(b), abs(c)
    z0 = A.crt(m % B_, B_, 0, A_ * C_) if B_ > 1 else 0
    y1 = A.crt(pow(n, -1, C_) if C_ > 1 else 0, C_, l * z0 % A_, A_) if A_ > 1 or C_ > 1 else 0
    y2 = A.crt(0, C_, l * B_ % A_, A_)
    B = [[1, y1, z0], [0, y2, B_], [0, A_ * C_, 0]]     # a basis of the lattice of the congruences (index |abc|)
    G = [[sum(abs(w) * B[i][k] * B[j][k] for k, w in enumerate((a, b, c))) for j in range(3)] for i in range(3)]
    U = A.lll_gram(G)
    R = [[sum(U[k][cc] * B[k][i] for k in range(3)) for i in range(3)] for cc in range(3)]
    for s in range(1, 4):
        for c1 in range(-s, s + 1):
            for c2 in range(-s, s + 1):
                for c3 in range(-s, s + 1):
                    if max(abs(c1), abs(c2), abs(c3)) < s:
                        continue
                    v = [c1 * R[0][i] + c2 * R[1][i] + c3 * R[2][i] for i in range(3)]
                    if a * v[0] ** 2 + b * v[1] ** 2 + c * v[2] ** 2 == 0:
                        return [v[i] * mul[i] for i in range(3)]
    raise RuntimeError('legendre: no solution among the small vectors')


def pairing(Ac, B, S, Xi):
    """Return [conic points, local points, matrix, rank] of Theta_1 on Sel(A, B) with basis Xi (tdpairing)."""
    return _reused('pairing', (Ac, B, tuple(S), tuple(Xi)), lambda: _pairing(Ac, B, S, Xi))


def _pairing(Ac, B, S, Xi):
    V = isog_places(S)
    A.tick()
    C = [conic(xi, Ac, B, S) for xi in Xi]
    Ls = [tangent(xi, Ac, B, P0) for xi, P0 in zip(Xi, C)]
    Pts = [[local_point(xi, Ac, B, L, p) for p in V] for xi, L in zip(Xi, Ls)]
    if not Xi:
        return C, Pts, [], 0
    M = theta(V, Xi, Ls, Pts, Ac, B)
    return C, Pts, M, rank2(M)


def chi(T, P, p, e):
    """Return the character at (p, e) of the point P (None for O), with X = 4 x (tdchi)."""
    if P is None:
        return 0
    X = 4 * Fraction(P[0])
    if X.denominator % p == 0:
        return 0
    k = A.kronecker((X.numerator * pow(X.denominator, -1, p) - e) % p, p)
    if k == 0:
        k = A.kronecker(A.peval(A.pderiv(T), e, p), p)
    return int(k == -1)


def char_primes(mm, T, R, PR, pmax):
    """Return the first primes, from PR then below pmax, where the characters of R have rank #R, or None."""
    return _char_scan(mm, T, R, PR, pmax, None)[0]


def _char_scan(mm, T, R, PR, pmax, slack):
    """Return (char_primes's primes or None, a kernel combination of R as a bitmask once the rank stays short of #R
    for slack usable primes in a row, else None)."""
    if len(R) == 0:
        return [], None
    D = cv.cinv(mm)[2]
    basis, keep, seen, idle = [], [], set(), 0
    for p in itertools.chain(PR, cv._fallback_primes(pmax + 1, set(PR))):
        A.tick()
        if p in seen:
            continue
        seen.add(p)
        if p < 3 or not A.is_prime(p) or D % p == 0:
            continue
        grew = False
        for e in A.roots_mod(T, p):
            grew |= A.insert(basis, sum(chi(T, P, p, e) << i for i, P in enumerate(R)))
        if grew:
            keep.append(p)
            idle = 0
            if len(basis) == len(R):
                return keep, None
        else:
            idle += 1
            if slack is not None and idle >= slack:
                return None, cv._kernel(basis, len(R))
    return None, None


def saturate(mm, T, gens, pts, PR, pmax):
    """Return (the points, 2-saturated where needed, the primes of char_primes for gens + points, or None).

    A kernel combination with a point in it lies in 2E(Q) (up to slack's chance): it is halved over Q and replaces
    one of its points, so the new points with the torsion span a lattice containing the old."""
    pr = char_primes(mm, T, gens + pts, PR, pmax)
    if pr is not None:
        return pts, pr
    F = lambda z: (Fraction(z[0]), Fraction(z[1]))
    gens, pts = [F(z) for z in gens], [F(z) for z in pts]
    for _ in range(4 * len(pts) + 4):
        R = gens + pts
        pr, c = _char_scan(mm, T, R, PR, pmax, cv.SAT_PRIMES)
        if c is None:
            return pts, pr
        mine = [i - len(gens) for i in range(len(gens), len(R)) if c >> i & 1]
        H = cv._halve(mm, cv._combination(mm, R, c)) if mine else None
        if H is None:
            break
        pts[mine[0]] = H
    return pts, char_primes(mm, T, gens + pts, PR, pmax)


def bound(Bs, w):
    B = min(Bs)
    return B, parity_floor(B, w)


def prove(mm, P, PTS, PR, pmax, r0, ct=None, ctsec=None, budget=None):
    """Return the descent's record (tdprove) for the minimal model mm, bad primes P and points PTS on mm.

    skipped lists the Cassels-Tate pairings left out, as (E or the isogeny's 1-based index, why, detail).
    """
    with reuse():
        return _prove(mm, P, PTS, PR, pmax, r0, ct, ctsec, budget or runtime.current())


CT_SECONDS = 60         # the least allowance of the Cassels-Tate pairings


def ct_allowance(descent_s, left, ctsec=None):
    """Return the seconds all Cassels-Tate pairings of a descent of descent_s seconds may take.

    By default they cost little more than the descent. It caps cost rather than being a time to fill: ctp.ctq stops a
    pairing once its projected cost exceeds what is left.
    """
    allow = ctsec if ctsec else max(CT_SECONDS, descent_s)
    return allow if left is None else min(allow, left - CT_MARGIN)


def _prove(mm, P, PTS, PR, pmax, r0, ct, ctsec, budget):
    t0 = time.monotonic()
    S = sorted({2} | set(P))
    b2, b4, b6, _ = cv.binv(mm)
    T = [16 * b6, 8 * b4, b2, 1]
    roots = sorted(cv.int_roots(T))
    if not roots:
        raise RuntimeError('tdprove: E(Q)[2] = 0')
    w = cv.rootno(mm, P)
    D = [isogeny(T, e, S) for e in roots]
    Th = [(pairing(d[1], d[2], S, d[3]), pairing(-2 * d[1], d[1] ** 2 - 4 * d[2], S, d[4])) for d in D]
    Bs = [d[7] - t1[3] - t2[3] for d, (t1, t2) in zip(D, Th)]
    skipped, CE, CI = [], None, [None] * len(D)
    # the pairings' own budget within the descent's; its end raises arith.TimeUp
    ctb = budget.child(seconds=max(0.0, ct_allowance(time.monotonic() - t0, budget.left(), ctsec)), stop=A.time_up,
                       masks=True)

    def timed(who, E):
        """Return ct's data for E, or None after recording why it was skipped."""
        try:
            ctb.check()
            with runtime.scope(ctb):
                return ct(E, S, ctb.left())
        except A.TimeUp as e:
            skipped.append((who, 'time', str(e).strip() or 'no time left'))
        except ctp.Costly as e:
            skipped.append((who, 'cost', '%.0f s' % e.seconds))
        except (RuntimeError, ValueError, ZeroDivisionError) as e:     # the search failed (CTSKIP in tower.gp)
            skipped.append((who, 'fail', str(e).strip()))
        return None
    # Most lifted quartics first, where a pairing can gain most; each runs only while the bound is above r0.
    todo = sorted([(-(D[0][5] - Th[0][0][3]), -1)] + [(-(d[6] - Th[i][1][3]), i) for i, d in enumerate(D)])
    for _, i in todo:
        if not ct or bound(Bs, w)[1] <= r0:
            break
        if i < 0:
            CE = timed('E', mm)
            if CE is not None:
                Bs = [min(Bs[t], D[t][7] - Th[t][0][3] - CE[4]) for t in range(len(D))]
            continue
        d = D[i]
        E2 = cv.minimal([0, -2 * d[1], 0, d[1] ** 2 - 4 * d[2], 0], S)[0]
        CI[i] = timed('%d' % (i + 1), E2)
        if CI[i] is not None:
            Bs[i] = min(Bs[i], d[7] - Th[i][1][3] - CI[i][4])
    B, Sb = bound(Bs, w)
    tor, gens = cv.torsion_group(mm)
    out = {'mm': list(mm), 'torsion': [int(x) for x in tor], 'w': w, 'conductor': cv.conductor(mm, P), 'isogenies': [],
           'cassels_tate': None if not CE else (CE[4],) + tuple(CE[:3]), 'skipped': skipped, 'B': B, 'S': Sb,
           'points': [], 'primes': None}
    for i, d in enumerate(D):
        (C1, P1, _, r1), (C2, P2, _, r2) = Th[i]
        out['isogenies'].append({'e': d[0], 'a': d[1], 'b': d[2], 'selmer': d[3], 'selmer_isogenous': d[4], 's': d[5],
                                 's_isogenous': d[6], 'dual': d[8], 'in_selmer': d[9], 'conic': C1, 'local': P1,
                                 'rank': r1, 'conic_isogenous': C2, 'local_isogenous': P2, 'rank_isogenous': r2,
                                 'cassels_tate': (CI[i][4],) + tuple(CI[i][:3]) if CI[i] else None, 'B': Bs[i]})
    if PTS:
        PTS, pr = saturate(mm, T, gens, list(PTS), PR, pmax)
        if pr is None:
            raise ValueError('The points do not prove rank >= %d: with the torsion, their characters are dependent at '
                             'the primes up to %d.' % (len(PTS), pmax))
        out['points'] = [(P[0], P[1]) for P in PTS]
        out['primes'] = [int(p) for p in pr]
    return out


# ---------- the 2-isogeny route: quartics spanning Sel_2(E) (in place of PARI's ell2cover)

def _class_x(R, wsq, W0, p):
    """Return loc_p(2 w / W0 + R) for a square root w of wsq, either sign, or None if both vanish."""
    for sg in (1, -1):
        if p == 0:
            with localcontext() as ctx:
                ctx.prec = 80 + len(str(wsq)) + len(str(R))
                w = (Decimal(wsq.numerator) / wsq.denominator).sqrt()
                x = 2 * sg * w + Decimal(R.numerator) / R.denominator * W0
            if abs(x) > Decimal(10) ** -30:
                return loc(Fraction(int(x > 0) * 2 - 1) * W0, 0)
            continue
        c = _class_xp(Fraction(R), Fraction(wsq), W0, p, sg)
        if c is not None:
            return c
    return None


def _class_xp(R, wsq, W0, p, sg, N=150):
    """Return _class_x at the prime p for the sign sg, exactly.

    The class needs 4 digits past the valuation; on cancellation the precision doubles, up to 2400.
    """
    RW = R * W0
    if wsq == 0:
        return None if RW == 0 else _plus(loc(RW, p), loc(Fraction(W0), p))
    v = A.val(wsq, p)
    if v % 2:
        raise RuntimeError('lift: the local value %s is not a square at %d' % (wsq, p))
    u = wsq / Fraction(p) ** v
    e = max(0, -(v // 2), -A.val(RW, p) if RW else 0)
    while N <= 2400:
        M = N - 1 + v // 2 + 2 * e
        m = p ** M
        r = sqrt_padic(u, p, N) * p ** (v // 2 + 2 * e)        # an integer: v // 2 + 2 e >= e >= 0
        Y = RW * Fraction(p) ** (2 * e)                                       # p-integral
        x = (Y.numerator * pow(Y.denominator, -1, m) + 2 * sg * r) % m
        if x and M - A.val(x, p) >= 4:
            return _plus(loc(x, p), loc(Fraction(W0), p))
        N *= 2
    return None


def _plus(a, b):
    return [(x + y) % 2 for x, y in zip(a, b)]


def _conic_point(d, a, b, S):
    """Return a point of W^2 = d X^2 + a X Y + (b/d) Y^2 with W0 != 0 and easy to factor."""
    P0 = conic(d, a, b, S)          # reused from the descent
    G = [[d, Fraction(a, 2), 0], [Fraction(a, 2), Fraction(b, d), 0], [0, 0, -1]]
    Q = A.param(G, P0)
    for n in range(1, 60):
        for u in range(-n, n + 1):
            v = n - abs(u)
            if math.gcd(u, v) != 1:
                continue
            X = A.primitive([q[0] * u * u + q[1] * u * v + q[2] * v * v for q in Q])
            r = abs(X[2])
            for p in S:
                r //= p ** A.val(r, p) if r else 1
            if X[2] and (r < 10 ** 18 or A.is_prime(r)):
                return X
    raise RuntimeError('lift: no conic point with W0 != 0 and W0 factored')


def lift(d, a, b, S):
    """Return the element of L of a class of Sel_2(E) above d in ker Theta_1, and |W0 c|."""
    X0, Y0, W0 = _conic_point(d, a, b, S)
    a2, b2 = -2 * a, a * a - 4 * b
    c_extra = 1
    for q, k in (A.factor(abs(W0)) if abs(W0) > 1 else []):
        if q in S:
            continue
        rt = A.roots_mod([b, a, 1], q)
        if len(rt) == 2:                    # split: the valuation of d X0 - theta_1 Y0 at the root theta_1
            K = 2 * k + 4
            th1 = A.padic_roots([b, a, 1], q, K)[0]
            e = A.val((d * X0 - th1 * Y0) % q ** K or q ** K, q) % 2
        else:
            e = k % 2
        c_extra *= q ** e
    G = [-1] + list(S)
    rows, rhs = [[] for _ in G], []
    Wp = dict(images(a2, b2, S))
    for p in [0] + list(S):
        s, t = local_point(d, a, b, tangent(d, a, b, [X0, Y0, W0]), p)
        zn = d * W0 * W0                    # N(z); rho = (d s^2 - theta t^2) conj(z) / N(z)
        u0, u1 = d * s * s, -t * t
        z0, z1 = d * X0, -Y0
        c0, c1 = z0 - a * z1, -z1           # conj(z0 + z1 theta) = z0 + z1 (-a - theta)
        p0 = u0 * c0 - b * u1 * c1          # (u0 + u1 th)(c0 + c1 th), th^2 = -a th - b
        p1 = u0 * c1 + u1 * c0 - a * u1 * c1
        tr = Fraction(2 * p0 - a * p1, zn)
        wsq = q4(d, a, b, s, t)
        cv = _class_x(tr, wsq, W0, p)
        if cv is None:
            raise RuntimeError('lift: c_v is 0 at %d' % p)
        t_v = [(x + y) % 2 for x, y in zip(cv, loc(c_extra, p))]
        for f in kernel2(Wp[p], len(basis(p))):
            for i, g in enumerate(G):
                rows[i].append(sum(x * y for x, y in zip(loc(g, p), f)) % 2)
            rhs.append(sum(x * y for x, y in zip(t_v, f)) % 2)
    x = solve2(rows, rhs) if rhs else [0] * len(G)
    if x is None:
        raise RuntimeError('lift: no c for %d' % d)
    c = c_extra * math.prod(g ** k for g, k in zip(G, x))
    u0, u1 = d * X0 * c - d, -Y0 * c       # (z c - d) / theta = (u0 + u1 th)(-(a + th)/b)
    return [Fraction(d), -Fraction(u0 * a - u1 * b, b), -Fraction(u0, b)], abs(W0 * c)


def twotor(mm, S):
    """Return integral quartics spanning Sel_2(E) of mm, where E(Q)[2] != 0, from the first root's 2-isogeny."""
    c4, c6, _ = cv.cinv(mm)
    I, J = c4, 2 * c6
    F = ctp.cubic(I, J)
    b2_, b4, b6, _ = cv.binv(mm)
    T = [16 * b6, 8 * b4, b2_, 1]
    e = sorted(cv.int_roots(T))[0]
    a, b = ab(T, e)
    a2, b2 = -2 * a, a * a - 4 * b
    Q = []
    for c in selmer(S, images(a2, b2, S)):
        g = ctp.minimise([c, 0, a2, 0, Fraction(b2, c)], I, J)
        if g is None:
            raise RuntimeError('twotor: the quartic of %d does not minimise' % c)
        Q.append(g)
    Sel = selmer(S, images(a, b, S))
    if Sel:
        M = pairing(a, b, S, Sel)[2]
        for v in kernel2(M, len(Sel)):
            d = math.prod(x ** k for x, k in zip(Sel, v))
            cl = tdclass(d, S)
            d = math.prod(g ** k for g, k in zip([-1] + list(S), cl))
            el, cf = lift(d, a, b, S)
            R = (Fraction(-b2_, 3) - e, Fraction(-1, 3), Fraction(0))           # X' = X - e with X = -(ph + b2)/3
            R2 = ctp.mul(F, R, R)
            xi = tuple(el[0] * u + el[1] * v_ + el[2] * w for u, v_, w in zip((1, 0, 0), R, R2))
            g, m = ctp.classg(xi, I, J, F, S, (sorted(set(S) | {3}), [cf]))     # N(xi) = (d W0 c)^2
            if m is None:
                raise RuntimeError('twotor: no quartic for the lift of %d' % d)
            Q.append(g)
    return [ctp.nice(ctp.reduce(g), F, S) for g in Q]


def twotor_ct(mm, S, left=None):
    """Return the Cassels-Tate data of mm by the 2-isogeny route.

    Raises Costly once the projected cost exceeds left seconds, and RuntimeError, ValueError or ZeroDivisionError when
    the search fails.
    """
    t = time.monotonic()
    Q = twotor(mm, S)
    return ctp.ctq(mm, S, Q, True, None if left is None else left - (time.monotonic() - t))
