"""The Cassels-Tate pairing on Sel_2(E) by binary quartics (Fisher 2022), added to a witness as cassels_tate.

The second half of the file computes the pairing itself, also for the 2-isogeny descent.
"""
import math, os, time
from fractions import Fraction
from decimal import Decimal, localcontext
from . import arith as A, curve as cv, field as fld
from .core import TimeLimit
from . import runtime
from .runtime import DETAIL, say, say_bound
from .witness import statement

LOG = DETAIL.info
HERE = os.path.dirname(os.path.abspath(__file__))
SECONDS = 60                    # the least time limit of the Cassels-Tate step (see after_selmer)


def strs(v):
    """Return the nested lists v with every number as a string, as the witness writes it."""
    if isinstance(v, (list, tuple)):
        return [strs(x) for x in v]
    return str(v)


def ct_block(rank, Q, loc, pairs):
    """Return the witness's Cassels-Tate entry from tdct's rank and its quartics, local points and pairs."""
    Q, loc, pairs = strs(Q), strs(loc), strs(pairs)
    return {'quartics': Q, 'local': loc,
            'pairs': [{'i': int(p[0]), 'j': int(p[1]), 'sum': p[2], 'm': p[3], 'sum_local': p[4], 'primes': p[5],
                       'points': p[6]} for p in pairs], 'rank': int(rank)}


def selmer_basis(wit):
    """Return the 2-descent's basis of Sel_2 cap V as (relation indices, character bits) pairs."""
    pd, els = wit['_pairing'], wit['elements']
    nch = len(pd['real']) + len(pd['primes'])
    out = []
    for comb in pd['sel']:
        J, ch = set(), 0
        for t in A.bits(comb):
            J ^= set(els[t])
            ch ^= pd['chars'][t]
        out.append((sorted(J), [(ch >> c) & 1 for c in range(nch)]))
    return out


def py_body(wit, basis):
    """Return ctprove's record for wit, from the relation records in its _pairing."""
    used = sorted({j for J, _ in basis for j in J})
    idx = {j: k for k, j in enumerate(used)}
    pd = wit['_pairing']
    rec = {j: (r, H) for j, r, H in pd['rels']}
    rels = [rec[j] for j in used]
    return ctprove([int(a) for a in wit['curve']['minimal_model']], [int(p) for p in wit['curve']['bad_primes']],
                      [int(c) for c in wit['field']['poly']], [Fraction(t) for t in wit['theta']], [r for r, _ in rels],
                      [H for _, H in rels], [[idx[j] for j in J] for J, _ in basis], [ch for _, ch in basis],
                      [list(z) for z in pd['primes']], points_x(wit))


def points_x(wit):
    """Return 4 x_E of the witness's points, on y^2 = W(x)."""
    rl = wit.get('rank_lower_bound') or {}
    return [4 * Fraction(p[0]) for p in rl.get('points', [])]


def _publish(wit, ct):
    """Set wit's cassels_tate to ct and make claims.S_ctp and the statement agree with it, at once."""
    c, rl = wit['claims'], wit.get('rank_lower_bound') or {}
    wit['cassels_tate'] = ct
    if ct:
        c['S_ctp'] = c['S'] - ct['rank']
    else:
        c.pop('S_ctp', None)
    src = None if 'points' in rl else rl.get('source')
    wit['statement'] = statement(c.get('S_ctp', c['S']), wit['hypothesis'], rl.get('r'), src)


def _kept(wit):
    """Return wit's Cassels-Tate section if it agrees with claims.S_ctp, else None."""
    ct, c = wit.get('cassels_tate'), wit['claims']
    if ct and isinstance(ct.get('rank'), int) and ct['rank'] > 0 and c.get('S_ctp') == c['S'] - ct['rank']:
        return ct
    return None


def wanted(wit):
    """Return whether the pairing could lower the bound; its rank is even, so S must be 2 above the lower bound."""
    c, rl = wit['claims'], wit.get('rank_lower_bound') or {}
    r = rl.get('r') or 0
    r1 = r if c['w'] == (-1) ** r else r + 1        # the rank's parity is w's
    return c['S'] - r1 >= 2 and bool(wit.get('_pairing')) and len(wit['_pairing']['sel']) >= 2


def add(wit, seconds=SECONDS, log=LOG):
    """Return a copy of wit with the Cassels-Tate step, within seconds.

    A new section replaces a coherent earlier one only if its rank is higher; a step that is skipped, fails or runs out
    of time keeps the earlier one.
    """
    wit = dict(wit, claims=dict(wit['claims']), _timing=dict(wit.get('_timing') or {}))
    if 'cassels_tate' not in wit:
        wit['cassels_tate'] = None
    wit['format'] = wit.pop('format')            # keep it the last key, as selmer writes it
    old = _kept(wit)
    _publish(wit, old)
    c = wit['claims']
    if not wanted(wit):
        return wit
    # ctprove pairs only a complement of the points' images in the basis, which they need not span.
    basis = selmer_basis(wit)
    t = time.time()
    try:
        with runtime.limited(seconds=seconds, error=A.time_up):
            info = py_body(wit, basis)
    except (TimeLimit, A.TimeUp):
        say('ct_time', seconds=round(seconds), S=c['S'])
        return wit
    except (RuntimeError, ValueError) as e:     # the search failed (CTSKIP in tower.gp); the bound without it stands
        say('ct_fail', S=c['S'], reason=(str(e).strip().splitlines() or [type(e).__name__])[-1][:300])
        return wit
    finally:
        wit['_timing']['ctp'] = round(time.time() - t, 2)
    sec = info.pop('section')
    ct = None if sec is None else ct_block(*sec)
    wit['_ctp'] = info
    n = info.get('paired', len(info.get('complement', [])))
    log('Cassels-Tate: %d of %d Selmer classes left by the points (rank %d), %d pairs, Cassels-Tate rank %d' % (
        n, info.get('basis', 0), info.get('points_rank', 0), n * (n - 1) // 2, info.get('rank', 0)))
    if ct and ct['rank'] and (old is None or ct['rank'] > old['rank']):
        _publish(wit, ct)
        say_bound(c['S_ctp'], 'ct', wit['hypothesis'] == 'GRH')
    return wit


def after_selmer(wit, budget=None, seconds=None, selmer_seconds=0.0):
    """Return a copy of wit with the Cassels-Tate step, costing little more than the 2-descent by default."""
    limit = seconds if seconds is not None else max(SECONDS, selmer_seconds or 0.0)
    left = (budget or runtime.current()).left()
    left = limit if left is None else min(limit, left)
    if left > 1:
        with runtime.scope(budget or runtime.current()):
            return add(wit, seconds=left)
    wit = dict(wit, claims=dict(wit['claims']))
    if 'cassels_tate' not in wit:
        wit['cassels_tate'] = None
    _publish(wit, _kept(wit))
    return wit


# ========== the pairing by binary quartics [Fisher, Res. Number Theory 8 (2022)], ported from the GP functions
# L = Q[ph]/(F), F = ph^3 - 3 I ph + J, in power coordinates; a quartic [a, b, c, d, e] is a x^4 + ... + e z^4.

Fr = Fraction


def inv_I(g):
    a, b, c, d, e = g
    return 12 * a * e - 3 * b * d + c * c


def inv_J(g):
    a, b, c, d, e = g
    return 72 * a * c * e - 27 * a * d * d - 27 * b * b * e + 9 * b * c * d - 2 * c ** 3


def cubic(I, J):
    return [J, -3 * I, 0, 1]


def z(g):
    a, b, c = g[0], g[1], g[2]
    return (Fr(3 * b * b - 8 * a * c, 3), Fr(4 * a, 3), Fr(0))


def mul(F, x, y):
    return fld.pmul(F, x, y)


def H(g, I):
    """Return the coefficients of x^2, x z, z^2 of H in L (tdH)."""
    a, b, c, d, e = g
    h = [3 * b * b - 8 * a * c, 4 * (b * c - 6 * a * d), 2 * (2 * c * c - 24 * a * e - 3 * b * d), 4 * (c * d - 6 * b * e),
         3 * d * d - 8 * c * e]
    G = [(Fr(h[k], 3), Fr(4 * g[k], 3), Fr(0)) for k in range(5)]
    return [G[0], tuple(x / 2 for x in G[1]), (G[2][0] / 6 + Fr(2 * I, 9), G[2][1] / 6, Fr(-2, 9))]


def gamma(g1, g2, g3, m, I, F):
    """Return the ph^2 coefficients of (z(g2) z(g3)/m) H1 (tdgamma)."""
    c = mul(F, mul(F, z(g2), z(g3)), fld.pinv(F, tuple(Fr(x) for x in m)))
    return [mul(F, c, h)[2] for h in H(g1, I)]


def q4(g, s, t):
    return sum(g[k] * s ** (4 - k) * t ** k for k in range(5))


def q2(f, s, t):
    return f[0] * s * s + f[1] * s * t + f[2] * t * t


def issq(v, p):
    return v != 0 and A.is_square_qp(v, p)


def mu2(g, I, J):
    """Return the rationals mu^2 with I(mu^2 g) = I and J(mu^2 g) = J (tdmu2)."""
    Ig, Jg = inv_I(g), inv_J(g)
    if I and J:
        return [Fr(J, Jg) / Fr(I, Ig)]
    if not I:
        r = _ratroot(Fr(J, Jg), 3)
        return [r] if r is not None else []
    r = _ratroot(Fr(I, Ig), 2)
    return [r, -r] if r is not None else []


def _ratroot(x, k):
    if x < 0 and k % 2 == 0:
        return None
    s = -1 if x < 0 else 1
    a, b = A.iroot(abs(x.numerator), k), A.iroot(x.denominator, k)
    return None if a is None or b is None else s * Fr(a, b)


def is_square_rat(x):
    return x >= 0 and _ratroot(Fr(x), 2) is not None


# ---------- square roots in L, quartic substitutions, minimisation and reduction

def sqrtL(x, F):
    """Return a square root of x in L, or None; candidates from the embeddings are checked exactly."""
    import itertools
    d = abs(fld.disc((F[3], F[2], F[1], F[0])))
    x = tuple(Fr(c) for c in x)
    D = 1
    for c in x:
        D = D * c.denominator // math.gcd(D, c.denominator)
    y = tuple(c * D * D * d * d for c in x)              # w = D d sqrt(x) has integral coordinates
    size = max(len(str(abs(c.numerator))) for c in y) + len(str(d)) + 40
    with localcontext() as ctx:
        ctx.prec = size
        rr, cc = A.roots(F, size)
        yd = [Decimal(c.numerator) for c in y]
        ev = lambda re, im: (yd[0] + yd[1] * re + yd[2] * (re * re - im * im), yd[1] * im + 2 * yd[2] * re * im)
        vals = [ev(r, 0)[0] for r in rr]
        if any(v < 0 for v in vals):
            return None
        sr = [v.sqrt() for v in vals]
        sc = [_csqrt(ev(*c)) for c in cc]
        for sg in itertools.product((1, -1), repeat=len(rr) + len(cc)):
            M = [[Decimal(1), r, r * r] for r in rr]
            v = [s_ * w for s_, w in zip(sg, sr)]
            for (re, im), (wr, wi), s_ in zip(cc, sc, sg[len(rr):]):
                M += [[Decimal(1), re, re * re - im * im], [Decimal(0), im, 2 * re * im]]
                v += [s_ * wr, s_ * wi]
            c = _solve_dec(M, v)
            if c is None:
                continue
            c = tuple(Fr(int(t.to_integral_value())) for t in c)
            if mul(F, c, c) == y:
                return tuple(t / (D * d) for t in c)
    return None


def _csqrt(v):
    re, im = v
    r = (re * re + im * im).sqrt()
    a = ((r + re) / 2).sqrt()
    b = ((r - re) / 2).sqrt()
    return (a, b if im >= 0 else -b)


def _solve_dec(M, v):
    M = [list(r) + [b] for r, b in zip(M, v)]
    n = 3
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if M[piv][c] == 0:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                t = M[r][c] / M[c][c]
                M[r] = [a - t * b for a, b in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def qsub(g, U):
    """Return g(al x + be z, ga x + de z) for U = [[al, be], [ga, de]] (tdqsub)."""
    (al, be), (ga, de) = U
    P1, P2 = [[1]], [[1]]
    for _ in range(4):
        P1.append(_pm(P1[-1], [be, al]))
        P2.append(_pm(P2[-1], [de, ga]))
    out = [0] * 5
    for k in range(5):
        if g[k]:
            for i, c in enumerate(_pm(P1[4 - k], P2[k])):
                out[4 - i] += g[k] * c
    return out


def _pm(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def minimise(g, I, J):
    """Return an integral quartic GL2(Q)-equivalent to g with invariants exactly I, J, or None."""
    D = _den(g)
    g = [int(Fr(c) * D * D) for c in g]
    m2 = mu2(g, I, J)
    lam = _ratroot(1 / m2[0], 2) if m2 and m2[0] > 0 else None
    if lam is None:
        return None
    for p, k in A.factor(lam.numerator):
        for _ in range(k):
            g = _step(g, p, 3)
            if g is None:
                return None
    for mu in mu2(g, I, J):
        h = [Fr(c) * mu for c in g]
        if mu > 0 and is_square_rat(mu) and all(c.denominator == 1 for c in h):
            return [int(c) for c in h]
    return None


def _step(g, p, depth):
    """Return g with one level less at p (see minimise), or None."""
    if all(x % (p * p) == 0 for x in g):
        return [x // (p * p) for x in g]
    moves = []
    rs = list(range(p)) if all(x % p == 0 for x in g) else A.roots_mod(g[::-1], p)    # g(r, 1) = 0 mod p,
    for r in rs + ([None] if g[0] % p == 0 else []):                                    # or g(1, 0)
        h = qsub(g, [[p, r], [0, 1]] if r is not None else [[1, 0], [0, p]])
        c = 0
        for x in h:
            c = math.gcd(c, x)
        v = A.val(c, p)
        if v >= 4:
            return [x // p ** 4 for x in h]
        if v >= 2:
            moves.append([x // (p * p) for x in h])
    for h in moves if depth else []:
        r = _step(h, p, depth - 1)
        if r is not None:
            return r
    return None


def reduce(g):
    """Return g o U, U in GL2(Z), reducing g's covariant quadratic form (Julia's) by LLL while g shrinks."""
    size = lambda h: max(abs(c) for c in h)
    while True:
        with localcontext() as ctx:
            ctx.prec = len(str(size(g))) + 30
            th = _croots4(g)
            if th is None:
                return g
            w = []
            for t in th:
                d = [sum((4 - k) * g[k] * _cp(t, 3 - k)[i] for k in range(4)) for i in (0, 1)]
                w.append(1 / (d[0] * d[0] + d[1] * d[1]).sqrt())
            G = [[sum(w), -sum(x * t[0] for x, t in zip(w, th))], [0, sum(x * (t[0] ** 2 + t[1] ** 2) for x, t in zip(w, th))]]
            G[1][0] = G[0][1]
        U = A.lll_gram([[Fr(x) for x in r] for r in G], eta=Fr(51, 100))
        h = qsub(g, [[U[0][0], U[0][1]], [U[1][0], U[1][1]]])
        if size(h) >= size(g):
            return g
        g = h


def _cp(t, k):
    """Return t^k for the complex t = (re, im)."""
    r, i = Decimal(1), Decimal(0)
    for _ in range(k):
        r, i = r * t[0] - i * t[1], r * t[1] + i * t[0]
    return r, i


def _croots4(g):
    """Return the 4 complex roots of g(x, 1) to the context's precision, or None."""
    if not g[0]:
        return None
    prec = getcontext_prec()
    with localcontext() as ctx:
        ctx.prec = min(prec, 40)
        zs = _dk4(g)
    if zs is None or prec <= 40:
        return zs
    return [_newton([Decimal(c) for c in g[::-1]], z, prec - 5) for z in zs]


def _dk4(g):
    """Return the roots of g(x, 1) by Durand-Kerner at the context's precision, or None."""
    a = [Decimal(c) / Decimal(g[0]) for c in g]
    R = 1 + max(abs(c) for c in a[1:])
    r0 = abs(a[4]) ** (Decimal(1) / 4) if a[4] else R           # the roots' geometric mean
    zs = [(r0 * Decimal(math.cos(0.4 + 1.6 * k)), r0 * Decimal(math.sin(0.4 + 1.6 * k))) for k in range(4)]
    eps = Decimal(10) ** (-getcontext_prec() // 2) * R
    for _ in range(2000):
        new = []
        for i, x in enumerate(zs):
            n = (Decimal(1), Decimal(0))
            for c in a[1:]:
                n = (n[0] * x[0] - n[1] * x[1] + c, n[0] * x[1] + n[1] * x[0])
            dn = (Decimal(1), Decimal(0))
            for j, y in enumerate(zs):
                if j != i:
                    u = (x[0] - y[0], x[1] - y[1])
                    dn = (dn[0] * u[0] - dn[1] * u[1], dn[0] * u[1] + dn[1] * u[0])
            q = dn[0] * dn[0] + dn[1] * dn[1]
            if not q:
                return None
            new.append((x[0] - (n[0] * dn[0] + n[1] * dn[1]) / q, x[1] - (n[1] * dn[0] - n[0] * dn[1]) / q))
        done = max(abs(u[0] - v[0]) + abs(u[1] - v[1]) for u, v in zip(new, zs)) < eps
        zs = new
        if done:
            return zs
    return zs


def getcontext_prec():
    from decimal import getcontext
    return getcontext().prec


def nice(g, F, S):
    """Return g(x, z + k x) for the first k with z(g) invertible and g(1, k) != 0 (tdnice)."""
    first = None
    for n in range(401):
        k = (n + 1) // 2 if n % 2 else -(n // 2)
        a = q4(g, 1, k)
        if not a:
            continue
        r = abs(Fr(a).numerator)
        for p in S:
            r //= p ** A.val(r, p)
        good = r == 1 or A.is_prime(r)
        if not (good or first is None):
            continue
        h = qsub(g, [[1, 0], [k, 1]])
        if not _invertible(z(h), F):
            continue
        if good or not S:
            return h
        first = h
    if first is None:
        raise RuntimeError('tdnice: no coordinates with z(g) invertible')
    return first


def _invertible(x, F):
    return A.det3(_mulmat(F, x)) != 0


# ---------- the quartic of a class (tdclassq, tdclassg)

def classq(xi, I, J, F, fa):
    """Return the candidate quartics of the class xi; fa = (P0, cofactors) saves factoring for qfsolve."""
    pw = [(Fr(1), Fr(0), Fr(0))]
    for _ in range(4):
        pw.append(mul(F, pw[-1], (0, 1, 0)))
    G = [[[mul(F, xi, pw[k + l])[c] for l in range(3)] for k in range(3)] for c in range(3)]
    if fa is None:
        d = A.det3([[x * 1 for x in r] for r in G[2]])
        den = 1
        for r in G[2]:
            for x in r:
                den = den * Fr(x).denominator // math.gcd(den, Fr(x).denominator)
        P0 = [p for p, _ in A.factor(abs(int(d * den ** 3)))] if d else []
        sol = A.solve(G[2], P0)
    else:
        sol = A.solve(G[2], fa[0], fa[1])
    if sol is None:
        raise RuntimeError('tdclassq: the conic has no rational point')
    P = A.param(G[2], sol)                          # rows: coefficients of x^2, x, 1 of the 3 coordinates
    q = [Fr(0)] * 5
    for i in range(3):
        for j in range(3):
            c = G[1][i][j]
            if c:
                for s in range(3):
                    for t in range(3):
                        q[s + t] += c * P[i][s] * P[j][t]
    g = q
    return [[mu * c for c in g] for mu in mu2(g, I, J)]


def classg(xi, I, J, F, S, fa=None):
    """Return [g, m] with g a minimised, reduced quartic of the class xi and m^2 = xi z(g), m None if none."""
    for h in classq(xi, I, J, F, fa):
        h2 = minimise(h, I, J)
        g = nice(h if h2 is None else reduce(h2), F, S)
        m = sqrtL(mul(F, xi, z(g)), F)
        if m is not None:
            return g, m
    return None, None


# ---------- local points and the pairing (tdqreal, tdqpoint, tdctpair, tdctq)

def qreal(g, gam):
    ok = lambda s, t: math.gcd(s, t) == 1 and q4(g, s, t) > 0 and (gam is None or q2(gam, s, t))
    for n in range(1, 31):
        for i in range(-n, n + 1):
            if ok(i, n - abs(i)):
                return [i, n - abs(i)]
    with localcontext() as ctx:
        ctx.prec = 100 + 2 * len(str(g))
        R = [Decimal(-10) ** 40] + [Decimal(z) / 2 ** 200 for z in A.real_roots_floor([int(Fr(c) * _den(g)) for c in g][::-1], 200)] + [Decimal(10) ** 40]
        for lo, hi in zip(R, R[1:]):
            mid = (lo + hi) / 2
            for c in A.convergents(mid, 2 ** 300):
                if ok(c.numerator, c.denominator):
                    return [c.numerator, c.denominator]
    raise RuntimeError('tdqreal: no real point on %s' % (g,))


def _den(g):
    D = 1
    for c in g:
        D = D * Fr(c).denominator // math.gcd(D, Fr(c).denominator)
    return D


def qpoint(g, p, gam=None):
    """Return a point [s, t] of g at the place p (0: R) with gamma(s, t) != 0 (tdqpoint)."""
    if p == 0:
        return qreal(g, gam)
    D = _den(g)
    gi = [int(Fr(c) * D * D) for c in g]
    w = A.wit(gi[::-1], p)
    if w is not None:
        s, t = w, 1
    else:
        w = A.wit([c * p ** i for i, c in enumerate(gi)], p)
        if w is None:
            raise RuntimeError('tdqpoint: the quartic is not soluble at %d' % p)
        s, t = 1, p * w
    K = A.val(q4(gi, s, t), p) + 4
    for j in range(201):
        s1, t1 = (s + j * p ** K, t) if t == 1 else (s, t + j * p ** K)
        if issq(q4(g, s1, t1), p) and (gam is None or q2(gam, s1, t1)):
            return [s1, t1]
    raise RuntimeError('tdqpoint: no point with gamma1 != 0 at %d' % p)


def ctpair(g1, g2, g3, m, I, F, S, X, Pts):
    """Return the pairing <[g1], [g2]> as 0 or 1 (tdctpair)."""
    gam = gamma(g1, g2, g3, m, I, F)
    W = A.isog_places(S) + list(X)
    e = 0
    for (s, t), p in zip(Pts, W):
        gv = q2(gam, s, t)
        if not issq(q4(g1, s, t), p) or not gv:
            raise RuntimeError('tdctpair: a local point is not one')
        e += A.hilbert(g2[0], gv, p) == -1
    return e % 2


def cof(g, F, P0):
    """Return n with |N(z(g))| = n^2 times primes of P0, or that part itself if not a square (tdcof)."""
    x = z(g)
    N = A.det3(_mulmat(F, x))
    r = abs(N.numerator) * N.denominator
    for p in P0:
        r //= p ** A.val(r, p)
    s = math.isqrt(r)
    return s if s * s == r else r


def _mulmat(F, x):
    cols = [mul(F, x, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    return [[cols[j][i] for j in range(3)] for i in range(3)]


class Costly(Exception):
    """Raised when a pairing's projected cost exceeds its allowance, so it is left out at once."""
    def __init__(self, seconds):
        super().__init__('projected cost %.0f s' % seconds)
        self.seconds = seconds


def ctq(mm, S, Q, fake, left=None):
    """Return [quartics, local points, pairs, matrix, rank] for the integral ELS quartics Q (tdctq).

    With left, raise Costly once the projected cost exceeds left, or pair k of N runs past k/N of it.
    """
    t0 = time.monotonic()
    fac = {}

    def factor(x):
        if x not in fac:
            fac[x] = A.factor(x)
        return fac[x]
    c4, c6, _ = cv.cinv(mm)
    I, J = c4, 2 * c6
    F = cubic(I, J)
    V = A.isog_places(S)
    P0 = sorted(set(S) | {3})
    Loc = [[qpoint(g, p) for p in V] for g in Q]
    cf = [cof(g, F, P0) for g in Q] if fake else None
    n = len(Q)
    M = [[0] * n for _ in range(n)]
    pairs = []

    def pair(i, j):
        zij = mul(F, z(Q[i]), z(Q[j]))
        g3, m = classg(zij, I, J, F, [], (P0, [cf[i], cf[j]]) if fake else None)
        if m is None:
            raise RuntimeError('tdct: no quartic for the sum of %s' % [i + 1, j + 1])
        gam = gamma(Q[i], Q[j], g3, m, I, F)
        cg = _content(gam)
        Xs = sorted({p for x in (abs(Q[j][0]), abs(cg.numerator), cg.denominator) if x > 1
                     for p, _ in factor(x)} - set(V))
        Pts = [qpoint(Q[i], p, gam) for p in V + Xs]
        pairs.append([i + 1, j + 1, g3, list(m), [qpoint(g3, p) for p in V], Xs, Pts])
        M[i][j] = M[j][i] = ctpair(Q[i], Q[j], g3, m, I, F, S, Xs, Pts)
    t1 = time.monotonic()           # the local points of the quartics done: the pairs next
    N = n * (n - 1) // 2
    for i in range(n):
        for j in range(i + 1, n):
            A.tick()
            if left is None:
                pair(i, j)
                continue
            share = (t0 + left - t1) * (len(pairs) + 1) / N
            with runtime.scope(runtime.current().child(seconds=max(0.0, t1 + share - time.monotonic()),
                                                       stop=lambda: Costly(math.inf))):
                pair(i, j)
            est = t1 - t0 + (time.monotonic() - t1) * N / len(pairs)
            if est > left:
                raise Costly(est)
    return [Q, Loc, pairs, M, A.rank2(M)]


def _content(v):
    v = [Fr(x) for x in v]
    n = 0
    for x in v:
        n = math.gcd(n, x.numerator)
    l = 1
    for x in v:
        l = l * x.denominator // math.gcd(l, x.denominator)
    return Fr(n, l)


# ---------- the Cassels-Tate step: the pairing on the 2-descent's elements (ctp.gp's ctprove)

def _scale(rows, den):
    """Return the fractional ideal (1/den) L(rows), rows an HNF, in lowest terms."""
    g = den
    for r in rows:
        for x in r:
            g = math.gcd(g, x)
    return [[x // g for x in r] for r in rows], den // g


def _det(H):
    return H[0][0] * H[1][1] * H[2][2]


def pinv_ideal(F, P):
    """Return P^-1 = (1/p) (p O + tau O), cached on P."""
    if not hasattr(P, 'inv'):
        P.inv = _scale(A.hnf([F.omul(P.tau, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))], P.p), P.p)
    return P.inv


_ONE = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def imul_f(F, X, Y):
    """Return the product of the fractional ideals X, Y = (HNF rows, den), as an HNF."""
    if X[0] == _ONE or Y[0] == _ONE:
        return _scale(Y[0] if X[0] == _ONE else X[0], X[1] * Y[1])
    return _scale(A.hnf([F.omul(x, y) for x in X[0] for y in Y[0]], _det(X[0]) * _det(Y[0])), X[1] * Y[1])


def leaf(k, r, Hs):
    """Return [y, A, E = D^-1] for the relation r times a square, (y) = A D^2, A the primes of odd valuation (ctleaf).
    """
    F = k
    yi, d = F.integral(fld.relation_el(r))
    ps = set(Hs) | set(k.lcp)
    if r[0] == 1:
        ps |= {p for p, _ in A.factor(r[4])}
    if r[0] == 2:
        ps.add(r[1])
    A_, E = set(), (_ONE, 1)
    if r[0] == 0 and math.gcd(r[1], r[2]) == 1:
        # At p not dividing lc(f), only the prime at (a : b) divides a - b alpha, to the power v_p(F(a, b)).
        a, b = r[1], r[2]
        Fab = sum(c * a ** i * b ** (3 - i) for i, c in enumerate(k.f))
        for p in sorted(ps - set(k.lcp)):
            v = A.val(Fab, p)
            if v:
                lab = (p, a * pow(b, -1, p) % p)
                if v % 2:
                    A_.add((p, lab))
                for _ in range(v // 2):
                    E = imul_f(F, E, pinv_ideal(F, F.prime(p, lab)))
        ps = set(k.lcp)
    for p in sorted(ps):
        for P in k.dec(p):
            v = F.val_int(yi, P)[0] - P.e * A.val(d, p)
            if v:
                if v % 2:
                    A_.add((p, P.label))
                for _ in range(abs(v // 2)):        # E = D^-1: P^-1 for each P in D, P for each P^-1
                    E = imul_f(F, E, pinv_ideal(F, P) if v > 0 else (P.hnf, 1))
    return [tuple(c * d for c in yi), A_, _scale(E[0], E[1] * d)]     # y d^2 = yi d in O, E / d


def _lg(x):
    """Return roughly log10 |x| for a nonzero int or Fraction."""
    x = Fr(x)
    return (abs(x.numerator).bit_length() - x.denominator.bit_length()) * 0.30103


def red(k, t):
    """Return [y b^2, A, E b^-1] for b in E small in the weighted norm sum_v c_v |y|_v |b|_v^2 (ctred).

    The Gram matrix is rounded at a scale bounded below by Hadamard and Fischer's inequalities. Precision only affects
    how well b reduces: the quartics are checked exactly.
    """
    F = k
    y, A_, (H0, den) = t
    lb = _basis_emb(k, 40)[1]
    g = int(max(_lg(x) for x in y if x))
    Y = g + lb + 1                                                      # |y|_v <= 10^Y
    py = 40 + int(lb)
    while True:                         # w_v to 20 digits: the cancellation in y's embeddings at most 10^(py - 20)
        w = _weights(k, y, py, g)
        if all(x and _dlg(x) > Y - g - py + 20 for x in w):
            break
        py *= 2
    s2, cv = _basis_emb(k, 40)[2:]
    V = A.lll_gram([[sum(c * a * b for c, a, b in zip(s2, r, q)) for q in H0] for r in H0])
    H = [[sum(V[r][c] * H0[r][i] for r in range(3)) for i in range(3)] for c in range(3)]
    T = [max(_lg(x) for x in r if x) + lb + 1 for r in H]               # |h|_v <= 10^T for the rows h of H
    with localcontext() as ctx:
        ctx.prec = 40
        lw = [_dlg(x) for x in w]
        Gi = [_dlg(sum(c * (abs(a) + abs(b) + Decimal(10) ** int(Ti - 37)) ** 2      # G_ii <= 10^Gi
                       for c, (a, b) in zip(w, _emb(k, h, 40)))) for h, Ti in zip(H, T)]
    dH, lp = _lg(_det(H0)), sum(c * x for c, x in zip(cv, lw)) - 0.602 * (3 - len(w))
    lam = 2 * dH + _lg(fld.disc(F.F)) + lp + lp / 3 + 2 * dH / 3 - sum(Gi)
    m = 30
    while True:
        e = int(m - lam)
        U = _red_lll(k, w, H, 12 + e + int(max(lw) + 2 * max(T)), e)
        if U is not None:
            break
        m *= 2
    be = [sum(U[r][0] * H[r][i] for r in range(3)) for i in range(3)]     # b = be / den
    # be' = N(be) be^-1 is in O, so E b^-1 = (1/|N(be)|) L be'
    y2 = [divmod(c, den * den) for c in F.omul(y, F.omul(be, be))]
    if any(r for _, r in y2):
        raise RuntimeError('ctred: y b^2 is not integral')
    M = F.omat(be)
    adj = (M[1][1] * M[2][2] - M[1][2] * M[2][1], M[1][2] * M[2][0] - M[1][0] * M[2][2],
           M[1][0] * M[2][1] - M[1][1] * M[2][0])
    n = sum(M[0][j] * adj[j] for j in range(3))
    E2 = _scale(A.hnf([F.omul(r, adj) for r in H0], _det(H0) * n * n), abs(n))
    return [tuple(q for q, _ in y2), A_, E2]


def _dlg(x):
    """Return log10 x for a Decimal x > 0, as a float."""
    a = x.adjusted()
    return a + math.log10(float(x.scaleb(-a)))


def _emb(k, x, prec):
    """Return the embeddings of x at the places, at the context's precision."""
    return [(sum(c * u for c, (u, _) in zip(x, v)), sum(c * w for c, (_, w) in zip(x, v)))
            for v in _basis_emb(k, prec)[0]]


def _weights(k, y, py, g):
    """Return the weights c_v |y|_v 10^-g at py digits."""
    sh = max(0, max(abs(c).bit_length() for c in y) - int(3.33 * py) - 20)
    with localcontext() as ctx:
        ctx.prec = py
        t = (Decimal(2) ** sh).scaleb(-g)
        ys = [x >> sh for x in y]
        return [(a * a + b * b).sqrt() * t * c for (a, b), c in zip(_emb(k, ys, py), _basis_emb(k, py)[3])]


def _red_lll(k, w, H, prec, e):
    """Return the LLL transformation of red's rounded Gram matrix, or None if not positive definite."""
    with localcontext() as ctx:
        ctx.prec = prec
        Em = [_emb(k, r, prec) for r in H]
        G = [[int(sum(c * (Em[i][v][0] * Em[j][v][0] + Em[i][v][1] * Em[j][v][1])
                      for v, c in enumerate(w)).scaleb(e).to_integral_value()) for j in range(3)] for i in range(3)]
    if not _pd(G):
        return None
    return A.lll_gram(G)


def _pd(G):
    """Return whether the 3 x 3 integer Gram matrix G is positive definite."""
    return G[0][0] > 0 and G[0][0] * G[1][1] - G[0][1] * G[1][0] > 0 and A.det3(G) > 0


_EMB = {}


def _newton(f, z, prec):
    """Return the root z of f refined by Newton's method to prec digits."""
    re, im = z
    for _ in range(200):
        v, d = (Decimal(0), Decimal(0)), (Decimal(0), Decimal(0))
        for c in reversed(f):           # Horner for f and f' at z
            d = (d[0] * re - d[1] * im + v[0], d[0] * im + d[1] * re + v[1])
            v = (v[0] * re - v[1] * im + c, v[0] * im + v[1] * re)
        n = d[0] * d[0] + d[1] * d[1]
        dr, di = (v[0] * d[0] + v[1] * d[1]) / n, (v[1] * d[0] - v[0] * d[1]) / n
        re, im = re - dr, im - di
        if max(abs(dr), abs(di)) <= (abs(re) + abs(im)).scaleb(-prec):
            break
    return re, im


def _basis_emb(k, prec):
    """Return the embeddings of O's basis at the places, cached per field: thousands of products share one."""
    key = tuple(k.f)
    if key not in _EMB:
        _EMB.clear()
        rr, cc = A.roots(k.f, 60)
        _EMB[key] = [0, [(r, Decimal(0)) for r in rr] + list(cc), None]
    c = _EMB[key]
    if c[0] < prec:
        c[0] = 2 * prec
        out = []
        with localcontext() as ctx:
            ctx.prec = c[0] + 20
            c[1] = [_newton(k.f, z, c[0] + 10) for z in c[1]]
            for re, im in c[1]:
                r2, i2 = re * re - im * im, 2 * re * im
                out.append([(Decimal(b[0]) + b[1] * re + b[2] * r2, b[1] * im + b[2] * i2) for b in k.B])
        c[2] = (out, float(max(abs(x).log10() for v in out for b in v for x in b if x)),
                [4 ** round(math.log2(max(abs(x) for v in out for x in v[i]))) for i in range(3)],
                [1 if v[1][1] == 0 else 2 for v in out])         # c_v is 1 at real, 2 at complex places
    return c[2]


def cmul(k, s, t):
    """Return the product of two [y, A, E], reduced once y or E is large (ctmul).

    Common primes go into D a few at a time, so each reduction removes a small skew and E stays small.
    """
    F = k
    x = [F.omul(s[0], t[0]), s[1] ^ t[1], imul_f(F, s[2], t[2])]
    C = sorted(s[1] & t[1], key=lambda pz: (pz[0], fld.label_text(pz[1])))     # the labels' text order
    while True:
        Ps = []
        while C and math.prod(P.q for P in Ps) < 10 ** 30:
            Ps.append(F.prime(*C.pop()))
        x[2] = imul_f(F, x[2], _inv_prod(F, Ps))
        if C or x[2][1] > 10 ** 15 or max(abs(c) for c in x[0]) > 10 ** 45:
            x = red(k, x)
        if not C:
            return x


def _inv_prod(F, Ps):
    """Return the product of P^-1 over primes Ps of distinct p, as one HNF by CRT."""
    if not Ps:
        return _ONE, 1
    if len({P.p for P in Ps}) < len(Ps):
        X = (_ONE, 1)
        for P in Ps:
            X = imul_f(F, X, pinv_ideal(F, P))
        return X
    T, q = [0, 0, 0], 1
    for P in Ps:
        u = pow(q, -1, P.p)
        T = [a + q * ((b - a) * u % P.p) for a, b in zip(T, P.tau)]
        q *= P.p
    return _scale(A.hnf([F.omul(T, e) for e in _ONE], q), q)


def cprod(k, L, a, b):
    """Return the product of the leaves L[a..b] as a balanced tree (ctprod)."""
    A.tick()
    if a == b:
        return L[a]
    c = (a + b) // 2
    return cmul(k, cprod(k, L, a, c), cprod(k, L, c + 1, b))


def ideal_norm(E):
    """Return the integer N(D) for E = D^-1."""
    rows, den = E
    return Fr(den ** 3, abs(A.det3(rows)))


def complement(f, th, X, Ech, CQ):
    """Return (basis indices of a complement of the points' images, located, rank) (ctcomplement)."""
    d = len(Ech)
    if not X or not d:
        return list(range(d)), 0, 0
    with localcontext() as ctx:
        ctx.prec = 110
        RR = A.roots(f, 100)[0]
        tc = [Decimal(c.numerator) / c.denominator for c in map(Fr, th)]
        rows = []
        for x0 in X:
            x0 = Fr(x0)
            xd = Decimal(x0.numerator) / x0.denominator
            pc = [int(xd - (tc[0] + tc[1] * z_ + tc[2] * z_ * z_) < 0) for z_ in RR]
            den = x0.denominator * math.prod(Fr(c).denominator for c in th)
            for q, c in CQ:
                if den % q == 0:
                    pc.append(-1)
                    continue
                v = (x0.numerator * pow(x0.denominator, -1, q) - sum(Fr(t).numerator * pow(Fr(t).denominator, -1, q) * c ** i
                                                                      for i, t in enumerate(th))) % q
                pc.append(-1 if v == 0 else int(A.kronecker(v, q) < 0))
            cols = [i for i, x in enumerate(pc) if x >= 0]
            sol = A.solve2([[Ech[r][c] for c in cols] for r in range(d)], [pc[c] for c in cols])
            if sol is not None:
                rows.append(sol)
    if not rows:
        return list(range(d)), 0, 0
    rk = A.rank2(rows)
    basis = []                                  # independent columns of the points' rows
    J = [c for c in range(d) if A.insert(basis, sum(rows[i][c] << i for i in range(len(rows))))]
    return [i for i in range(d) if i not in J], len(rows), rk


def ctprove(mm, P, f, th, R, Hs, SelJ, Ech, CQ, X):
    """Return ctprove's record; its section is a subset W with nonsingular Gram matrix, or None at rank 0."""
    import time
    c4, c6, _ = cv.cinv(mm)
    I, J = c4, 2 * c6
    F = cubic(I, J)
    S = sorted({2} | set(P))
    P0 = sorted(set(S) | {3})
    k = fld.form_field(f)
    b2 = cv.binv(mm)[0]
    # alpha as a polynomial in theta, then theta -> -(ph + b2)/3
    K3 = k.f
    t1 = tuple(Fr(c) for c in th)
    t2 = fld.pmul(K3, t1, t1)
    al = A.solve3([[1, t1[0], t2[0]], [0, t1[1], t2[1]], [0, t1[2], t2[2]]], (0, 1, 0))
    thL = (Fr(-b2, 3), Fr(-1, 3), Fr(0))
    alL = tuple(al[0] * e0 + al[1] * e1 + al[2] * e2 for e0, e1, e2 in zip((1, 0, 0), thL, mul(F, thL, thL)))
    pws = [(Fr(1), Fr(0), Fr(0)), alL, mul(F, alL, alL)]
    comp, located, prk = complement(f, th, X, Ech, CQ)
    out = {'basis': len(SelJ), 'located': located, 'points_rank': prk, 'complement': [int(i) for i in comp]}
    t0 = time.time()
    leaves, Q = {}, []
    if len(comp) < 2:                           # fewer than two classes left by the points: no pair to compute
        comp = []
    for t in comp:
        L = []
        for j in SelJ[t]:
            if j not in leaves:
                leaves[j] = leaf(k, R[j], Hs[j])
            L.append(leaves[j])
        y = cprod(k, L, 0, len(L) - 1)
        if any(p not in S for p, _ in y[1]):
            raise RuntimeError('ctprove: a Selmer element has an odd valuation outside S')
        yp = k.power(y[0])
        xi = tuple(sum(Fr(yp[i]) * pws[i][c] for i in range(3)) for c in range(3))
        nd = ideal_norm(y[2])
        g, m = classg(xi, I, J, F, S, (P0, [int(nd)]))
        if m is None:
            raise RuntimeError('ctprove: no quartic in the class of a Selmer element')
        if any(Fr(c).denominator != 1 for c in g):
            raise RuntimeError('ctprove: the reduced quartic of a Selmer element is not integral: %s' % g)
        Q.append([int(c) for c in g])
    # A quartic with g[0] a nonzero square has a rational point, so its class is in the kernel; drop it.
    Q = [g for g in Q if not (g[0] > 0 and math.isqrt(g[0]) ** 2 == g[0])]
    out.update(quartics_ms=int(1000 * (time.time() - t0)), paired=len(Q))
    if len(Q) < 2:
        out.update(pairs_ms=0, rank=0, section=None)
        return out
    t0 = time.time()
    res = ctq(mm, S, Q, True)
    M, R2 = res[3], res[4]
    out.update(pairs_ms=int(1000 * (time.time() - t0)), rank=int(R2), section=None)
    if not R2:
        return out
    basis = []
    Wi = [i for i, row in enumerate(M) if A.insert(basis, sum(b << t for t, b in enumerate(row)))]
    pos = {w + 1: n + 1 for n, w in enumerate(Wi)}
    QW, LW = [res[0][i] for i in Wi], [res[1][i] for i in Wi]
    PW = [[pos[pr[0]], pos[pr[1]]] + pr[2:] for pr in res[2] if pr[0] in pos and pr[1] in pos]
    out['section'] = (R2, QW, LW, PW)
    return out


