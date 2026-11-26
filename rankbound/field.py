"""The cubic field K of a curve with E(Q)[2] = 0, without PARI: its data and its arithmetic."""
import copy, functools, gzip, json, math, os
from types import SimpleNamespace
from fractions import Fraction
from decimal import Decimal, localcontext
from . import runtime
from .core import NotApplicable, parity_bound
from .runtime import read_json
from . import arith as A, curve as cv
from .curve import NOT_APPLICABLE, _ainvs5, canon, check_name

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT = 'rankbound-selmer-field-1'


# ---------- the field

_MEMO = {}


def compute(ainvs, bad_primes):
    """Return (the field data, the small labels), memoised for one curve."""
    key = (tuple(str(a) for a in ainvs), tuple(str(p) for p in bad_primes))
    if key not in _MEMO:
        _MEMO.clear()
        _MEMO[key] = field_data([int(p) for p in (canon(p, 'bad prime') for p in bad_primes)], _ainvs5(ainvs))
    F, small = _MEMO[key]
    return copy.deepcopy(F), list(small)


def check_bad_primes(a, P):
    """Return (the minimal model, [u, r, s, t]) of a; raise ValueError unless P is exactly its bad primes."""
    for p in P:
        if p < 2 or not A.is_prime(p):
            raise ValueError('The bad prime %d is not prime.' % p)
    if len(set(P)) != len(P):
        raise ValueError('The bad prime %d is listed twice.' % next(p for p in P if P.count(p) > 1))
    mm, v = cv.minimal(a, P)
    D = cv.cinv(mm)[2]
    for p in P:
        if D % p:
            raise ValueError('The bad prime %d does not divide the minimal discriminant.' % p)
    r = abs(D)
    for p in P:
        r //= p ** A.val(r, p)
    if r != 1:
        raise ValueError('The bad primes given do not account for the whole discriminant.')
    return mm, v


def ftypes(f, p):
    """Return ([(r, e)] of the degree-1 primes above p, [f] of all primes above p)."""
    pts = roots_p1(f, p)
    rest = 3 - sum(m for _, m in pts)
    return pts, [1] * len(pts) + ([rest] if rest else [])


EULER, CATALAN = 0.5772156649015329, 0.915965594177219


def bdf(f, dK, r1, Tmax, types=None):
    """Return the least T at which [BDF08] (10) holds in floats with margin 1e-8 (the checker proves it exactly).

    types, if given, is a cache p -> ftypes(f, p)."""
    split = {} if types is None else types

    def ok(T):
        cD = math.log(abs(dK)) - 3 * (EULER + math.log(8 * math.pi)) - r1 * math.pi / 2
        cN = 3 * math.pi ** 2 / 2 + r1 * 4 * CATALAN
        lT, SA, SB = math.log(T), 0.0, 0.0
        for p in A.primes(2, T):
            if p not in split:
                split[p] = ftypes(f, p)
            lp = math.log(p)
            for fp in split[p][1]:
                lN = fp * lp
                if lN >= lT:
                    continue
                q = 1 / math.sqrt(p ** fp)
                M = math.ceil(lT / lN) - 1
                a_ = lN * q
                b_ = lN * a_
                if M > 1:
                    i1 = 1 / (1 - q)
                    a_ *= (1 - q ** M) * i1
                    b_ *= (1 - q ** M * (M + 1 - M * q)) * i1 * i1
                SA += a_
                SB += b_
        return cD + (cN + 2 * SB) / lT - 2 * SA < -1e-8

    lo, hi = 1, 16
    while not ok(hi):
        lo, hi = hi, 2 * hi
        if hi > 2 * Tmax:
            raise RuntimeError('BDF bound > Tmax')
    while hi - lo > 1:
        t = (lo + hi) // 2
        if ok(t):
            hi = t
        else:
            lo = t
    return hi


def field_data(P, a):
    """Return (the field.json record, the small labels) of the curve a with bad primes P."""
    mm, _ = check_bad_primes(a, P)
    b2, b4, b6, _ = cv.binv(mm)
    if cv.int_roots([16 * b6, 8 * b4, b2, 1]):
        raise NotApplicable(NOT_APPLICABLE)
    f, _ = field_of(mm, [2, 3] + P)
    dK = disc((f[3], f[2], f[1], f[0]))
    roots = A.real_roots_floor(f, 128)
    bach = math.floor(12 * math.log(abs(dK)) ** 2)
    types = {}

    def ft(p):
        if p not in types:
            types[p] = ftypes(f, p)
        return types[p]
    GB = bdf(f, dK, len(roots), bach, types)
    lsize, skew = size((f[3], f[2], f[1], f[0]))
    D = cv.cinv(mm)[2]
    bk, n, tam = [], 0, {}
    for p in P:
        fp, kod, c = cv.localred(mm, p)
        t = int(A.val(D, p) % 2 == 0) if fp == 1 else (len(ft(p)[1]) - 1 if fp >= 2 else 0)
        n += t
        bk.append({'p': str(p), 'reduction': str(fp), 'v_disc': str(A.val(D, p)), 'contribution': str(t)})
        tam[str(p)] = c
    small = [(p, r) for p in A.primes(2, GB) for r, _ in ft(p)[0]]
    F = {
        'format': FORMAT,
        'minimal_model': [str(x) for x in mm],
        'bad_primes': [str(p) for p in P],
        'poly': [int(c) for c in f],
        'dK': str(dK),
        'log2_dK': math.log2(abs(dK)),
        'r1': len(roots),
        'GB': GB,
        'bach': bach,
        'ns': len(small),
        'log2_size': float(lsize),
        'log2_skew': math.log2(skew),
        'real_roots': [(z + (1 << 127)) >> 128 for z in roots],
        'J': [[p, p] for p in Field(f).lcp if A.val(f[3], p) % 2],
        'J_large': [],
        'ramified': [{'p': str(p), 'all_degree_1': all(x == 1 for x in ft(p)[1]), 'labels': [[r, e] for r, e in ft(p)[0]]}
                     for p in sorted(set([2, 3] + P)) if dK % p == 0],
        'u': 2 if D > 0 else 1,
        'n': n,
        'bk_primes': bk,
        'w': cv.rootno(mm, P),
        'sign_disc': 1 if D > 0 else -1,
        'torsion': [int(x) for x in cv.torsion_group(mm)[0]],
        'tamagawa': tam,
    }
    F['conductor'] = conductor(F)
    return F, small


def conductor(F):
    """Return the conductor from the field data."""
    N = 1
    for b in F['bk_primes']:
        N *= int(b['p']) ** int(b['reduction'])
    return str(N)


def target(F, r):
    """Return g* = r - u - n, the least dim Cl(K)[2] that rank >= r allows, or None without r."""
    return None if r is None else r - F['u'] - F['n']


def goal(F):
    """Return the largest G that gives the least possible S (S = r needs w = (-1)^r)."""
    r, u, n, w = F.get('rank_lower_bound'), F['u'], F['n'], F['w']
    best = parity_bound(0, u, n, w)
    if r is not None:
        best = max(best, r if w == (-1) ** r else r + 1)
    return best + 1 - u - n


def fingerprint(ainvs, bad_primes, r, name):
    """Return the fingerprint of the artifact 'field'."""
    return runtime.fingerprint(format=FORMAT, ainvs=[str(a) for a in ainvs], bad_primes=sorted(str(p) for p in bad_primes),
                               r=r, name=name)


def prepare(state, ainvs, bad_primes, r=None, name=None, minimal_model=None):
    """Commit the artifact 'field' unless it is already valid; return the field data.

    Raise ValueError if minimal_model is wrong or r exceeds ns + u + n."""
    name = check_name(name or os.path.basename(os.path.abspath(state)))
    fp = fingerprint(ainvs, bad_primes, r, name)
    if runtime.load(state, 'field', FORMAT, fp) is not None:
        return load(state)
    F, small = compute(ainvs, bad_primes)
    if minimal_model is not None and [str(x) for x in minimal_model] != F['minimal_model']:
        raise ValueError('The minimal_model given for %s is wrong; it should be [%s]. Correct it or leave it out.' % (
            name, ','.join(F['minimal_model'])))
    if r is not None and r > F['ns'] + F['u'] + F['n']:
        raise ValueError('r = %d is impossible for this curve: under GRH its rank is at most %d. Check r.' % (
            r, F['ns'] + F['u'] + F['n']))
    F['name'] = name
    F['input_ainvs'] = [str(a) for a in ainvs]
    F['rank_lower_bound'] = r
    F['g_target'] = target(F, r)
    with runtime.transaction(state, 'field', FORMAT, fp, {'ns': F['ns']}) as tx:
        tx.write('small.txt', ''.join('%d %d\n' % s for s in small))
        tx.write('field.json', json.dumps(F, indent=1) + '\n')
    return F


def load(state):
    return read_json(os.path.join(state, 'field.json'))


def write_split(poly, pmax, out, extra=()):
    """Write to out, in makefb's format, the primes below pmax or in extra where F splits completely."""
    extra = sorted({int(p) for p in extra if int(p) >= pmax})
    f = [int(c) for c in poly]
    tmp = '%s.tmp%d.gz' % (out, os.getpid())
    with gzip.open(tmp, 'wt') as fo:
        fo.write('# primes below %d (and %d larger ones) where F splits completely (rankbound, field.write_split)\n' % (
            pmax, len(extra)))
        for p in A.primes(2, pmax) + extra:
            if any(c % p for c in f):
                rs = roots_p1(f, p)
                if len(rs) == 3:
                    fo.write('%d: %s\n' % (p, ','.join(str(r) for r, _ in rs)))
    os.replace(tmp, out)
    return out


def load_small(state):
    with open(os.path.join(state, 'small.txt')) as f:
        return [tuple(map(int, l.split())) for l in f]


def norm(poly, a, b):
    c0, c1, c2, c3 = poly
    return ((c3 * a + c2 * b) * a + c1 * b * b) * a + c0 * b ** 3


# ========== cubic fields
# K = Q(alpha), f = F(x, 1) = [c0, c1, c2, c3] the index form, O_K = <1, c3 alpha, c3 alpha^2 + c2 alpha>.  Elements
# are 3-tuples on 1, alpha, alpha^2 ("power" coordinates) or on that basis ("O" coordinates).

# ---------- Q[x]/(g), g a cubic

def pmul(g, x, y):
    """Return x y in Q[x]/(g), g = [g0, g1, g2, g3]."""
    z = [0] * 5
    for i in range(3):
        for j in range(3):
            z[i + j] += x[i] * y[j]
    for k in (4, 3):
        c = Fraction(z[k]) / g[3]
        for i in range(4):
            z[k - 3 + i] -= c * g[i]
    return tuple(z[:3])


def pinv(g, x):
    """Return 1/x in Q[x]/(g)."""
    cols = [pmul(g, x, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    return A.solve3([[cols[j][i] for j in range(3)] for i in range(3)], (1, 0, 0))


# ---------- the maximal order

def _apply(F, M):
    """Return F(m11 x + m12 y, m21 x + m22 y), F = (a, b, c, d) from x^3 to y^3."""
    (m11, m12), (m21, m22) = M
    X, Y = [m12, m11], [m22, m21]               # x and y in t = X/Y, low degree first
    out = [0, 0, 0, 0]
    for k, c in enumerate(F):
        p = [1]
        for _ in range(3 - k):
            p = _pm(p, X)
        for _ in range(k):
            p = _pm(p, Y)
        for i, v in enumerate(p):
            out[i] += c * v
    return tuple(out[::-1])


def _pm(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def disc(F):
    a, b, c, d = F
    return b * b * c * c - 4 * a * c ** 3 - 4 * b ** 3 * d - 27 * a * a * d * d + 18 * a * b * c * d


def multiple_root(F, p):
    """Return (r, multiplicity) of a multiple root of F mod p, or None."""
    f = [F[3], F[2], F[1], F[0]]
    for r, m in roots_p1(f, p):
        if m > 1:
            return r, m
    return None


def roots_p1(f, p):
    """Return [(r, multiplicity)] of the zeros in P^1(F_p) of the form with F(x, 1) = f; r = p is (1:0), last."""
    g = A.pmod(f, p)
    out = []
    for r in A.roots_mod(g, p):
        m, h = 0, g
        while len(h) > 1 and A.peval(h, r, p) == 0:
            h = A.pdivmod(h, [-r, 1], p)[0]
            m += 1
        out.append((r, m))
    if len(g) < 4:
        out.append((p, 4 - len(g)))
    return out


def maximal_form(T, primes):
    """Return (an index form F of O_K for K = Q[x]/(T), a root of F(x, 1)); primes must hold every p^2 | disc(T).

    An overring of index p exists iff a multiple root of F mod p at (1:0) gives p^2 | a [Belabas 1997]."""
    F = (T[3], T[2], T[1], T[0])
    rho = (0, 1, 0)
    for p in sorted(set(primes)):
        while disc(F) % (p * p) == 0:
            if all(c % p == 0 for c in F):
                F = tuple(c // p for c in F)
                continue
            mr = multiple_root(F, p)
            if mr is None:
                break
            r = mr[0]
            if r != p:                          # (r:1) to (1:0): F(r x + y, x), rho -> 1/(rho - r)
                F = _apply(F, ((r, 1), (1, 0)))
                rho = pinv(T, (rho[0] - r, rho[1], rho[2]))
            if F[0] % (p * p):
                break
            F = (F[0] // (p * p), F[1] // p, F[2], F[3] * p)
            rho = tuple(p * c for c in rho)
    return F, rho


# ---------- GL2(Z)-reduction

def _croots(F, prec=60):
    """Return the roots of F(x, 1) as (re, im) Decimals."""
    rr, cc = A.roots([F[3], F[2], F[1], F[0]], prec)
    return [(z, Decimal(0)) for z in rr] + [(re, s * im) for re, im in cc for s in (1, -1)]


def reduce_form(F):
    """Return (G, M) with G = F o M reduced, the smaller of the two weightings."""
    with localcontext() as ctx:
        ctx.prec = 60 + 2 * max(len(str(abs(c))) for c in F)         # keeps the Gram matrix positive definite
        th = _croots(F, ctx.prec)
        a, b, c, d = F
        def dF(z):                              # |F'(z, 1)|^2
            re, im = z
            r2, i2 = re * re - im * im, 2 * re * im
            return (3 * a * r2 + 2 * b * re + c) ** 2 + (3 * a * i2 + 2 * b * im) ** 2
        cands = []
        for w in ([Decimal(1)] * 3, [dF(z) ** (Decimal(-1) / 3) for z in th]):
            G = [[sum(w), -sum(wi * z[0] for wi, z in zip(w, th))], [0, sum(wi * (z[0] ** 2 + z[1] ** 2) for wi, z in zip(w, th))]]
            G[1][0] = G[0][1]
            U = A.lll_gram([[Fraction(x) for x in r] for r in G], eta=Fraction(51, 100))
            M = ((U[0][0], U[0][1]), (U[1][0], U[1][1]))
            H = _apply(F, M)
            if H[0] < 0:                        # F o (-M) = -(F o M)
                H, M = tuple(-x for x in H), tuple(tuple(-x for x in r) for r in M)
            cands.append((H, M))
    return cands[0] if size(cands[0][0])[0] <= size(cands[1][0])[0] else cands[1]


def size(F):
    """Return (log2 of the skewed size, the skew) of F."""
    cs = [F[3], F[2], F[1], F[0]]
    l = [math.log2(abs(c)) if c else None for c in cs]
    g = lambda t: max(l[i] + (i - 1.5) * t for i in range(4) if l[i] is not None)
    gb, t0 = math.inf, 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            if l[i] is not None and l[j] is not None:
                t = (l[i] - l[j]) / (j - i)
                v = g(t)
                if v < gb:
                    gb, t0 = v, t
    # g is convex: the best grid point is next to t0
    best, sk = math.inf, 1.0
    t0 = min(100.0, max(-100.0, t0))
    for k in range(max(-400, math.floor(4 * t0) - 6), min(400, math.ceil(4 * t0) + 6) + 1):
        m = g(k / 4)
        if m < best:
            best, sk = m, 2.0 ** (k / 4)
    return best, sk


# ---------- the field K = Q(alpha) and its primes

def label_text(z):
    """Return the prime label z as 'p:r' (degree 1) or 'p:kK:fF'; their order fixes the witness's lists."""
    return '%d:%d' % z if len(z) == 2 else '%d:k%d:f%d' % z


class Prime:
    """A prime P of K above p: label, phi (O_K -> F_p[t]/(g)), hnf, and tau not in p O_K with tau P in p O_K."""
    def __init__(self, K, p, e, f, r, g, k):
        self.p, self.e, self.f, self.r, self.g = p, e, f, r, g
        self.label = (p, r) if f == 1 else (p, k, f)
        a, b, c = K.F[0], K.F[1], K.F[2]
        if f == 1:
            self.phi = [[1], [a * r % p], [(a * r * r + b * r) % p]] if r < p else [[1], [-b % p], [-c % p]]
        else:
            self.phi = [[1], A.pmod([0, a], p), A.pdivmod(A.pmod([0, b, a], p), g, p)[1]]
        self.q = p ** f
        # P = ker(O_K -> F_q), which contains p O_K
        rows = [[p if i == j else 0 for j in range(3)] for i in range(3)]
        cols = [self.phi[i] + [0] * (f - len(self.phi[i])) for i in range(3)]
        for v in A._kernel_modp([[cols[i][j] for i in range(3)] for j in range(f)], 3, p):
            rows.append(v)
        self.hnf = A.hnf(rows)
        # tau h in p O_K for each basis element h of P
        eqs = []
        for h in self.hnf:
            Mh = K.omat(h)
            eqs += [[Mh[r][i] for i in range(3)] for r in range(3)]
        tau = A._kernel_modp(eqs, 3, p)
        self.tau = next(v for v in tau if any(x % p for x in v))

    def red(self, x):
        """Return the image in F_q of x in O_K."""
        z = [0] * self.f
        for xi, ph in zip(x, self.phi):
            for j, c in enumerate(ph):
                z[j] += xi * c
        return A.pmod(z, self.p)


def lat_reduce(x, H):
    """Return the canonical residue of x modulo the HNF lattice H."""
    x = list(x)
    for i in range(3):
        q = x[i] // H[i][i]
        x = [a - q * b for a, b in zip(x, H[i])]
    return tuple(x)


class Field:
    """K = Q(alpha), alpha a root of the index form f."""

    def __init__(self, f):
        self.f = [int(c) for c in f]
        c0, c1, c2, c3 = self.f
        self.F = (c3, c2, c1, c0)               # (a, b, c, d)
        if not c3:
            raise ValueError('not a cubic form: its x^3 coefficient is 0')
        B = [(1, 0, 0), (0, c3, 0), (0, c2, c3)]        # the basis of O_K
        self.B = B
        # w1 = a alpha, w2 = a alpha^2 + b alpha:
        #   w1^2 = -b w1 + a w2,   w1 w2 = -a d - c w1,   w2^2 = -b d - d w1 - c w2
        a, b, c, d = self.F
        self.T = [[(1, 0, 0), (0, 1, 0), (0, 0, 1)],
                  [(0, 1, 0), (0, -b, a), (-a * d, -c, 0)],
                  [(0, 0, 1), (-a * d, -c, 0), (-b * d, -d, -c)]]
        self.dec_cache, self.pcache = {}, {}

    @functools.cached_property
    def lcp(self):
        """Return the prime divisors of c3 (lazily: factoring may be costly)."""
        c3 = self.f[3]
        return A.prime_divisors(abs(c3)) if abs(c3) > 1 else []

    def ocoords(self, x):
        """Return the O coordinates of x from its power coordinates."""
        c0, c1, c2, c3 = self.f
        n3 = Fraction(x[2]) / c3
        n2 = (Fraction(x[1]) - n3 * c2) / c3
        out = (Fraction(x[0]), n2, n3)
        return tuple(int(v) if v.denominator == 1 else v for v in out)

    def power(self, n):
        """Return the power coordinates of n from its O coordinates."""
        return (n[0], n[1] * self.f[3] + n[2] * self.f[2], n[2] * self.f[3])

    def omul(self, x, y):
        z = [0, 0, 0]
        for i in range(3):
            if x[i]:
                for j in range(3):
                    if y[j]:
                        c = x[i] * y[j]
                        t = self.T[i][j]
                        z[0] += c * t[0]; z[1] += c * t[1]; z[2] += c * t[2]
        return tuple(z)

    def omat(self, x):
        """Return the matrix of multiplication by x; column j is x b_j."""
        u, v, w = x
        a, b, c, d = self.F
        return [[u, -a * d * w, -a * d * v - b * d * w],
                [v, u - b * v - c * w, -c * v - d * w],
                [w, a * v, u - c * w]]

    def norm(self, x):
        """Return N(x) = det omat(x) for x in O coordinates."""
        (a, b, c), (d, e, f), (g, h, i) = self.omat(x)
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    def integral(self, x):
        """Return (y, d), y in O_K and d > 0 least with x = y / d."""
        n = self.ocoords(x)
        d = 1
        for v in n:
            if isinstance(v, Fraction):
                d = d * v.denominator // math.gcd(d, v.denominator)
        return tuple(int(v * d) for v in n), d

    def dec(self, p):
        """Return the primes above p, degree 1 first (in the order of roots_p1)."""
        if p not in self.dec_cache:
            pts = roots_p1(self.f, p)
            out = [Prime(self, p, m, 1, r, None, 0) for r, m in pts]
            rest = 3 - sum(m for _, m in pts)
            if rest:
                g = A.pmod(self.f, p)
                for r, m in pts:
                    if r < p:
                        for _ in range(m):
                            g = A.pdivmod(g, [-r, 1], p)[0]
                inv = pow(g[-1], -1, p)
                g = [c * inv % p for c in g]
                out.append(Prime(self, p, 1, rest, None, g, 2 if pts else 1))
            self.dec_cache[p] = out
        return self.dec_cache[p]

    def prime(self, p, label):
        """Return the prime above p with this label.

        For an unramified degree-1 label prime to lc(F), only (p, q, label, tau) with tau = (f(x) / (x - r))(alpha),
        which vanishes at the other primes above p only."""
        if (p, label) not in self.pcache:
            c0, c1, c2, c3 = self.f
            if p not in self.dec_cache and len(label) == 2 and disc(self.F) % p and c3 % p:
                r = label[1]
                tau = ((c1 + c2 * r + c3 * r * r) % p, r, 1)
                self.pcache[p, label] = SimpleNamespace(p=p, q=p, label=label, tau=tau)
            else:
                self.pcache[p, label] = next(P for P in self.dec(p) if P.label == label)
        return self.pcache[p, label]

    def val_int(self, x, P):
        """Return (v_P(x), x tau^v / p^v) for x != 0 in O_K."""
        v, p = 0, P.p
        while True:
            y = self.omul(x, P.tau)
            if any(c % p for c in y):
                return v, x
            x, v = tuple(c // p for c in y), v + 1

    def val(self, x, P):
        """Return v_P(x) for x != 0 in power coordinates."""
        y, d = self.integral(x)
        return self.val_int(y, P)[0] - P.e * A.val(d, P.p)


_FIELDS = {}


def field_of(ainvs, primes):
    """Return (the reduced index form of the field of W, a root of W), W the 2-division cubic; as maximal_form."""
    key = tuple(ainvs)
    if key not in _FIELDS:
        _FIELDS.clear()
        _FIELDS[key] = _field_of(ainvs, primes)
    return _FIELDS[key]


def _field_of(ainvs, primes):
    from .curve import binv
    b2, b4, b6, _ = binv(ainvs)
    T = [16 * b6, 8 * b4, b2, 1]
    F, rho = maximal_form(T, primes)
    G, M = reduce_form(F)
    (m11, m12), (m21, m22) = M
    alpha = pmul(T, (m22 * rho[0] - m12, m22 * rho[1], m22 * rho[2]), pinv(T, (m11 - m21 * rho[0], -m21 * rho[1], -m21 * rho[2])))
    a2 = pmul(T, alpha, alpha)
    theta = A.solve3([[1, alpha[0], a2[0]], [0, alpha[1], a2[1]], [0, alpha[2], a2[2]]], (0, 1, 0))
    return [G[3], G[2], G[1], G[0]], theta


_FORM = {}


def form_field(f):
    """Return the Field of f, cached so that callers share its prime decompositions."""
    key = tuple(int(c) for c in f)
    if key not in _FORM:
        _FORM.clear()
        _FORM[key] = Field(key)
    return _FORM[key]


def relation_el(r):
    """Return the element of the relation r ([0, a, b], [1, c0, c1, c2, d] or [2, p])."""
    if r[0] == 0:
        return (r[1], -r[2], 0)
    if r[0] == 1:
        return tuple(Fraction(c, r[4]) for c in r[1:4])
    return (r[1], 0, 0)
