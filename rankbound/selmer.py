"""The bound rank E(Q) <= dim Sel_2(E) for E(Q)[2] = 0, and its witness.

K = Q(alpha) is the cubic field of the index form F.  Relations come from the sieve (sieve_witness, under the GRH)
or, unconditionally, from SPLIT up to Minkowski's bound (prove).  Selmer and Job are the work folders of the
routes SPLIT and SIEVE.
"""
import bisect, concurrent.futures, datetime, itertools, json, math, multiprocessing, os, random, time
from decimal import Decimal, localcontext
from fractions import Fraction
from . import ctp, curve as cv, field as fld, linalg, sieve
from .core import FactorisationNeeded, ProofData, RankboundError, Stopped, TimeLimit, parity_bound, parity_floor
from . import arith as A
from .messages import about, big, core_hours, duration, plural, text
from . import runtime
from .runtime import DETAIL, Folder, read_json, say, say_curve, say_lower
from .witness import FORMAT6, statement

LOG = DETAIL.info
METHOD = '2-Selmer group'


# ---------- time: the current budget, polled by the loops (arith.tick)

def run_py(fn, *args, seconds=3600):
    """Return fn(*args) within the current budget and seconds."""
    try:
        with runtime.limited(seconds=seconds, error=A.time_up):
            return fn(*args)
    except A.TimeUp as e:
        runtime.raise_if_stopped()
        raise TimeLimit(str(e), seconds)

HERE = os.path.dirname(os.path.abspath(__file__))
PI40 = Fraction(31415926535897932384626433832795028841971, 10 ** 40)    # pi truncated to 40 digits (check/check.gp)
MINK_AUTO = 3 * 10 ** 6         # the Minkowski bound SPLIT reaches in about MINK_SECONDS on one core
MINK_SECONDS = 600
UNCOND_HOURS = 3                # uncond: refused when the estimate is above this many hours on the given cores
FB_MIN = 30                     # the factor base bound B0 is at least this (or B)
NQ = 40                         # character primes beyond U
ROUNDS = 10                     # rounds of the a,b search, each on further shells, until v = U
SHELLS, GROW = 300, 1.6         # the first round's shells max(|a|, b), and the growth of the later ones
LP = 30                         # the large prime bound of the a,b relations, times B0 (when B0 = B)


# ---------- the primes S

REDUCE_S = True         # leave out of S the odd bad primes with c_p odd (False: every bad prime)


def s_primes(F):
    """Return (S, dropped): the sorted primes of the descent, and [[p, c_p], ...] for the bad primes left out.

    An odd bad prime with odd c_p can be left out: its local image is unramified (Schaefer-Stoll).
    """
    S = sorted({2} | {int(p) for p in F['bad_primes']})
    if not REDUCE_S:
        return S, []
    if 'tamagawa' in F:
        cp = {int(p): c for p, c in F['tamagawa'].items()}
    else:
        cp = {p: cv.localred([int(a) for a in F['minimal_model']], p)[2] for p in S if p > 2}
    dropped = [[p, cp[p]] for p in S if p > 2 and cp[p] % 2]
    return [p for p in S if p == 2 or cp[p] % 2 == 0], dropped


# ---------- the bound B

def uncond_seconds(Bm, cores=1):
    """Return the estimated seconds of SPLIT up to Bm on cores; the cost is about linear in the bound."""
    return MINK_SECONDS * Bm / (MINK_AUTO * max(1, cores))


def uncond_estimate(curve, cores=1):
    """Return (Minkowski's bound B, log10 |d_K|, estimated seconds of SPLIT on cores) for curve."""
    dK = field_disc(curve)
    Bm = minkowski(dK, 1 if dK < 0 else 0)
    return Bm, len(str(abs(dK))) - 1, uncond_seconds(Bm, cores)


def minkowski(dK, r2):
    """Return an integer B above Minkowski's bound; pi40 <= pi keeps it an upper bound."""
    m2 = (16 / PI40 ** 2) ** r2 * Fraction(4, 81) * abs(int(dK))         # the square of the upper bound
    return math.isqrt(m2.numerator // m2.denominator) + 1


# ---------- F_2 linear algebra: f2elim for the relation matrices, Python ints (bit i: coordinate i) for small ones

def rank(rows):
    return len(echelon(rows))


def xor(vals, idx):
    """Return the XOR of vals[i] over the indices i in idx."""
    s = 0
    for i in idx:
        s ^= vals[i]
    return s


def reduce_mod(x, basis):
    """Return x reduced modulo the echelon basis {leading bit: vector}, zero exactly on the span."""
    for h in sorted(basis, reverse=True):
        if (x >> h) & 1:
            x ^= basis[h]
    return x


def echelon(vecs):
    piv = {}
    for v in vecs:
        x = reduce_mod(v, piv)
        if x:
            piv[x.bit_length() - 1] = x
    return piv


def kernel(vecs):
    """Return a basis of the combinations (ints, bit i for vecs[i]) whose XOR of vecs is 0."""
    piv, out = {}, []
    for i, v in enumerate(vecs):
        x, c = v, 1 << i
        for h in sorted(piv, reverse=True):
            if (x >> h) & 1:
                x ^= piv[h][0]
                c ^= piv[h][1]
        if x:
            piv[x.bit_length() - 1] = (x, c)
        else:
            out.append(c)
    return out


def small_pivots(state, ops, piv, nsmall):
    """Return the combinations (bit t for piv[t]) of pivot rows without large columns: a basis of V."""
    L, Sm = {}, {}

    def st(i):
        if i not in L:
            L[i], Sm[i] = state(i)
    for dst, src in ops:
        st(dst)
        st(src)
        L[dst] = L[dst] ^ L[src]
        Sm[dst] ^= Sm[src]
    for i in piv:
        st(i)
    npv = len(piv)
    pos = {c: t for t, c in enumerate(sorted(set().union(*(L[i] for i in piv))))}
    base, comb = {}, []
    for t, i in enumerate(piv):         # large columns on top, then small ones, then the pivot's own bit
        x = (((sum(1 << pos[c] for c in L[i]) << nsmall) | Sm[i]) << npv) | (1 << t)
        while True:
            h = x.bit_length() - 1
            if h < npv:
                raise RuntimeError('selmer: the pivot rows of f2elim are dependent: a bug')
            if h not in base:
                base[h] = x
                if h < npv + nsmall:
                    comb.append(x & ((1 << npv) - 1))
                break
            x ^= base[h]
    return comb


# ---------- the relation lines of a witness, and the local data of the 2-descent (selmer.gp's m1head, m1base)

def rel_text(r, H):
    """Return the witness line of the relation r with norm primes H."""
    hx = ','.join('%x' % p for p in H)
    if r[0] == 0:
        return '%d,%d:%s' % (r[1], r[2], hx)
    if r[0] == 1:
        return 'e:%d,%d,%d,%d:%s' % (r[1], r[2], r[3], r[4], hx)
    return 'f:%d' % r[1]


def purge(d, B, S):
    """Remove in place the relations of d with a large prime no other relation has."""
    big = lambda z: is_deg1(z) and z[0] >= B and z[0] not in S
    keep = list(range(len(d['rels'])))
    while True:
        wt = {}
        for i in keep:
            for z in d['rels'][i][2]:
                if big(z):
                    wt[z] = wt.get(z, 0) + 1
        new = [i for i in keep if all(wt.get(z, 2) > 1 for z in d['rels'][i][2] if big(z))]
        if len(new) == len(keep):
            break
        keep = new
    d['rels'] = [d['rels'][i] for i in keep]
    d['ch'] = [d['ch'][i] for i in keep]


def local_setup(F, P, S, B0, nq, L=0, seconds=3600):
    """Return the setup the rounds of the a,b search share, computed once."""
    return run_py(lambda *a: sel_local(*a)[0], [int(a) for a in F['minimal_model']], F['poly'], [int(p) for p in P],
                  B0, 3, SMAX, nq, L, S, seconds=seconds)


def shells(F, P, S, B0, need, slo, shi, L=0, seconds=3600):
    """Return a round's a,b relations from the shells slo..shi, their number and the last shell."""
    return run_py(lambda: fb_shells(kfield(F['poly'], [int(p) for p in P]), S, B0, need, slo, shi, L), seconds=seconds)


def head(F, P, S):
    """Return selmer.gp's m1head as a record: theta, W, the signature, the real roots and the places of S."""
    return sel_head([int(a) for a in F['minimal_model']], F['poly'], [int(p) for p in P], S)[0]


def base(F, P, S, qmin, nq, pmin, lim, seconds=3600):
    """Return selmer.gp's m1sieve as a record."""
    return run_py(sel_sieve, [int(a) for a in F['minimal_model']], F['poly'], [int(p) for p in P], qmin,
                  nq, pmin, lim, NSPLIT, SMAX_S, S, seconds=seconds)


# ---------- the characters and local classes of the relations, as selmer.gp's m1sgn, m1chi and m1cls1 define them:
# in Python where that is exact and quick (tests/test_selmer.py checks them against GP), else by m1batch

def nonresidue(n, p):
    """Return whether n, prime to the odd prime p, is not a square mod p."""
    n %= p
    if not n:
        raise RuntimeError('selmer: %d divides a value whose residue symbol was asked for' % p)
    return pow(n, (p - 1) // 2, p) != 1


def batch(F, S, rels, items, chars):
    """Return selmer.gp's m1batch records (j, place or -1, bits) for items [j, i, places or -1] of rels."""
    if not items:
        return []
    return sel_batch(F['poly'], F['bad_primes'], rels, [[j, i + 1, t] for j, i, t in items], chars, S)


_NR = {}


def _nonres_table(q):
    """Return a cached table mod the odd prime q: 1 at non-residues, 0 at nonzero squares, 2 at 0."""
    if q not in _NR:
        t = bytearray([1]) * q
        t[0] = 2
        for x in range(1, (q + 1) // 2):
            t[x * x % q] = 0
        _NR[q] = bytes(t)
    return _NR[q]


class Work:
    """The function and chunks of one par_map, inherited by forked workers so nothing is pickled per task."""

    def __init__(self, f, chunks):
        self.f, self.chunks = f, chunks

    def run(self, k):
        return self.f(self.chunks[k])


_WORK = None                    # the running par_map's Work


def _par_run(k):
    return _WORK.run(k)


def par_map(f, items, jobs, least=20000):
    """Return f(items), for f mapping a list to an equally long list, in jobs forked processes when large."""
    global _WORK
    if jobs <= 1 or len(items) < least:
        return f(items)
    n = 4 * jobs
    chunks = [items[i * len(items) // n:(i + 1) * len(items) // n] for i in range(n)]
    old, _WORK = _WORK, Work(f, chunks)
    try:
        with multiprocessing.get_context('fork').Pool(jobs) as p:
            parts = p.map(_par_run, range(n), chunksize=1)
    finally:
        _WORK = old
    return [x for part in parts for x in part]


def characters(d, rels, F):
    """Return each relation's real signs (bits 0..nr-1) and characters (bits nr..) as an int."""
    M, R = d['realq']
    sh = 1 << M
    nr = len(R)
    Q = d['chars']
    nonres = [_nonres_table(q) if q < 1 << 16 else None for q, _ in Q]
    out, todo = [], []
    for j, (r, H, _) in enumerate(rels):
        if r[0] == 0:
            a, b = r[1], r[2]
            x = 0
            for i, N in enumerate(R):
                lo, hi = a * sh - b * N, a * sh - b * (N + 1)
                if (lo < 0) != (hi < 0) or not lo or not hi:
                    raise RuntimeError('selmer: the sign of relation %d,%d at a real place needs more precision' % (a, b))
                x |= (lo < 0) << i
            for t, (q, c) in enumerate(Q):
                if nonres[t] is None:
                    x |= nonresidue(a - b * c, q) << (nr + t)
                    continue
                v = nonres[t][(a - b * c) % q]
                if v == 2:
                    raise RuntimeError('selmer: %d divides a value whose residue symbol was asked for' % q)
                x |= v << (nr + t)
        elif r[0] == 2:
            x = 0
            for t, (q, c) in enumerate(Q):
                x |= nonresidue(r[1], q) << (nr + t)
        else:
            x = None
            todo.append(j)
        out.append(x)
    if todo:
        for j, _, x in batch(F, d['S'], [rels[j][0] for j in todo], [[j, k, -1] for k, j in enumerate(todo)], Q):
            out[j] = x
    return out


class Places:
    """The places of K above S, in m1places order, with their widths and selmer.gp's place data."""

    def __init__(self, d, F):
        self.pld, self.S = d['pld'], d['S']
        self.width = [w for _, w in d['cls']]
        self.off = [sum(self.width[:i]) for i in range(len(self.width))]
        self.poly = [int(c) for c in F['poly']]
        self.rho = {}               # place -> (k, the root of its cubic mod p^k)

    def form(self, i):
        """Return the cubic with root al at place i: F(x, 1), or F(t x - 1, x) when the place has t."""
        t = self.pld[i]['t']
        if t is None:
            return self.poly
        c0, c1, c2, c3 = self.poly       # F(t x - 1, x) = sum c_k (t x - 1)^k x^(3 - k)
        out = [0, 0, 0, 0]
        for k, c in enumerate(self.poly):
            for j in range(k + 1):          # (t x - 1)^k = sum_j C(k, j) t^j x^j (-1)^(k - j)
                out[j + 3 - k] += c * math.comb(k, j) * t ** j * (-1) ** (k - j)
        return out

    def root(self, i, k):
        """Return the root mod p^k of the cubic at the unramified degree-1 place i."""
        pl = self.pld[i]
        p = pl['p']
        have = self.rho.get(i)
        if have and have[0] >= k:
            return have[1] % p ** k
        c0, c1, c2, c3 = self.form(i)
        x, n = (have[1], have[0]) if have else ((-pl['g'][0]) % p, 1)
        while n < k:
            n = min(2 * n, k)
            m = p ** n
            fx = ((c3 * x + c2) * x + c1) * x + c0
            dfx = (3 * c3 * x + 2 * c2) * x + c1
            x = (x - fx * pow(dfx, -1, m)) % m
        self.rho[i] = (n, x)
        return x

    def one(self, i, r, H):
        """Return the class of the relation r at place i as m1cls1's bits, or None to ask GP."""
        pl = self.pld[i]
        p, e, f, tab, g = pl['p'], pl['e'], pl['f'], pl['tab'], pl['g']
        if r[0] == 2:
            q = r[1]
            if q == p:
                return None
            if p == 2:
                return tab[q % 8]               # the integer q mod 8 (b0 = 0)
            return (f % 2 and nonresidue(q, p)) << 1
        if r[0] != 0:
            return None
        a, b = r[1], r[2]
        if pl['t'] is not None:             # a - b al = (a' - b' al') (t - al)
            a, b, c = b, pl['t'] * b - a, pl['ct']
        else:
            c = 0
        d = len(g) - 1
        inP = d == 1 and (a + b * g[0]) % p == 0          # g = x - r: a - b r = a + b g0
        if not inP:
            if p == 2:
                x = tab[8 * (b % 8) + a % 8]
                return None if x is None else x ^ c
            N = sum(co * a ** k * b ** (d - k) for k, co in enumerate(g))
            return ((f // d) % 2 and nonresidue(N, p)) << 1 ^ c
        if e > 1:                           # ramified, a - b al in pr, small valuations
            if not pl['rtab']:
                return None
            M, rt = pl['rtab']
            m = p ** M
            x = rt[m * (b % m) + a % m]
            return None if x is None else x ^ c
        if f > 1:
            return None
        # K_P = Q_p, al -> rho: the class of a - b rho is v (the class of p) plus the unit's
        k = H.count(p) + 8
        x = (a - b * self.root(i, k)) % p ** k
        if not x:
            return None
        v = 0
        while x % p == 0:
            x //= p
            v += 1
        if v + (3 if p == 2 else 1) > k:
            return None
        if p == 2:
            u = tab[x % 8]
            return (u ^ pl['cp'] if v % 2 else u) ^ c
        return ((v % 2) | ((nonresidue(x, p) ^ (v % 2 and pl['cp'] >> 1 & 1)) << 1)) ^ c

    def classes(self, rels, idx, F, chars):
        """Return {j: the classes of rels[j] at every place, place i at bit off[i]} for j in idx."""
        out, todo = {}, {}
        for j in idx:
            r, H, _ = rels[j]
            x = 0
            for i in range(len(self.pld)):
                c = self.one(i, r, H)
                if c is None:
                    todo.setdefault(j, []).append(i)
                else:
                    x |= c << self.off[i]
            out[j] = x
        if todo:
            js = sorted(todo)
            for j, i, x in batch(F, self.S, [rels[j][0] for j in js], [[j, k, todo[j]] for k, j in enumerate(js)], chars):
                out[j] |= x << self.off[i]
        return out


# ---------- the algebra

def prime_factors(n):
    """Return the prime factors of the small positive integer n by trial division."""
    out, p = [], 2
    while p * p <= n:
        if n % p == 0:
            out.append(p)
            while n % p == 0:
                n //= p
        p += 1
    return out + ([n] if n > 1 else [])


def is_deg1(z):
    """Return whether the prime label z is of degree 1."""
    return len(z) == 2


def algebra(d, tri, free, ns, B, S, F, cores=1, least=None):
    """Return the claims and witness data from the setup d and the triangular relations tri.

    The matrices are eliminated untrimmed, so the dimensions are exact.
    """
    Sset = set(S)
    rels = list(d['rels'])
    nfb = len(rels)
    # the free rows of [B0, B) that a triangular row needs (it has a prime of degree > 1 above q)
    need_q = sorted({z[0] for _, _, lab in tri for z in lab if not is_deg1(z)} - {r[1] for r, _, _ in rels if r[0] == 2})
    for q in need_q:
        if q not in free:
            raise RuntimeError('selmer: no free row for %d' % q)
        rels.append(([2, q], [q], free[q]))
    rels += tri
    # columns: small = degree 1 with p < B, numbered first; then large
    cols = {}
    labs = sorted({z for _, _, lab in rels for z in lab} | set(d['fb']), key=lambda z: (not (is_deg1(z) and z[0] < B), z[0], fld.label_text(z)))
    for z in labs:
        cols[z] = len(cols)
    nsmall = sum(1 for z in labs if is_deg1(z) and z[0] < B)
    s_columns = [z for z in labs if is_deg1(z) and z[0] in Sset]
    # Each triangular row has its own top prime outside the factor base; no other row may have one, or
    # dim V = dim V_G + ntri below would count a dimension twice.
    tops = set(d['fb'])
    lcp = set(prime_factors(abs(int(F['poly'][3]))))
    for _, _, lab in tri:
        top = max(z[0] for z in lab)
        at = [z for z in lab if z[0] == top]
        if len(at) != 1 or not is_deg1(at[0]) or at[0] in tops or top in Sset or top in lcp:
            raise RuntimeError('selmer: a triangular relation is not triangular: %r' % (lab,))
        tops.add(at[0])
    ttops = tops - set(d['fb'])
    if any(z in ttops for _, _, lab in rels[:nfb] for z in lab):
        raise RuntimeError('selmer: a factor base relation meets the top of a triangular relation: a bug')
    ntri, nrel = len(tri), len(rels)
    nblk = nrel - ntri
    # Three blocks of one block-diagonal f2elim run; each block's dim V is its number of small pivots.
    # G: the relations, small columns of degree 1 with p < B.  S: the same with the unit rows of the S columns (k).
    # K: the characters (small) against the columns outside S (large); its small pivots are the elements.
    nr = len(d['real'])
    nch = nr + len(d['chars'])
    chars = d['ch'][:nfb]
    blk = [[cols[z] for z in lab] for _, _, lab in rels[:nblk]]
    used = sorted({c for r in blk for c in r} | {cols[z] for z in s_columns})
    nu, nsm = len(used), sum(1 for c in used if c < nsmall)
    loc = {c: i for i, c in enumerate(used)}            # small columns first
    NS = nch + 2 * nsm                                  # small: characters, G's, S's; then large: G's, S's, K's
    gm = {c: nch + i if i < nsm else NS + i - nsm for c, i in loc.items()}
    sm = {c: nch + nsm + i if i < nsm else NS + nu - nsm + i - nsm for c, i in loc.items()}
    kid = {}
    for z in labs:
        if z[0] not in Sset:
            kid[z] = NS + 2 * (nu - nsm) + len(kid)
    unit = [[cols[z]] for z in s_columns]
    rows = [[gm[c] for c in r] for r in blk] + [[sm[c] for c in r] for r in blk + unit]
    k0 = len(rows)
    for i in range(nfb):
        if chars[i] >> nch:
            raise RuntimeError('selmer: relation %d has more than %d characters' % (i, nch))
        rows.append([kid[z] for z in rels[i][2] if z in kid])
    E = linalg.eliminate(rows, NS + 2 * (nu - nsm) + len(kid), NS, cores, bits=[0] * k0 + chars)
    del rows
    part = {'G': ([], []), 'S': ([], []), 'K': ([], [])}
    which = lambda i: 'G' if i < nblk else 'S' if i < k0 else 'K'
    for o in E.ops:
        if which(o[0]) != which(o[1]):
            raise RuntimeError('selmer: an op of f2elim joins two blocks: a bug')
        part[which(o[0])][0].append(o)
    for i in E.pivots:
        part[which(i)][1].append(i)
    opsG, pivG = part['G']
    opsS = [(x - nblk, y - nblk) for x, y in part['S'][0]]
    pivS = [i - nblk for i in part['S'][1]]
    opsK = [(x - k0, y - k0) for x, y in part['K'][0]]
    pivK = [i - k0 for i in part['K'][1]]

    def gstate(r):          # (large columns, small ones as an int) of a row of G or S
        return frozenset(loc[c] for c in r if loc[c] >= nsm), sum(1 << loc[c] for c in r if loc[c] < nsm)
    gs = blk + unit
    dimG = len(small_pivots(lambda i: gstate(gs[i]), opsG, pivG, nsm))
    dimS = len(small_pivots(lambda i: gstate(gs[i]), opsS, pivS, nsm))
    comb = small_pivots(lambda i: (frozenset(kid[z] for z in rels[i][2] if z in kid), chars[i]), opsK, pivK, nch)
    v = len(comb)
    if dimG + dimS + v != E.dimV:
        raise RuntimeError('selmer: the small pivots (%d, %d, %d) do not add up to f2elim\'s dim V %d: a bug' % (
            dimG, dimS, v, E.dimV))
    # The pivots are f2elim's, every triangular row and the free rows Q its primes of degree > 1 need.  Each triangular
    # row plus its free rows has no large column and keeps its top, so if the ops leave Q unchanged the pivots span
    # dim V_G + ntri.  Otherwise every row is a pivot, with no ops.
    freerow = {r[1]: i for i, (r, _, _) in enumerate(rels[:nblk]) if r[0] == 2}
    Q, tidy = set(), True
    for _, _, lab in tri:
        out = set(lab)
        for q in {z[0] for z in lab if not is_deg1(z)}:
            Q.add(freerow[q])
            out ^= set(rels[freerow[q]][2])
        tidy = tidy and all(is_deg1(z) and z[0] < B for z in out)
    tail = Q | set(range(nblk, nrel))
    shift = lambda i: i + ntri if i >= nblk else i         # the unit rows follow every relation
    if tidy and not any(o[0] in Q for o in opsG):
        ops, pivots = [list(o) for o in opsG], sorted(set(pivG) | tail)
    else:
        ops, pivots = [], list(range(nrel))
    if tidy and not any(o[0] in Q for o in opsS):
        ops_s, pivots_s = [[shift(i) for i in o] for o in opsS], sorted({shift(i) for i in pivS} | tail)
    else:
        ops_s, pivots_s = [], list(range(nrel + len(s_columns)))
    dimV, dimV_s = dimG + ntri, dimS + ntri
    G, k = ns - dimV, ns - dimV_s
    U = d['r1'] + d['r2'] + d['SK'] + k
    data = dict(rels=rels, nfb=nfb, ntri=ntri, cols=cols, s_columns=s_columns, ops=ops, pivots=pivots, ops_s=ops_s,
                pivots_s=pivots_s)
    if least is not None and norm_bound(0, U, len(S), 0, F['w']) <= least:
        return dict(no_elements(G, dimV, k, U, len(S), F), **data)
    els = elements_of([[pivK[j] for j in A.bits(cb)] for cb in comb], [x for o in opsK for x in o])
    for e in els:
        out = set()
        for j in e:
            out ^= set(rels[j][2])
        if not e or any(z[0] not in Sset for z in out):
            raise RuntimeError('selmer: an element has an odd valuation outside S: a bug')
    return dict(selmer_part(d, rels, els, chars, F, S, G, dimV, k, U), **data)


def xrel_row(labs, col, Sset, kid, base):
    """Return the kernel-matrix columns of an xrel element with odd-valuation labels labs."""
    row = set()
    for z in labs or ():
        if z[0] in Sset:
            continue
        c = col.get(z)
        row.add(kid[c] if c in kid else kid.setdefault(z, base + len(kid)))
    return sorted(row)


class TagBits:
    """The small columns of kernel_rows as f2elim's bits: seeded random tags, and one column per xrel row."""

    def __init__(self, T, nx, n):
        self.T, self.nx, self.n = T, nx, n

    def __len__(self):
        return self.n

    def __iter__(self):
        rnd, T, nx = random.Random(1), self.T, self.nx
        for t in range(self.n):
            yield rnd.getrandbits(T) | ((1 << (T + t)) if t < nx else 0)


class Kid:
    """The kernel matrix's columns: M's outside S, then the xrel rows' own prime ideals, without a table."""

    def __init__(self, scols, ncol, base):
        self.s, self.sset, self.ncol, self.base, self.own = list(scols), set(scols), ncol, base, {}
        self.top = self.s[-1] if self.s else -1

    def __contains__(self, c):
        return type(c) is int and 0 <= c < self.ncol and c not in self.sset

    def __getitem__(self, c):
        if c not in self:
            return self.own[c]
        return self.base + c - (len(self.s) if c > self.top else bisect.bisect_left(self.s, c))

    def setdefault(self, z, v):
        return self.own.setdefault(z, v)

    def __len__(self):
        return self.ncol - len(self.s) + len(self.own)

    def row(self, r):
        """Return the kernel matrix's row of M's row r: its columns outside S, in order."""
        s, top, ls, base, sset = self.s, self.top, len(self.s), self.base, self.sset
        return [base + c - (ls if c > top else bisect.bisect_left(s, c)) for c in r if c not in sset]


class KernelRows:
    """The kernel matrix: the xrel rows in memory, then M's rows outside S, read from disk on demand."""

    def __init__(self, xr, rows, nrows, kid):
        self.xr, self.rows, self.nrows, self.kid = xr, rows, nrows, kid

    def __len__(self):
        return len(self.xr) + self.nrows

    def __iter__(self):
        yield from self.xr
        for t, r in enumerate(self.rows):
            if t == self.nrows:
                break
            yield self.kid.row(r)

    def __getitem__(self, t):
        return self.xr[t] if t < len(self.xr) else self.kid.row(self.rows[t - len(self.xr)])


def kernel_rows(rk, nx, T, ncols, threads=1, keep=None, seed=1, work=None):
    """Return (combinations of rk whose large columns cancel, whether they may be partial).

    Random small tags keep rows from vanishing in the merge; each xrel row has its own small column, so the trim never
    removes it.
    """
    bits = TagBits(T, nx, len(rk))
    Kr = linalg.kernel(rk, ncols, T + nx, threads, bits=bits, keep=T if keep is None else keep, seed=seed, work=work)
    # partial: Lanczos's vectors of a larger kernel, or only the pool's empty rows; the caller tries the next seed
    part = (' method lanczos ' in Kr.info and int(Kr.info.split(' vectors ')[1].split()[0]) >= 48) or \
        Kr.info.rstrip().endswith(' status partial')
    return elements_of(Kr.vectors, Kr.ops), part


def elements_of(vecs, ops):
    """Return the elements of the combinations vecs of rows after the flat ops (dst, src, ...)."""
    mask = {}                           # row -> the elements (bits) containing it
    for t, v in enumerate(vecs):
        for j in v:
            mask[j] = mask.get(j, 0) ^ (1 << t)
    for k in range(len(ops) - 2, -1, -2):
        m = mask.get(ops[k])
        if m:
            src = ops[k + 1]
            mask[src] = mask.get(src, 0) ^ m
    els = [[] for _ in vecs]
    for i in sorted(mask):
        for t in A.bits(mask[i]):
            els[t].append(i)
    return els


def selmer_part(d, rels, els, chars, F, S, G, dimV, k, U, jobs=1):
    """Return the characters used and the claims from the Selmer conditions on the elements els."""
    v = len(els)
    nr = len(d['real'])
    nch = nr + len(d['chars'])
    if v > U:
        raise RuntimeError('selmer: %d independent elements of K(S,2) but U = %d: a bug' % (v, U))
    ech = [xor(chars, e) for e in els]
    if rank(ech) != v:
        raise RuntimeError('selmer: the elements are not independent by characters: a bug')
    # the characters used: columns of rank v
    use, piv = [], {}
    for c in range(nch):
        col = sum(((x >> c) & 1) << i for i, x in enumerate(ech))
        y = reduce_mod(col, piv)
        if y:
            piv[y.bit_length() - 1] = y
            use.append(c)
        if len(use) == v:
            break
    if len(use) < v:
        raise RuntimeError('selmer: the characters used have rank %d < %d' % (len(use), v))
    # local conditions per place group: the classes modulo the image of the points
    used = sorted({j for e in els for j in e})
    cls = dict(zip(used, par_map(lambda js: [c for _, c in sorted(Places(d, F).classes(rels, js, F, d['chars']).items())],
                                 used, jobs)))
    res = [0] * v
    off, pos = 0, 0
    widths = []
    for p in S:
        w = sum(n for q, n in d['cls'] if q == p)
        widths.append((p, off, w))
        off += w
    ecls = [xor(cls, e) for e in els]
    for p, o, w in widths:
        basis = echelon(d['pts'][str(p)][1])
        if len(basis) != dims_of(d, p):
            raise RuntimeError('selmer: local image at %d has rank %d, not %d' % (p, len(basis), dims_of(d, p)))
        for i in range(v):
            res[i] |= reduce_mod((ecls[i] >> o) & ((1 << w) - 1), basis) << pos
        pos += w
    binf = echelon(d['pts']['inf'][1])
    if len(binf) != (1 if nr == 3 else 0):
        raise RuntimeError('selmer: the image at infinity')
    for i in range(v):
        res[i] |= reduce_mod(ech[i] & ((1 << nr) - 1), binf) << pos
    pos += nr
    dsel = v - rank(res)
    sel = kernel(res)                   # Sel_2 cap V as combinations of the elements, for ctp.py
    # The norm condition N(x) square is implied by the local ones; a violation is a bug.
    nres = list(res)
    nv = {j: norm_class(rels[j][0], F['poly'], S) for j in used}
    nvec = [xor(nv, e) for e in els]
    for i in range(v):
        nres[i] |= nvec[i] << pos
    if v - rank(nres) != dsel:
        raise RuntimeError('selmer: the norm condition is not implied by the local ones: a bug')
    vkn = v - rank(nvec)                # dim(V cap ker N)
    Sbk = parity_bound(G, F['u'], F['n'], F['w'])
    Sn = norm_bound(dsel, U, len(S), vkn, F['w'])
    bound = min(Sn, Sbk)
    return {
        'elements': els,
        'characters': {'real': [d['real'][c] for c in use if c < nr],
                       'primes': [d['chars'][c - nr] for c in use if c >= nr]},
        'claims': {'G': G, 'dimV': dimV, 'k': k, 'U': U, 'v': v, 'dim_sel_cap_V': dsel, 'dim_V_cap_ker_N': vkn,
                   'S_norm': Sn, 'S_bk': Sbk, 'S': bound, 'w': F['w']},
        'pairing': {'sel': sel, 'chars': ech, 'real': d['real'], 'primes': d['chars'],
               'rels': [[j, list(rels[j][0]), list(rels[j][1])] for j in used]},     # for the Cassels-Tate step
    }


def norm_class(r, poly, S):
    """Return the class in Q(S, 2) of the exact norm of the relation r: bit 0 its sign, bit t + 1 v_{S[t]}."""
    c = [Fraction(int(x)) for x in poly]
    if r[0] == 2:
        n = Fraction(int(r[1]) ** 3)
    elif r[0] == 0:
        a, b = int(r[1]), int(r[2])
        n = sum(c[i] * a ** i * b ** (3 - i) for i in range(4)) / c[3]
    else:
        g, dd = [int(x) for x in r[1:4]], int(r[4])
        C = [[Fraction(0), Fraction(0), -c[0] / c[3]], [Fraction(1), Fraction(0), -c[1] / c[3]],
             [Fraction(0), Fraction(1), -c[2] / c[3]]]       # multiplication by alpha
        mul = lambda A, B: [[sum(A[i][t] * B[t][j] for t in range(3)) for j in range(3)] for i in range(3)]
        C2 = mul(C, C)
        M = [[g[0] * (i == j) + g[1] * C[i][j] + g[2] * C2[i][j] for j in range(3)] for i in range(3)]
        det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
               M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
        n = det / dd ** 3
    if n == 0:
        raise RuntimeError('selmer: a relation of norm 0: a bug')
    vec = int(n < 0)
    for t, q in enumerate(S):
        e = 0
        for x, sg in ((n.numerator, 1), (n.denominator, -1)):
            x = abs(x)
            while x % q == 0:
                x //= q
                e += sg
        vec |= (e & 1) << (t + 1)
    return vec


def norm_bound(dsel, U, nS, vkn, w):
    """Return S_norm from dsel = dim(Sel_2 cap V), U, nS = #S, vkn = dim(V cap ker N) and w.

    Sel_2 lies in ker N and dim ker N <= U - 1 - #S, so dim Sel_2 <= dsel + U - 1 - #S - vkn.
    """
    slack = U - 1 - nS - vkn
    if slack < 0:
        raise RuntimeError('selmer: dim(V cap ker N) = %d > U - 1 - #S = %d: a bug' % (vkn, U - 1 - nS))
    if slack == 0:
        if (-1) ** dsel != w:
            raise RuntimeError('selmer: dim Sel_2 = %d but the root number is %+d: a bug' % (dsel, w))
        return dsel
    return parity_floor(dsel + slack, w)


def least(r, w):
    """Return the least bound possible given r and w: r, or r + 1 by parity; None without r."""
    return None if r is None else (r if w == (-1) ** r else r + 1)


def no_elements(G, dimV, k, U, nS, F):
    """Return the claims and empty element data of a witness without elements (v = 0)."""
    Sbk = parity_bound(G, F['u'], F['n'], F['w'])
    Sn = norm_bound(0, U, nS, 0, F['w'])
    return {'elements': [], 'characters': {'real': [], 'primes': []}, 'pairing': None,
            'claims': {'G': G, 'dimV': dimV, 'k': k, 'U': U, 'v': 0, 'dim_sel_cap_V': 0, 'dim_V_cap_ker_N': 0,
                       'S_norm': Sn, 'S_bk': Sbk, 'S': min(Sn, Sbk), 'w': F['w']}}


def norm_target(F, r):
    """Return the largest G at which the sieve's witness needs no elements, or None without r."""
    lo = least(r, F['w'])
    if lo is None:
        return None
    S = s_primes(F)[0]
    c = int(F['r1']) + (3 - int(F['r1'])) // 2 + s_count(F, S) - 1 - len(S)        # bound: parity_floor(c + G)
    G = lo - c + 1
    while G >= 0 and parity_floor(c + G, F['w']) > lo:
        G -= 1
    return max(G, -1)


def s_count(F, S):
    """Return #S_K, the number of primes of K above S."""
    f = [int(x) for x in F['poly']]
    return sum(len(fld.ftypes(f, p)[1]) for p in S)


def dims_of(d, p):
    return next(pl['dim'] for pl in d['places'] if pl['p'] == p)


# ---------- the witness

def local_prec(x, W, p):
    """Return v_p(W(x)) + 1 (+ 3 at p = 2): the p-adic precision at which W(x) is seen to be a square."""
    x = Fraction(x)
    w = sum(Fraction(int(c)) * x ** i for i, c in enumerate(W))
    v, n = 0, w
    while n.numerator % p == 0:
        n /= p
        v += 1
    while n.denominator % p == 0:
        n *= p
        v -= 1
    return v + (3 if p == 2 else 1)


def is_square_qp(t, p):
    """Return whether the nonzero rational t is a square in Q_p (exact)."""
    t = Fraction(t)
    if t == 0:
        return False
    v = 0
    while t.numerator % p == 0:
        t /= p; v += 1
    while t.denominator % p == 0:
        t *= p; v -= 1
    if v % 2:
        return False
    u = t.numerator * t.denominator
    if p == 2:
        return u % 8 == 1
    return pow(u % p, (p - 1) // 2, p) == 1


def prove(curve, hypothesis='GRH', cores=1, log=None, B0=None, state=None, budget=None):
    """Return the unwritten format-6 witness of curve from SPLIT's relations.

    hypothesis 'none' uses Minkowski's bound, refused if estimated above UNCOND_HOURS. With state, the triangular
    relations found so far are kept for a resume.
    """
    b = budget or runtime.current()
    with runtime.scope(b):
        try:
            return _prove(curve, hypothesis, cores, log or LOG, B0, state)
        except TimeLimit:
            runtime.raise_if_stopped(b)
            raise


TRI_SECONDS = 3600              # the triangular search's limit, or 3 times an unconditional run's estimate if more


def _prove(curve, hypothesis, cores, log, B0, state):
    t0 = time.time()
    tm = {}
    runtime.raise_if_stopped(slack=0)
    F, small = fld.compute(curve.ainvs, curve.bad_primes)
    if F['torsion'] and any(n % 2 == 0 for n in F['torsion']):
        raise RankboundError('2-descent over the cubic field needs E(Q)[2] = 0.')
    if curve.minimal_model is not None and [str(x) for x in curve.minimal_model] != F['minimal_model']:
        raise ValueError('The minimal_model given for %s is wrong; it should be [%s]. Correct it or leave it out.' % (
            curve.name, ','.join(F['minimal_model'])))
    tm['field'] = time.time() - t0
    say_curve(F['conductor'], len(F['bad_primes']), F['w'], F['torsion'])
    say_lower(curve.r, bool(curve.proof))
    P = [str(p) for p in F['bad_primes']]
    S, dropped = s_primes(F)
    r2 = (3 - F['r1']) // 2
    Bm = minkowski(F['dK'], r2)
    if hypothesis == 'none':
        est = uncond_seconds(Bm, cores)
        if est > UNCOND_HOURS * 3600:
            raise RankboundError(text('uncond_far', B=big(Bm), time=duration(est), cores=plural(cores, 'core'),
                                      limit=UNCOND_HOURS))
        say('how_uncond', time=duration(est), cores=plural(cores, 'core'))
    if hypothesis == 'none':
        B, btype = Bm, 'minkowski'
    else:
        B, btype = F['GB'], 'BDF'
    if B0 is None:
        B0 = min(B, max(F['GB'], FB_MIN))
    B0 = min(B0, B)
    need = 2 * (sum(1 for p, _ in small if p < B0) + len(S)) + 60
    L = LP * B0 if B0 == B else 0       # large primes only above B, where no triangular relation meets them
    t = time.time()
    lim = max(TRI_SECONDS, 3 * uncond_seconds(B, cores)) if hypothesis == 'none' else TRI_SECONDS
    tri, free, nst, fail = triangular(F, P, S, B0, B, cores, seconds=lim, state=state)
    tm['triangular'] = time.time() - t
    if fail:
        log('Selmer: %d prime ideals without a triangular relation (they weaken k)' % len(fail))
    nq = 3 + 3 * len(S) + sum(1 for p, _ in small if p < B0) + max(0, B0 - F['GB']) + len(fail) + NQ
    slo, shi, d, nab = 1, SHELLS, None, 0
    t = time.time()
    loc = local_setup(F, P, S, B0, nq, L, seconds=lim)
    tm['setup'] = time.time() - t
    for rnd in range(ROUNDS):
        t = time.time()
        new, fbn, shell = shells(F, P, S, B0, need if rnd == 0 else 10 ** 9, slo, shi, L, seconds=lim)
        tm['setup'] = tm.get('setup', 0) + time.time() - t
        nab += fbn
        if d is None:
            d = dict(loc, rels=loc['rels'] + new)
            d['ch'] = characters(d, d['rels'], F)
        else:                   # later rounds add only their new a,b relations
            d['rels'] += new
            d['ch'] += characters(d, new, F)
        dp = dict(d, rels=list(d['rels']), ch=list(d['ch']))     # the pool d keeps the unmatched large primes
        purge(dp, B, set(S))
        ns = d['ns0'] + nst
        if btype == 'BDF' and ns != len(small):
            raise RuntimeError('selmer: ns = %d but the field has %d' % (ns, len(small)))
        t = time.time()
        a = algebra(dp, tri, free, ns, B, S, F, cores, least=least(curve.r, F['w']))
        tm['algebra'] = tm.get('algebra', 0) + time.time() - t
        c = a['claims']
        log('2-descent round %d (shells to %d): %d relations (%d a,b found), k = %d, U = %d, v = %d, dim Sel_2 cap V = %d, '
            'S = %d' % (rnd + 1, shell, len(a['rels']), nab, c['k'], c['U'], c['v'], c['dim_sel_cap_V'], c['S']))
        if a['pairing'] is None:     # no elements needed, so no local images
            dp['pts'] = {'inf': ([], [])}
            break
        if c['v'] == c['U'] or c['U'] - 1 - len(S) - c['dim_V_cap_ker_N'] == 0 or (curve.r is not None and
                                                                                 c['S'] <= curve.r):
            break               # complete, ker N in V, or S at the known lower bound
        slo, shi = shell + 1, int(max(shell, slo) * GROW) + 1
    who = {'name': curve.name, 'r': curve.r, 'source': curve.source,
           'proof': dict(curve.proof, ainvs=curve.proof.get('ainvs') or curve.ainvs) if curve.r is not None and curve.proof else None}
    a = dict(a, s_dropped=dropped, relations=[rel_text(rr, H) for rr, H, _ in a['rels']],
             s_columns=[list(z) for z in a['s_columns']])
    wit = witness_of(who, F, dp, a, B, btype, ns, S)
    tm['total'] = time.time() - t0
    wit['_timing'] = {k: round(v, 2) for k, v in tm.items()}
    wit['_conductor'] = F['conductor']
    if a['pairing'] is not None:
        wit['_pairing'] = a['pairing']       # for ctp.py; not written
    return wit


# ---------- the 2-descent on the relations of the sieve (sieve.py), or given ones

PROGRESS = 300                  # seconds between the 2-descent's progress lines in run.log
NQS = 12                        # character primes beyond the bound on U
KERNEL_TRIES = 3                # the 2-descent's kernel: at most this many seeds of f2kernel
NSPLIT, SMAX_S = 2, 8            # extra elements of each prime above S, and the shells searched for them


def k_of(res, ns, scols):
    """Return k = ns - dim V_s for the solve res, from its basis of V_H, without another elimination.

    No held column is a sparse pivot's, so dim V_s is #(small S columns) plus the echelon rows of V_H led by a small
    column outside S, in the order (held outside S) < (small outside S) < (S columns).
    """
    held = list(res.held)
    sc = set(scols)
    if not sc <= set(range(ns)) | set(held):
        raise RuntimeError('selmer: a large s_column is not held: a bug')
    hw = (len(held) + 63) // 64
    rw = 8 * linalg.vs_words(len(held), ns)
    small_sc = [c for c in sc if c < ns]
    A = sum(1 << (ns + j) for j, c in enumerate(held) if c not in sc)   # small columns, then held[j] at ns + j
    B = ((1 << ns) - 1) ^ sum(1 << c for c in small_sc)
    with open(res.vs, 'rb') as f:
        data = f.read()
    piv, rest, inV = {}, [], 0
    for o in range(0, len(data), rw):
        h, s = int.from_bytes(data[o:o + 8 * hw], 'little'), int.from_bytes(data[o + 8 * hw:o + rw], 'little')
        inV += not h
        low = s & -s
        if not h and low & B:           # still f2elim's first column: already echelon
            piv[low.bit_length() - 1] = s
        else:
            rest.append(s | h << ns)
    if inV != res.dimV:
        raise RuntimeError('selmer: f2elim\'s basis of V has %d rows, not dim V = %d: a bug' % (inV, res.dimV))
    for x in rest:
        while x:
            y = x & A or x & B or x
            t = (y & -y).bit_length() - 1
            if t not in piv:
                piv[t] = x
                break
            x ^= piv[t]
    return ns - len(small_sc) - sum(1 for t in piv if B >> t & 1)


def sieve_relation(line):
    """Return the relation line 'a,b:hexprimes' or 'f:p' of the sieve as (selmer.gp's vector, its primes, None)."""
    if line.startswith('f:'):
        p = int(line[2:])
        return [2, p], [p], None
    a, b, H = linalg.parse_line(line)
    return [0, a, b], H, None


def sieve_witness(state, M, res, nrows, proof, source, threads=1, log=LOG):
    """Return the unwritten format-6 witness from the sieve's first nrows rows of M and their solve res."""
    t0 = time.time()
    F = fld.load(state)
    S, dropped = s_primes(F)
    Sset = set(S)
    kept, piv, need = linalg.cone(res.ops, res.piv)
    G, dimV = res.bound, res.dimV
    r = F.get('rank_lower_bound')
    lo = least(0 if r is None else r, F['w'])
    if SKIP and parity_bound(G, F['u'], F['n'], F['w']) <= lo:
        return bk_witness(F, M, res, kept, piv, need, S, dropped, proof, source, log, t0)
    ns, ncol = M.ns, len(M.col)
    # k: G's pivots plus the unit rows of the S columns, with G's ops
    scols = M.col.with_prime(Sset)      # degree 1
    k = k_of(res, ns, scols)
    tk = time.time()
    log('Selmer: k = %d from the solve\'s basis of V and %d unit rows (%.0f s)' % (k, len(scols), tk - t0))
    if SKIP:
        dh = head(F, [str(p) for p in F['bad_primes']], S)
        if norm_bound(0, dh['r1'] + dh['r2'] + dh['SK'] + k, len(S), 0, F['w']) <= lo:
            return bk_witness(F, M, res, kept, piv, need, S, dropped, proof, source, log, t0,
                              k=k, scols=[list(M.col.label(c)) for c in scols], d=dh)
    # the local data, with character primes above every prime of every relation
    qmin = 2
    for l in itertools.islice(M.src, nrows):
        qmin = max([qmin] + ([int(l[2:])] if l.startswith('f:') else [int(h, 16) for h in l.split(':', 1)[1].split(',') if h]))
    P = [str(p) for p in F['bad_primes']]
    nq = max(4, (3 - F['r1']) // 2 + 3 * len(S) + k + NQS)
    small_field = len(str(abs(int(F['dK'])))) - 1 < PMIN_S_BELOW
    d = base(F, P, S, qmin, nq, 2 if small_field else qmin + 1, qmin + 1)
    U = d['r1'] + d['r2'] + d['SK'] + k
    tb = time.time()
    log('Selmer: the local data, U = %d (%.0f s)' % (U, tb - tk))
    # more rows: the ideals (p) of S, and extra elements of the primes above S
    xrel = [([2, p], [p], None) for p in S] + d['rels']
    xline = ['f:%d' % p for p in S] + [rel_text(rr, H) for rr, H, _ in d['rels']]
    M_rel = lambda i: sieve_relation(M.src[i]) if i < nrows else xrel[i - nrows]
    # The xrel rows come first, so the dense phase takes them; row t is xrel[t] for t < nx, else M.rows[t - nx].
    nx = len(xrel)
    key = lambda t: nrows + t if t < nx else t - nx
    T = 64 * ((U + 24 + 63) // 64)
    base_ = T + nx                  # small columns: the tags, then one per xrel row
    kid = Kid(scols, ncol, base_)
    rk = KernelRows([xrel_row(labs, M.col, Sset, kid, base_) for _, _, labs in xrel], M.rows, nrows, kid)
    tf = time.time()
    log('Selmer: kernel vectors of %d rows outside S, %d large columns (%.0f s to set up)' % (len(rk), len(kid), tf - tb))
    # Take kernel vectors of independent characters, smallest first, until U.  Lanczos gives part of a larger kernel,
    # so with fewer than U another seed is tried, up to KERNEL_TRIES, stopping after a seed that adds nothing (the
    # characters' rank on the kernel can be below U).  la/selmer keeps the files for a resume.
    els, basis, ncand, te = [], {}, 0, tf
    chars = dict(zip(range(nrows, nrows + nx), characters(d, xrel, F)))
    for seed in range(1, KERNEL_TRIES + 1):
        before = len(els)
        vecs, part = kernel_rows(rk, nx, T, base_ + len(kid), threads, seed=seed,
                                 work=os.path.join(state, 'la', 'selmer'))
        cand = sorted(([key(t) for t in e] for e in vecs), key=len)
        del vecs
        ncand += len(cand)
        te = time.time()
        log('Selmer: %d kernel vectors, %d relations in all (seed %d, %.0f s)' % (len(cand), sum(map(len, cand)), seed,
                                                                             te - tf))
        said = time.time()
        for tried, e in enumerate(cand):
            if time.time() - said > PROGRESS:
                said = time.time()
                log('Selmer: %d of U = %d elements from %d kernel vectors, the characters of %d relations (%.0f s)' % (
                    len(els), U, tried, len(chars), said - te))
            new = sorted({j for j in e if j not in chars})
            if new:
                chars.update(zip(new, par_map(lambda rs: characters(d, rs, F), [M_rel(j) for j in new], threads)))
            x = reduce_mod(xor(chars, e), basis)
            if x:
                out = set()
                for j in e:
                    out ^= set(rk[j - nrows if j >= nrows else j + nx])
                if not e or out:
                    raise RuntimeError('selmer: a kernel vector has an odd valuation outside S: a bug')
                basis[x.bit_length() - 1] = x
                els.append(e)
                if len(els) == U:
                    break
        if len(els) == U or not part or (seed > 1 and len(els) == before):
            break
    del rk
    tc = time.time()
    log('Selmer: %d of U = %d elements, the characters of %d relations (%.0f s); the Selmer conditions next' % (
        len(els), U, len(chars), tc - te))
    # the relations of the witness: G's, then those of the elements
    extra = sorted({j for e in els for j in e} - set(need))
    order = list(need) + extra
    pos = {i: t for t, i in enumerate(order)}
    lines = [M.src[i] if i < nrows else xline[i - nrows] for i in order]
    rels = [M_rel(i) for i in order]
    chars = {pos[j]: x for j, x in chars.items() if j in pos}
    els = [sorted(pos[j] for j in e) for e in els]
    use = sorted({j for e in els for j in e})
    sp = selmer_part(d, rels, els, chars, F, S, G, dimV, k, U, jobs=threads)
    nrel = len(order)
    ops = [[pos[x], pos[y]] for x, y in kept]
    pivots = [pos[i] for i in piv]
    a = dict(sp, relations=lines, ops=ops, pivots=pivots, ops_s=ops,
             pivots_s=pivots + list(range(nrel, nrel + len(scols))), s_columns=[list(M.col.label(c)) for c in scols],
             s_dropped=dropped)
    who = {'name': F['name'], 'r': F.get('rank_lower_bound'), 'source': source, 'proof': proof}
    wit = witness_of(who, F, d, a, F['GB'], 'BDF', ns, S)
    wit['_pairing'] = a['pairing']           # for ctp.py; not written
    c = wit['claims']
    log('2-descent on the sieve\'s relations: %d in the witness (%d of G, %d more of the elements; %d elements of primes '
        'above S found), %d kernel vectors of %d relations; G = %d, k = %d, U = %d, v = %d, dim Sel_2 cap V = %d, '
        'S = %d.  %.1f s: k %.1f, local data %.1f, kernel %.1f (f2elim %.1f), characters %.1f, Selmer conditions '
        '%.1f' % (nrel, len(need), len(extra), len(d['rels']), ncand, len(use), G, k, U, c['v'],
                  c['dim_sel_cap_V'], c['S'], time.time() - t0, tk - t0, tb - tk, te - tb, te - tf, tc - te,
                  time.time() - tc))
    return wit


# ---------- the 2-descent's checkpoint before the Cassels-Tate step

SELMER_SCHEMA = 'rankbound-selmer-state-1'


def selmer_fingerprint(F, proof, source, solve):
    """Return the fingerprint of the 2-descent's witness: its field, lower bound, solve, switches and format."""
    return runtime.fingerprint(schema=SELMER_SCHEMA, field=F, proof=proof, source=source, solve=solve,
                               reduce_s=REDUCE_S, skip=SKIP, format=FORMAT6)


def selmer_save(folder, wit, fingerprint):
    """Commit the 2-descent's witness, private keys included, so a stop in the Cassels-Tate step resumes there."""
    import gzip
    with runtime.transaction(folder, 'selmer', SELMER_SCHEMA, fingerprint, {'relations': len(wit['relations'])}) as tx:
        with gzip.open(tx.path('selmer.json.gz'), 'wt') as f:
            json.dump(wit, f)


def selmer_load(folder, fingerprint):
    """Return the 2-descent's witness saved for fingerprint, or None."""
    import gzip
    if runtime.load(folder, 'selmer', SELMER_SCHEMA, fingerprint) is None:
        return None
    with gzip.open(os.path.join(folder, 'selmer.json.gz'), 'rt') as f:
        return json.load(f)


PMIN_S_BELOW = 40              # below this log10 |d_K|, search elements of every degree-1 prime above S, since the
                                # sieve's relations alone can miss an S-unit class (LMFDB 1888.d1, 753139.a1)
SKIP = True                     # make no elements when S_bk or S_norm is already the least possible


def bk_witness(F, M, res, kept, piv, need, S, dropped, proof, source, log=LOG, t0=None, k=None, scols=None, d=None):
    """Return the unwritten witness without elements when min(S_norm, S_bk) is already least(r, w)."""
    t0 = time.time() if t0 is None else t0
    d = d or head(F, [str(p) for p in F['bad_primes']], S)
    d['pts'] = {'inf': ([], [])}
    G, dimV = res.bound, res.dimV
    kk = G if k is None else k
    U = d['r1'] + d['r2'] + d['SK'] + kk
    a = no_elements(G, dimV, kk, U, len(S), F)
    Sbk, Sn = a['claims']['S_bk'], a['claims']['S_norm']
    pos = {i: t for t, i in enumerate(need)}
    ops, pivots = [[pos[x], pos[y]] for x, y in kept], [pos[i] for i in piv]
    a.update({'s_columns': [] if k is None else scols, 'relations': [M.src[i] for i in need], 'ops': ops,
              'pivots': pivots, 'ops_s': None if k is None else ops,
              'pivots_s': None if k is None else pivots + list(range(len(need), len(need) + len(scols))),
              's_dropped': dropped})
    who = {'name': F['name'], 'r': F.get('rank_lower_bound'), 'source': source, 'proof': proof}
    wit = witness_of(who, F, d, a, F['GB'], 'BDF', M.ns, S)
    rr = 'r = %d' % who['r'] if who['r'] is not None else 'no r'
    if k is None:
        log('2-descent not needed: S_bk = %d is already the least bound (%s); a witness without elements, %d relations '
            '(%.1f s)' % (Sbk, rr, len(need), time.time() - t0))
    else:
        log('2-descent not needed: the norm-kernel bound S_norm = %d (k = %d, U = %d, #S = %d) is already the least bound '
            '(%s); a witness without elements, %d relations (%.1f s)' % (Sn, k, U, len(S), rr, len(need),
                                                                       time.time() - t0))
    return wit


def witness_of(who, F, d, a, B, btype, ns, S):
    """Return the unpacked format-6 witness dict; who is (name, r, source, proof)."""
    r, src, proof = who['r'], who['source'], who['proof']
    c = a['claims']
    hyp = 'GRH' if btype == 'BDF' else 'none'
    rl = None
    if r is not None and proof:
        rl = cv.rank_proof(F['minimal_model'], F['bad_primes'], r, proof)
        src = None
    elif r is not None:
        rl = {'r': r, 'source': src}
    if r is not None and c['S'] < r:
        raise RankboundError(text('points_high', S=c['S'], n=r) if proof else text('r_high', S=c['S'], r=r))
    loc = {'u': F['u'], 'n': F['n'], 'inf': d['pts']['inf'][0], 'p': {}}
    for p in (p for p in S if str(p) in d['pts']):     # none without elements
        xs = d['pts'][str(p)][0]
        loc['p'][str(p)] = {'points': xs, 'prec': max([local_prec(x, d['W'], p) for x in xs], default=1)}
    wit = {'name': who['name'],
            'curve': {'ainvs': F['minimal_model'], 'minimal_model': F['minimal_model'], 'bad_primes': F['bad_primes']},
            'statement': statement(c['S'], hyp, r, src),
            'torsion': F['torsion'], 'hypothesis': hyp}
    if rl:
        wit['rank_lower_bound'] = rl
    wit.update({
        'field': {'poly': [str(x) for x in F['poly']], 'dK': F['dK'], 'bound': {'type': btype, 'B': B, 'ns': ns}},
        'weierstrass': d['W'], 'theta': d['theta'],
        's_primes': [str(p) for p in S],
        's_columns': a['s_columns'],
        **({'s_dropped': a['s_dropped']} if a.get('s_dropped') else {}),
        'local': loc,
        'characters': a['characters'],
        'claims': c,
        'created': datetime.date.today().isoformat(),
        'relations': a['relations'],
        'ops': a['ops'], 'pivots': a['pivots'],
        **({'ops_s': a['ops_s'], 'pivots_s': a['pivots_s']} if a['ops_s'] is not None else {}),
        'elements': a['elements'],
        'format': FORMAT6})
    return wit


# ---------- the work folder

def field_disc(curve):
    """Return d_K, the discriminant of the cubic field of curve."""
    P = [int(p) for p in curve.bad_primes]
    b2, b4, b6, _ = cv.binv(cv.minimal(cv._ainvs5(curve.ainvs), P)[0])
    return fld.disc(fld.maximal_form([16 * b6, 8 * b4, b2, 1], [2, 3] + P)[0])


def say_bound(wit):
    """Report the 2-descent's bound, and the root number where parity lowered it."""
    from .runtime import say_bound as say_
    c = wit['claims']
    raw = c['dim_sel_cap_V'] + c['U'] - c['v']
    root = c['S'] < raw and (c['S'] < c['G'] + int(wit['local']['u']) + int(wit['local']['n']) or c['v'] < c['U'])
    say_(c['S'], 'parity' if root else 'selmer', wit['hypothesis'] == 'GRH')


class Selmer(Folder):
    """The work folder of the route SPLIT: a curve without 2-torsion, bounded unconditionally (--uncond)."""
    method = 'selmer'
    route = 'SPLIT'

    def precheck(self, curve):
        """Refuse the curve before any work (ValueError)."""
        cv.checked(curve)

    def estimate(self, curve, cores=1, ecm_hours=1.0, quiet=False, given=None):
        """Return the unconditional bound's estimate in core-hours, reported unless quiet."""
        Bm, x, sec = uncond_estimate(curve, cores)
        if not quiet:
            say('est_uncond' if sec <= UNCOND_HOURS * 3600 else 'est_uncond_far', name=curve.name, B=big(Bm),
                time=duration(sec), cores=plural(cores, 'core'), limit=UNCOND_HOURS)
        return sec * cores / 3600

    def prove(self, curve, req, resources, budget, **_):
        """Bound the rank unconditionally and return the unchecked witness as ProofData; raise Stopped at the end."""
        cores, ecm_hours = resources.cores, resources.ecm_hours
        budget.check()
        if curve.composites:
            if not ecm_hours:
                raise FactorisationNeeded('The discriminant is not fully factored. Give its prime factors with '
                                          '--bad-primes.', [{'n': c, 'digits': len(c)} for c in curve.composites])
            cv.check_composites(curve)
            curve = cv.factor_in(self.dir, curve, ecm_hours, cores, 1.0, budget)
        if curve.proof and curve.r:         # check the points before any work
            cv.check_cyclic(field_disc(curve), curve.r)
        cv.save(self.dir, curve)
        path = self.path(curve.name + '.witness.json.gz')
        self.note(stage='relations')
        t = time.time()
        with runtime.alarm(budget, on_fire=lambda: say('late_witness')):     # for loops that do not poll
            wit = prove(curve, hypothesis='none', cores=cores, state=self.dir, budget=budget)
        say_bound(wit)
        wit = ctp.after_selmer(wit, budget, selmer_seconds=wit.get('_timing', {}).get('total', 0.0))
        c = wit['claims']
        r = curve.r
        rl = wit.get('rank_lower_bound') or {}
        res = {'name': curve.name, 'outcome': 'selmer', 'GRH': False, 'rank_bound': c.get('S_ctp', c['S']),
               'witness': path, 'rank_lower_bound': r, 'r_proved': r is not None and (r == 0 or 'points' in rl),
               'statement': wit['statement'],
               'claims': c, 'G': c['G'], 'u': wit['local']['u'], 'n': wit['local']['n'], 'w': c['w'],
               'torsion': wit['torsion'], 'minimal_model': wit['curve']['minimal_model'],
               'conductor': wit['_conductor'], 'core_hours': round((time.time() - t) * cores / 3600, 3),
               'cost': wit['_timing']}
        return ProofData(witness=path, data=wit, rank_bound=res['rank_bound'], hypothesis='none',
                         statement=wit['statement'], method=self.method, rank_lower_bound=r, r_proved=res['r_proved'],
                         record=res, cost=res['cost'], outcome='selmer')


# ========== the route SIEVE: the 2-Selmer group on the sieve's relations (sieve.py), under the GRH

def torsion_list(t):
    """Return the torsion as a list of ints, from a list or a string '2,4' or '[2, 4]'."""
    if isinstance(t, list):
        return [int(x) for x in t]
    return [int(x) for x in str(t or '').strip('[]').split(',') if x.strip()]


class Job(Folder):
    """The work folder of the route SIEVE: prepare, tune, sieve and check, then the witness, each stage resumable.

    slices/NAME.rels.gz, NAME.json finished special-q slices (given.rels.gz: given relations) history.json,
    checking.json, STOP the solves, the pending check, the outcome la/ the last elimination; la/best.* and order.txt the
    solve to certify NAME.witness.json.gz, selmer.json the witness, and its stage before the Cassels-Tate step
    """
    method = 'selmer'
    route = 'sieve'

    def precheck(self, curve):
        """Refuse the curve before any work (ValueError)."""
        cv.checked(curve)

    def __init__(self, workdir):
        super().__init__(workdir)
        self.solver = None                     # the Solver after a sieve or solve in this process
        self.outcome = None
        self.rebound = False                   # prepare changed the folder's r, so the stop is decided again

    def prepare(self, curve, ecm_hours=1.0, cores=1, rebind=False):
        """Write the field data and curve.json once, under a lock, since shards may start together.

        The relations do not depend on r, so a folder prepared for another r is reused; with rebind it takes this r.
        """
        cv.checked(curve)
        fd = sieve.State(self.dir).lock('prepare.lock', wait=3600)
        try:
            old = cv.saved(self.dir, curve)
            if old is not None and (not rebind or (old.r, old.proof, old.source) == (curve.r, curve.proof,
                                                                                      curve.source)):
                return fld.load(self.dir)
            F = other_r_field(self.dir, curve)
            if F is not None and not rebind:
                return F
            if old is not None or F is not None:
                sieve.State(self.dir).clear_stop()
                self.rebound = True
                F = rebind_field(self.dir, curve, F) if F is not None else fld.prepare(
                    self.dir, curve.ainvs, curve.bad_primes, r=curve.r, name=curve.name,
                    minimal_model=curve.minimal_model)
                self.check_proof(curve)
                cv.save(self.dir, curve)
                return F
            if curve.composites:
                if not ecm_hours:
                    raise FactorisationNeeded('The discriminant is not fully factored. Give its prime factors with '
                                              '--bad-primes.', [{'n': c, 'digits': len(c)} for c in curve.composites])
                cv.check_composites(curve)
                curve = cv.factor_in(self.dir, curve, ecm_hours, cores, 1.0, runtime.current())
            F = fld.prepare(self.dir, curve.ainvs, curve.bad_primes, r=curve.r, name=curve.name,
                            minimal_model=curve.minimal_model)
            self.check_proof(curve)
            cv.save(self.dir, curve)
            return F
        finally:
            os.close(fd)

    def check_proof(self, curve):
        """Raise RankboundError, before any sieving, if the points do not prove the rank lower bound."""
        F = fld.load(self.dir)
        if not curve.proof or F.get('rank_lower_bound') is None:
            return
        cv.check_cyclic(int(F['dK']), F['rank_lower_bound'])
        try:
            cv.rank_proof(F['minimal_model'], F['bad_primes'], F['rank_lower_bound'],
                          dict(curve.proof, ainvs=curve.proof.get('ainvs') or curve.ainvs))
        except RuntimeError as e:
            raise RankboundError(str(e))

    def give(self, given):
        """Commit the given relations as a finished slice, factoring F(a, b) where needed."""
        S = sieve.State(self.dir)
        if S.given() == given.digest:
            return False
        fd = S.lock('run.lock')
        try:
            if S.given() != given.digest:
                F = fld.load(self.dir)
                sieve.check_given(F, given)
                lines, _ = sieve.given_lines(F, given.rels)
                S.write_given(lines, given.digest)
        finally:
            os.close(fd)
        return True

    def tune(self, lpb_min=None):
        """Choose the sieve's parameters (the table, or the trial sieve) and write params.json, once."""
        fd = sieve.State(self.dir).lock('prepare.lock', wait=3600)
        try:
            prm = sieve.tune(self.dir, fld.load(self.dir), lpb_min=lpb_min)
        finally:
            os.close(fd)
        self.note(estimate=sieve.sieve_estimate(prm))
        return prm

    def retune(self, err):
        """Re-tune at a larger lpb after the sieve dried up; the finished slices stay valid."""
        S = sieve.State(self.dir)
        prm = S.params()
        if prm is None:
            raise err
        lpb = prm['lpb0'] + 2
        if prm.get('siever') == 'given' or lpb > sieve.LPB_MAX:
            raise err
        say('retune')
        S.drop_params(aside='params_lpb%d.json' % prm['lpb0'])
        new = self.tune(lpb_min=lpb)
        if new['lpb0'] <= prm['lpb0']:           # never re-tune to the same lpb
            raise err
        return new

    def say_start(self, F):
        """Report the curve and its rank lower bound."""
        say_curve(fld.conductor(F), len(F['bad_primes']), F['w'], torsion_list(F.get('torsion')))
        cur = read_json(self.path('curve.json'), {})
        say_lower(F.get('rank_lower_bound'), bool(cur.get('proof')))

    def estimate(self, curve, cores=1, ecm_hours=1.0, quiet=False, given=None):
        """Prepare and tune, and return the sieve's estimate in core-hours (0 when finished or given)."""
        if self.info().get('result'):
            if not quiet and self.info().get('key') == self.key:
                self.earlier()                  # a finished folder: report its result
            return 0.0
        if given is not None:
            self.prepare(curve, ecm_hours, cores)
            return 0.0
        with self.logging():
            self.prepare(curve, ecm_hours, cores)
            prm = sieve.State(self.dir).params() or self.tune()
        h = sieve.sieve_estimate(prm)
        if not quiet:
            say('est_sieve', name=curve.name, about=about(h), wall='' if h < 0.01 else ', %s on %s' % (
                duration(h * 3600 / max(1, cores)), plural(cores, 'core')))
        return h

    def sieve_shard(self, curve, cores=1, shard=(1, 1), budget=None, budget_core_hours=None):
        """Sieve share K of N until the folder has the next check's relations, within the budgets."""
        b = budget if budget is not None else runtime.Budget()
        with runtime.scope(b):
            h = self.estimate(curve, cores, quiet=True)
        if self.info().get('result') or sieve.State(self.dir).stopped():
            say('other_done')
            return 0
        k, n = shard
        with runtime.metered(self.store) as meter, runtime.scope(b):
            meter.budget = self.store.set_budget(h, budget_core_hours)
            while True:
                sieve.size_slices(self.dir, cores * n)
                try:
                    sieve.sieve_run(self.dir, jobs=cores, shard=(k - 1, n), budget=b, meter=meter)
                    return 0
                except sieve.DriedUp as e:
                    if n > 1:
                        raise
                    self.retune(e)

    def witness(self, jobs=1, budget=None, outcome=None):
        """Make the witness of the best solve and return it as unchecked ProofData.

        A final stop certifies only a solve it already has, never a new elimination.
        """
        b = runtime.current() if budget is None else budget
        S = sieve.State(self.dir)
        fd = S.lock('run.lock')
        try:
            with runtime.scope(b):
                return self._witness(S, jobs, b, outcome)
        finally:
            os.close(fd)

    def _witness(self, S, jobs, b, outcome=None):
        if b.expired():
            raise b.error()
        F = fld.load(self.dir)
        cur = read_json(self.path('curve.json'))
        prm = S.params()
        cpu, nrel = S.slice_stats()
        self.note(stage='solving', relations=nrel, core_hours=round(cpu / 3600, 3))
        path = self.path(F['name'] + '.witness.json.gz')
        t = time.time()
        proof = cur.get('proof')
        if proof and not proof.get('ainvs'):      # the points are on the curve's given model
            proof = dict(proof, ainvs=cur['ainvs'])
        source = cur.get('source', 'cited')
        # the 2-descent's checkpoint for this solve
        fp = selmer_fingerprint(F, proof, source, read_json(self.path('STOP'), None))
        c = selmer_load(self.dir, fp) if S.stopped() else None
        if c is None:
            V = self.solver
            la = self.path('la')
            # Resumed after STOP: re-solve the slices of la/best in order at its trim, reusing its elimination.
            if V is None or V.last is None:
                names, rec = S.best(la)
                if names is None and outcome in sieve.FINAL:      # no solve to certify
                    raise runtime.final_stop(self.dir, outcome, final_reason(S, jobs))
                V = sieve.Solver.of(S, F, prm, jobs)
                V.budget = b
                V.load(names)
                if rec:         # that solve's column count
                    V.M.pad = rec.get('ncols', 0)
                V.last = linalg.solve(V.M, la, V.target, keep=rec['keep'] if rec else V.M.ns, threads=jobs,
                                      name='best' if rec else 'elim', budget=b)
                V.last_rows = len(V.M.rows)
                if V.last.late is not None:
                    raise Stopped(self.dir)
                if V.last.bound is None:
                    raise RankboundError(sieve.too_dense_text(V.last, self.dir))
                if rec:
                    S.commit_best(la)
            V.budget = b
            V.shrink(la)
            t = time.time()
            say('witness_start')
            with runtime.alarm(b, on_fire=lambda: say('late_witness')):
                c = sieve_witness(self.dir, V.M, V.last, V.last_rows, proof, source, threads=jobs)
            # the 2-descent's wall time on the curve, which sets the Cassels-Tate step's allowance
            wall = cpu / max(1, jobs) + sum(h.get('solve_s') or 0 for h in S.history()) + time.time() - t
            c.setdefault('_timing', {})['selmer_wall'] = round(wall, 1)
            if ctp.wanted(c):               # checkpoint, so a stop in the Cassels-Tate step resumes there
                selmer_save(self.dir, c, fp)
        say_bound(c)
        # the Cassels-Tate step within the budget
        c = ctp.after_selmer(c, b, selmer_seconds=c.get('_timing', {}).get('selmer_wall', 0.0))
        r = F['rank_lower_bound']
        res = {'name': F['name'], 'outcome': S.stop_outcome(), 'GRH': True,
               'rank_bound': c['claims'].get('S_ctp', c['claims']['S']), 'claims': c['claims'], 'witness': path,
               'rank_lower_bound': r, 'r_proved': r is not None and (r == 0 or 'points' in c.get('rank_lower_bound', {})),
               'statement': c['statement'],
               'relations': nrel, 'core_hours': round(cpu / 3600, 3), 'estimate_core_hours': sieve.sieve_estimate(prm),
               'G': c['claims']['G'], 'u': F['u'], 'n': F['n'], 'w': F['w'], 'minimal_model': F['minimal_model'],
               'conductor': fld.conductor(F), 'torsion': F.get('torsion'),
               'cost': {'sieve': round(cpu, 1), 'solve': round(sum(h.get('solve_s') or 0 for h in S.history()), 1),
                        'witness': round(time.time() - t, 1)},
               'spent': self.store.spent()}
        final = {}
        if res['outcome'] in sieve.FINAL:
            final = {'final': res['outcome'], 'reason': final_reason(S, jobs)}
        final['data'] = c                       # the runner writes it
        return ProofData(witness=path, rank_bound=res['rank_bound'], hypothesis='GRH', statement=c['statement'],
                         method=self.method, rank_lower_bound=r, r_proved=res['r_proved'], record=res, cost=res['cost'],
                         outcome=res['outcome'], **final)

    def retry(self, S, window, cores, budget_ch):
        """Clear a final STOP that this run has more time, cores or budget for; its pending check stays."""
        st = S.stop_record()
        oc = st.get('outcome')
        if oc == 'too_long':
            more = window is None or (st.get('window') is not None and window > 1.05 * st['window']) or \
                cores > (st.get('threads') or cores)
        elif oc == 'budget':
            more = budget_ch is None or budget_ch > (st.get('budget_core_hours') or 0)
        else:
            more = False
        if more:
            S.clear_stop()
        return more

    def prove(self, curve, req, resources, budget, given=None):
        """Run every stage up to the witness, resuming; return ProofData, or None if another process published."""
        with runtime.metered(self.store) as meter, runtime.scope(budget):
            return self._prove(curve, req, resources, budget, given, meter)

    def _prove(self, curve, req, resources, budget, given, meter):
        cores, ecm_hours = resources.cores, resources.ecm_hours
        window = budget.left()
        self.prepare(curve, ecm_hours, cores, rebind=True)
        F = fld.load(self.dir)
        self.say_start(F)
        S = sieve.State(self.dir)
        if given is not None:
            self.give(given)
            sieve.write_given_params(self.dir)
            if not S.stopped():
                self.note(stage='solving')
                self.outcome, self.solver = sieve.solve_given(self.dir, budget, threads=cores, window=window)
            return self.witness(cores, budget)
        prm = S.params() or self.tune()
        if prm.get('siever') == 'given':            # relations were given before; now sieve on
            S.drop_params(aside='params_given.json')
            S.clear_stop()
            prm = self.tune()
        meter.budget = self.store.set_budget(sieve.sieve_estimate(prm), resources.budget_core_hours)
        if S.stopped() and S.stop_outcome() in sieve.FINAL:
            self.retry(S, window, cores, meter.budget)
        if not S.stopped():
            say('how_sieve', about=about(sieve.sieve_estimate(prm)))
        self.note(stage='sieving')
        norm = norm_target(F, F.get('rank_lower_bound'))
        while True:
            sieve.size_slices(self.dir, cores)
            try:
                self.outcome, self.solver = sieve.sieve_run(self.dir, jobs=cores, budget=budget, window=window,
                                                            meter=meter, redecide=self.rebound, norm=norm)
                break
            except sieve.DriedUp as e:                # each re-tune raises lpb by at least 2, up to LPB_MAX
                self.retune(e)
        if not self.outcome:
            if not S.stopped():
                raise Stopped(self.dir)
            if self.stored():                   # another process finished meanwhile
                say('other_done')
                return None
            if self.solver is None:
                say('other_sieved')
        return self.witness(cores, budget, S.stop_outcome())


def other_r_field(state, curve):
    """Return the folder's field data if it is this curve's with another r or points, else None."""
    try:
        F = fld.load(state)
    except (OSError, ValueError):
        return None
    if not isinstance(F, dict) or F.get('rank_lower_bound') == curve.r:
        return None
    fp = fld.fingerprint(curve.ainvs, curve.bad_primes, F.get('rank_lower_bound'), curve.name)
    return F if runtime.load(state, 'field', fld.FORMAT, fp) is not None else None


def rebind_field(state, curve, F):
    """Commit field.json for curve's r from F, the same curve's field for another r; only r and g_target change."""
    r = curve.r
    if r is not None and r > F['ns'] + F['u'] + F['n']:
        raise ValueError('r = %d is impossible for this curve: under GRH its rank is at most %d. Check r.' % (
            r, F['ns'] + F['u'] + F['n']))
    F = dict(F, rank_lower_bound=r, g_target=fld.target(F, r))
    with runtime.transaction(state, 'field', fld.FORMAT, fld.fingerprint(curve.ainvs, curve.bad_primes, r, curve.name),
                             {'ns': F['ns']}) as tx:
        tx.adopt('small.txt')
        tx.write('field.json', json.dumps(F, indent=1) + '\n')
    return F


def final_reason(S, cores):
    """Return the text of the final stop recorded in STOP."""
    st = S.stop_record()
    if st.get('outcome') == 'budget':
        b = core_hours(st.get('budget_core_hours') or 0)
        if S.best(os.path.join(S.path, 'la'))[0] is None:
            return text('budget_none', budget=b)
        return text('budget_spent', budget=b, spent=core_hours(st.get('spent_core_hours') or st.get('budget_core_hours')
                                                                or 0))
    return sieve.too_long_text(st.get('need_s'), st.get('window'), st.get('threads') or cores)



# ========== SPLIT: relations found in the field itself, for the unconditional bound.  triangular() finds, for each
# degree-1 prime P with B0 <= N(P) < B, an element of P whose other primes are smaller.

SMAX = 12                       # shells of the small-element search per prime ideal


def tri_ranges(B0, B, parts):
    """Split [B0, B) into parts ranges of about equal numbers of primes."""
    if B <= B0:
        return []
    li = lambda t: t / max(math.log(t), 1.0)
    tot, cuts, lo = li(B) - li(max(B0, 2)), [], B0
    for i in range(1, parts):
        target = li(max(B0, 2)) + tot * i / parts
        a, b = lo, B
        while b - a > 1:
            m = (a + b) // 2
            if li(m) < target:
                a = m
            else:
                b = m
        if b > lo:
            cuts.append(b)
            lo = b
    edges = [B0] + cuts + [B]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1) if edges[i] < edges[i + 1]]


def triangular(F, P, S, B0, B, cores=1, seconds=3600, state=None):
    """Return (triangular relations, free rows, degree-1 count, failures) for the primes in [B0, B).

    With state, each finished range is an artifact, so a stopped run resumes on any number of cores.
    """
    if B <= B0:
        return [], {}, 0, []
    tb = runtime.current().child(seconds=seconds, stop=lambda: TimeLimit('The time limit was reached.', seconds))
    tdir = fp = None
    rngs = None
    if state is not None:
        tdir = os.path.join(state, 'tri')
        fp = runtime.fingerprint(schema=TRI_SCHEMA, poly=[str(c) for c in F['poly']], P=[str(p) for p in P],
                                 S=[int(p) for p in S], B0=int(B0), B=int(B), smax=SMAX)
        if runtime.inspect(tdir, 'plan', TRI_SCHEMA, fp)[0] == 'valid':
            with open(os.path.join(tdir, 'plan.json')) as f:
                rngs = [tuple(r) for r in json.load(f)['ranges']]
    if rngs is None:
        rngs = tri_ranges(B0, B, max(1, 4 * cores) if B - B0 > 10 ** 4 else 1)
        if tdir is not None:
            if os.path.isdir(tdir):
                for n in os.listdir(tdir):          # ranges of another curve or bound
                    os.remove(os.path.join(tdir, n))
            with runtime.transaction(tdir, 'plan', TRI_SCHEMA, fp, {'ranges': len(rngs)}) as tx:
                tx.write('plan.json', json.dumps({'ranges': rngs}))

    def one(rg):
        if tdir is not None:
            rec = tri_load(tdir, fp, rg)
            if rec is not None:
                return rec
        tb.check()
        args = (F['poly'], [int(p) for p in P], S, rg[0], rg[1], SMAX)
        try:        # a worker process gets the seconds left
            rec = procs.submit(tri, *args, tb.left()).result() if procs else tri(*args, tb.left())
        except Exception:
            tb.check()                          # the end may surface in a worker
            raise
        if tdir is not None:
            tri_save(tdir, fp, rg, rec)
        return rec

    procs = None
    if cores > 1 and len(rngs) > 1:
        try:
            procs = concurrent.futures.ProcessPoolExecutor(min(cores, len(rngs)))
        except OSError:                 # no process pool (a sandbox): one core
            pass
    try:
        if procs is None:       # serially, since tri sets a global time limit; like ex.map, raise the first error
            outs, err = [], None
            for rg in rngs:
                try:
                    outs.append(one(rg))
                except Exception as e:
                    err = err or e
            if err is not None:
                raise err
        else:
            with concurrent.futures.ThreadPoolExecutor(max(1, cores)) as ex:
                outs = list(ex.map(one, rngs))  # raises the first error once every range has ended
    finally:
        if procs:
            procs.shutdown(cancel_futures=True)
    tb.check()
    rels, free, ns, fail = [], {}, 0, []
    for rec in outs:
        rels += [tuple(r) for r in rec['rels']]
        free.update((int(p), list(lab)) for p, lab in rec['free'])
        ns += rec['ns']
        fail += rec['fail']
    return rels, free, ns, fail


TRI_SCHEMA = 'rankbound-tri-3'


def tri_name(rg):
    return 'range_%d_%d' % tuple(rg)


def tri_save(tdir, fp, rg, rec):
    """Commit the range rg's record rec as an artifact of tdir."""
    with runtime.transaction(tdir, tri_name(rg), TRI_SCHEMA, runtime.fingerprint(plan=fp, range=list(rg)),
                             {'relations': len(rec['rels'])}) as tx:
        tx.write('%d_%d.json' % tuple(rg), json.dumps(rec))


def tri_load(tdir, fp, rg):
    """Return the record of the range rg kept in tdir, or None if not valid for this plan."""
    if runtime.inspect(tdir, tri_name(rg), TRI_SCHEMA, runtime.fingerprint(plan=fp, range=list(rg)))[0] != 'valid':
        return None
    try:
        with open(os.path.join(tdir, '%d_%d.json' % tuple(rg))) as f:
            rec = json.load(f)
        labels = lambda lab: [tuple(z) for z in lab]     # prime labels are tuples
        return {'rels': [(r, H, labels(lab)) for r, H, lab in rec['rels']],
                'free': [[int(p), labels(lab)] for p, lab in rec['free']], 'ns': int(rec['ns']),
                'fail': labels(rec['fail'])}
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ========== the data of the 2-descent and SPLIT's relations, ported from selmer.gp and split.gp
# alpha is a root of f = F(x, 1).  A relation is [0, a, b] (a - b alpha), [1, c0, c1, c2, d]
# ((c0 + c1 alpha + c2 alpha^2)/d) or [2, p] (p).  At P above p, x = pi^v u has class v mod 2 and the class of u in
# (O/P)^*/squares, or (O/P^(2e+1))^*/squares at p = 2.

_K = {}


def kfield(f, P):
    """Return the K of f and P, cached for one curve at a time."""
    key = (tuple(int(c) for c in f), tuple(sorted(int(p) for p in P)))
    if key not in _K:
        _K.clear()
        _K[key] = K(*key)
    return _K[key]


class K:
    """The field of f with the bad primes P, which factorisation divides out first as PARI's addprimes does."""

    def __init__(self, f, P):
        self.f = [int(c) for c in f]
        self.F = fld.form_field(self.f)
        self.P = sorted({int(p) for p in P})
        self.lcp = self.F.lcp
        self.t2 = None
        self.loc = {}

    def dec(self, p):
        return self.F.dec(p)

    # ---------- local classes

    def res(self, x, P):
        """Return the residue in F_q of x with v_P(x) >= 0, as a polynomial mod (g, p)."""
        y, d = self.F.integral(x)
        vy, uy = self.F.val_int(y, P)
        vd, ud = self.F.val_int((d, 0, 0), P)
        if vy > vd:
            return []
        a, b = P.red(uy), P.red(ud)
        return _fq_mul(a, _fq_pow(b, P.q - 2, P), P)

    def unit_bits(self, u, P):
        """Return the class bits of the P-unit u."""
        if P.p == 2:
            tab, H = self.table2(P)
            return tab[fld.lat_reduce(u, H)]
        z = P.red(u)
        return [int(_fq_pow(z, (P.q - 1) // 2, P) != [1])]

    def cls_int(self, y, P):
        v, u = self.F.val_int(y, P)
        return [v % 2] + self.unit_bits(u, P)

    def cls(self, x, P):
        """Return the class of x != 0 at P as a list of bits (m1cls1)."""
        y, d = self.F.integral(x)
        out = self.cls_int(y, P)
        if d > 1:
            vd = A.val(d, P.p)
            out = _xor(out, self.cls_int((d // P.p ** vd, 0, 0), P))
            if vd % 2:
                out = _xor(out, self.cls_int((P.p, 0, 0), P))
        return out

    def table2(self, P):
        """Return ({residue: class bits} of (O/P^(2e+1))^*/squares, the HNF of P^(2e+1)) for P above 2."""
        if id(P) not in self.loc:
            H = P.hnf
            for _ in range(2 * P.e):
                H = A.hnf([self.F.omul(x, y) for x in H for y in P.hnf])
            d = [H[i][i] for i in range(3)]
            units = [r for r in ((i, j, k) for i in range(d[0]) for j in range(d[1]) for k in range(d[2])) if P.red(r)]
            mulr = lambda x, y: fld.lat_reduce(self.F.omul(x, y), H)
            co = {mulr(u, u): 0 for u in units}
            n = 0
            for u in units:
                if u not in co:
                    new = {mulr(w, u): c | 1 << n for w, c in co.items()}
                    co.update(new)
                    n += 1
            self.loc[id(P)] = ({r: [c >> i & 1 for i in range(n)] for r, c in co.items()}, H)
        return self.loc[id(P)]

    def width(self, P):
        return 2 if P.p > 2 else 1 + len(next(iter(self.table2(P)[0].values())))

    # ---------- relations

    def row(self, x, H, ab=None):
        """Return the labels of the primes of odd valuation of x, supported on H and lc(f) (m1row).

        For a - b alpha, at p not dividing lc(f) only the prime at (a : b) divides it, to the power v_p(F(a, b)).
        """
        out = []
        if ab:
            a, b = ab
            for p in sorted(set(H) - set(self.lcp)):
                if H.count(p) % 2:
                    out.append((p, a * pow(b, -1, p) % p))
            H = []
        y, d = self.F.integral(x)
        for p in sorted(set(H) | set(self.lcp)):
            for P in self.dec(p):
                if (self.F.val_int(y, P)[0] - P.e * A.val(d, p)) % 2:
                    out.append(P.label)
        return out

    def coords(self, y):
        """Return [c0, c1, c2, d] with y = (c0 + c1 alpha + c2 alpha^2)/d, d > 0 least."""
        x = [Fraction(c) for c in self.F.power(y)]
        d = 1
        for c in x:
            d = d * c.denominator // math.gcd(d, c.denominator)
        return [int(c * d) for c in x] + [d]

    def t2basis(self, P):
        """Return P's basis reduced by LLL in an integer T2 form; T2 only orders the search."""
        if self.t2 is None:
            self.t2 = t2int(t2gram(self.f, self.F))
        G0, H = self.t2, P.hnf
        GH = [[sum(G0[a][b] * H[j][b] for b in range(3)) for a in range(3)] for j in range(3)]
        G = [[sum(H[i][a] * GH[j][a] for a in range(3)) for j in range(3)] for i in range(3)]
        U = A.lll_gram(G, eta=Fraction(51, 100))
        return [[sum(U[k][c] * P.hnf[k][i] for k in range(3)) for i in range(3)] for c in range(3)]

    def search(self, P, smax, test):
        """Yield test(el, N(el)) when not None, for el in P in field.gp's forvec order up to the shell smax."""
        R = self.t2basis(P)
        for s in range(1, smax + 1):
            for c0 in range(-s, s + 1):
                for c1 in range(-s, s + 1):
                    for c2 in range(-s, s + 1):
                        c = (c0, c1, c2)
                        if max(map(abs, c)) < s or next(x for x in c if x) < 0:
                            continue
                        el = tuple(c0 * R[0][i] + c1 * R[1][i] + c2 * R[2][i] for i in range(3))
                        v = test(el, self.F.norm(el))
                        if v is not None:
                            yield v


def _xor(a, b):
    return [x ^ y for x, y in zip(a, b)]


def _fq_mul(a, b, P):
    return A.pmulmod(a, b, P.p) if P.f == 1 else A.pdivmod(A.pmulmod(a, b, P.p), P.g, P.p)[1]


def _fq_pow(a, k, P):
    r = [1]
    while k:
        if k & 1:
            r = _fq_mul(r, a, P)
        a = _fq_mul(a, a, P)
        k >>= 1
    return r


def bits_int(v):
    """Return the bit list v as an int, v[0] its bit 0."""
    return sum(b << i for i, b in enumerate(v))


def smooth_ok(n, B, allowed=(), extra=()):
    """Return the primes of n if each is < B or in allowed, else None; an unsplit cofactor is not smooth."""
    out = []
    for p, e in A.factor(n, min(B, 1000), extra):
        fa = [p] if p < 1000 or p in extra else _small(p, B, allowed)
        if fa is None or any(q >= B and q not in allowed for q in fa):
            return None
        out += fa * e
    return sorted(out)


def _small(c, B, allowed):
    """Return the primes of c if all are < B or in allowed, by a short Pollard rho, else None."""
    if c in allowed or A.is_prime(c):
        return [c]
    g = A._rho(c, 4, 4 * math.isqrt(B) + 64)
    if g is None:
        return None
    a, b = _small(g, B, allowed), _small(c // g, B, allowed)
    return None if a is None or b is None else a + b


# ---------- split.gp

def tri(f, P, S, lo, hi, smax, seconds=None):
    """Return m1tri's record for the primes in [lo, hi): rels, free, ns and fail; arith.TimeUp after seconds."""
    with runtime.limited(seconds=seconds, error=A.time_up):
        return _tri(f, P, S, lo, hi, smax)


def _tri(f, P, S, lo, hi, smax):
    k = kfield(f, P)
    Ss = set(S) | set(k.lcp)
    rels, free, fail, ns = [], [], [], 0
    for p in A.primes(lo, hi):
        A.tick()
        dec = k.dec(p)
        if any(Q.f > 1 for Q in dec):
            free.append([int(p), k.row((p, 0, 0), [p])])
        for Q in dec:
            if Q.f != 1:
                continue
            ns += 1
            if p in Ss:
                continue

            def test(el, N, p=p):
                m = abs(N) // p
                if m == 0 or m % p == 0:
                    return None
                H = smooth_ok(m, p)
                return None if H is None else (el, sorted(H + [p]))
            z = next(k.search(Q, smax, test), None)
            if z is None:
                fail.append(Q.label)
                continue
            r = [1] + k.coords(z[0])
            rels.append(rel(r, z[1], k.row(fld.relation_el(r), z[1])))
    return {'rels': rels, 'free': free, 'ns': ns, 'fail': fail}


def sunits(k, Ss, pmin, nsplit, smax, ok):
    """Return nsplit relations of each degree-1 prime above p in Ss, p >= pmin, with norms passing ok (m1sunits)."""
    out = []
    for p in sorted(Ss):
        if p < pmin:
            continue
        for Q in k.dec(p):
            if Q.f != 1:
                continue

            def test(el, N):
                if N == 0:
                    return None
                H = ok(abs(N))
                return None if H is None else (el, H)
            for el, H in _take(k.search(Q, smax, test), nsplit):
                r = [1] + k.coords(el)
                out.append(rel(r, H, k.row(fld.relation_el(r), H)))
    return out


def _take(gen, n):
    """Return the first n values of gen, asking for no more, so a search stops at its quota."""
    return list(itertools.islice(gen, max(0, n)))


def fb_fixed(k, S, B0, nsplit, smax):
    """Return m1fb's relations that do not depend on the shells: free relations and S-units."""
    Ss = set(S) | set(k.lcp)
    out = [rel([2, p], [p], k.row((p, 0, 0), [p])) for p in A.primes(2, B0) if p not in Ss]
    out += [rel([2, p], [p], k.row((p, 0, 0), [p])) for p in sorted(Ss)]
    out += sunits(k, Ss, B0, nsplit, smax, lambda n: smooth_ok(n, B0, Ss, k.P))
    return out


def fb_shells(k, S, B0, need, slo, shi, L):
    """Return (a - b alpha relations on the shells slo..shi, their number, the last shell) (m1fb)."""
    Ss = set(S) | set(k.lcp)
    out = []
    c0, c1, c2, c3 = k.f
    cnt, last, top = 0, slo, slo - 1
    for s in range(slo, shi + 1):
        if cnt >= need or s > 2 * last + 40:
            break
        A.tick()
        top = s
        for b in range(1, s + 1):
            for a in (range(-s, s + 1) if b == s else (-s, s)):
                if cnt >= need:
                    break
                if math.gcd(a, b) != 1:
                    continue
                N = c0 * b ** 3 + c1 * a * b * b + c2 * a * a * b + c3 * a ** 3
                if N == 0:
                    continue
                H = smoothlp(abs(N), B0, Ss, L)
                if H is None:
                    continue
                out.append(rel([0, a, b], H, k.row((a, -b, 0), H, (a, b))))
                cnt, last = cnt + 1, s
    return out, cnt, top


_PB = {}


def smoothlp(n, B0, Ss, L):
    """Return the primes of n if all are < B0 or in Ss, except at most one simple prime in [B0, L), else None."""
    key = (B0, tuple(sorted(Ss)))
    if key not in _PB:
        _PB.clear()
        _PB[key] = math.prod(A.primes(2, B0)) * math.prod(q for q in Ss if q >= B0)
    m = n
    g = math.gcd(m, _PB[key])
    while g > 1:
        m //= g
        g = math.gcd(m, g)
    if m > 1 and (m in Ss or m < B0 or m >= L or not A.is_prime(m)):
        return None
    fa = A.factor(n // m, B0, [q for q in Ss if q >= B0])
    return sorted([q for q, e in fa for _ in range(e)] + ([m] if m > 1 else []))


# ---------- selmer.gp

def realroots(f, digits=100):
    return A.roots(f, digits)[0]


def record():
    """Return an empty record of the 2-descent's data; relations are (vector, norm primes, odd-valuation labels)."""
    return {'rels': [], 'places': [], 'fb': [], 'pts': {}, 'pld': []}


def rel(r, H, lab):
    """Return a relation (vector, norm primes, labels) as fresh lists."""
    return [int(z) for z in r], [int(z) for z in H], list(lab)


def sel_head(mm, f, P, S):
    """Return (m1head's record, (the field, W, theta, the real roots of f times 2^640, floored))."""
    k = kfield(f, P)
    b2, b4, b6, _ = cv.binv(mm)
    W = [16 * b6, 8 * b4, b2, 1]
    th = theta(k, W, mm, P)
    rr = A.real_roots_floor(f, 640)
    d = record()
    d.update(theta=[str(x) for x in th], W=[str(x) for x in W], r1=len(rr), r2=(3 - len(rr)) // 2,
             real=list(range(len(rr))), S=[int(p) for p in sorted(S)])
    nsk = 0
    for p in sorted(S):
        dec = k.dec(p)
        t = sum(1 for Q in dec if Q.e == 1 and Q.f == 1)
        nsk += len(dec)
        d['places'].append({'p': int(p), 'n': len(dec), 'dim': (2 if t == 3 else t) + (p == 2),
                            'labels': [Q.label for Q in dec]})
    d['SK'] = nsk
    return d, (k, W, th, rr)


def is_root(k, W, th):
    """Return whether W(th) = 0 exactly in K."""
    z, pw = (0, 0, 0), (1, 0, 0)
    for c in W:
        z = tuple(a + c * b for a, b in zip(z, pw))
        pw = fld.pmul(k.f, pw, th)
    return not any(z)


def theta(k, W, mm=None, P=()):
    """Return theta on 1, alpha, alpha^2 with W(theta) = 0, checked exactly."""
    import itertools
    if mm is not None:
        f, th = fld.field_of([int(a) for a in mm], [2, 3] + [int(p) for p in P])
        c0, c1, c2, c3 = W
        dW = c1 * c1 * c2 * c2 - 4 * c0 * c2 ** 3 - 4 * c1 ** 3 * c3 - 27 * c0 * c0 * c3 * c3 + 18 * c0 * c1 * c2 * c3
        th = tuple(Fraction(x) for x in th)
        if [int(c) for c in f] == k.f and not (dW >= 0 and math.isqrt(dW) ** 2 == dW) and is_root(k, W, th):
            return list(th)
    with localcontext() as ctx:
        ctx.prec = 80
        rf, cf = A.roots(k.f, 60)
        rw, cw = A.roots(W, 60)
        ev = lambda j, re, im: _emb(k.F.power(tuple(int(i == j) for i in range(3))), re, im)
        for perm in itertools.permutations(rw):
            for sg in ((1, -1) if cw else (1,)):
                M, v = [], []
                for z, w in zip(rf, perm):
                    M.append([ev(j, z, 0)[0] for j in range(3)])
                    v.append(w)
                for (re, im), (wr, wi) in zip(cf, cw):
                    M.append([ev(j, re, im)[0] for j in range(3)])
                    M.append([ev(j, re, im)[1] for j in range(3)])
                    v += [wr, sg * wi]
                try:
                    n = A.solve3([[Fraction(x) for x in r] for r in M], [Fraction(x) for x in v])
                except (StopIteration, ZeroDivisionError):
                    continue
                th = k.F.power(tuple(round(x) for x in n))
                if is_root(k, W, th):
                    return [Fraction(c) for c in th]
    raise RuntimeError('m1head: W has no root in K')


def t2gram(f, F, digits=40):
    """Return an exact rational Gram matrix approximating T2 on O's basis, checked positive definite.

    The digits double until it is (ArithmeticError past 640).
    """
    while digits <= 640:
        try:
            rr, cc = A.roots(f, digits)
        except ValueError:          # roots closer than the precision
            digits *= 2
            continue
        emb = [(z, Decimal(0), 1) for z in rr] + [(re, im, 2) for re, im in cc]
        with localcontext() as ctx:
            ctx.prec = digits + 10
            E = [[(Fraction(a), Fraction(b), w) for re, im, w in emb
                  for a, b in [_emb(F.power(tuple(int(i == j) for i in range(3))), re, im)]] for j in range(3)]
        G = [[sum(w * (a[0] * b[0] + a[1] * b[1]) for a, b in zip(E[i], E[j]) for w in [a[2]]) for j in range(3)]
             for i in range(3)]
        if G[0][0] > 0 and G[0][0] * G[1][1] - G[0][1] ** 2 > 0 and A.det3(G) > 0:
            return G
        digits *= 2
    raise ArithmeticError('t2gram: no positive definite T2 form on the basis of O')


T2_BITS = 64           # t2int's first scale; T2's diagonal entries are >= 1


def t2int(G):
    """Return round(2^s G) for the least s = T2_BITS * 2^i at which it is positive definite (some s is, as G is)."""
    s = T2_BITS
    while True:
        Gi = [[round(Fraction(x) * 2 ** s) for x in r] for r in G]
        if Gi[0][0] > 0 and Gi[0][0] * Gi[1][1] - Gi[0][1] * Gi[1][0] > 0 and A.det3(Gi) > 0:
            return Gi
        s *= 2


def _emb(x, re, im):
    """Return (Re, Im) of x at the root re + i im of f."""
    return (x[0] + x[1] * re + x[2] * (re * re - im * im), x[1] * im + 2 * x[2] * re * im)


def places(k, S):
    return [(p, Q) for p in sorted(S) for Q in k.dec(p)]


def chars(f, S, qmin, nq):
    out, q = [], max(qmin, 3)
    while len(out) < nq:
        q = A.next_prime(q + 1)
        if q in S or f[3] % q == 0:
            continue
        rt = A.roots_mod(f, q)
        if rt:
            out.append([q, rt[0]])
    return out


def pld(k, p, Q):
    """Return the place data of Q above p for Places.one; a class missing from its tables is None."""
    F = k.F
    t, ct = None, 0
    be = (0, 1, 0)
    if k.f[3] % p == 0:
        t = 0
        while F.val((t, -1, 0), Q) > 0:
            t += 1
        be = fld.pinv(k.f, (t, -1, 0))
        ct = bits_int(k.cls((t, -1, 0), Q))
    z = k.res(be, Q)
    g = _minpoly(z, Q)
    tab = None
    if p == 2:
        tab = [_cls_or_none(k, (a0 - b0 * be[0], -b0 * be[1], -b0 * be[2]), Q, lambda v: v == 0)
               for b0 in range(8) for a0 in range(8)]
    M = 0 if Q.e < 2 else 4 if p == 2 else 1 if p <= 64 else 0
    rtab = None
    if M:
        q, c = p ** M, (2 * Q.e + 1 if p == 2 else 1)
        rtab = (M, [_cls_or_none(k, (a0 - b0 * be[0], -b0 * be[1], -b0 * be[2]), Q, lambda v: v + c <= Q.e * M)
                    for b0 in range(q) for a0 in range(q)])
    return {'p': int(p), 'e': Q.e, 'f': Q.f, 't': t, 'g': [int(c) for c in g], 'cp': bits_int(k.cls((p, 0, 0), Q)),
            'ct': ct, 'tab': tab, 'rtab': rtab}


def _cls_or_none(k, x, Q, okv):
    if not any(x):
        return None
    return bits_int(k.cls(x, Q)) if okv(k.F.val(x, Q)) else None


def _minpoly(z, Q):
    """Return the monic minimal polynomial over F_p of z in F_q, low to high."""
    p = Q.p
    z = z + [0] * (Q.f - len(z))
    if Q.f == 1 or not any(z[1:]):
        return [-z[0] % p, 1]
    M = []
    for j in range(Q.f):                         # multiplication by z on 1, t, ...
        b = [0] * Q.f
        b[j] = 1
        v = _fq_mul(z, A.ptrim(b), Q)
        M.append(v + [0] * (Q.f - len(v)))
    M = [[M[j][i] for j in range(Q.f)] for i in range(Q.f)]
    tr = sum(M[i][i] for i in range(Q.f)) % p
    if Q.f == 2:
        det = (M[0][0] * M[1][1] - M[0][1] * M[1][0]) % p
        return [det, -tr % p, 1]
    m2 = sum(M[i][i] * M[j][j] - M[i][j] * M[j][i] for i in range(3) for j in range(i + 1, 3)) % p
    (a, b, c), (d, e_, f_), (g, h, i_) = M
    det = (a * (e_ * i_ - f_ * h) - b * (d * i_ - f_ * g) + c * (d * h - e_ * g)) % p
    return [-det % p, m2, -tr % p, 1]


def inf_points(W, th, f):
    """Return ([x], [signs of x - theta at the real roots]) if W has three real roots, else ([], []) (m1inf)."""
    with localcontext() as ctx:
        ctx.prec = 110
        RR = realroots(f)
        if len(RR) < 3:
            return [], []
        e = realroots(W)
        xx = A.bestappr((e[0] + e[1]) / 2, 2 ** 20)
        x = Decimal(xx.numerator) / xx.denominator
        tc = [Decimal(c.numerator) / c.denominator for c in map(Fraction, th)]
        return [xx], [[int(x - (tc[0] + tc[1] * z + tc[2] * z * z) < 0) for z in RR]]


def local_points(k, W, th, p, PLp, t, rng, tries=200000):
    """Return (points, class bits): x with W(x) a nonzero square in Q_p whose classes span dimension t."""
    if t == 0:
        return [], []
    J = A.val(_disc3(W), p) // 2 + 4
    Z = [r for g in (W, A.pderiv(W)) for r in A.padic_roots(g, p, 2 * J + 10)]
    X, C, basis, kk = [], [], [], 4
    for i in range(1, tries + 1):
        if i % 200 == 0:
            kk += 1
        if Z and i % 2:
            xx = Fraction(Z[rng.randrange(len(Z))] + p ** (rng.randrange(J) + 1) * (rng.randrange(p ** 4) + 1) * rng.choice((1, -1)))
        else:
            xx = Fraction(rng.randrange(p ** kk), p ** (2 * rng.randrange(2))) * rng.choice((1, -1))
        w = A.peval(W, xx)
        if not w or not A.is_square_qp(w, p):
            continue
        cl = sum((k.cls((xx - th[0], -th[1], -th[2]), Q) for Q in PLp), [])
        if A.insert(basis, sum(b << i for i, b in enumerate(cl))):
            X.append(xx)
            C.append(cl)
        if len(X) == t:
            return X, C
    raise RuntimeError('m1points: the local image at %d has rank %d < %d after %d tries' % (p, len(X), t, tries))


def _disc3(W):
    d, c, b, a = W
    return fld.disc((a, b, c, d))


def sel_base(mm, f, P, qmin, nq, S):
    """Return (m1base's record, the field)."""
    d, (k, W, th, rr) = sel_head(mm, f, P, S)
    d['realq'] = (640, [int(x) for x in rr])
    d['chars'] = [[int(q), int(c)] for q, c in chars(f, set(S), qmin, nq)]
    PL = places(k, S)
    d['cls'] = [[int(p), k.width(Q)] for p, Q in PL]
    d['pld'] = [pld(k, p, Q) for p, Q in PL]
    xi, ci = inf_points(W, th, f)
    d['pts']['inf'] = ([str(x) for x in xi], [bits_int(c) for c in ci if c])
    rng = random.Random(1)
    for p in sorted(S):
        dec = k.dec(p)
        t = sum(1 for Q in dec if Q.e == 1 and Q.f == 1)
        X, C = local_points(k, W, th, p, dec, (2 if t == 3 else t) + (p == 2), rng)
        d['pts'][str(p)] = ([str(x) for x in X], [bits_int(c) for c in C if c])
    return d, k


def sel_local(mm, f, P, B0, nsplit, smax, nq, L, S):
    """Return (m1setup's record without the shells, the field)."""
    d, k = sel_base(mm, f, P, max(B0, L), nq, S)
    ns0 = 0
    for p in A.primes(2, B0):
        for Q in k.dec(p):
            if Q.f == 1:
                ns0 += 1
                d['fb'].append(Q.label)
    d['ns0'] = ns0
    d['rels'] += fb_fixed(k, S, B0, nsplit, smax)
    return d, k


def sel_sieve(mm, f, P, qmin, nq, pmin, lim, nsplit, smax, S):
    """m1sieve: sel_base's record with the S-unit relations of the primes above S from pmin."""
    d, k = sel_base(mm, f, P, qmin, nq, S)
    Ss = set(S)
    d['rels'] += sunits(k, Ss, pmin, nsplit, smax, lambda n: smooth_ok(n, lim, Ss, k.P))
    return d


def sel_batch(f, P, rels, items, Q, S):
    """Return m1batch's records: (j, -1, signs and characters) for an int t, else (j, s, class) per place."""
    k = kfield(f, P)
    PL = places(k, S)
    out = []
    RR = None
    for j, i, t in items:
        r = rels[i - 1]
        if isinstance(t, int):
            if RR is None:
                with localcontext() as ctx:
                    ctx.prec = 110
                    RR = realroots(f)
            out.append((j, -1, bits_int(sgn(r, RR) + [chi(r, q) for q in Q])))
        else:
            x = fld.relation_el(r)
            out += [(j, s, bits_int(k.cls(x, PL[s][1]))) for s in t]
    return out


def sgn(r, RR):
    out = []
    with localcontext() as ctx:
        ctx.prec = 110
        for z in RR:
            v = r[1] - r[2] * z if r[0] == 0 else r[1] + r[2] * z + r[3] * z * z if r[0] == 1 else Decimal(r[1])
            if abs(v) < Decimal(10) ** -30:
                raise RuntimeError('m1sgn: a relation is too close to 0 at a real place')
            out.append(int(v < 0))
    return out


def chi(r, Q):
    q, z = Q
    v = (r[1] - r[2] * z if r[0] == 0 else (r[1] + r[2] * z + r[3] * z * z) * pow(r[4], -1, q) if r[0] == 1
         else r[1]) % q
    if v == 0:
        raise RuntimeError('m1chi: %d divides a relation' % q)
    return int(A.kronecker(v, q) < 0)
