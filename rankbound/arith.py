"""Integer and polynomial arithmetic without PARI; polynomials are coefficient lists, low degree first."""
import math, random, sys
from . import runtime as _runtime
from fractions import Fraction
from decimal import Decimal, localcontext

_SIEVE = bytearray([1]) * 1000
_SIEVE[0] = _SIEVE[1] = 0
for _i in range(2, 32):
    if _SIEVE[_i]:
        _SIEVE[_i * _i::_i] = bytearray(len(range(_i * _i, 1000, _i)))
SMALL = [i for i in range(1000) if _SIEVE[i]]


class TimeUp(Exception):
    """A runtime.limited budget ran out."""


def time_up():
    """Return the TimeUp for a runtime.limited budget's error=."""
    return TimeUp('The time limit was reached.')


def tick():
    """Raise the current budget's exception if it has ended."""
    _runtime.check()


def primes(lo, hi):
    """Return the primes p with lo <= p < hi."""
    lo = max(lo, 2)
    if hi <= lo:
        return []
    s = bytearray([1]) * (hi - lo)
    for p in primes(2, math.isqrt(hi - 1) + 1) if hi > 1000 else SMALL:
        if p * p >= hi:
            break
        st = max(p * p, (lo + p - 1) // p * p)
        s[st - lo::p] = bytearray(len(range(st - lo, hi - lo, p)))
    return [lo + i for i in range(hi - lo) if s[i]]


_PL = [[]]


def plist(top):
    """Return a cached list of primes that contains all those below top."""
    if not _PL[0] or _PL[0][-1] < top:
        _PL[0] = primes(2, max(2 * top, 10 ** 5))
    return _PL[0]


def _sprp(n, a):
    d, s = n - 1, 0
    while not d & 1:
        d >>= 1; s += 1
    x = pow(a, d, n)
    if x in (1, n - 1):
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def _lucas(n):
    """Return whether the odd non-square n is a strong Lucas probable prime."""
    D = 5
    while kronecker(D, n) != -1:
        D = -D - 2 if D > 0 else -D + 2
    Q = (1 - D) // 4
    d, s = n + 1, 0
    while not d & 1:
        d >>= 1; s += 1
    U, V, Qk = 1, 1, Q % n                      # P = 1
    inv2 = (n + 1) // 2
    for b in bin(d)[3:]:
        U, V, Qk = U * V % n, (V * V - 2 * Qk) % n, Qk * Qk % n
        if b == '1':
            U, V = (U + V) * inv2 % n, (D * U + V) * inv2 % n
            Qk = Qk * Q % n
    if U == 0 or V == 0:
        return True
    for _ in range(s - 1):
        V, Qk = (V * V - 2 * Qk) % n, Qk * Qk % n
        if V == 0:
            return True
    return False


def is_prime(n):
    """Return whether n is a BPSW probable prime (exact below 2^64)."""
    if n < 1000:
        return n >= 2 and bool(_SIEVE[n])
    for p in SMALL[:40]:
        if n % p == 0:
            return False
    return _sprp(n, 2) and math.isqrt(n) ** 2 != n and _lucas(n)


def next_prime(n):
    """Return the least prime >= n."""
    n = max(n, 2)
    while not is_prime(n):
        n += 1
    return n


def kronecker(a, n):
    if n == 0:
        return 1 if a in (1, -1) else 0
    k = 1
    if n < 0:
        n = -n
        if a < 0:
            k = -k
    v = (n & -n).bit_length() - 1
    n >>= v
    if v and not a & 1:
        return 0
    if v & 1 and a % 8 in (3, 5):
        k = -k
    a %= n
    while a:
        while not a & 1:
            a >>= 1
            if n % 8 in (3, 5):
                k = -k
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            k = -k
        a %= n
    return k if n == 1 else 0


def val(x, p):
    """Return v_p(x) for a nonzero int or Fraction."""
    if isinstance(x, Fraction):
        return val(x.numerator, p) - val(x.denominator, p)
    v = 0
    while x % p == 0:
        x //= p; v += 1
    return v


def iroot(n, k):
    """Return the integer k-th root of n >= 0 if exact, else None (Newton from above, never below the root)."""
    if n < 2 or k == 1:
        return n
    if k == 2:
        x = math.isqrt(n)
        return x if x * x == n else None
    x = 1 << -(-n.bit_length() // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            break
        x = y
    return x if x ** k == n else None


def sqrt_mod(a, p):
    """Return a square root of the square a mod the prime p (Tonelli-Shanks)."""
    a %= p
    if p == 2 or a < 2:
        return a
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while not q & 1:
        q >>= 1; s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(a, q, p), pow(a, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p; i += 1
        if i == m:
            raise ValueError('not a square')
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


def is_square_qp(x, p):
    """Return whether the rational x != 0 is a square in Q_p (p = 0 for R)."""
    x = Fraction(x)
    if p == 0:
        return x > 0
    v = val(x, p)
    if v % 2:
        return False
    u = x.numerator * x.denominator // p ** abs(v)
    return u % 8 == 1 if p == 2 else kronecker(u % p, p) == 1


def hilbert(a, b, p):
    """Return the Hilbert symbol (a, b)_p of nonzero rationals (p = 0 for R)."""
    a, b = Fraction(a), Fraction(b)
    if p == 0:
        return -1 if a < 0 and b < 0 else 1
    a, b = a.numerator * a.denominator, b.numerator * b.denominator
    al, be = val(a, p), val(b, p)
    u, v = a // p ** al, b // p ** be
    if p == 2:
        e = lambda x: (x - 1) // 2 % 2
        w = lambda x: (x * x - 1) // 8 % 2
        return -1 if (e(u) * e(v) + al * w(v) + be * w(u)) % 2 else 1
    s = -1 if al * be % 2 and p % 4 == 3 else 1
    return s * kronecker(u, p) ** be * kronecker(v, p) ** al


def crt(r1, m1, r2, m2):
    return (r1 + m1 * ((r2 - r1) * pow(m1, -1, m2) % m2)) % (m1 * m2)


# ---------- factoring

def _time_out():
    return TimeoutError('factor: time limit')


def _rho(n, tries=40, limit=1 << 22):
    """Return a proper factor of the odd composite n by Pollard-Brent, or None."""
    for c in range(1, tries + 1):
        y, m, g, r, q = 2, 128, 1, 1, 1
        x = ys = 2
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                tick()
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
            if r > limit:
                break
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if 1 < g < n:
            return g
    return None


def _ecm(n, B1, curves):
    """Return a proper factor of n by ECM (Suyama's curves, stage 2 to 50 B1), or None."""
    ps = primes(2, B1 + 1)              # stage 2 has its own wheel
    for _ in range(curves):
        tick()
        sg = random.randrange(6, 1 << 30)
        u, v = (sg * sg - 5) % n, 4 * sg % n
        x, z = pow(u, 3, n), pow(v, 3, n)
        g = math.gcd(16 * x * v * u % n, n)
        if g != 1:
            if g != n:
                return g
            continue
        a24 = pow(v - u, 3, n) * (3 * u + v) % n * pow(16 * x * v % n, -1, n) % n     # (A + 2)/4

        def add(P, Q, D):               # D = P - Q
            s, t = (P[0] - P[1]) * (Q[0] + Q[1]) % n, (P[0] + P[1]) * (Q[0] - Q[1]) % n
            return D[1] * (s + t) ** 2 % n, D[0] * (s - t) ** 2 % n

        def dbl(P):
            s, d = (P[0] + P[1]) ** 2 % n, (P[0] - P[1]) ** 2 % n
            return s * d % n, (s - d) * (d + a24 * (s - d)) % n

        def mul(k, P):
            R0, R1 = P, dbl(P)
            for b in bin(k)[3:]:
                R0, R1 = (add(R1, R0, P), dbl(R1)) if b == '1' else (dbl(R0), add(R1, R0, P))
            return R0

        Q = (x, z)
        for i, p in enumerate(ps):
            if p > B1:
                break
            if i % 512 == 0:
                tick()
            q = p
            while q * p <= B1:
                q *= p
            Q = mul(q, Q)
        g = math.gcd(Q[1], n)
        if g == n:
            continue
        if g > 1:
            return g
        # stage 2: [m w] Q = +-[j] Q for q = m w +- j
        w = 210
        Q2 = dbl(Q)
        B = {1: Q, 3: add(Q2, Q, Q)}
        for j in range(5, w // 2, 2):
            B[j] = add(B[j - 2], Q2, B[j - 4])
        B = [B[j] for j in B if math.gcd(j, w) == 1]
        W = mul(w, Q)
        m1 = max(1, B1 // w)
        G, Gp = mul(m1 * w, Q), (mul((m1 - 1) * w, Q) if m1 > 1 else None)
        acc = 1
        for m in range(m1, (50 * B1) // w + 2):
            if m > m1:
                G, Gp = (add(G, W, Gp) if Gp else dbl(G)), G
            if m % 64 == 0:
                tick()
            for b in B:
                acc = acc * (G[0] * b[1] - b[0] * G[1]) % n
        g = math.gcd(acc, n)
        if 1 < g < n:
            return g
    return None


# Pollard-Brent's steps before ECM: factors to about 10 digits, beyond which ECM is faster
RHO_FIRST = 1 << 16


def _split(n, effort=1, timed=False):
    """Return a proper factor of the composite n, or None; with timed, ECM runs until the budget ends."""
    tick()
    if n % 2 == 0:
        return 2
    for k in primes(2, n.bit_length()):     # a k-th power is a q-th power for q | k
        r = iroot(n, k)
        if r:
            return r
    if n < 1 << 80 or effort < 1:
        return _rho(n)
    g = _rho(n, 1, RHO_FIRST)
    B1 = 2000
    while g is None and (timed or B1 <= 3 * 10 ** 6 * effort):
        tick()
        g = _ecm(n, B1, 25)
        B1 = min(3 * B1, 10 ** 8)
    return g


def factor(n, lim=None, extra=(), seconds=None):
    """Return the factorisation [(p, e), ...] of |n| != 0 into BPSW probable primes.

    Raise ValueError if a composite cannot be split (TimeoutError after seconds).  With lim, as PARI's
    factor(n, lim): only primes below lim or in extra, then the cofactor."""
    n = abs(n)
    out = {}
    for p in sorted(set(extra)):
        while n % p == 0:
            out[p] = out.get(p, 0) + 1; n //= p
    top = 1000 if lim is None else lim
    for p in (SMALL if top <= 1000 else plist(top)):
        if p >= top or p * p > n:
            break
        while n % p == 0:
            out[p] = out.get(p, 0) + 1; n //= p
    if n > 1 and n < top * top and (lim is None or n < lim):
        out[n] = out.get(n, 0) + 1; n = 1
    if lim is not None:
        if n > 1:
            out[n] = out.get(n, 0) + 1
        return sorted(out.items())
    todo = [n] if n > 1 else []
    with _runtime.scope(_runtime.current().child(seconds, stop=_time_out) if seconds is not None else _runtime.current()):
        while todo:
            m = todo.pop()
            if is_prime(m):
                out[m] = out.get(m, 0) + 1
                continue
            g = _split(m, timed=seconds is not None)
            if g is None:
                raise ValueError('cannot factor %d' % m)
            todo += [g, m // g]
    return sorted(out.items())


def insert(basis, v):
    """Add the F_2 vector v to the echelon basis if independent; return whether it was."""
    for w in basis:
        v = min(v, v ^ w)
    if v:
        basis.append(v)
        basis.sort(reverse=True)
    return bool(v)


def rank2(rows):
    """Return the F_2 rank of rows (lists of bits)."""
    basis = []
    for r in rows:
        insert(basis, sum(b << i for i, b in enumerate(r)))
    return len(basis)


def prime_divisors(n):
    return [p for p, _ in factor(n)]


# ---------- polynomials mod p and over Z_p

def ptrim(f):
    while f and f[-1] == 0:
        f = f[:-1]
    return f


def pmod(f, p):
    return ptrim([c % p for c in f])


def pmulmod(f, g, p):
    if not f or not g:
        return []
    h = [0] * (len(f) + len(g) - 1)
    for i, a in enumerate(f):
        if a:
            for j, b in enumerate(g):
                h[i + j] += a * b
    return pmod(h, p)


def pdivmod(f, g, p):
    """Return (q, r) with f = q g + r mod p, for g != 0 mod p."""
    f, g = pmod(f, p), pmod(g, p)
    inv = pow(g[-1], -1, p)
    q = [0] * max(len(f) - len(g) + 1, 0)
    while len(f) >= len(g):
        c = f[-1] * inv % p
        k = len(f) - len(g)
        q[k] = c
        for i, b in enumerate(g):
            f[k + i] = (f[k + i] - c * b) % p
        f = ptrim(f)
    return q, f


def pgcd(f, g, p):
    f, g = pmod(f, p), pmod(g, p)
    while g:
        f, g = g, pdivmod(f, g, p)[1]
    return [c * pow(f[-1], -1, p) % p for c in f] if f else f


def ppowmod(f, k, m, p):
    r, f = [1], pdivmod(f, m, p)[1]
    while k:
        if k & 1:
            r = pdivmod(pmulmod(r, f, p), m, p)[1]
        f = pdivmod(pmulmod(f, f, p), m, p)[1]
        k >>= 1
    return r


def peval(f, x, m=None):
    r = 0
    for c in reversed(f):
        r = r * x + c
        if m:
            r %= m
    return r


def pderiv(f):
    return [i * c for i, c in enumerate(f)][1:]


def roots_mod(f, p):
    """Return the distinct roots in [0, p) of f mod the prime p, increasing."""
    f = pmod(f, p)
    if len(f) <= 1:
        return []
    if p < 64:
        return [x for x in range(p) if peval(f, x, p) == 0]
    g = pgcd(f, _sub(ppowmod([0, 1], p, f, p), [0, 1], p), p)
    out, todo = [], [g]
    while todo:
        h = todo.pop()
        if len(h) <= 1:
            continue
        if len(h) == 2:
            out.append(-h[0] * pow(h[1], -1, p) % p)
            continue
        while True:
            a = random.randrange(p)
            w = pgcd(h, _sub(ppowmod([a, 1], (p - 1) // 2, h, p), [1], p), p)
            if 1 < len(w) < len(h):
                todo += [w, pdivmod(h, w, p)[0]]
                break
    return sorted(out)


def _sub(f, g, p):
    n = max(len(f), len(g))
    return pmod([(f[i] if i < len(f) else 0) - (g[i] if i < len(g) else 0) for i in range(n)], p)


def has_odd_factor(f, p):
    """Return whether a factor of f mod p > deg f has odd multiplicity (Yun)."""
    f = pmod(f, p)
    a = pgcd(f, pmod(pderiv(f), p), p)
    b = pdivmod(f, a, p)[0]
    k = 1
    while len(b) > 1:
        c = pgcd(a, b, p)
        if k % 2 and len(pdivmod(b, c, p)[0]) > 1:
            return True
        a, b, k = pdivmod(a, c, p)[0], c, k + 1
    return False


def padic_roots(f, p, k):
    """Return the roots in Z_p of f mod p^k (multiple roots only as far as refined)."""
    out = []

    def go(g, r0, m, depth):            # x = r0 + p^m y, g(y) = f(r0 + p^m y) / p^c
        c = min(val(x, p) for x in g if x)
        g = [x // p ** c for x in g]
        if m >= k or depth > 4 * k:
            out.append(r0 % p ** k)
            return
        dg = pderiv(g)
        for y in roots_mod(g, p):
            if peval(dg, y, p) % p:     # simple root: Newton doubles the precision
                z, n = y, 1
                while n < k - m:
                    n = min(2 * n, k - m)
                    M = p ** n
                    z = (z - peval(g, z, M) * pow(peval(dg, z, M), -1, M)) % M
                out.append((r0 + p ** m * z) % p ** k)
            else:
                h = _shift(g, y, p)
                go(h, r0 + p ** m * y, m + 1, depth + 1)
    go(list(f), 0, 0, 0)
    return sorted(set(out))


def _shift(g, y, p):
    """Return g(y + p x) as a polynomial in x."""
    n = len(g)
    h = [0] * n
    for i, c in enumerate(g):
        for j in range(i + 1):
            h[j] += c * math.comb(i, j) * y ** (i - j) * p ** j
    return h


# ---------- rationals

def convergents(x, B):
    """Yield the convergents of x with denominator <= B, in order."""
    x = Fraction(x)
    p0, q0, p1, q1 = 0, 1, 1, 0
    y = x
    while True:
        t = math.floor(y)
        p0, q0, p1, q1 = p1, q1, t * p1 + p0, t * q1 + q0
        if q1 > B:
            return
        yield Fraction(p1, q1)
        if y == t:
            return
        y = 1 / (y - t)


def bestappr(x, B):
    """Return the last convergent of the real x with denominator <= B, as PARI's bestappr."""
    x = Fraction(x)
    p0, q0, p1, q1 = 0, 1, 1, 0
    y = x
    while True:
        a = math.floor(y)
        p2, q2 = a * p1 + p0, a * q1 + q0
        if q2 > B:
            return Fraction(p1, q1)
        p0, q0, p1, q1 = p1, q1, p2, q2
        if y == a:
            return Fraction(p1, q1)
        y = 1 / (y - a)


# ---------- LLL

def lll_gram(G, delta=Fraction(99, 100), eta=Fraction(1, 2)):
    """Return the unimodular U (columns: the reduced basis) of the exact LLL of the Gram G (Cohen, Alg. 2.6.7)."""
    n = len(G)
    if all(type(x) is int for r in G for x in r):
        B = G
    else:
        den = 1
        for r in G:
            for x in r:
                den = den * Fraction(x).denominator // math.gcd(den, Fraction(x).denominator)
        B = [[int(Fraction(x) * den) for x in r] for r in G]
    H = [[int(i == j) for j in range(n)] for i in range(n)]       # H[i]: vector i on the old basis
    lam = [[0] * n for _ in range(n)]
    d = [1] + [0] * n                                             # d[i + 1] = d_i of Cohen
    dn, dd, en, ed = delta.numerator, delta.denominator, eta.numerator, eta.denominator

    def gram(i, j):
        return sum(H[i][a] * B[a][b] * H[j][b] for a in range(n) if H[i][a] for b in range(n) if H[j][b])

    def red(k, l):
        if ed * 2 * abs(lam[k][l]) > en * 2 * d[l + 1]:
            q = (2 * lam[k][l] + d[l + 1]) // (2 * d[l + 1])
            H[k] = [x - q * y for x, y in zip(H[k], H[l])]
            lam[k][l] -= q * d[l + 1]
            for i in range(l):
                lam[k][i] -= q * lam[l][i]

    k, kmax, steps = 1, 0, 0
    d[1] = gram(0, 0)
    while k < n:
        steps += 1
        if steps > 10 ** 5:                     # only if G is not positive definite
            raise ArithmeticError('lll_gram: no end (the Gram matrix is not positive definite)')
        if k > kmax:
            kmax = k
            for j in range(k + 1):
                u = gram(k, j)
                for i in range(j):
                    u = (d[i + 1] * u - lam[k][i] * lam[j][i]) // d[i]
                if j < k:
                    lam[k][j] = u
                else:
                    d[k + 1] = u
        red(k, k - 1)
        if dd * d[k + 1] * d[k - 1] < dn * d[k] * d[k] - dd * lam[k][k - 1] ** 2:
            H[k], H[k - 1] = H[k - 1], H[k]
            for j in range(k - 1):
                lam[k][j], lam[k - 1][j] = lam[k - 1][j], lam[k][j]
            l = lam[k][k - 1]
            Bn = (d[k - 1] * d[k + 1] + l * l) // d[k]
            for i in range(k + 1, kmax + 1):
                t = lam[i][k]
                lam[i][k] = (d[k + 1] * lam[i][k - 1] - l * t) // d[k]
                lam[i][k - 1] = (Bn * t + l * lam[i][k]) // d[k + 1]
            d[k] = Bn
            k = max(1, k - 1)
        else:
            for l in range(k - 2, -1, -1):
                red(k, l)
            k += 1
    return [[H[j][i] for j in range(n)] for i in range(n)]


# ---------- real and complex roots

def root_bound_bits(f):
    """Return e + 1 with |z| < 2^(e + 1) for every complex root z of f (Fujiwara, from bit lengths)."""
    d, lead = len(f) - 1, abs(f[-1]).bit_length()
    return 1 + max([0] + [-((lead - 1 - abs(c).bit_length()) // (d - i)) for i, c in enumerate(f[:-1]) if c])


def _sign_at(f, N, s):
    """Return the sign of f(N / 2^s), exactly."""
    d = len(f) - 1                      # Horner on the homogenised f(N, 2^s)
    v = f[d]
    for i in range(d - 1, -1, -1):
        v = v * N + (f[i] << (s * (d - i)))
    return (v > 0) - (v < 0)


def sturm(f):
    """Return an integer Sturm sequence of the squarefree integer polynomial f."""
    seq = [f, pderiv(f)]
    while len(seq[-1]) > 1:
        a, b = seq[-2], seq[-1]
        delta = len(a) - len(b)
        lc = b[-1]
        r = [c * lc ** (delta + 1) for c in a]
        for k in range(len(r) - len(b), -1, -1):        # r = lc^(delta+1) a mod b
            q = r[k + len(b) - 1] // lc
            for j, c in enumerate(b):
                r[k + j] -= q * c
        r = ptrim(r[:len(b) - 1])
        if not r:
            break
        g = 0
        for c in r:
            g = math.gcd(g, c)
        sgn = -1 if lc < 0 and (delta + 1) % 2 else 1
        seq.append([-sgn * c // g for c in r])
    return seq


def _variations(seq, N, s):
    """Return the sign variations of the Sturm sequence seq at N / 2^s (zeros skipped)."""
    v, last = 0, 0
    for g in seq:
        x = _sign_at(g, N, s)
        if x:
            v += last and x != last
            last = x
    return v


def real_roots_floor(f, M):
    """Return floor(z 2^M) for the real roots z of the squarefree f, increasing (exact, by Sturm)."""
    f = ptrim([int(c) for c in f])
    d = len(f) - 1
    if d < 1:
        return []
    if d == 1:
        return [(-f[0] << M) // f[1]]
    S = sturm(f)
    eb = root_bound_bits(f)
    count = lambda lo, hi, s: _variations(S, lo, s) - _variations(S, hi, s)     # roots in (lo, hi] / 2^s
    out = []
    todo = [(-(1 << eb), 1 << eb, 0, count(-(1 << eb), 1 << eb, 0))]
    while todo:
        lo, hi, s, n = todo.pop()
        if n == 0:
            continue
        if n == 1 and _sign_at(f, lo, s) and _sign_at(f, hi, s):
            t = max(s, M)                               # one simple root: bisect the sign change
            lo, hi = lo << (t - s), hi << (t - s)
            a = _sign_at(f, lo, t)
            while hi - lo > 1:
                m = (lo + hi) // 2
                v = _sign_at(f, m, t)
                if v == 0:
                    lo = hi = m
                elif v == a:
                    lo = m
                else:
                    hi = m
            out.append(lo >> (t - M))
            continue
        if n == 1 and _sign_at(f, hi, s) == 0:
            out.append(hi << (M - s) if s <= M else hi >> (s - M))
            continue
        if hi - lo == 1:
            lo, hi, s = 2 * lo, 2 * hi, s + 1
        m = (lo + hi) // 2
        c = count(lo, m, s)
        todo += [(m, hi, s, n - c), (lo, m, s, c)]
    return sorted(out)


def roots(f, digits=60):
    """Return (the real roots, the complex roots with im > 0) of f, squarefree with at most one complex pair."""
    M = int(digits * 3.33) + 8
    with localcontext() as ctx:
        ctx.prec = digits + 20
        fl = real_roots_floor(f, M)
        if any(x == y for x, y in zip(fl, fl[1:])):
            raise ValueError('roots: two real roots closer than 2^-%d' % M)
        rr = [Decimal(z) / Decimal(2) ** M for z in fl]
        q = [Decimal(int(c)) for c in ptrim(list(f))]
        if len(rr) == len(q) - 1:
            return [+z for z in rr], []
        for z in rr:                    # deflate from the end that does not cancel: z's error is relative to |z|
            n = len(q) - 1
            b = [Decimal(0)] * n
            if abs(z) > 1:
                b[0] = -q[0] / z
                for i in range(1, n):
                    b[i] = (b[i - 1] - q[i]) / z
            else:
                b[n - 1] = q[n]
                for i in range(n - 1, 0, -1):
                    b[i - 1] = q[i] + z * b[i]
            q = b
        if len(q) != 3:
            raise ValueError('roots: more than one complex pair')
        c, b, a = q
        return [+z for z in rr], [(+(-b / (2 * a)), +((4 * a * c - b * b).sqrt() / (2 * abs(a))))]


if __name__ == '__main__':              # curve.py's 'pari' task
    print([p for p, _ in factor(int(sys.argv[1]))])


# ========== integer linear algebra

def solve3(M, v):
    """Return x with M x = v for a nonsingular rational 3 x 3 M."""
    R = []
    for r, b in zip(M, v):
        r = [Fraction(c) for c in r] + [Fraction(b)]
        L = 1
        for c in r:
            L = L * c.denominator // math.gcd(L, c.denominator)
        R.append([c.numerator * (L // c.denominator) for c in r])
    d = det3([r[:3] for r in R])
    if d:
        return tuple(Fraction(det3([[R[i][3] if j == k else R[i][j] for j in range(3)] for i in range(3)]), d)
                     for k in range(3))
    M = [[Fraction(c) for c in r] + [Fraction(b)] for r, b in zip(M, v)]     # singular: raise below
    for c in range(3):
        piv = next(r for r in range(c, 3) if M[r][c])
        M[c], M[piv] = M[piv], M[c]
        for r in range(3):
            if r != c and M[r][c]:
                t = M[r][c] / M[c][c]
                M[r] = [a - t * b for a, b in zip(M[r], M[c])]
    return tuple(M[i][3] / M[i][i] for i in range(3))


def _kernel_modp(rows, n, p):
    """Return a basis of {x in F_p^n : rows x = 0}."""
    M = [[c % p for c in r] for r in rows if any(c % p for c in r)]
    piv, R = [], []
    for col in range(n):
        k = next((i for i in range(len(R), len(M)) if M[i][col]), None)
        if k is None:
            continue
        M[len(R)], M[k] = M[k], M[len(R)]
        r = M[len(R)]
        try:
            inv = pow(r[col], -1, p)
        except ValueError:              # p is a fake prime (see conics)
            raise ZeroDivisionError(r[col])
        r = [c * inv % p for c in r]
        M[len(R)] = r
        for i in range(len(M)):
            if i != len(R) and M[i][col]:
                t = M[i][col]
                M[i] = [(a - t * b) % p for a, b in zip(M[i], r)]
        R.append(r)
        piv.append(col)
    out = []
    for free in [c for c in range(n) if c not in piv]:
        v = [0] * n
        v[free] = 1
        for r, pc in zip(M, piv):
            v[pc] = -r[free] % p
        out.append(v)
    return out


def _xgcd(a, b):
    """Return (g, u, v) with g = gcd(a, b) = u a + v b, for b != 0."""
    g = math.gcd(a, b)
    m = abs(b // g)
    u = pow(a // g, -1, m) if m > 1 else 0
    return g, u, (g - u * a) // b


def hnf(rows, d=None):
    """Return the upper triangular HNF of the full-rank lattice spanned by rows (and d Z^3, mod d: Cohen 2.4.8)."""
    if d is not None:
        return _hnf_mod(rows, abs(d))
    rows = [list(r) for r in rows]
    out = []
    for col in range(3):
        rows = [r for r in rows if any(r)]
        while True:
            nz = [r for r in rows if r[col]]
            if len(nz) <= 1:
                break
            nz.sort(key=lambda r: abs(r[col]))
            piv = nz[0]
            for r in nz[1:]:
                q = r[col] // piv[col]
                for j in range(3):
                    r[j] -= q * piv[j]
        piv = next(r for r in rows if r[col])
        rows.remove(piv)
        if piv[col] < 0:
            piv = [-c for c in piv]
        out.append(piv)
    return _hnf_top(out)


def _hnf_top(out):
    """Reduce the entries above the diagonal of out into [0, diagonal)."""
    for i in range(3):
        for k in range(i):
            q = out[k][i] // out[i][i]
            out[k] = [a - q * b for a, b in zip(out[k], out[i])]
    return out


def _hnf_mod(rows, d):
    """Return hnf(rows, d), by extended-gcd steps with entries mod d."""
    pool = [[x % d for x in r] for r in rows]
    out = []
    for c in range(3):
        piv, rest = [0, 0, 0], []
        piv[c] = d
        for r in pool:
            if r[c] % piv[c] == 0:
                q = r[c] // piv[c]
                r2 = [(y - q * x) % d for x, y in zip(piv, r)]
            else:
                g, u, v = _xgcd(piv[c], r[c])
                a, b = piv[c] // g, r[c] // g
                piv, r2 = [(u * x + v * y) % d for x, y in zip(piv, r)], [(a * y - b * x) % d for x, y in zip(piv, r)]
                piv[c] = g
            if any(r2):
                rest.append(r2)
        out.append(piv)
        pool = rest
    return _hnf_top(out)


# ========== conics, as PARI's qfsolve and qfparam
# [Simon 2005; Cremona-Rusin 2003].  The "primes" may be unfactored: an impossible inverse or square root mod one
# raises Composite with a factor.

class Composite(Exception):
    """n, treated as a prime, is composite: g is a proper factor."""
    def __init__(self, n, g):
        super().__init__('composite %d' % n)
        self.n, self.g = n, g


def det3(G):
    (a, b, c), (d, e, f), (g, h, i) = G
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def tmul(B, G):
    """Return B^T G B (the columns of B are the new basis)."""
    n = len(B[0])
    GB = [[sum(G[i][k] * B[k][j] for k in range(3)) for j in range(n)] for i in range(3)]
    return [[sum(B[k][i] * GB[k][j] for k in range(3)) for j in range(n)] for i in range(n)]


def mmul(A_, B):
    return [[sum(A_[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A_))]


def cols(rows):
    return [[rows[j][i] for j in range(len(rows))] for i in range(len(rows[0]))]


def kern(G, p):
    try:
        return _kernel_modp(G, 3, p)
    except ZeroDivisionError as e:
        raise Composite(p, math.gcd(e.args[0], p))


def sqrtm(x, p):
    x %= p
    if p == 2 or x == 0:
        return x
    if pow(x, (p - 1) // 2, p) != 1:
        if is_prime(p):
            return None
        g = math.gcd(pow(x, (p - 1) // 2, p) - 1, p)
        raise Composite(p, g if 1 < g < p else _rho(p) or p)
    r = sqrt_mod(x, p)
    if r * r % p != x:
        raise Composite(p, math.gcd(r * r - x, p))
    return r


def solve(G, P0=(), cof=()):
    """Return a nonzero integer zero of G, or None; outside P0, the primes of det G divide cof."""
    den = 1
    for r in G:
        for x in r:
            den = den * Fraction(x).denominator // math.gcd(den, Fraction(x).denominator)
    G = [[int(Fraction(x) * den) for x in r] for r in G]
    c = 0
    for r in G:
        for x in r:
            c = math.gcd(c, x)
    G = [[x // c for x in r] for r in G]
    d = abs(det3(G))
    P0 = sorted(set(P0) | {p for p, _ in factor(den)} | {p for p, _ in factor(c)} | {2, 3})
    r = d
    for p in P0:
        r //= p ** val(r, p)
    base = coprime([math.gcd(r, x) for x in cof] + [r])
    for _ in range(8):
        try:
            return _solve(G, P0 + base)
        except Composite as e:
            base = coprime(base + [e.g, e.n // e.g])
    return _solve(G, P0 + sorted({p for b in base for p, _ in factor(b)}))


def coprime(L):
    """Return the coprime base of the integers > 1 in L."""
    B = sorted({x for x in L if x > 1})
    while True:
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                g = math.gcd(B[i], B[j])
                if g > 1:
                    B = sorted({x for x in B[:i] + B[i + 1:j] + B[j + 1:] + [g, B[i] // g, B[j] // g] if x > 1})
                    break
            else:
                continue
            break
        else:
            return B


def _solve(G, ps):
    M = [[int(i == j) for j in range(3)] for i in range(3)]
    for p in ps:
        while det3(G) % (p * p) == 0:
            k = kern(G, p)
            if len(k) == 3:
                G = [[x // p for x in r] for r in G]
                continue
            if len(k) == 2:                     # B^T G B / p on the lattice x = k mod p
                B = cols(hnf([[p * (i == j) for j in range(3)] for i in range(3)] + k))
                G = [[x // p for x in r] for r in tmul(B, G)]
                M = mmul(M, B)
                continue
            if len(k) != 1:
                raise RuntimeError('qfsolve: no kernel mod %d' % p)
            B = cols(hnf([[p * (i == j) for j in range(3)] for i in range(3)] + k))      # p (Z^3 + Z k/p)
            H = tmul(B, G)
            if any(x % (p * p) for r in H for x in r):
                return None
            G = [[x // (p * p) for x in r] for r in H]
            M = [[Fraction(x, p) for x in r] for r in mmul(M, B)]
    d = det3(G)
    if d == 0:
        return None
    # the isotropic line mod each p | d, as a linear form, joined by CRT
    phi, mod = [0, 0, 0], 1
    for p in ps:
        if d % p:
            continue
        k = kern(G, p)[0]
        v = None
        if p == 2:
            v = next((u for u in ([a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1))
                      if any(u) and _indep(k, u, 2) and qv(G, u) % 2 == 0), None)
        else:
            u, w = [e for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1]) if _indep(k, e, p)][:2]
            if not _indep3(k, u, w, p):
                w = next(e for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1]) if _indep3(k, u, e, p))
            a, b, c = qv(G, u), bil(G, u, w), qv(G, w)
            if a % p == 0:
                v = u
            else:
                r = sqrtm(b * b - a * c, p)
                if r is None:
                    return None
                s = (r - b) * _inv(a, p) % p
                v = [(s * x + y) % p for x, y in zip(u, w)]
        if v is None:
            return None
        f = [(k[1] * v[2] - k[2] * v[1]) % p, (k[2] * v[0] - k[0] * v[2]) % p, (k[0] * v[1] - k[1] * v[0]) % p]
        phi = [crt(x, mod, y, p) for x, y in zip(phi, f)]
        mod *= p
    gens = [[mod * (i == j) for j in range(3)] for i in range(3)]
    gens += [[phi[j] if t == i else -phi[i] if t == j else 0 for t in range(3)] for i in range(3) for j in range(i + 1, 3)]
    B = cols(hnf(gens))
    H = majorant(tmul(B, G))
    if H is None:
        return None
    U = lll_gram(H)
    R = mmul(B, U)
    for s in range(1, 5):
        for c in ((a, b, c) for a in range(-s, s + 1) for b in range(-s, s + 1) for c in range(-s, s + 1)
                  if max(abs(a), abs(b), abs(c)) == s):
            x = [sum(R[i][j] * c[j] for j in range(3)) for i in range(3)]
            if qv(G, x) == 0:
                y = [sum(M[i][j] * x[j] for j in range(3)) for i in range(3)]
                return primitive(y)
    raise RuntimeError('qfsolve: no zero among the small vectors')


def _inv(a, p):
    try:
        return pow(a, -1, p)
    except ValueError:
        raise Composite(p, math.gcd(a, p))


def _indep(k, u, p):
    return any((k[i] * u[j] - k[j] * u[i]) % p for i in range(3) for j in range(i + 1, 3))


def _indep3(k, u, w, p):
    return det3([k, u, w]) % p != 0


def qv(G, x):
    return sum(G[i][j] * x[i] * x[j] for i in range(3) for j in range(3))


def bil(G, x, y):
    return sum(G[i][j] * x[i] * y[j] for i in range(3) for j in range(3))


def majorant(H):
    """Return a positive definite majorant of H (from an LDL^T), or None if H is definite."""
    import random
    rng = random.Random(1)
    T = [[int(i == j) for j in range(3)] for i in range(3)]
    for _ in range(50):
        K = tmul(T, [[Fraction(x) for x in r] for r in H])
        L, D = [[Fraction(int(i == j)) for j in range(3)] for i in range(3)], []
        for i in range(3):
            D.append(K[i][i] - sum(L[i][k] ** 2 * D[k] for k in range(i)))
            if D[i] == 0:
                break
            for j in range(i + 1, 3):
                L[j][i] = (K[j][i] - sum(L[j][k] * L[i][k] * D[k] for k in range(i))) / D[i]
        if len(D) == 3 and D[2]:
            if all(x > 0 for x in D) or all(x < 0 for x in D):
                return None
            M = [[sum(L[a][k] * abs(D[k]) * L[b][k] for k in range(3)) for b in range(3)] for a in range(3)]
            return tmul(_inverse(T), M)
        T = [[int(i == j) + (rng.randrange(-2, 3) if j > i else 0) for j in range(3)] for i in range(3)]
        T = mmul(T, [[int(i == j) + (rng.randrange(-2, 3) if j < i else 0) for j in range(3)] for i in range(3)])
    raise RuntimeError('majorant: no nonzero pivots')


def _inverse(T):
    """Return T^-1, exactly."""
    T = [[Fraction(c) for c in r] for r in T]
    d = det3(T)
    if not d:
        cols_ = [solve3(T, e) for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]      # raises
        return [[cols_[j][i] for j in range(3)] for i in range(3)]
    cof = lambda i, j: (T[(i + 1) % 3][(j + 1) % 3] * T[(i + 2) % 3][(j + 2) % 3]
                        - T[(i + 1) % 3][(j + 2) % 3] * T[(i + 2) % 3][(j + 1) % 3])
    return [[cof(j, i) / d for j in range(3)] for i in range(3)]


def primitive(v):
    v = [Fraction(x) for x in v]
    d = 1
    for x in v:
        d = d * x.denominator // math.gcd(d, x.denominator)
    v = [int(x * d) for x in v]
    g = 0
    for x in v:
        g = math.gcd(g, x)
    return [x // g for x in v]


def param(G, s):
    """Return forms [q_0, q_1, q_2] in x^2, x z, z^2 parametrising x^T G x = 0 from its zero s."""
    B = _complete(s)
    H = tmul(B, [[Fraction(x) for x in r] for r in G])
    # Q(X s + Y v + Z w) = 2 X (H01 Y + H02 Z) + q(Y, Z); point = -q s + l (Y v + Z w)
    b1, b2 = H[0][1], H[0][2]
    q = [H[1][1], 2 * H[1][2], H[2][2]]
    l = [2 * b1, 2 * b2]
    out = []
    for i in range(3):
        s_i, v_i, w_i = B[i][0], B[i][1], B[i][2]
        out.append([-q[0] * s_i + l[0] * v_i, -q[1] * s_i + l[0] * w_i + l[1] * v_i, -q[2] * s_i + l[1] * w_i])
    return out


def _complete(s):
    """Return a unimodular matrix with first column s."""
    v, U = list(s), [[int(i == j) for j in range(3)] for i in range(3)]
    while sum(1 for x in v if x) > 1:
        i, j = sorted((t for t in range(3) if v[t]), key=lambda t: -abs(v[t]))[:2]
        q = v[i] // v[j]
        v[i] -= q * v[j]
        U[i] = [x - q * y for x, y in zip(U[i], U[j])]
    k = next(t for t in range(3) if v[t])
    U[0], U[k] = U[k], U[0]
    if v[k] < 0:
        U[0] = [-x for x in U[0]]
    Ui = _inverse(U)
    return [[int(x) for x in r] for r in Ui]


# ========== local arithmetic, ported from check.gp (names in parentheses)

ENUM = 64                   # square_mod tries every residue below this prime (check.gp's TDENUM)


# ---------- square classes of Q_v

def eps(p):
    e = 2
    while kronecker(e, p) != -1:
        e += 1
    return e


def basis(p):
    return [-1] if p == 0 else [-1, 5, 2] if p == 2 else [eps(p), p]


def loc(d, p):
    """Return the square class of the rational d != 0 at p (tdloc)."""
    d = Fraction(d)
    if p == 0:
        return [int(d < 0)]
    v = val(d, p)
    u = d / Fraction(p) ** v
    if p == 2:
        u = u.numerator * pow(u.denominator, -1, 8) % 8
        return [int(u % 4 == 3), int(u in (3, 5)), v % 2]
    return [int(kronecker(u.numerator * u.denominator % p, p) == -1), v % 2]


def rep(c, p):
    return math.prod(b ** x for b, x in zip(basis(p), c))


def isog_places(S):
    return [0] + sorted(set(S) | {3, 5, 7})


# ---------- local solubility and its witnesses

def content_val(H, p):
    return min(val(c, p) for c in H if c)


def shift(H, r, q):
    """Return H(r + q x)."""
    return ptrim(_shift(list(H), r, 1) if q == 1 else [c * q ** i for i, c in enumerate(_shift(list(H), r, 1))])


def square_mod(H, p):
    """Return whether H(s) is a nonzero square mod p for some s (tdsquare)."""
    h = pmod(H, p)
    if p < ENUM:
        return any(z and kronecker(z, p) == 1 for z in (peval(h, s, p) for s in range(p)))
    if len(h) <= 1:
        return kronecker(h[0], p) == 1
    return has_odd_factor(h, p) or kronecker(h[-1], p) == 1


def _drive(H, p, frames):
    """Return the s found by a depth-first search of frames, or None (explicit stack: the depth is unbounded).

    frames(H, p) yields (s,) or a child (r, q, H(r + q x))."""
    stack, n = [(frames(H, p), 0, 1)], 0
    while stack:
        n += 1
        if n % 256 == 0:
            tick()
        try:
            x = next(stack[-1][0])
        except StopIteration:
            stack.pop()
            continue
        if len(x) == 1:
            s = x[0]
            for _, r, q in reversed(stack):
                s = r + q * s
            return s
        stack.append((frames(x[2], p), x[0], x[1]))
    return None


def _zp_frames(H, p):
    c = content_val(H, p)
    if c >= 2:
        H = [x // p ** (2 * (c // 2)) for x in H]
        c = c % 2
    if c == 1:
        R = roots_mod([x // p for x in H], p)
    else:
        if p == 2:
            for s0 in range(8):
                h = peval(H, s0)
                if h == 0 or h % 8 == 1:
                    yield (0,)
                    return
                if h % 2 == 0:
                    yield 0, 1, shift(H, s0, 8)
            return
        if square_mod(H, p):
            yield (0,)
            return
        R = roots_mod(H, p)
    for r in R:
        yield 0, 1, shift(H, r, p)


def zp(H, p):
    """Return whether H(s) is a square in Q_p for some s in Z_p (tdzp)."""
    return _drive(H, p, _zp_frames) is not None


def soluble(g, p):
    """Return whether y^2 = g(u, v) has a Q_p-point (tdquartic)."""
    return zp(g[::-1], p) or zp([c * p ** i for i, c in enumerate(g)], p)


def sqwit(H, p):
    """Return s with H(s) a nonzero square mod p, or None (tdsqwit)."""
    h = pmod(H, p)
    if p < ENUM:
        return next((s for s in range(p) if peval(h, s, p) and kronecker(peval(h, s, p), p) == 1), None)
    if not square_mod(H, p):
        return None
    rng = random.Random(p)                      # deterministic witnesses
    for _ in range(10 ** 4):
        s = rng.randrange(p)
        z = peval(h, s, p)
        if z and kronecker(z, p) == 1:
            return s
    raise RuntimeError('tdsqwit: no residue found mod %d' % p)


def _wit_frames(H, p):
    c = content_val(H, p)
    q = p
    if c >= 2:
        H = [x // p ** (2 * (c // 2)) for x in H]
        c = c % 2
    if c == 1:
        R = roots_mod([x // p for x in H], p)
    elif p == 2:
        for s in range(8):
            if peval(H, s) % 8 == 1:
                yield (s,)
                return
        R, q = [s for s in range(8) if peval(H, s) % 2 == 0], 8
    else:
        w = sqwit(H, p)
        if w is not None:
            yield (w,)
            return
        R = roots_mod(H, p)
    for r in R:
        yield r, q, shift(H, r, q)


def wit(H, p):
    """Return an integer s with H(s) a nonzero square in Q_p, or None (tdwit)."""
    return _drive(H, p, _wit_frames)


# ---------- F_2: set bits, kernels and solutions

def bits(x):
    """Return the indices of the set bits of x."""
    out = []
    while x:
        low = x & -x
        out.append(low.bit_length() - 1)
        x ^= low
    return out


def kernel2(rows, n):
    """Return a basis of the F_2 kernel of rows."""
    M = []
    for r in rows:
        v = sum(b << i for i, b in enumerate(r))
        for pc, w in M:
            if v >> pc & 1:
                v ^= w
        if v:
            pc = v.bit_length() - 1
            M = [(q, w ^ v if w >> pc & 1 else w) for q, w in M] + [(pc, v)]
    out = []
    for f in [c for c in range(n) if c not in {q for q, _ in M}]:
        x = 1 << f
        for q, w in M:
            if w >> f & 1:
                x |= 1 << q
        out.append([x >> i & 1 for i in range(n)])
    return out


def solve2(rows, v):
    """Return c with sum c_i rows_i = v over F_2, or None."""
    n = len(rows)
    k = kernel2([[r[j] for r in rows] + [v[j]] for j in range(len(v))], n + 1)
    x = next((w for w in k if w[n]), None)
    return None if x is None else x[:n]
