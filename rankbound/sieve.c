/* sieve: rankbound's special-q lattice siever on one algebraic side, for balanced to lopsided fields.
   libc and libm only.  Nothing printed is trusted: the checker refactors every relation.

   F(x, y) = c0 y^3 + c1 x y^2 + c2 x^2 y + c3 x^3 (CADO's poly0 order).  A relation is (a, b), b > 0, gcd(a, b) = 1,
   with |F(a, b)| a product of primes <= LIM and a cofactor <= 2^MFB of primes <= 2^LPB.  Roots r = p mean (1 : 0).
   A special-q's lattice points are (a, b) = i u + j v (for skew >= q, u = (q, 0) and v = (r, 1), so j = b), and a
   root (R : 1) of p hits where i = rho j mod p, rho = -x1/x0, x0 = a_u - R b_u, x1 = a_v - R b_v (for (1 : 0):
   x0 = b_u, x1 = b_v), or on the lines j = 0 mod p when x0 = 0.

   Output, las's format: "a,b:p1,p2,...:" per relation (the primes of |F(a, b)| in hex with multiplicity, q included,
   ascending), then "# Total cpu time Ts" and "# Total N reports [Xs/r, Yr/sq]".
     sieve roots c0,c1,c2,c3 LIM OUT     "p: r1,r2,..." for the primes p <= LIM (makefb's format)
     sieve sieve -poly c0,c1,c2,c3 -lim L -lpb B -mfb M -N N [-skew S] [-roots FILE] [-q0 Q0 -q1 Q1]
                 [-sample n -seed X] [-out FILE] [-server] [-v]
         q prime in [Q0, Q1), r an affine root, 2^N positions each (las's -I I: N = 2 I - 1); -sample: n of them.
         -server: each stdin line "Q0 Q1" is answered with its relations, the totals and "# done"; SIGUSR1 ends the
         slice before its next special-q with "# stopped before special-q Q", the totals and "# done".  With
         RANKBOUND_PARENT (a pid) set, it dies with its parent even if the parent is killed. */
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#else
#include <pthread.h>
#endif

typedef unsigned __int128 u128;
typedef __int128 i128;

#define LOGB 16                 /* a bucket region: 2^LOGB positions (one sieve array) */
#define BR (1 << LOGB)
#define PSKIP 32                /* primes below this are not sieved */
#define SEG 32                  /* positions per threshold segment */
#define NPR 64                  /* most primes in a relation */

static void die(const char *m) { fprintf(stderr, "sieve: %s\n", m); exit(2); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory"); return p; }
static double cpu_now(void) { struct timespec t; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* ---------- 64-bit arithmetic: primality, rho, inverses */

static uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m) { return (uint64_t)((u128)a * b % m); }
static uint64_t powmod64(uint64_t a, uint64_t e, uint64_t m) {
    uint64_t r = 1 % m; a %= m;
    for (; e; e >>= 1) { if (e & 1) r = mulmod64(r, a, m); a = mulmod64(a, a, m); }
    return r;
}
static uint64_t bgcd(uint64_t a, uint64_t b) {
    if (!a || !b) return a | b;
    int k = __builtin_ctzll(a | b); a >>= __builtin_ctzll(a);
    do { b >>= __builtin_ctzll(b); if (a > b) { uint64_t t = a; a = b; b = t; } b -= a; } while (b);
    return a << k;
}
/* Montgomery arithmetic mod an odd n < 2^63, R = 2^64 (u128 % dominates the cofactor kernels otherwise) */
typedef struct { uint64_t n, ni, r2, one; } mont_t;
static inline uint64_t mredc(const mont_t *M, u128 t) {
    uint64_t m = (uint64_t)t * M->ni;
    u128 u = (t + (u128)m * M->n) >> 64;
    return u >= M->n ? (uint64_t)(u - M->n) : (uint64_t)u;
}
static inline uint64_t mmul(const mont_t *M, uint64_t a, uint64_t b) { return mredc(M, (u128)a * b); }
static void mont_init(mont_t *M, uint64_t n) {
    uint64_t x = n;
    for (int i = 0; i < 6; i++) x *= 2 - n * x;
    u128 r = ((u128)1 << 64) % n;
    M->n = n; M->ni = (uint64_t)0 - x; M->one = (uint64_t)r; M->r2 = (uint64_t)(r * r % n);
}
static uint64_t mpow(const mont_t *M, uint64_t a, uint64_t e) {
    uint64_t r = M->one;
    for (; e; e >>= 1) { if (e & 1) r = mmul(M, r, a); a = mmul(M, a, a); }
    return r;
}
static int isprime64(uint64_t n) {             /* n < 2^63; deterministic (Sinclair's 7 bases) */
    static const uint64_t SP[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}, B[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    if (n < 2) return 0;
    for (int i = 0; i < 12; i++) if (n % SP[i] == 0) return n == SP[i];
    if (n < 1369) return 1;
    mont_t M; mont_init(&M, n);
    uint64_t d = n - 1, mone = n - M.one; int s = 0;
    while (!(d & 1)) { d >>= 1; s++; }
    for (int i = 0; i < 7; i++) {
        uint64_t a = B[i] % n;
        if (!a) continue;
        uint64_t x = mpow(&M, mmul(&M, a, M.r2), d);
        if (x == M.one || x == mone) continue;
        int k = 1;
        for (; k < s; k++) { x = mmul(&M, x, x); if (x == mone) break; }
        if (k == s) return 0;
    }
    return 1;
}
static uint64_t rho64(uint64_t n) {            /* a nontrivial factor of the odd composite n < 2^63 (Brent, gcd per 64) */
    mont_t M; mont_init(&M, n);
    for (uint64_t c0 = 1;; c0++) {
        uint64_t c = mmul(&M, c0, M.r2), y = mmul(&M, 2, M.r2), x = y, ys = y, q = M.one, g = 1;
        for (uint64_t r = 1; g == 1 && r < (1ULL << 30); r <<= 1) {
            x = y;
            for (uint64_t i = 0; i < r; i++) { y = mmul(&M, y, y) + c; if (y >= n) y -= n; }
            for (uint64_t k = 0; k < r && g == 1; k += 64) {
                ys = y;
                for (uint64_t i = 0; i < 64 && i < r - k; i++) { y = mmul(&M, y, y) + c; if (y >= n) y -= n; q = mmul(&M, q, x > y ? x - y : y - x); }
                g = bgcd(q, n);
            }
        }
        if (g == n) do { ys = mmul(&M, ys, ys) + c; if (ys >= n) ys -= n; g = bgcd(x > ys ? x - ys : ys - x, n); } while (g == 1);
        if (g != n && g != 1) return g;
    }
}
static uint32_t invmod32(uint32_t a, uint32_t p) {
    int64_t t = 0, nt = 1; uint32_t r = p, nr = a;
    while (nr) { uint32_t k = r / nr, x = r - k * nr; r = nr; nr = x; int64_t y = t - (int64_t)k * nt; t = nt; nt = y; }
    return (uint32_t)(t < 0 ? t + p : t);
}
static uint64_t inv2_64(uint64_t p) {                /* mod 2^64, p odd */
    uint64_t x = p;
    for (int i = 0; i < 6; i++) x *= 2 - p * x;
    return x;
}

/* ---------- fixed-width integers: magnitude in NL 64-bit limbs (d[0..n)), and a sign */

#define NL 8
typedef struct { int n, neg; uint64_t d[NL]; } big;
static int big_over;                            /* a product needed more than NL limbs */

static void big_set_i64(big *x, int64_t v) {
    x->neg = v < 0; x->d[0] = v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v; x->n = x->d[0] != 0;
}
static void big_mul_i64(big *r, const big *x, int64_t m) {
    uint64_t um = m < 0 ? (uint64_t)0 - (uint64_t)m : (uint64_t)m;
    u128 c = 0; int n = um ? x->n : 0;
    for (int i = 0; i < n; i++) { c += (u128)x->d[i] * um; r->d[i] = (uint64_t)c; c >>= 64; }
    if (c) { if (n == NL) big_over = 1; else r->d[n++] = (uint64_t)c; }
    r->n = n; r->neg = n ? x->neg ^ (m < 0) : 0;
}
static int mag_cmp(const big *x, const big *y) {
    if (x->n != y->n) return x->n < y->n ? -1 : 1;
    for (int i = x->n - 1; i >= 0; i--) if (x->d[i] != y->d[i]) return x->d[i] < y->d[i] ? -1 : 1;
    return 0;
}
static void mag_sub(big *r, const big *x, const big *y) {     /* |x| >= |y| */
    uint64_t br = 0; int n = x->n, top = 0;
    for (int i = 0; i < n; i++) {
        uint64_t yi = i < y->n ? y->d[i] : 0, t = x->d[i] - yi - br;
        br = (x->d[i] < yi) | ((x->d[i] - yi) < br);
        r->d[i] = t;
        if (t) top = i + 1;
    }
    r->n = top;
}
static void big_add(big *r, const big *x, const big *y) {
    if (x->neg == y->neg) {
        const big *a = x->n >= y->n ? x : y, *b = x->n >= y->n ? y : x;
        u128 c = 0; int n = a->n;
        for (int i = 0; i < n; i++) { c += (u128)a->d[i] + (i < b->n ? b->d[i] : 0); r->d[i] = (uint64_t)c; c >>= 64; }
        if (c) { if (n == NL) big_over = 1; else r->d[n++] = 1; }
        r->n = n; r->neg = x->neg;
    } else if (mag_cmp(x, y) >= 0) { int s = x->neg; mag_sub(r, x, y); r->neg = r->n ? s : 0; }
    else { int s = y->neg; mag_sub(r, y, x); r->neg = r->n ? s : 0; }
}
static int big_parse(big *x, const char *s, const char *end) {    /* decimal [s, end), optional sign */
    int neg = 0;
    memset(x, 0, sizeof *x);
    if (s < end && (*s == '-' || *s == '+')) neg = *s++ == '-';
    if (s == end) return -1;
    for (; s < end; s++) {
        if (*s < '0' || *s > '9') return -1;
        big t; big_mul_i64(&t, x, 10); *x = t; x->neg = 0;
        big d; big_set_i64(&d, *s - '0'); big_add(x, x, &d);
        if (big_over || x->n == NL) return -1;
    }
    x->neg = neg && x->n;
    return 0;
}
static uint32_t big_mod32(const big *x, uint32_t p) {
    uint64_t r = 0;
    for (int i = x->n - 1; i >= 0; i--) { r = ((r << 32) | (x->d[i] >> 32)) % p; r = ((r << 32) | (x->d[i] & 0xffffffffu)) % p; }
    return (uint32_t)r;
}
static int big_bits(const big *x) { return x->n ? 64 * (x->n - 1) + 64 - __builtin_clzll(x->d[x->n - 1]) : 0; }
/* |x| /= p if p divides it (p odd, Hensel's exact division); return whether it did. */
static int big_divexact(big *x, uint64_t p, uint64_t pinv) {
    uint64_t c = 0, qd[NL];
    for (int i = 0; i < x->n; i++) {
        uint64_t s = x->d[i], l = s - c, bo = s < c;
        uint64_t qi = l * pinv;
        qd[i] = qi;
        c = (uint64_t)(((u128)qi * p) >> 64) + bo;
    }
    if (c) return 0;
    int n = x->n;
    for (int i = 0; i < n && i < NL; i++) x->d[i] = qd[i];
    while (n && !x->d[n - 1]) n--;
    x->n = n;
    return 1;
}
static int big_div2(big *x) {
    if (!x->n || (x->d[0] & 1)) return 0;
    for (int i = 0; i < x->n; i++) x->d[i] = (x->d[i] >> 1) | (i + 1 < x->n ? x->d[i + 1] << 63 : 0);
    if (!x->d[x->n - 1]) x->n--;
    return 1;
}

/* ---------- the polynomial */

static big CB[4];
static double CD[4];
static int parse_poly(const char *s) {
    for (int i = 0; i < 4; i++) {
        const char *e = strchr(s, ',');
        if (!e) e = s + strlen(s);
        if ((i < 3) != (*e == ',') || big_parse(&CB[i], s, e)) return -1;
        char buf[256]; size_t n = e - s < 255 ? (size_t)(e - s) : 255; memcpy(buf, s, n); buf[n] = 0;
        CD[i] = strtod(buf, 0);
        s = e + 1;
    }
    return CB[3].n ? 0 : -1;
}
static void normF(big *N, int64_t a, int64_t b) {
    big t, u;
    big_mul_i64(&t, &CB[3], a);
    big_mul_i64(&u, &CB[2], b); big_add(&t, &t, &u);
    big_mul_i64(&t, &t, a);
    big_mul_i64(&u, &CB[1], b); big_mul_i64(&u, &u, b); big_add(&t, &t, &u);
    big_mul_i64(&t, &t, a);
    big_mul_i64(&u, &CB[0], b); big_mul_i64(&u, &u, b); big_mul_i64(&u, &u, b); big_add(N, &t, &u);
}

/* ---------- roots of F mod p */

static uint32_t P_;
static inline uint32_t mm(uint32_t a, uint32_t b) { return (uint32_t)((uint64_t)a * b % P_); }
static inline uint32_t sb(uint32_t a, uint32_t b) { return a >= b ? a - b : a + (P_ - b); }
/* r = x y mod f, f = t^3 + f[2] t^2 + f[1] t + f[0], x, y of degree < 3 */
static void pmul(uint32_t *r, const uint32_t *x, const uint32_t *y, const uint32_t *f) {
    uint64_t z[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) z[i + j] += (uint64_t)x[i] * y[j] % P_;
    uint32_t w[5];
    for (int i = 0; i < 5; i++) w[i] = (uint32_t)(z[i] % P_);
    for (int d = 4; d >= 3; d--) if (w[d]) for (int k = 0; k < 3; k++) w[d - 3 + k] = sb(w[d - 3 + k], mm(w[d], f[k]));
    r[0] = w[0]; r[1] = w[1]; r[2] = w[2];
}
static void ppow(uint32_t *r, const uint32_t *b0, uint64_t e, const uint32_t *f) {
    uint32_t b[3] = {b0[0], b0[1], b0[2]}; r[0] = 1; r[1] = r[2] = 0;
    for (; e; e >>= 1) { if (e & 1) pmul(r, r, b, f); pmul(b, b, b, f); }
}
static int pdeg(const uint32_t *f, int n) { while (n >= 0 && !f[n]) n--; return n; }
static int pgcd(uint32_t *g, const uint32_t *x, const uint32_t *y) {   /* monic gcd, degrees <= 3; returns degree */
    uint32_t a[4], b[4]; memcpy(a, x, sizeof a); memcpy(b, y, sizeof b);
    int da = pdeg(a, 3), db = pdeg(b, 3);
    while (db >= 0) {
        uint32_t ib = invmod32(b[db], P_);
        while (da >= db) {
            uint32_t c = mm(a[da], ib);
            for (int k = 0; k <= db; k++) a[da - db + k] = sb(a[da - db + k], mm(c, b[k]));
            da = pdeg(a, da - 1);
        }
        uint32_t t[4]; memcpy(t, a, sizeof t); memcpy(a, b, sizeof a); memcpy(b, t, sizeof b);
        int td = da; da = db; db = td;
    }
    if (da < 0) return -1;
    uint32_t ia = invmod32(a[da], P_);
    for (int k = 0; k < 4; k++) g[k] = k <= da ? mm(a[k], ia) : 0;
    return da;
}
static uint32_t sqrtm(uint32_t a) {             /* a square root of the square a mod the odd prime P_ */
    uint32_t p = P_;
    if (!a) return 0;
    if (p % 4 == 3) return (uint32_t)powmod64(a, (p + 1) / 4, p);
    uint32_t q = p - 1; int s = 0;
    while (!(q & 1)) { q >>= 1; s++; }
    uint32_t z = 2;
    while (powmod64(z, (p - 1) / 2, p) != p - 1) z++;
    uint32_t m = s, c = (uint32_t)powmod64(z, q, p), t = (uint32_t)powmod64(a, q, p), r = (uint32_t)powmod64(a, (q + 1) / 2, p);
    while (t != 1) {
        uint32_t i = 0, tt = t;
        while (tt != 1) { tt = mm(tt, tt); i++; }
        uint32_t b = c;
        for (uint32_t k = 0; k + 1 < m - i; k++) b = mm(b, b);
        m = i; c = mm(b, b); t = mm(t, c); r = mm(r, b);
    }
    return r;
}
/* The roots of the monic squarefree g of degree d, which splits into linear factors (Cantor-Zassenhaus). */
static int split(const uint32_t *g, int d, uint32_t *out, uint32_t seed) {
    if (d <= 0) return 0;
    if (d == 1) { out[0] = g[0] ? P_ - g[0] : 0; return 1; }
    if (d == 2) {
        uint32_t i2 = (P_ + 1) / 2, b = g[1], disc = sb(mm(b, b), mm(4 % P_, g[0])), s = sqrtm(disc);
        out[0] = mm(sb(s, b), i2); out[1] = mm(sb(P_ - s == P_ ? 0 : P_ - s, b), i2);
        return 2;
    }
    uint32_t f[3] = {g[0], g[1], g[2]};
    for (uint32_t dl = seed % P_;; dl = (dl + 1) % P_) {
        uint32_t base[3] = {dl, 1, 0}, r[3];
        ppow(r, base, (P_ - 1) / 2, f);
        uint32_t h[4] = {sb(r[0], 1), r[1], r[2], 0}, gg[4] = {g[0], g[1], g[2], 1}, f1[4];
        int d1 = pgcd(f1, h, gg);
        if (d1 >= 1 && d1 < 3) {
            int n = split(f1, d1, out, seed + 1);
            uint32_t qq[4] = {0, 0, 0, 0}, rem[4]; memcpy(rem, gg, sizeof rem);
            for (int k = 3 - d1; k >= 0; k--) {
                uint32_t c = rem[k + d1]; qq[k] = c;
                for (int t = 0; t <= d1; t++) rem[k + t] = sb(rem[k + t], mm(c, f1[t]));
            }
            return n + split(qq, 3 - d1, out + n, seed + 7);
        }
    }
}
static int cmpu32(const void *x, const void *y) { uint32_t a = *(const uint32_t *)x, b = *(const uint32_t *)y; return a < b ? -1 : a > b; }
/* The distinct roots of F mod p in P^1, ascending, with r = p for (1 : 0). */
static int roots_mod(uint32_t p, uint32_t *out) {
    uint32_t f[4]; int n = 0;
    for (int i = 0; i < 4; i++) { uint32_t m = big_mod32(&CB[i], p); f[i] = CB[i].neg && m ? p - m : m; }
    P_ = p;
    if (p < 128) {
        for (uint32_t x = 0; x < p; x++) if (((mm(f[3], x) + f[2]) % p * x % p + f[1]) % p * x % p == (p - f[0]) % p) out[n++] = x;
    } else {
        int d = pdeg(f, 3);
        if (d >= 1) {
            uint32_t iv = invmod32(f[d], p), m[4] = {0, 0, 0, 0};
            for (int k = 0; k <= d; k++) m[k] = mm(f[k], iv);
            if (d == 3) {
                uint32_t x[3] = {0, 1, 0}, r[3], g[4];
                ppow(r, x, p, m);
                uint32_t h[4] = {r[0], sb(r[1], 1), r[2], 0};
                int dg = pgcd(g, h, m);
                if (dg < 0) { memcpy(g, m, sizeof g); dg = 3; }
                n = split(g, dg, out, 3);
            } else if (d == 2) {
                uint32_t b = m[1], disc = sb(mm(b, b), mm(4 % p, m[0]));
                if (!disc) { out[0] = mm(b ? p - b : 0, (p + 1) / 2); n = 1; }
                else if (powmod64(disc, (p - 1) / 2, p) == 1) n = split(m, 2, out, 3);
            } else n = split(m, 1, out, 3);
        }
        qsort(out, n, sizeof *out, cmpu32);
        int k = 0;
        for (int i = 0; i < n; i++) if (!k || out[i] != out[k - 1]) out[k++] = out[i];
        n = k;
    }
    if (!f[3]) out[n++] = p;
    return n;
}

static uint8_t *prime_table(uint64_t n) {       /* c[i] = 1 for i composite or < 2 */
    uint8_t *c = calloc(n < 1 ? 2 : n + 1, 1);   /* n = 0 still needs c[1] */
    if (!c) die("out of memory");
    c[0] = c[1] = 1;
    for (uint64_t p = 2; p * p <= n; p++) if (!c[p]) for (uint64_t m = p * p; m <= n; m += p) c[m] = 1;
    return c;
}

/* ---------- the factor base: entries (p, r) ascending (p, r = p for (1 : 0)) */

static uint32_t *FP, *FR; static size_t NFB, CAPFB;
static void fb_add(uint32_t p, uint32_t r) {
    if (NFB == CAPFB) { CAPFB = CAPFB ? 2 * CAPFB : 1 << 14; FP = realloc(FP, CAPFB * 4); FR = realloc(FR, CAPFB * 4); if (!FP || !FR) die("out of memory"); }
    FP[NFB] = p; FR[NFB] = r; NFB++;
}
static void fb_compute(uint64_t lim) {
    uint8_t *c = prime_table(lim);
    uint32_t rr[4];
    for (uint64_t p = 2; p <= lim; p++) if (!c[p]) { int n = roots_mod((uint32_t)p, rr); for (int i = 0; i < n; i++) fb_add((uint32_t)p, rr[i]); }
    free(c);
}
static void fb_read(const char *path, uint64_t lim) {    /* the file's own limit is in its header */
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    char line[512]; uint64_t flim = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') { char *t = strstr(line, "p <= "); if (t) flim = strtoull(t + 5, 0, 10); continue; }
        char *s = line; uint64_t p = strtoull(s, &s, 10);
        if (*s != ':' || p > lim) continue;
        s++;
        for (;;) {
            while (*s == ' ' || *s == ',') s++;
            if (*s < '0' || *s > '9') break;
            fb_add((uint32_t)p, (uint32_t)strtoull(s, &s, 10));
        }
    }
    fclose(f);
    if (flim < lim) { fprintf(stderr, "sieve: %s stops at %llu < lim\n", path, (unsigned long long)flim); exit(1); }
}

/* ---------- the sieve */

static int LPB, MFB, LOGN;              /* LOGN: log2 of the positions per special-q */
static uint64_t LIM;
static int LIM_SQ;      /* LIM^2 >= 2^LPB */
static double SKEW = 1, SCALE;
static int VERBOSE;
static uint8_t *FL;                     /* round(SCALE log2 p) per entry */
static uint64_t *FBM;                   /* floor((2^64 - 1) / p) per entry (Barrett) */
static size_t NT0, NBR;                 /* entries [0, NT0) tiny (p < PSKIP), [NT0, NBR) p < 2^16, [NBR, NFB) larger */
static uint32_t TINY[16]; static int NTINY; static uint64_t TINYM;   /* the tiny primes with roots, their product */
static uint32_t *SPINV, *SMAXQ;         /* entries [NT0, NBR): p^-1 mod 2^32, (2^32 - 1) / p (divisibility tests) */
static size_t NEW;                      /* the entries below the rectangle's width: [NT0, NEW) */
/* the prime powers PSKIP <= p^k < 2^16, entry e hitting where QA a + QB b = 0 mod QM (roots lifted from p) */
static uint32_t *QM, *QA, *QB, *QRHO; static uint8_t *QL; static size_t NQ, CAPQ;
static uint32_t *ST, *QST;              /* per entry: the start in the current line, or the next hit (carried) */
static uint32_t *BRHO;                  /* rho of every entry [NT0, NFB), for KB special-q */
#define NOPOS 0xffffffffu               /* x0 = 0: the lines j = 0 mod p */
#define SKIPV 0xfffffffeu               /* not sieved (the special-q's own root) */
/* slices of the entries [NT0, NFB): runs of at most 2^16 with one log; bucket updates are 16-bit position | 16-bit index */
static size_t *SLB; static int NSL;
static uint32_t *BK; static size_t BKCAP; static uint32_t **BKP; static uint32_t *BKOFF; static int NBKMAX;
static double TM[10];
static long NSURV, NCAND;
static FILE *OUT;
static long NREL;

static inline double flog2(double x) {         /* log2 x for x > 0, within 0.01 */
    union { double d; uint64_t u; } v = {x};
    int e = (int)((v.u >> 52) & 0x7ff) - 1023;
    v.u = (v.u & 0x000fffffffffffffULL) | 0x3ff0000000000000ULL;
    double m = v.d - 1;
    return e + m * (1.4142135 - 0.4142135 * m);
}
static inline uint64_t bmod(uint64_t x, uint64_t p, uint64_t m) {     /* x mod p, m = (2^64 - 1) / p */
    uint64_t r = x - (uint64_t)(((u128)x * m) >> 64) * p;      /* the quotient estimate is short by at most 2 */
    r = r >= p ? r - p : r;
    return r >= p ? r - p : r;
}
static inline uint64_t smod(int64_t x, uint64_t p, uint64_t m) {
    if (x >= 0 && (uint64_t)x < p) return (uint64_t)x;
    uint64_t r = bmod(x < 0 ? (uint64_t)0 - (uint64_t)x : (uint64_t)x, p, m);
    return x < 0 && r ? p - r : r;
}

typedef struct { int64_t a0, b0, a1, b1; } qlat;
/* Gauss reduction of {(q, 0), (r, 1)} for the inner product a a' + s^2 b b' */
static qlat reduce_q(uint64_t q, uint64_t r) {
    int64_t u0 = (int64_t)q, u1 = 0, v0 = (int64_t)r, v1 = 1;
    long double s2 = (long double)SKEW * SKEW;
#define NRM(x, y) ((long double)(x) * (x) + s2 * (long double)(y) * (y))
    for (;;) {
        long double nu = NRM(u0, u1), nv = NRM(v0, v1);
        if (nu < nv) { int64_t t = u0; u0 = v0; v0 = t; t = u1; u1 = v1; v1 = t; long double x = nu; nu = nv; nv = x; }
        long double dot = (long double)u0 * v0 + s2 * (long double)u1 * v1;
        int64_t k = (int64_t)llroundl(dot / nv);
        if (!k) break;
        u0 -= k * v0; u1 -= k * v1;
        if (NRM(u0, u1) >= nv) break;
    }
    qlat L = {v0, v1, u0, u1};          /* i along the shorter vector */
    return L;
}

/* ---------- the region: the positions of small |G(i, j)|, G(i, j) = F(i u + j v) = j^3 g(i / j)
   At level X: the largest rectangle -W/2 <= i < W/2, 0 <= j < J (W a power of 2) with |G| <= X on its boundary
   (BMAX), and on each line j >= 1 the intervals of |g(i/j)| <= X/j^3 outside it, at least LMIN long.  A line's
   length does not grow with j (sublevel sets of a cubic grow at least like y^(1/3)), so the lines stop at the first
   shorter than LMIN. */

static double G[4], TR[3], QP, QQ, TC[2], BMAX[7][61], GB; static int NTR, NTC;     /* GB: a bound on g's roots */
static int RW, RJ;                      /* the rectangle (RW = 0: none) */
typedef struct { int64_t lo; uint64_t x; uint32_t len, j; } seg_t;     /* line j, i in [lo, lo + len), positions x.. */
static seg_t *SEGS; static int NSEG, CAPSEG;
static uint64_t NPOS, LMIN;
#define JCAP (1 << 15)

static inline double gf(double t) {                     /* g(t), factored: stable near the roots */
    double v = G[3];
    for (int r = 0; r < NTR; r++) v *= t - TR[r];
    return NTR == 1 ? v * (t * t + QP * t + QQ) : v;
}
static inline double Gabs(double i, double j) {        /* |G(i, j)|, factored */
    double v = fabs(G[3]);
    for (int r = 0; r < NTR; r++) v *= fabs(i - TR[r] * j);
    return NTR == 1 ? v * fabs(i * i + QP * i * j + QQ * j * j) : v;
}
static long double gl(long double t) { return ((G[3] * t + G[2]) * t + G[1]) * t + G[0]; }
static void g_roots(void) {             /* g's real roots (with one, its quadratic factor) and critical points */
    long double B = 1, A = 3 * (long double)G[3], Bq = 2 * (long double)G[2], C = G[1], D = Bq * Bq - 4 * A * C, pts[4];
    for (int i = 0; i < 3; i++) { long double t = fabsl(G[i] / (long double)G[3]); if (1 + t > B) B = 1 + t; }
    GB = (double)B;
    int np = 0; NTC = 0;
    pts[np++] = -B;
    if (D > 0) {
        long double s = sqrtl(D), x1 = -(Bq + (Bq >= 0 ? s : -s)) / (2 * A), x2 = C / (A * x1);
        if (x1 > x2) { long double t = x1; x1 = x2; x2 = t; }
        pts[np++] = x1; pts[np++] = x2; TC[NTC++] = (double)x1; TC[NTC++] = (double)x2;
    }
    pts[np++] = B;
    NTR = 0;
    for (int k = 0; k + 1 < np; k++) {
        long double lo = pts[k], hi = pts[k + 1], flo = gl(lo);
        if ((flo < 0) == (gl(hi) < 0)) continue;
        for (int it = 0; it < 200; it++) {
            long double m = (lo + hi) / 2;
            if (m == lo || m == hi) break;
            if ((gl(m) < 0) == (flo < 0)) lo = m; else hi = m;
        }
        TR[NTR++] = (double)((lo + hi) / 2);
    }
    if (NTR == 1) { QP = G[2] / G[3] + TR[0]; QQ = G[1] / G[3] + TR[0] * QP; }
}
/* The t with g(t) = v on the monotone piece [a, b]. */
static double solve(double a, double b, double v) {
    int up = gf(b) > gf(a);
    for (int it = 0; it < 80; it++) { double m = (a + b) / 2; if ((gf(m) < v) == up) a = m; else b = m; }
    return (a + b) / 2;
}
/* The intervals of t where |g(t)| <= y (at most 3). */
static int sublevel(double y, double *lo, double *hi) {
    double B = GB + cbrt(y / fabs(G[3])), pts[4] = {-B, 0, 0, 0}; int np = 1, n = 0;
    for (int c = 0; c < NTC; c++) if (TC[c] > -B && TC[c] < B) pts[np++] = TC[c];
    pts[np++] = B;
    for (int k = 0; k + 1 < np; k++) {
        double a = pts[k], b = pts[k + 1], va = gf(a), vb = gf(b);
        if (fmax(va, vb) < -y || fmin(va, vb) > y) continue;
        double s = fabs(va) <= y ? a : solve(a, b, va < 0 ? -y : y), e = fabs(vb) <= y ? b : solve(a, b, vb < 0 ? -y : y);
        if (n && s <= hi[n - 1]) hi[n - 1] = e; else { lo[n] = s; hi[n] = e; n++; }
    }
    return n;
}
/* The number of positions at level 2^lx, at most cap (the last line trimmed to its middle); with build, also set
   the rectangle and the segments. */
static uint64_t region(double lx, int build, uint64_t cap) {
    int W = 0, J = 0;
    for (int w = 9; w <= 15 && (256ULL << w) <= cap; w++) {
        int k = 0, per = BR >> w;
        while (k < 61 && log2(BMAX[w - 9][k]) <= lx) k++;
        double f = k == 0 || k == 61 ? 0 : (lx - log2(BMAX[w - 9][k - 1])) / (log2(BMAX[w - 9][k]) - log2(BMAX[w - 9][k - 1]) + 1e-9);
        int jj = k ? (int)exp2((k - 1 + f) / 4.0) / per * per : 0;
        if (jj >= 256 && (uint64_t)jj << w > (uint64_t)W * J) { W = 1 << w; J = jj; }   /* fewer lines do not pay a walk per prime */
    }
    if ((uint64_t)W * J > cap) J = (int)(cap / W) / (BR / W) * (BR / W);      /* monotone in X: the lines trimmed */
    uint64_t n = (uint64_t)W * J;
    if (build) { RW = W; RJ = J; NSEG = 0; }
    for (int j = 1; j < JCAP; j++) {
        double lo[3], hi[3], tot = 0, y = exp2(lx) / ((double)j * j * j);
        if (2 * (GB + cbrt(y / fabs(G[3]))) * j < LMIN) break;          /* an outer bound of the line */
        int k = sublevel(y, lo, hi);
        for (int x = 0; x < k; x++) { lo[x] = floor(lo[x] * j); hi[x] = ceil(hi[x] * j); tot += hi[x] - lo[x]; }
        if (tot < LMIN || n + LMIN > cap) break;
        for (int x = 0; x < k; x++) {           /* minus the rectangle's line */
            double pa[2] = {lo[x], hi[x]}, pb[2] = {0, 0}; int np = 1;
            if (j < J && hi[x] > -W / 2 && lo[x] < W / 2) {
                np = 0;
                if (lo[x] < -W / 2) { pa[np] = lo[x]; pb[np++] = -W / 2; }
                if (hi[x] > W / 2) { pa[np] = W / 2; pb[np++] = hi[x]; }
            } else pb[0] = hi[x];
            for (int y = 0; y < np; y++) for (double a = pa[y]; pb[y] - a >= LMIN && n + LMIN <= cap; a += 2147483648.0) {
                double len = pb[y] - a < 2147483648.0 ? pb[y] - a : 2147483648.0;
                if (n + len > cap) { a += floor((len - (double)(cap - n)) / 2); len = (double)(cap - n); }
                if (build) {
                    if (NSEG == CAPSEG) { CAPSEG = CAPSEG ? 2 * CAPSEG : 64; SEGS = realloc(SEGS, CAPSEG * sizeof *SEGS); if (!SEGS) die("out of memory"); }
                    SEGS[NSEG++] = (seg_t){(int64_t)a, n, (uint32_t)len, (uint32_t)j};
                }
                n += (uint64_t)len;
            }
        }
    }
    if (build) NPOS = n;
    return n;
}
/* Set G for the lattice L, its roots and the region. */
static void shape(const qlat *L) {
    double x[2] = {(double)L->a1, (double)L->a0}, y[2] = {(double)L->b1, (double)L->b0};
    memset(G, 0, sizeof G);
    for (int k = 0; k < 4; k++) {
        double m[4] = {1, 0, 0, 0};
        for (int e = 0; e < 3; e++) {
            double l0 = e < k ? x[0] : y[0], l1 = e < k ? x[1] : y[1], n[4] = {0, 0, 0, 0};
            for (int d = 0; d < 3; d++) { n[d] += m[d] * l0; n[d + 1] += m[d] * l1; }
            memcpy(m, n, sizeof m);
        }
        for (int d = 0; d < 4; d++) G[d] += CD[k] * m[d];
    }
    g_roots();
    for (int w = 9; w <= 15 && (256ULL << w) <= (1ULL << LOGN); w++) {     /* BMAX: |G| on the rectangle's boundary */
        double H = (1 << w) / 2, m = fabs(G[3]) * H * H * H;
        for (int k = 0; k < 61; k++) {
            double J = floor(exp2(k / 4.0));
            m = fmax(m, fmax(Gabs(-H, J), Gabs(H, J)));
            for (int s = 0; s <= 16; s++) m = fmax(m, Gabs(-H + H * s / 8, J));
            BMAX[w - 9][k] = m;
        }
    }
    double lo = 0, hi = 1100, target = exp2(LOGN);
    for (int it = 0; it < 32; it++) { double mid = (lo + hi) / 2; if ((double)region(mid, 0, 1ULL << LOGN) >= target) hi = mid; else lo = mid; }
    region(hi, 1, 1ULL << LOGN);
}

/* The thresholds of a piece, per SEG positions: floor(SCALE (log2 min |G| - log2 q - MFB)), the minimum from the
   segment's ends and critical points, 0 across a root. */
static void thresholds(int j, int64_t i0, uint32_t len, double off, uint8_t *T) {
    uint32_t ns = (len + SEG - 1) / SEG;
    double v = Gabs((double)i0, j);
    for (uint32_t k = 0; k < ns; k++) {
        double w = Gabs((double)i0 + (double)(k + 1 < ns ? (k + 1) * SEG : len), j), t = SCALE * (flog2(v < w ? v : w) - off) - 0.5;
        T[k] = t <= 0 ? 0 : t >= 255 ? 255 : (uint8_t)t;
        v = w;
    }
    for (int r = 0; r < NTR + NTC; r++) {       /* the segments with a root (0) or a critical point inside */
        double c = (r < NTR ? TR[r] : TC[r - NTR]) * j - (double)i0;
        if (c < 0 || c >= len) continue;
        double t = r < NTR ? 0 : SCALE * (flog2(Gabs(c + (double)i0, j)) - off) - 0.5;
        uint32_t k = (uint32_t)(c / SEG);
        if (t < T[k]) T[k] = t <= 0 ? 0 : (uint8_t)t;
    }
}

/* ---------- survivors */

typedef struct { int64_t i; uint32_t x, j, plim; uint8_t nb; uint32_t bp[16]; } surv_t;
static uint8_t HITS[1 << 15];           /* the entries [NT0, NBR) whose root test holds */
static void divout(big *N, uint64_t p, uint64_t *PR, int *np) {     /* to full multiplicity */
    if (p == 2) { while (*np < NPR && big_div2(N)) PR[(*np)++] = 2; return; }
    uint64_t pi = inv2_64(p);
    while (*np < NPR && big_divexact(N, p, pi)) PR[(*np)++] = p;
}

/* Factor a survivor; print it if it is a relation. */
static void survivor(uint64_t q, const qlat *L, const surv_t *sv, const uint32_t *rho) {
    int64_t i = sv->i, j = sv->j;
    if (!(((uint64_t)i | (uint64_t)j) & 1)) return;
    int small = i > -(1LL << 31) && i < (1LL << 31) && L->a0 > -(1LL << 31) && L->a0 < (1LL << 31) && L->a1 > -(1LL << 31) && L->a1 < (1LL << 31);
    i128 A = small ? i * L->a0 + j * L->a1 : (i128)i * L->a0 + (i128)j * L->a1, Bb = small ? i * L->b0 + j * L->b1 : (i128)i * L->b0 + (i128)j * L->b1;
    if (Bb < 0) { A = -A; Bb = -Bb; }
    if (Bb == 0 || A >= ((i128)1 << 62) || A <= -((i128)1 << 62) || Bb >= ((i128)1 << 62)) return;
    int64_t a = (int64_t)A, b = (int64_t)Bb;
    if (bgcd((uint64_t)(a < 0 ? -a : a), (uint64_t)b) != 1) return;
    big N; big_over = 0; normF(&N, a, b);
    if (!N.n || big_over) return;
    uint64_t PR[NPR]; int np = 0;
    if (!big_divexact(&N, q, inv2_64(q))) return;
    PR[np++] = q;
    u128 r = 0;
    for (int k = N.n - 1; k >= 0; k--) r = ((r << 64) | N.d[k]) % TINYM;
    for (int t = 0; t < NTINY; t++) if ((uint64_t)r % TINY[t] == 0) divout(&N, TINY[t], PR, &np);
    /* a branch-free root test over the entries sieved here, then the hits */
    size_t ne = sv->plim == BR ? NBR : NEW;
    if (i > -(1 << 15) && i < (1 << 15) && j < (1 << 15))     /* rho j + p 2^15 - i < 2^32 */
        for (size_t e = NT0; e < ne; e++) HITS[e - NT0] = (rho[e] * (uint32_t)j + (FP[e] << 15) - (uint32_t)i) * SPINV[e] <= SMAXQ[e];
    else
        for (size_t e = NT0; e < ne; e++) HITS[e - NT0] = (rho[e] * (uint64_t)j + FP[e] - smod(i, FP[e], FBM[e])) % FP[e] == 0;
    for (size_t e = NT0; e < ne; e++) {
        uint32_t p = FP[e];
        if (rho[e] == NOPOS ? j % p == 0 : HITS[e - NT0] && rho[e] != SKIPV) if (PR[np - 1] != p) divout(&N, p, PR, &np);
    }
    for (int k = 0; k < sv->nb; k++) if (PR[np - 1] != sv->bp[k]) divout(&N, sv->bp[k], PR, &np);
    if (N.n > 1 || big_bits(&N) > MFB || np >= NPR - 4) return;
    if (N.n == 1 && N.d[0] > 1) {
        uint64_t st[8]; int ns = 0;
        st[ns++] = N.d[0];
        while (ns) {
            uint64_t n = st[--ns];
            if (n == 1) continue;
            if (isprime64(n)) { if ((LPB < 64 && n >> LPB) || np >= NPR) return; PR[np++] = n; continue; }
            if ((LIM_SQ && !(n >> LPB)) || ns + 2 > 8) return;       /* with LIM_SQ, a composite <= 2^lpb has a prime <= lim */
            uint64_t d = rho64(n); st[ns++] = d; st[ns++] = n / d;
        }
    }
    for (int k = 1; k < np; k++) for (int m = k; m > 0 && PR[m - 1] > PR[m]; m--) { uint64_t t = PR[m]; PR[m] = PR[m - 1]; PR[m - 1] = t; }
    char buf[NPR * 17 + 48], *o = buf + snprintf(buf, 48, "%lld,%lld:", (long long)a, (long long)b);
    for (int k = 0; k < np; k++) {
        if (k) *o++ = ',';
        int nd = (67 - __builtin_clzll(PR[k] | 1)) / 4;
        for (int d = nd - 1; d >= 0; d--) *o++ = "0123456789abcdef"[PR[k] >> (4 * d) & 15];
    }
    *o++ = ':'; *o++ = '\n';
    fwrite(buf, 1, o - buf, OUT);
    NREL++;
}

/* ---------- the bucket entries' hits */

/* Franke-Kleinjung's walk (the rectangle, p >= W): successive hits differ by (a0, b0), (a1, b1) or both, with
   -W < a0 <= 0 <= a1 < W, a1 - a0 >= W; position x = j W + i + W/2.  Leaves w at the first hit. */
typedef struct { uint64_t x; int64_t ia, ic; uint32_t ba, bc, h, mask; } walk_t;
static inline void step(walk_t *w) { uint32_t i0 = (uint32_t)w->x & w->mask; w->x += (i0 >= w->ba ? w->ia : 0) + (i0 < w->bc ? w->ic : 0); }
#define PUSH(w) (*BKP[(w).x >> LOGB]++ = (uint32_t)((w).x & (BR - 1)) | (w).h)
static int fk_init(walk_t *w, int32_t W, uint32_t pp, uint32_t rho, uint32_t hint) {
    if (rho >= SKIPV || rho == 0) return 0;
    int32_t p = (int32_t)pp, a0 = -p, b0 = 0, a1 = (int32_t)rho, b1 = 1;
    while (a1 >= W) {
        int32_t k = a0 / a1; a0 -= k * a1; b0 -= k * b1;
        if (a0 > -W) break;
        k = a1 / a0; a1 -= k * a0; b1 -= k * b0;
    }
    if (a1 >= W) { if (!a0) return 0; int32_t k = (a1 - W) / (-a0) + 1; a1 += k * a0; b1 += k * b0; }
    else if (a0 <= -W) { if (!a1) return 0; int32_t k = (a1 - W - a0) / a1; a0 += k * a1; b0 += k * b1; }
    w->ia = (int64_t)b0 * W + a0; w->ic = (int64_t)b1 * W + a1; w->ba = (uint32_t)(W - a1); w->bc = (uint32_t)(-a0); w->h = hint;
    w->mask = (uint32_t)W - 1; w->x = (uint64_t)(W >> 1);
    step(w);
    return 1;
}

/* rho of the entries for KB special-q at once: one inverse per prime (Montgomery's trick). */
#define KB 16
static void entry_rho(const qlat *Ls, int k) {
    size_t nb = NFB - NT0;
    for (size_t e = NT0, f; e < NFB; e = f) {
        uint32_t p = FP[e];
        uint64_t m = FBM[e], am0[KB], bm0[KB], am1[KB], bm1[KB], x0[3 * KB], x1[3 * KB], pre[3 * KB], acc = 1;
        for (f = e; f < NFB && FP[f] == p; f++) ;
        for (int t = 0; t < k; t++) { am0[t] = smod(Ls[t].a0, p, m); bm0[t] = smod(Ls[t].b0, p, m); am1[t] = smod(Ls[t].a1, p, m); bm1[t] = smod(Ls[t].b1, p, m); }
        int n = 0;
        for (size_t g = e; g < f && g < e + 3; g++) for (int t = 0; t < k; t++, n++) {
            uint64_t R = FR[g];
            x0[n] = R == p ? bm0[t] : am0[t] + p - (bm0[t] < 2 ? R * bm0[t] : bmod(R * bm0[t], p, m)); if (x0[n] >= p) x0[n] -= p;
            x1[n] = R == p ? bm1[t] : am1[t] + p - (bm1[t] < 2 ? R * bm1[t] : bmod(R * bm1[t], p, m)); if (x1[n] >= p) x1[n] -= p;
            pre[n] = acc; if (x0[n]) acc = bmod(acc * x0[n], p, m);
        }
        uint64_t inv = invmod32((uint32_t)acc, p);
        for (int g = n / k - 1, u = n - 1; g >= 0; g--) for (int t = k - 1; t >= 0; t--, u--) {
            uint32_t *rho = BRHO + t * nb + (e + g - NT0);
            if (!x0[u]) { *rho = x1[u] ? NOPOS : SKIPV; continue; }
            *rho = (uint32_t)bmod((x1[u] ? p - x1[u] : 0) * bmod(inv * pre[u], p, m), p, m);
            inv = bmod(inv * x0[u], p, m);
        }
    }
}

/* The first hit at or after i0 on line j: (rho j - i0) mod p. */
static inline uint64_t first(uint64_t rho, uint32_t j, int64_t i0, uint64_t p, uint64_t m) {
    uint64_t a = j == 1 ? rho : bmod(rho * j, p, m), b = smod(i0, p, m);
    return a >= b ? a - b : a + p - b;
}

static inline uint32_t sieve_ap(uint8_t *S, uint32_t x, uint32_t len, uint32_t p, uint8_t lg) {   /* returns the next hit */
    for (; x + 3 * p < len; x += 4 * p) { S[x] += lg; S[x + p] += lg; S[x + 2 * p] += lg; S[x + 3 * p] += lg; }
    for (; x < len; x += p) S[x] += lg;
    return x;
}
/* One entry on the rectangle's lines [j0, j0 + n) (st: the first hit on line j0; recomputed at j0 = 0). */
static void rect_one(uint8_t *S, uint32_t j0, uint32_t n, uint32_t p, uint32_t r, uint32_t *st, uint8_t lg) {
    if (r == SKIPV) return;
    if (r == NOPOS) { for (uint32_t l = 0; l < n; l++) if ((j0 + l) % p == 0) for (int x = 0; x < RW; x++) S[l * RW + x] += lg; return; }
    uint32_t s = j0 ? *st : (uint32_t)(RW / 2) % p;
    for (uint32_t l = 0; l < n; l++, S += RW) { sieve_ap(S, s, RW, p, lg); s += r; if (s >= p) s -= p; }
    *st = s;
}
/* One entry on a segment's piece (st carries the next hit between pieces). */
static void seg_one(uint8_t *S, uint32_t j, int64_t i0, uint32_t len, int first_piece, uint32_t p, uint32_t r, uint32_t *st, uint8_t lg, uint64_t m) {
    if (r == SKIPV) return;
    if (r == NOPOS) { if (j % p == 0) for (uint32_t x = 0; x < len; x++) S[x] += lg; return; }
    uint32_t x = first_piece ? (uint32_t)first(r, j, i0, p, m) : *st;
    *st = sieve_ap(S, x, len, p, lg) - len;
}

/* ---------- one special-q */

static uint8_t SV[BR];
static uint8_t TH[BR / SEG + 1];
static uint64_t MARK[BR / 64];          /* survivors */
static surv_t *SURV; static int NSV, CAPSV;
typedef struct { uint32_t off, len, j, plim; int64_t i0; int mode; } piece_t;   /* mode: 0, 1 a rectangle line; 2, 3 a
                                                                                  segment's first, next piece */
static piece_t *PC; static int CAPPC;

static void sieve_q(uint64_t q, qlat L, const uint32_t *rho) {
    double t0 = cpu_now(), t;
    shape(&L);
    t = cpu_now(); TM[7] += t - t0; t0 = t;
    const uint64_t XR = (uint64_t)RW * RJ;
    for (NEW = NT0; NEW < NBR && FP[NEW] < (uint32_t)RW; NEW++) ;
    int nbk = (int)((NPOS + BR - 1) >> LOGB);
    if (nbk > NBKMAX) die("region too large");
    for (size_t f = 0; f < NQ; f++) {
        uint32_t m = QM[f];
        uint64_t x0 = (QA[f] * (uint64_t)smod(L.a0, m, UINT64_MAX / m) + QB[f] * (uint64_t)smod(L.b0, m, UINT64_MAX / m)) % m;
        uint64_t x1 = (QA[f] * (uint64_t)smod(L.a1, m, UINT64_MAX / m) + QB[f] * (uint64_t)smod(L.b1, m, UINT64_MAX / m)) % m;
        QRHO[f] = bgcd(x0, m) == 1 ? (uint32_t)((m - x1) * invmod32((uint32_t)x0, m) % m) : SKIPV;   /* rare */
    }
    t = cpu_now(); TM[0] += t - t0; t0 = t;
    /* the bucket hits: in the rectangle two walks at once, so their latencies overlap */
    for (int k = 0; k < nbk; k++) BKP[k] = BK + (size_t)k * BKCAP;
    for (int s = 0; s < NSL; s++) {
        for (int k = 0; k < nbk; k++) {         /* checked per slice: the guard after the last bucket holds one */
            size_t n = BKP[k] - (BK + (size_t)k * BKCAP);
            if (n > BKCAP) die("bucket overflow");
            BKOFF[(size_t)s * NBKMAX + k] = (uint32_t)n;
        }
        walk_t A, B; int na = 0;
        for (size_t e = SLB[s]; e < SLB[s + 1]; e++) {
            uint32_t p = FP[e], r = rho[e], h = (uint32_t)(e - SLB[s]) << 16;
            if (RW && p >= (uint32_t)RW && fk_init(na ? &B : &A, RW, p, r, h)) {
                if (!na) na = 1;
                else {
                    na = 0;
                    while (A.x < XR && B.x < XR) { PUSH(A); PUSH(B); step(&A); step(&B); }
                    for (; A.x < XR; step(&A)) PUSH(A);
                    for (; B.x < XR; step(&B)) PUSH(B);
                }
            }
        }
        for (int g = 0; g < NSEG; g++) {            /* segment by segment: fewer bucket streams at once */
            const seg_t *sg = SEGS + g;
            for (size_t e = SLB[s] > NBR ? SLB[s] : NBR; e < SLB[s + 1]; e++) {
                uint32_t p = FP[e], r = rho[e], h = (uint32_t)(e - SLB[s]) << 16;
                if (r >= SKIPV) continue;
                for (uint64_t x = sg->x + first(r, sg->j, sg->lo, p, FBM[e]), end = sg->x + sg->len; x < end; x += p)
                    *BKP[x >> LOGB]++ = (uint32_t)(x & (BR - 1)) | h;
            }
        }
        for (; na && A.x < XR; step(&A)) PUSH(A);
    }
    for (int k = 0; k < nbk; k++) {
        size_t n = BKP[k] - (BK + (size_t)k * BKCAP);
        if (n > BKCAP) die("bucket overflow");
        BKOFF[(size_t)NSL * NBKMAX + k] = (uint32_t)n;
    }
    t = cpu_now(); TM[1] += t - t0; t0 = t;
    double off = log2((double)q) + MFB;
    int sg = 0;
    for (int k = 0; k < nbk; k++) {
        uint64_t X0 = (uint64_t)k << LOGB;
        int npc = 0;
        if (CAPPC < BR / 512 + NSEG + 1) { CAPPC = BR / 512 + NSEG + 1; PC = realloc(PC, CAPPC * sizeof *PC); if (!PC) die("out of memory"); }
        if (X0 < XR)
            for (uint64_t x = X0; x < X0 + BR && x < XR; x += RW) {
                uint32_t j = (uint32_t)(x / RW);
                PC[npc++] = (piece_t){(uint32_t)(x - X0), (uint32_t)RW, j, (uint32_t)RW, -RW / 2, j == 0 ? 0 : 1};
            }
        while (sg < NSEG && SEGS[sg].x + SEGS[sg].len <= X0) sg++;
        for (int g = sg; g < NSEG && SEGS[g].x < X0 + BR; g++) {
            uint64_t a = SEGS[g].x > X0 ? SEGS[g].x : X0, b = SEGS[g].x + SEGS[g].len;
            if (b > X0 + BR) b = X0 + BR;
            PC[npc++] = (piece_t){(uint32_t)(a - X0), (uint32_t)(b - a), SEGS[g].j, BR, SEGS[g].lo + (int64_t)(a - SEGS[g].x), a == SEGS[g].x ? 2 : 3};
        }
        uint32_t len = NPOS - X0 < BR ? (uint32_t)(NPOS - X0) : BR;
        memset(SV, 0, len);
        if (X0 < XR) {
            uint32_t j0 = (uint32_t)(X0 / RW), n = (uint32_t)((XR - X0 < BR ? XR - X0 : BR) / RW);
            for (size_t e = NT0; e < NEW; e++) rect_one(SV, j0, n, FP[e], rho[e], &ST[e], FL[e]);
            for (size_t f = 0; f < NQ && QM[f] < (uint32_t)RW; f++) rect_one(SV, j0, n, QM[f], QRHO[f], &QST[f], QL[f]);
        }
        for (int c = 0; c < npc; c++) {
            const piece_t *pc = PC + c;
            if (pc->mode < 2) continue;
            for (size_t e = NT0; e < NBR; e++) seg_one(SV + pc->off, pc->j, pc->i0, pc->len, pc->mode == 2, FP[e], rho[e], &ST[e], FL[e], FBM[e]);
            for (size_t f = 0; f < NQ; f++) seg_one(SV + pc->off, pc->j, pc->i0, pc->len, pc->mode == 2, QM[f], QRHO[f], &QST[f], QL[f], UINT64_MAX / QM[f]);
        }
        t = cpu_now(); TM[2] += t - t0; t0 = t;
        uint32_t *base = BK + (size_t)k * BKCAP;
        for (int s = 0; s < NSL; s++) {
            uint8_t lg = FL[SLB[s]];
            const uint32_t *u = base + BKOFF[(size_t)s * NBKMAX + k], *ue = base + BKOFF[(size_t)(s + 1) * NBKMAX + k];
            for (; u + 4 <= ue; u += 4) { SV[u[0] & 0xffff] += lg; SV[u[1] & 0xffff] += lg; SV[u[2] & 0xffff] += lg; SV[u[3] & 0xffff] += lg; }
            for (; u < ue; u++) SV[*u & 0xffff] += lg;
        }
        t = cpu_now(); TM[3] += t - t0; t0 = t;
        NSV = 0;
        for (int c = 0; c < npc; c++) {
            const piece_t *pc = PC + c;
            if (pc->j == 0) continue;
            thresholds((int)pc->j, pc->i0, pc->len, off, TH);
            for (uint32_t g = 0; g * SEG < pc->len; g++) {
                const uint8_t *P = SV + pc->off + g * SEG, th = TH[g];
                uint32_t n = pc->len - g * SEG < SEG ? pc->len - g * SEG : SEG;
                uint8_t mx = 0;
                for (uint32_t x = 0; x < n; x++) mx = P[x] > mx ? P[x] : mx;
                if (mx < th) continue;
                for (uint32_t x = 0; x < n; x++) {
                    if (P[x] < th) continue;
                    NCAND++;
                    int64_t i = pc->i0 + g * SEG + x;
                    if (!(((uint64_t)i | pc->j) & 1)) continue;         /* 2 | gcd(a, b) */
                    double v = Gabs((double)i, pc->j);
                    if (v > 0 && P[x] < SCALE * (flog2(v) - off) - 0.5) continue;
                    if (NSV == CAPSV) { CAPSV = CAPSV ? 2 * CAPSV : 4096; SURV = realloc(SURV, CAPSV * sizeof *SURV); if (!SURV) die("out of memory"); }
                    uint32_t pos = pc->off + g * SEG + x;
                    surv_t *sv = SURV + NSV++;
                    sv->i = i; sv->x = pos; sv->j = pc->j; sv->plim = pc->plim; sv->nb = 0;
                    MARK[pos >> 6] |= 1ULL << (pos & 63);
                }
            }
        }
        t = cpu_now(); TM[4] += t - t0; t0 = t;
        if (!NSV) continue;
        for (int s = 0; s < NSL; s++) {     /* a second pass for the survivors' bucket primes */
            const uint32_t *u = base + BKOFF[(size_t)s * NBKMAX + k], *ue = base + BKOFF[(size_t)(s + 1) * NBKMAX + k];
            for (; u < ue; u++) {
                uint32_t pos = *u & 0xffff;
                if (!(MARK[pos >> 6] >> (pos & 63) & 1)) continue;
                int lo = 0, hi = NSV - 1;
                while (lo < hi) { int m = (lo + hi) / 2; if (SURV[m].x < pos) lo = m + 1; else hi = m; }
                surv_t *v = &SURV[lo];
                if (v->nb < 16) v->bp[v->nb++] = FP[SLB[s] + (*u >> 16)];
            }
        }
        t = cpu_now(); TM[5] += t - t0; t0 = t;
        for (int v = 0; v < NSV; v++) { MARK[SURV[v].x >> 6] = 0; survivor(q, &L, &SURV[v], rho); }
        NSURV += NSV;
        t = cpu_now(); TM[6] += t - t0; t0 = t;
    }
}

/* ---------- setup */

static void q_add(uint32_t m, uint32_t a, uint32_t b, uint8_t lg) {
    if (NQ == CAPQ) {
        CAPQ = CAPQ ? 2 * CAPQ : 256;
        QM = realloc(QM, CAPQ * 4); QA = realloc(QA, CAPQ * 4); QB = realloc(QB, CAPQ * 4); QL = realloc(QL, CAPQ);
        if (!QM || !QA || !QB || !QL) die("out of memory");
    }
    QM[NQ] = m; QA[NQ] = a; QB[NQ] = b; QL[NQ] = lg; NQ++;
}

static void setup(void) {
    NT0 = 0; while (NT0 < NFB && FP[NT0] < PSKIP) NT0++;
    NBR = NT0; while (NBR < NFB && FP[NBR] < BR) NBR++;
    NTINY = 0;
    for (size_t e = 0; e < NT0; e++) if (!NTINY || TINY[NTINY - 1] != FP[e]) TINY[NTINY++] = FP[e];
    TINYM = 1; for (int t = 0; t < NTINY; t++) TINYM *= TINY[t];
    FL = xmalloc(NFB); FBM = xmalloc(NFB * 8);
    for (size_t e = 0; e < NFB; e++) {
        double l = SCALE * log2((double)FP[e]);
        FL[e] = (uint8_t)(l + 0.5 > 255 ? 255 : l + 0.5); FBM[e] = UINT64_MAX / FP[e];
    }
    SPINV = xmalloc(NBR * 4 + 4); SMAXQ = xmalloc(NBR * 4 + 4); ST = xmalloc(NBR * 4 + 4);
    for (size_t e = NT0; e < NBR; e++) { SPINV[e] = (uint32_t)inv2_64(FP[e]); SMAXQ[e] = UINT32_MAX / FP[e]; }
    if (NBR - NT0 > sizeof HITS) die("too many small primes");
    /* the prime powers p^k < 2^16, roots lifted by trying the p lifts of each; the first power >= PSKIP of a tiny
       prime counts k log p, for its unsieved lower powers */
    for (size_t e = 0; e < NBR;) {
        uint32_t p = FP[e], m = p, k = 1, r[4096], n = 0, aff = 0;
        size_t f = e;
        for (; f < NBR && FP[f] == p; f++) r[n++] = FR[f] == p ? 0 : FR[f], aff += FR[f] != p;
        for (e = f; (uint64_t)m * p < BR && n; k++) {
            uint32_t m2 = m * p, r2[4096], n2 = 0, c[4];
            for (int i = 0; i < 4; i++) { uint32_t t = big_mod32(&CB[i], m2); c[i] = CB[i].neg && t ? m2 - t : t; }
            for (uint32_t z = 0; z < n; z++) for (uint32_t t = 0; t < p && n2 < 4096; t++) {
                uint64_t x = r[z] + (uint64_t)t * m, v = z < aff ? ((c[3] * x % m2 + c[2]) * x % m2 + c[1]) * x % m2 + c[0]
                                                                 : ((c[0] * x % m2 + c[1]) * x % m2 + c[2]) * x % m2 + c[3];
                if (v % m2 == 0) r2[n2++] = (uint32_t)x;
                if (z + 1 == aff && t + 1 == p) aff = n2;      /* the affine lifts come first */
            }
            memcpy(r, r2, n2 * 4); n = n2; m = m2;
            if (m < PSKIP) continue;
            uint8_t lg = (uint8_t)(SCALE * log2(p) * (p < PSKIP && m / p < PSKIP ? k + 1 : 1) + 0.5);
            for (uint32_t z = 0; z < n; z++) z < aff ? q_add(m, 1, m - r[z], lg) : q_add(m, r[z] ? m - r[z] : 0, 1, lg);
        }
    }
    for (size_t f = 1; f < NQ; f++)
        for (size_t g = f; g > 0 && QM[g - 1] > QM[g]; g--) {
            uint32_t t = QM[g]; QM[g] = QM[g - 1]; QM[g - 1] = t; t = QA[g]; QA[g] = QA[g - 1]; QA[g - 1] = t;
            t = QB[g]; QB[g] = QB[g - 1]; QB[g - 1] = t; uint8_t l = QL[g]; QL[g] = QL[g - 1]; QL[g - 1] = l;
        }
    QRHO = xmalloc(NQ * 4 + 4); QST = xmalloc(NQ * 4 + 4);
    BRHO = xmalloc((NFB - NT0 + 1) * 4 * KB);
    SLB = xmalloc((NFB - NT0 + 2) * sizeof *SLB); NSL = 0;
    for (size_t e = NT0; e < NFB;) {
        SLB[NSL++] = e;
        size_t f = e + 1;
        while (f < NFB && FL[f] == FL[e] && f - e < 65536) f++;
        e = f;
    }
    SLB[NSL] = NFB;
    /* buckets: the expected hits per region (primes >= 2^9) plus a margin */
    NBKMAX = (int)(((1ULL << LOGN) + BR - 1) >> LOGB) + 1;
    double hits = 0;
    for (size_t e = NT0; e < NFB; e++) if (FP[e] >= 512) hits += (double)BR / FP[e];
    BKCAP = (size_t)(hits * 1.15 + 6 * sqrt(hits) + 4096);
    BK = xmalloc(((size_t)NBKMAX + 1) * BKCAP * 4 + 64);   /* one bucket of guard: a slice's overflow stays inside */
    BKP = xmalloc(NBKMAX * sizeof *BKP);
    BKOFF = xmalloc(((size_t)NSL + 1) * NBKMAX * 4);
}

/* The largest log2 |G| / q over the region of a special-q, from which SCALE is set. */
static double est_maxbits(uint64_t q, uint64_t r) {
    qlat L = reduce_q(q, r);
    shape(&L);
    double best = 1;
    for (int s = 0; RW && s <= 64; s++)
        best = fmax(best, fmax(Gabs(-RW / 2 + RW * s / 64.0, RJ), fmax(Gabs(-RW / 2, RJ * s / 64.0), Gabs(RW / 2, RJ * s / 64.0))));
    for (int g = 0; g < NSEG; g++) best = fmax(best, fmax(Gabs((double)SEGS[g].lo, SEGS[g].j), Gabs((double)SEGS[g].lo + SEGS[g].len, SEGS[g].j)));
    return log2(best) - log2((double)q);
}

/* the special-q of [q0, q1): primes with their affine roots */
static uint64_t *SQ, *SQR; static size_t NSQ, CAPSQ;
static void sq_add(uint64_t q, uint64_t r) {
    if (NSQ == CAPSQ) { CAPSQ = CAPSQ ? 2 * CAPSQ : 1024; SQ = realloc(SQ, CAPSQ * 8); SQR = realloc(SQR, CAPSQ * 8); if (!SQ || !SQR) die("out of memory"); }
    SQ[NSQ] = q; SQR[NSQ] = r; NSQ++;
}
static void special_q(uint64_t q0, uint64_t q1) {
    if (q1 <= q0) return;
    if (q1 > (1ULL << 32)) die("special-q above 2^32");
    uint64_t sq = (uint64_t)sqrt((double)q1) + 2;
    uint8_t *c = prime_table(sq), *seg = calloc(q1 - q0, 1);
    if (!seg) die("out of memory");
    for (uint64_t p = 2; p <= sq; p++) if (!c[p]) for (uint64_t m = (q0 + p - 1) / p * p; m < q1; m += p) if (m != p) seg[m - q0] = 1;
    uint32_t rr[4];
    for (uint64_t q = q0 < 2 ? 2 : q0; q < q1; q++) {
        if (seg[q - q0]) continue;
        if (q <= LIM && NFB) {
            size_t lo = 0, hi = NFB;
            while (lo < hi) { size_t m = (lo + hi) / 2; if (FP[m] < q) lo = m + 1; else hi = m; }
            for (; lo < NFB && FP[lo] == q; lo++) if (FR[lo] < q) sq_add(q, FR[lo]);
        } else {
            int n = roots_mod((uint32_t)q, rr);
            for (int i = 0; i < n; i++) if (rr[i] < q) sq_add(q, rr[i]);
        }
    }
    free(c); free(seg);
}

static volatile sig_atomic_t STOP_SLICE;     /* SIGUSR1 (server mode): end the slice before its next special-q */
static void on_usr1(int s) { (void)s; STOP_SLICE = 1; }

static void run_range(uint64_t q0, uint64_t q1, long nsample, unsigned long seed) {
    double c0 = cpu_now();
    NSQ = 0;
    long r0 = NREL;
    if (nsample <= 0) special_q(q0, q1);
    else {
        /* A draw (x, j), j < 3, gives x's j-th affine root if x is prime and has one, so every pair (q, r) is
           equally likely (the next prime after x would weight q by its prime gap). */
        srand48((long)seed); NSQ = 0;
        long draws = 0, cap = (long)(64.0 * (double)nsample * (3 * log((double)q1 + 2) + 1)), rep = 0, nr[4] = {0};
        for (; draws < cap && (long)NSQ < nsample && q1 > q0; draws++) {
            uint64_t q = q0 + (uint64_t)(drand48() * (double)(q1 - q0)), n = NSQ;
            int j = (int)(drand48() * 3);
            if (q >= q1 || q < 2 || !isprime64(q)) continue;
            special_q(q, q + 1);
            size_t got = NSQ - n;
            if ((size_t)j >= got) { NSQ = n; continue; }
            uint64_t r = SQR[n + j];
            int seen = 0;
            for (size_t t = 0; t < n && !seen; t++) seen = SQ[t] == q && SQR[t] == r;
            NSQ = n;
            if (seen) { rep++; continue; }
            sq_add(q, r);
            nr[got]++;
        }
        fprintf(OUT, "# special-q sample: %zu pairs of [%llu, %llu) (from primes with 1, 2, 3 roots: %ld %ld %ld), "
                     "%ld repeats drawn again, %ld draws, seed %lu\n", NSQ, (unsigned long long)q0,
                (unsigned long long)q1, nr[1], nr[2], nr[3], rep, draws, seed);
    }
    size_t done = NSQ;
    for (size_t s = 0; s < NSQ && done == NSQ; s += KB) {
        if (STOP_SLICE) { done = s; break; }
        int k = NSQ - s < KB ? (int)(NSQ - s) : KB;
        qlat Ls[KB];
        double t0 = cpu_now();
        for (int t = 0; t < k; t++) Ls[t] = reduce_q(SQ[s + t], SQR[s + t]);
        entry_rho(Ls, k);
        TM[0] += cpu_now() - t0;
        for (int t = 0; t < k; t++) {
            if (STOP_SLICE) { done = s + t; break; }
            sieve_q(SQ[s + t], Ls[t], BRHO + t * (NFB - NT0) - NT0);
        }
    }
    if (done < NSQ) {
        fprintf(OUT, "# stopped before special-q %llu\n", (unsigned long long)SQ[done]);
        NSQ = done;
    }
    double cpu = cpu_now() - c0;
    long n = NREL - r0;
    if (VERBOSE)
        fprintf(OUT, "# sieve profile (s): region %.3f rho %.3f fill %.3f small %.3f apply %.3f scan %.3f rescan %.3f factor %.3f; "
                     "candidates %ld survivors %ld; scale %.2f; last region: rectangle %d x %d, %d segments, 2^%.2f positions\n",
                TM[7], TM[0], TM[1], TM[2], TM[3], TM[4], TM[5], TM[6], NCAND, NSURV, SCALE, RW, RJ, NSEG, log2((double)NPOS + 1));
    fprintf(OUT, "# Total cpu time %.2fs\n", cpu);
    fprintf(OUT, "# Total %ld reports [%.3gs/r, %.1fr/sq]\n", n, n ? cpu / n : 0.0, NSQ ? (double)n / NSQ : 0.0);
}

static int usage(void) {
    fprintf(stderr, "usage: sieve roots c0,c1,c2,c3 LIM OUT\n"
                    "       sieve sieve -poly c0,c1,c2,c3 -lim L -lpb B -mfb M -N N [-skew S] [-roots FILE] [-q0 Q0 -q1 Q1]\n"
                    "                    [-sample N -seed X] [-out FILE] [-server] [-v]\n");
    return 2;
}

#ifndef __linux__
static pid_t PARENT;
static void *watch_parent(void *arg) {     /* exit once the parent is gone (getppid changes on reparenting) */
    sigset_t all;
    (void)arg;
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, 0);    /* SIGUSR1 must reach the main thread */
    for (;;) {
        sleep(1);
        if (getppid() != PARENT) _exit(1);
    }
    return 0;
}
#endif

int main(int argc, char **argv) {
    if (argc < 2) return usage();
    if (!strcmp(argv[1], "roots")) {
        if (argc != 5 || parse_poly(argv[2])) return usage();
        char *end; const char *a = argv[3];
        uint64_t lim = strtoull(a, &end, 10);
        if (!*a || *end || *a == '-' || lim >= (1ULL << 32)) die("LIM must be an integer in 0..2^32-1");
        FILE *o = fopen(argv[4], "w");
        if (!o) { perror(argv[4]); return 1; }
        fb_compute(lim);
        fprintf(o, "# roots of F mod p, p <= %llu (rankbound sieve; r = p for (1:0))\n", (unsigned long long)lim);
        for (size_t e = 0; e < NFB;) {
            fprintf(o, "%u: %u", FP[e], FR[e]);
            size_t f = e + 1;
            for (; f < NFB && FP[f] == FP[e]; f++) fprintf(o, ",%u", FR[f]);
            fputc('\n', o);
            e = f;
        }
        if (fclose(o)) { perror(argv[4]); return 1; }
        return 0;
    }
    if (strcmp(argv[1], "sieve")) return usage();
    const char *parent = getenv("RANKBOUND_PARENT");
#ifdef __linux__
    if (parent && (prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != (pid_t)atol(parent))) return 1;
#else
    if (parent) {
        pthread_t t;
        PARENT = (pid_t)atol(parent);
        if (getppid() != PARENT || pthread_create(&t, 0, watch_parent, 0) || pthread_detach(t)) return 1;
    }
#endif
    const char *poly = 0, *rootsf = 0, *outf = 0;
    uint64_t q0 = 0, q1 = 0; long nsample = 0; unsigned long seed = 1; int server = 0;
    LPB = MFB = LOGN = 0;
    for (int k = 2; k < argc; k++) {
        const char *o = argv[k], *v = k + 1 < argc ? argv[k + 1] : 0;
        if (!strcmp(o, "-server")) { server = 1; continue; }
        if (!strcmp(o, "-v")) { VERBOSE = 1; continue; }
        if (!v) return usage();
        k++;
        if (!strcmp(o, "-poly")) poly = v;
        else if (!strcmp(o, "-lim")) LIM = strtoull(v, 0, 10);
        else if (!strcmp(o, "-lpb")) LPB = atoi(v);
        else if (!strcmp(o, "-mfb")) MFB = atoi(v);
        else if (!strcmp(o, "-N")) LOGN = atoi(v);
        else if (!strcmp(o, "-skew")) SKEW = strtod(v, 0);
        else if (!strcmp(o, "-roots")) rootsf = v;
        else if (!strcmp(o, "-q0")) q0 = strtoull(v, 0, 10);
        else if (!strcmp(o, "-q1")) q1 = strtoull(v, 0, 10);
        else if (!strcmp(o, "-sample")) nsample = atol(v);
        else if (!strcmp(o, "-seed")) seed = strtoul(v, 0, 10);
        else if (!strcmp(o, "-out")) outf = v;
        else return usage();
    }
    if (!poly || parse_poly(poly)) die("bad -poly");
    if (LIM < 2 || LIM >= (1ULL << 32) || LPB < 2 || LPB > 63 || MFB < 0 || MFB > 62 || LOGN < 16 || LOGN > 30 || !(SKEW > 0))
        die("parameters out of range");
    LIM_SQ = LIM * LIM >= (1ULL << LPB);
    if (rootsf) fb_read(rootsf, LIM); else fb_compute(LIM);
    OUT = stdout;
    if (outf && !(OUT = fopen(outf, "w"))) { perror(outf); return 1; }
    static char obuf[1 << 20];
    setvbuf(OUT, obuf, _IOFBF, sizeof obuf);
    /* SCALE from special-q near the top of the range (or LIM) */
    {
        for (size_t e = 0; e < NFB; e++) if (FP[e] >= BR) LMIN += 8;   /* a segment must pay a first hit per entry */
        LMIN = LMIN > (1ULL << LOGN) / 2 ? (1ULL << LOGN) / 2 : LMIN < BR ? BR : LMIN;
        uint64_t qt = server ? (LIM > 3 ? LIM : 3) : (q1 > 3 ? q1 - 1 : 3);
        double mb = 0;
        uint32_t rr[4];
        for (int t = 0, k = 0; t < 400 && k < 4; t++) {
            uint64_t qq = qt - t;
            if (qq < 3 || !isprime64(qq)) continue;
            int n = roots_mod((uint32_t)qq, rr);
            for (int i = 0; i < n; i++) if (rr[i] < qq) { double b = est_maxbits(qq, rr[i]); if (b > mb) mb = b; k++; }
        }
        if (mb < 20) mb = 20;
        SCALE = 250.0 / (mb + 6);
        if (SCALE > 4) SCALE = 4;
    }
    setup();
    if (!server) run_range(q0, q1, nsample, seed);
    else {
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = on_usr1;
        sa.sa_flags = SA_RESTART;
        sigaction(SIGUSR1, &sa, 0);
        char line[256];
        while (fgets(line, sizeof line, stdin)) {
            unsigned long long a, b;
            if (sscanf(line, "%llu %llu", &a, &b) != 2) continue;
            run_range(a, b, 0, 0);
            fputs("# done\n", OUT);
            fflush(OUT);
        }
    }
    if (fflush(OUT) || (OUT != stdout && fclose(OUT))) { perror("sieve: output"); return 1; }
    return 0;
}
