/* check: the checker's C program; it shares no code with the prover.

     check rel                       the relations a - b alpha and (p), in exact integer arithmetic
     check rank [THREADS [PREFIX]]   dim V = rank(M) - rank(M_L) over F_2

   ---------- check rel
   F = c3 x^3 + c2 x^2 y + c1 x y^2 + c0 y^3 is primitive with disc F = d_K, so R_F = O_K (checked in PARI) and, by
   Dedekind-Kummer, the primes above p are the factors of F mod p.  A degree-1 prime is named "p:r" by its point (r : 1),
   or "p:p" for (1 : 0); one of degree f > 1 is "p::f".  For gcd(a, b) = 1 and J^-1 = O_K + alpha O_K, (a - b alpha) J
   is integral of norm |F(a, b)| and lies only in the primes at the point (a : b), while J = prod_(p | c3) P_(1:0)^v_p(c3).
   So v_P(a - b alpha) is v_p(F(a, b)) at P_(a:b), less v_p(c3) at P_(1:0), and 0 elsewhere; and (p) = prod P^e.
   Checked: F(a, b) != 0 is the product of the given primes, each prime (Miller-Rabin to 12 bases, a proof below 2^64),
   none dividing both a and b (so gcd(a, b) = 1).
   stdin, one item per line, integers in canonical decimal, separated exactly as shown:
     F c0 c1 c2 c3     first
     L p e             each prime factor of |c3|, proved prime by PARI; their product is checked here
     [a,b,[h1,..,hk]]  a relation, h > 0
     [p]               a free relation
   stdout: "R i col ..." per relation, its primes of odd valuation.  Exit 1 with "E i reason" on a failed check; 3 with
   "X i reason" beyond what is checked here (a prime of 2^64 or more, a number over MAXL limbs: gp checks those); 2 on
   malformed input.

   ---------- check rank
   M is the npiv pivot rows after the ops, M_L its columns 0..nl-1 (the large ones).  stdin, uint32 in native order:
   nrows, ncols, nl, nops, npiv; each row as its length and its distinct columns; nops pairs (dst, src), applied in
   order as row[dst] ^= row[src]; npiv row indices.  stdout: "rank R rankL RL large NL dimV D".  Exit 2 on bad input
   or no memory, 4 if the dense matrix fits neither the memory nor, in blocks, the temporary file (nothing decided).
   With the column order L < S, rank(M_L) is the number of echelon rows whose first column is in L, so dim V is the
   number whose first column is in S.  (1) Peeling, while a column c lies in one row r alone: if c is in L, r adds 1
   to both ranks; if r has no L column, it adds 1 to dim V; r is dropped.  (2) The rest by Gaussian elimination, one
   64-column word at a time, word-major so that each step reads in order, with "four Russians" tables of 8 pivots.
   PREFIX: also the counts for the first PREFIX pivot rows alone ("prefix P rank ..."), from the same run.  Echelon
   rows are reduced only by earlier rows, and a column in one row of M is in one of the first rows or none, so both
   steps restrict to them. */
#define _POSIX_C_SOURCE 200809L        /* getline */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

/* ==================== check rel */

typedef unsigned __int128 u128;

/* ---------- integers: sign and magnitude, 32-bit limbs, least significant first */

#define MAXL 4096
typedef struct { int n, neg; uint32_t d[MAXL]; } Big;

static int toobig;                      /* a result needed more than MAXL limbs */
static void bset_to(Big *r, const Big *x) { r->n = x->n; r->neg = x->neg; memcpy(r->d, x->d, (size_t)x->n * 4); }
static void bnorm(Big *x) { while (x->n && !x->d[x->n - 1]) x->n--; if (!x->n) x->neg = 0; }
static void bset(Big *x, uint64_t v) { x->neg = 0; x->d[0] = (uint32_t)v; x->d[1] = (uint32_t)(v >> 32); x->n = 2; bnorm(x); }
/* x = x m + c, m, c < 2^32 */
static void bmuladd(Big *x, uint32_t m, uint32_t c) {
  uint64_t k = c;
  for (int i = 0; i < x->n; i++) { k += (uint64_t)x->d[i] * m; x->d[i] = (uint32_t)k; k >>= 32; }
  if (k) { if (x->n == MAXL) { toobig = 1; return; } x->d[x->n++] = (uint32_t)k; }
}
/* x = |x| m, m < 2^64 */
static void bmul64(Big *x, uint64_t m) {
  uint32_t lo = (uint32_t)m, hi = (uint32_t)(m >> 32);
  if (!hi) { bmuladd(x, lo, 0); bnorm(x); return; }
  static Big t; bset_to(&t, x);
  bmuladd(x, lo, 0);
  bmuladd(&t, hi, 0);
  if (t.n + 1 > MAXL) { toobig = 1; return; }
  /* x += t 2^32 */
  uint64_t k = 0; int n = t.n + 1 > x->n ? t.n + 1 : x->n;
  for (int i = x->n; i < n; i++) x->d[i] = 0;
  for (int i = 0; i < n; i++) { k += (uint64_t)x->d[i] + (i >= 1 && i - 1 < t.n ? t.d[i - 1] : 0); x->d[i] = (uint32_t)k; k >>= 32; }
  x->n = n;
  if (k) { if (n == MAXL) { toobig = 1; return; } x->d[x->n++] = (uint32_t)k; }
  bnorm(x);
}
/* r = a b (r distinct from a and b) */
static void bmul(Big *r, const Big *a, const Big *b) {
  if (!a->n || !b->n) { r->n = 0; r->neg = 0; return; }
  if (a->n + b->n > MAXL) { toobig = 1; r->n = 0; return; }
  memset(r->d, 0, (size_t)(a->n + b->n) * 4);
  for (int i = 0; i < a->n; i++) {
    uint64_t k = 0;
    for (int j = 0; j < b->n; j++) { k += (uint64_t)a->d[i] * b->d[j] + r->d[i + j]; r->d[i + j] = (uint32_t)k; k >>= 32; }
    r->d[i + b->n] = (uint32_t)k;
  }
  r->n = a->n + b->n; r->neg = a->neg ^ b->neg; bnorm(r);
}
static int bcmpabs(const Big *a, const Big *b) {
  if (a->n != b->n) return a->n < b->n ? -1 : 1;
  for (int i = a->n - 1; i >= 0; i--) if (a->d[i] != b->d[i]) return a->d[i] < b->d[i] ? -1 : 1;
  return 0;
}
/* r = a + b (r may be a or b) */
static void badd(Big *r, const Big *a, const Big *b) {
  static Big t;
  const Big *x = a, *y = b;
  if (a->neg == b->neg) {
    int n = a->n > b->n ? a->n : b->n; uint64_t k = 0;
    for (int i = 0; i < n; i++) { k += (uint64_t)(i < a->n ? a->d[i] : 0) + (i < b->n ? b->d[i] : 0); t.d[i] = (uint32_t)k; k >>= 32; }
    t.n = n; if (k) { if (n == MAXL) { toobig = 1; return; } t.d[t.n++] = (uint32_t)k; }
    t.neg = a->neg;
  } else {
    if (bcmpabs(a, b) < 0) { x = b; y = a; }
    int64_t k = 0;
    for (int i = 0; i < x->n; i++) { k += (int64_t)x->d[i] - (i < y->n ? y->d[i] : 0); t.d[i] = (uint32_t)k; k = k < 0 ? -1 : 0; }
    t.n = x->n; t.neg = x->neg;
  }
  bnorm(&t); bset_to(r, &t);
}
/* |x| mod m, m >= 1 */
static uint64_t bmodu(const Big *x, uint64_t m) {
  u128 k = 0;
  for (int i = x->n - 1; i >= 0; i--) k = ((k << 32) | x->d[i]) % m;
  return (uint64_t)k;
}
/* x mod m in 0..m-1, x signed */
static uint64_t bmods(const Big *x, uint64_t m) { uint64_t r = bmodu(x, m); return x->neg && r ? m - r : r; }
/* the end of the canonical decimal integer at s ("0", or [1-9][0-9]*, after '-' if sign), or NULL */
static const char *canon(const char *s, int sign) {
  if (sign && *s == '-' && s[1] != '0') s++;
  if (*s == '0') return s + 1;
  if (*s < '1' || *s > '9') return NULL;
  while (*s >= '0' && *s <= '9') s++;
  return s;
}
/* x = the integer of the len chars at s, which canon() has checked (toobig if over MAXL limbs) */
static void bparse(Big *x, const char *s, size_t len) {
  x->n = 0; x->neg = *s == '-';
  for (size_t i = x->neg; i < len && !toobig; i++) {
    if (!x->n) { if (s[i] != '0') { x->d[0] = (uint32_t)(s[i] - '0'); x->n = 1; } }
    else bmuladd(x, 10, (uint32_t)(s[i] - '0'));
  }
  bnorm(x);
}
/* whether s is a relation line [a,b,[h1,...,hk]] (k >= 0) or a free relation [p] in canonical form */
static int wellformed(const char *s) {
  if (*s++ != '[') return 0;
  if (!strchr(s, ',')) return *s != '0' && (s = canon(s, 0)) && !strcmp(s, "]");
  if (!(s = canon(s, 1)) || *s++ != ',' || !(s = canon(s, 1)) || *s++ != ',' || *s++ != '[') return 0;
  if (*s != ']') for (;;) { if (*s == '0' || !(s = canon(s, 0))) return 0; if (*s != ',') break; s++; }
  return !strcmp(s, "]]");
}

/* ---------- arithmetic mod a prime p < 2^64 */

static uint64_t mulm(uint64_t a, uint64_t b, uint64_t m) { return (uint64_t)((u128)a * b % m); }
static uint64_t addm(uint64_t a, uint64_t b, uint64_t m) { u128 s = (u128)a + b; return (uint64_t)(s >= m ? s - m : s); }
static uint64_t subm(uint64_t a, uint64_t b, uint64_t m) { return a >= b ? a - b : (uint64_t)((u128)a + m - b); }
/* a^-1 mod p (p prime, a != 0 mod p), by the extended Euclidean algorithm; checked */
static uint64_t invm(uint64_t a, uint64_t p) {
  if (p < (1ULL << 62)) {
    int64_t r0 = (int64_t)p, r1 = (int64_t)(a % p), s0 = 0, s1 = 1;
    while (r1) { int64_t q = r0 / r1, t = r0 - q * r1; r0 = r1; r1 = t; t = s0 - q * s1; s0 = s1; s1 = t; }
    uint64_t x = (uint64_t)((s0 % (int64_t)p + (int64_t)p) % (int64_t)p);
    if (r0 != 1 || mulm(x, a, p) != 1 % p) { fprintf(stderr, "check rel: no inverse mod %llu\n", (unsigned long long)p); exit(2); }
    return x;
  }
  __int128 r0 = p, r1 = a % p, s0 = 0, s1 = 1;
  while (r1) { __int128 q = r0 / r1, t = r0 - q * r1; r0 = r1; r1 = t; t = s0 - q * s1; s0 = s1; s1 = t; }
  uint64_t x = (uint64_t)(((s0 % (__int128)p) + p) % p);
  if (r0 != 1 || mulm(x, a, p) != 1 % p) { fprintf(stderr, "check rel: no inverse mod %llu\n", (unsigned long long)p); exit(2); }
  return x;
}
/* Whether n is prime, by Miller-Rabin to the bases 2, 3, ..., 37: correct for every n < 3.18 10^23
   (Sorenson-Webster 2017), so for every n < 2^64. */
static uint64_t mulm32(uint64_t a, uint64_t b, uint64_t m) { return a * b % m; }
static int mr(uint64_t n) {
  static const uint64_t B[12] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 };
  if (n < 2) return 0;
  for (int i = 0; i < 12; i++) { if (n == B[i]) return 1; if (n % B[i] == 0) return 0; }
  uint64_t d = n - 1; int s = 0;
  while (!(d & 1)) { d >>= 1; s++; }
  uint64_t (*mul)(uint64_t, uint64_t, uint64_t) = n < (1ULL << 32) ? mulm32 : mulm;
  for (int i = 0; i < 12; i++) {
    uint64_t x = 1, a = B[i], e = d;
    while (e) { if (e & 1) x = mul(x, a, n); a = mul(a, a, n); e >>= 1; }
    if (x == 1 || x == n - 1) continue;
    int ok = 0;
    for (int k = 1; k < s; k++) { x = mul(x, x, n); if (x == n - 1) { ok = 1; break; } }
    if (!ok) return 0;
  }
  return 1;
}
/* mr, remembered: the primes of the relations repeat */
#define PCACHE (1u << 20)
static uint64_t pc[PCACHE]; static int8_t pv[PCACHE];
static int isprime64(uint64_t n) {
  uint32_t h = (uint32_t)((n * 0x9E3779B97F4A7C15ULL) >> 44);
  for (uint32_t t = 0; t < 16; t++, h = (h + 1) & (PCACHE - 1)) {
    if (pv[h] && pc[h] == n) return pv[h] > 0;
    if (!pv[h]) { int r = mr(n); pc[h] = n; pv[h] = r ? 1 : -1; return r; }
  }
  return mr(n);
}

/* polynomials mod p of degree <= 6, coefficients low to high; d = -1 for 0 */
typedef struct { int d; uint64_t c[7]; } Pol;
static void pnorm(Pol *a) { while (a->d >= 0 && !a->c[a->d]) a->d--; }
/* a mod m, m != 0 */
static void pmod(Pol *a, const Pol *m, uint64_t p) {
  uint64_t il = invm(m->c[m->d], p);
  pnorm(a);
  while (a->d >= m->d) {
    uint64_t q = mulm(a->c[a->d], il, p); int s = a->d - m->d;
    for (int i = 0; i <= m->d; i++) a->c[s + i] = subm(a->c[s + i], mulm(q, m->c[i], p), p);
    pnorm(a);
  }
}
static void pmulmod(Pol *r, const Pol *a, const Pol *b, const Pol *m, uint64_t p) {
  Pol t; t.d = a->d < 0 || b->d < 0 ? -1 : a->d + b->d;
  for (int i = 0; i <= t.d; i++) t.c[i] = 0;
  for (int i = 0; i <= a->d; i++) for (int j = 0; j <= b->d; j++) t.c[i + j] = addm(t.c[i + j], mulm(a->c[i], b->c[j], p), p);
  pmod(&t, m, p); *r = t;
}
static void ppowmod(Pol *r, const Pol *a, uint64_t e, const Pol *m, uint64_t p) {
  Pol x = *a, y; y.d = 0; y.c[0] = 1; pmod(&y, m, p); pmod(&x, m, p);
  while (e) { if (e & 1) pmulmod(&y, &y, &x, m, p); pmulmod(&x, &x, &x, m, p); e >>= 1; }
  *r = y;
}
/* the monic gcd of a and b */
static void pgcd(Pol *r, Pol a, Pol b, uint64_t p) {
  pnorm(&a); pnorm(&b);
  while (b.d >= 0) { pmod(&a, &b, p); Pol t = a; a = b; b = t; }
  if (a.d >= 0) { uint64_t il = invm(a.c[a.d], p); for (int i = 0; i <= a.d; i++) a.c[i] = mulm(a.c[i], il, p); }
  *r = a;
}
/* a / m, m divides a exactly */
static void pdiv(Pol *q, Pol a, const Pol *m, uint64_t p) {
  uint64_t il = invm(m->c[m->d], p);
  Pol r; r.d = a.d - m->d; for (int i = 0; i <= r.d; i++) r.c[i] = 0;
  pnorm(&a);
  while (a.d >= m->d) {
    uint64_t c = mulm(a.c[a.d], il, p); int s = a.d - m->d; r.c[s] = c;
    for (int i = 0; i <= m->d; i++) a.c[s + i] = subm(a.c[s + i], mulm(c, m->c[i], p), p);
    pnorm(&a);
  }
  pnorm(&r); *q = r;
}
static uint64_t peval(const Pol *a, uint64_t x, uint64_t p) { uint64_t v = 0; for (int i = a->d; i >= 0; i--) v = addm(mulm(v, x, p), a->c[i], p); return v; }
/* Put the roots of g (monic, a product of distinct linear factors) into out, by gcd((x + t)^((p-1)/2) - 1, g) for
   t = 0, 1, ... (p odd). */
static int splitroots(const Pol *g, uint64_t p, uint64_t *out) {
  if (g->d <= 0) return 0;
  if (g->d == 1) { out[0] = subm(0, mulm(g->c[0], invm(g->c[1], p), p), p); return 1; }
  for (uint64_t t = 0; t < p; t++) {
    Pol x; x.d = 1; x.c[0] = t % p; x.c[1] = 1;
    Pol h, d; ppowmod(&h, &x, (p - 1) / 2, g, p);
    if (h.d < 0) { h.d = 0; h.c[0] = 0; }
    h.c[0] = subm(h.c[0], 1, p); pnorm(&h);
    pgcd(&d, h, *g, p);
    if (d.d > 0 && d.d < g->d) { Pol e; pdiv(&e, *g, &d, p); int k = splitroots(&d, p, out); return k + splitroots(&e, p, out + k); }
  }
  return 0;
}

/* ---------- the field and the checks */

static Big C[4];                        /* c0..c3 */
#define MAXLC 256
static char lcs[MAXLC][4100]; static uint64_t lcp[MAXLC]; static int lce[MAXLC], nlc;   /* the primes of c3 */

typedef struct { uint64_t p; uint64_t r; int f; int par; } Lab;
#define MAXLAB 4096
static Lab lab[MAXLAB]; static int nlab;
static int over;
static void toggle(uint64_t p, uint64_t r, int f, long e) {
  if (!(e & 1)) return;
  for (int i = 0; i < nlab; i++) if (lab[i].p == p && lab[i].r == r && lab[i].f == f) { lab[i].par ^= 1; return; }
  if (nlab == MAXLAB) { over = 1; return; }
  lab[nlab].p = p; lab[nlab].r = r; lab[nlab].f = f; lab[nlab].par = 1; nlab++;
}

static long idx;
static _Noreturn void fail(const char *why) { printf("E %ld %s\n", idx, why); exit(1); }
static _Noreturn void limit(const char *why) { printf("X %ld %s\n", idx, why); exit(3); }
static _Noreturn void bad(const char *why) { fprintf(stderr, "check rel: line %ld: %s\n", idx, why); exit(2); }

/* the factors of F mod p: the points of P^1(F_p) with multiplicities, and the degree left (0, 2 or 3) */
static int factor_mod(uint64_t p, uint64_t *rt, int *mult, int *left) {
  Pol f; f.d = 3;
  for (int i = 0; i < 4; i++) f.c[i] = bmods(&C[i], p);
  pnorm(&f);
  if (f.d < 0) bad("F = 0 mod p: F is not primitive");
  int minf = 3 - f.d, k = 0;
  if (minf) { rt[k] = p; mult[k] = minf; k++; }
  uint64_t r[3]; int nr = 0;
  if (f.d >= 1) {
    if (p < 1024) { for (uint64_t x = 0; x < p; x++) if (!peval(&f, x, p)) r[nr++] = x; }
    else {
      Pol xp, X; X.d = 1; X.c[0] = 0; X.c[1] = 1;
      ppowmod(&xp, &X, p, &f, p);
      if (xp.d < 1) { for (int i = xp.d + 1; i <= 1; i++) xp.c[i] = 0; xp.d = 1; }
      xp.c[1] = subm(xp.c[1], 1, p); pnorm(&xp);
      Pol g; pgcd(&g, f, xp, p);
      nr = splitroots(&g, p, r);
      if (nr != (g.d > 0 ? g.d : 0)) bad("the roots of F mod p were not all found");
    }
  }
  int used = minf;
  for (int i = 0; i < nr; i++) {
    if (peval(&f, r[i], p)) bad("not a root");
    Pol a = f; int m = 0;
    while (a.d >= 1 && !peval(&a, r[i], p)) {     /* divide by x - r */
      Pol l; l.d = 1; l.c[0] = subm(0, r[i], p); l.c[1] = 1; pdiv(&a, a, &l, p); m++;
    }
    rt[k] = r[i]; mult[k] = m; k++; used += m;
  }
  *left = 3 - used;
  if (*left == 1 || *left < 0) bad("F mod p has a degree-1 factor without a root");
  return k;
}

static void out_labels(void) {
  printf("R %ld", idx);
  for (int i = 0; i < nlab; i++) if (lab[i].par) {
    if (lab[i].f == 1) {
      if (lab[i].p == UINT64_MAX) printf(" %s:%s", lcs[lab[i].r], lcs[lab[i].r]);      /* (1:0) above a large prime of c3 */
      else printf(" %llu:%llu", (unsigned long long)lab[i].p, (unsigned long long)lab[i].r);
    } else printf(" %llu::%d", (unsigned long long)lab[i].p, lab[i].f);
  }
  printf("\n");
}

/* the J part: P_(1:0)^-v_p(c3) for every prime of c3 */
static void jpart(void) {
  for (int i = 0; i < nlc; i++) {
    if (lcp[i]) toggle(lcp[i], lcp[i], 1, lce[i]);
    else toggle(UINT64_MAX, (uint64_t)i, 1, lce[i]);
  }
}

static uint64_t H[1 << 16];
static int cmp64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return (x > y) - (x < y); }

static void relation(const char *s) {
  /* [a,b,[h1,...]], wellformed */
  const char *p = s + 1, *c1 = strchr(p, ','), *c2 = strchr(c1 + 1, ',');
  static Big a, b, n, t, u, v, prod;
  toobig = 0;
  bparse(&a, p, (size_t)(c1 - p)); bparse(&b, c1 + 1, (size_t)(c2 - c1 - 1));
  if (toobig) limit("has a number too large to check");
  int k = 0; const char *q = c2 + 2;
  while (*q != ']') {
    uint64_t h = 0;
    while (*q >= '0' && *q <= '9') {
      unsigned d = (unsigned)(*q - '0');
      if (h > (UINT64_MAX - d) / 10) limit("has a prime of 2^64 or more");
      h = h * 10 + d; q++;
    }
    if (k == (1 << 16)) limit("has too many primes");
    H[k++] = h;
    if (*q == ',') q++;
  }
  /* n = F(a, b) = ((c3 a + c2 b) a + c1 b^2) a + c0 b^3 */
  bmul(&t, &C[3], &a); bmul(&u, &C[2], &b); badd(&t, &t, &u);
  bmul(&n, &t, &a);
  bmul(&t, &b, &b); bmul(&u, &C[1], &t); badd(&n, &n, &u);
  bmul(&v, &n, &a);
  bmul(&u, &t, &b); bmul(&t, &C[0], &u); badd(&n, &v, &t);
  if (toobig) limit("has F(a, b) too large to check");
  if (!n.n) fail("has F(a, b) = 0");
  qsort(H, (size_t)k, 8, cmp64);
  bset(&prod, 1);
  for (int i = 0; i < k; i++) {
    bmul64(&prod, H[i]);
    if (toobig || bcmpabs(&prod, &n) > 0) fail("does not factor as claimed");
  }
  if (bcmpabs(&prod, &n)) fail("does not factor as claimed");
  nlab = 0; over = 0;
  jpart();
  for (int i = 0; i < k;) {
    int j = i; while (j < k && H[j] == H[i]) j++;
    uint64_t h = H[i];
    if (!isprime64(h)) { char m[96]; snprintf(m, sizeof m, "has %llu, which is not prime", (unsigned long long)h); fail(m); }
    uint64_t am = bmods(&a, h), bm = bmods(&b, h);
    if (!am && !bm) fail("has a and b not coprime");
    uint64_t r = bm ? mulm(am, invm(bm, h), h) : h;
    toggle(h, r, 1, j - i);
    i = j;
  }
  if (over) limit("has too many primes");
  out_labels();
}

static void freerel(const char *s) {
  /* [p], wellformed */
  const char *q = s + 1; uint64_t p = 0;
  while (*q >= '0' && *q <= '9') {
    unsigned d = (unsigned)(*q - '0');
    if (p > (UINT64_MAX - d) / 10) limit("has a prime of 2^64 or more");
    p = p * 10 + d; q++;
  }
  if (!isprime64(p)) { char m[96]; snprintf(m, sizeof m, "is (%llu), and %llu is not prime", (unsigned long long)p, (unsigned long long)p); fail(m); }
  uint64_t rt[4]; int mult[4], left;
  int k = factor_mod(p, rt, mult, &left);
  nlab = 0;
  for (int i = 0; i < k; i++) toggle(p, rt[i], 1, mult[i]);
  if (left) toggle(p, 0, left, 1);
  out_labels();
}

static int rel_main(void) {
  char *line = NULL; size_t cap = 0; ssize_t got;
  int haveF = 0; long nrel = 0;
  static Big lc;
  bset(&lc, 1);
  while ((got = getline(&line, &cap, stdin)) > 0) {
    size_t len = (size_t)got;
    idx = nrel;
    if (len > (1 << 20)) limit("is too long to check");
    if (memchr(line, 0, len)) bad("a NUL byte");
    while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
    if (!len) continue;
    if (line[0] == 'F') {
      const char *s = line + 1, *e;
      toobig = 0;
      for (int i = 0; i < 4; i++, s = e) {
        if (*s++ != ' ' || !(e = canon(s, 1))) bad("bad F line");
        bparse(&C[i], s, (size_t)(e - s));
      }
      if (*s || toobig) bad("bad F line");
      if (!C[3].n) bad("c3 = 0");
      haveF = 1;
    } else if (line[0] == 'L') {
      const char *ps = line + 2, *q = line[1] == ' ' && *ps != '0' ? canon(ps, 0) : NULL;
      const char *r = q && *q == ' ' && q[1] != '0' ? canon(q + 1, 0) : NULL;
      if (nlc == MAXLC || !r || *r || q - ps >= 4100 || r - q > 7) bad("bad L line");
      static Big pb; toobig = 0;
      bparse(&pb, ps, (size_t)(q - ps));
      if (toobig) bad("bad L prime");
      int e = atoi(q + 1);
      memcpy(lcs[nlc], ps, (size_t)(q - ps)); lcs[nlc][q - ps] = 0; lce[nlc] = e;
      lcp[nlc] = pb.n <= 2 ? (uint64_t)pb.d[0] | (pb.n == 2 ? (uint64_t)pb.d[1] << 32 : 0) : 0;
      if (lcp[nlc] && !isprime64(lcp[nlc])) bad("L: not prime");
      for (int t = 0; t < e; t++) {
        static Big r; bmul(&r, &lc, &pb); bset_to(&lc, &r);
        if (toobig) bad("L: too large");
      }
      nlc++;
    } else if (line[0] == '[') {
      if (!haveF) bad("no F line");
      if (bcmpabs(&lc, &C[3])) bad("the L lines are not the factorization of |c3|");
      if (!wellformed(line)) bad("malformed relation");
      if (strchr(line, ',')) relation(line); else freerel(line);
      nrel++;
    } else bad("unknown line");
  }
  if (haveF && bcmpabs(&lc, &C[3])) bad("the L lines are not the factorization of |c3|");
  return 0;
}

/* ==================== check rank */

typedef struct { uint32_t n, *c; } Row;     /* sorted columns */

static void die(const char *msg) { fprintf(stderr, "check rank: %s\n", msg); exit(2); }
static void *need(void *p) { if (!p) die("out of memory"); return p; }
static void *alloc(size_t n, size_t sz) { return need(calloc(n ? n : 1, sz)); }
static void get(void *p, size_t n) { if (n && fread(p, 4, n, stdin) != n) die("short input"); }
static int cmpu(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return (x > y) - (x < y); }

/* d ^= s (sorted merge; d == s gives the empty row) */
static uint32_t *buf; static size_t bufcap;
static void xor_into(Row *d, const Row *s) {
  if (bufcap <= (size_t)d->n + s->n) { bufcap = 2 * ((size_t)d->n + s->n) + 1; buf = need(realloc(buf, bufcap * 4)); }
  uint32_t i = 0, j = 0, k = 0;
  while (i < d->n && j < s->n) {
    if (d->c[i] < s->c[j]) buf[k++] = d->c[i++]; else if (d->c[i] > s->c[j]) buf[k++] = s->c[j++]; else { i++; j++; }
  }
  while (i < d->n) buf[k++] = d->c[i++];
  while (j < s->n) buf[k++] = s->c[j++];
  free(d->c); d->c = alloc(k, 4); memcpy(d->c, buf, (size_t)k * 4); d->n = k;
}

/* ---------- (2) the echelon form of the dense rows, BW words of every row in memory at a time
   Each block is built from the sparse rows, the records of the earlier words (pivots, their combinations, the rows
   cleared and their combinations) are replayed on it from a temporary file, and its own words are eliminated.
   Why the result is the full elimination's, by induction on the words w: when word w is examined, the open rows,
   their order, the counts, the value of word w of every row and the records before w are those of the full run.
   A clearing of word t reads and writes word t alone, as a function of the record, and both runs apply the same
   records to it in the same order; examining word w reads only word w of the open rows, in order.  So both stop at
   the same word with the same counts.  A record with no rows to clear changes nothing and is not written; threads
   clear disjoint words.
   The file is unlinked at creation, so it is removed on every exit; each record read back is checked for its word
   and bounds. */
static uint64_t *A; static size_t W, N, B0;   /* word t of row r is A[(t - B0) * N + r] */
/* A clearing of nr rows on `words` words costs about words (nr + 2048) table reads (about 1 ns each); each thread
   gets at least PAR_WORK of them, as starting one costs microseconds (0: always every thread). */
#ifndef PAR_WORK
#define PAR_WORK (1u << 17)
#endif
static uint64_t nclear, nthreaded, nstarted; static int maxT = 1;      /* for the mode line */
typedef struct { const uint32_t *row; const uint64_t *combo; size_t n; const uint32_t *piv; const uint64_t *pm; int np; size_t lo, hi; } Job;
/* clear the rows on the words lo..hi-1; T[g][v] is the xor of the pivots 8g + j over the bits j of v */
static void *clear(void *arg) {
  const Job *J = arg; uint64_t T[8][256], P[64];
  for (size_t t = J->lo; t < J->hi; t++) {
    uint64_t *a = A + (t - B0) * N;
    for (int j = 0; j < J->np; j++) {
      P[j] = a[J->piv[j]];
      for (uint64_t m = J->pm[j]; m; m &= m - 1) P[j] ^= P[__builtin_ctzll(m)];
    }
    for (int g = 0; 8 * g < J->np; g++) {
      T[g][0] = 0;
      for (int v = 1; v < 1 << (J->np - 8 * g < 8 ? J->np - 8 * g : 8); v++) T[g][v] = T[g][v & (v - 1)] ^ P[8 * g + __builtin_ctz(v)];
    }
    for (int g = (J->np + 7) / 8; g < 8; g++) T[g][0] = 0;
    for (size_t r = 0; r < J->n; r++) {
      uint64_t m = J->combo[r];
      a[J->row[r]] ^= T[0][m & 255] ^ T[1][m >> 8 & 255] ^ T[2][m >> 16 & 255] ^ T[3][m >> 24 & 255] ^
                      T[4][m >> 32 & 255] ^ T[5][m >> 40 & 255] ^ T[6][m >> 48 & 255] ^ T[7][m >> 56];
    }
  }
  return NULL;
}
/* one record's clearing of the words lo..hi-1, split over the threads */
static void clear_words(const uint32_t *row, const uint64_t *combo, size_t nr, const uint32_t *piv, const uint64_t *pm,
                        int np, size_t lo, size_t hi, int threads) {
  size_t words = hi - lo;
  double work = (double)words * ((double)nr + 2048), k = PAR_WORK > 0 ? work / PAR_WORK : threads;
  int T = k >= threads ? threads : k >= 2 ? (int)k : 1;
  if ((size_t)T > words) T = words ? (int)words : 1;
  nclear++; nthreaded += T > 1; nstarted += T - 1; if (T > maxT) maxT = T;
  pthread_t th[256]; Job J[256]; int up[256];
  for (int t = 0; t < T; t++) J[t] = (Job){ row, combo, nr, piv, pm, np, lo + words * t / T, lo + words * (t + 1) / T };
  for (int t = 1; t < T; t++) up[t] = !pthread_create(&th[t], NULL, clear, &J[t]);
  clear(&J[0]);
  for (int t = 1; t < T; t++) { if (up[t]) pthread_join(th[t], NULL); else clear(&J[t]); }
}
/* the temporary file of the records */
static FILE *lg; static uint64_t lgbytes, lgmax = UINT64_MAX;
static void nofit(const char *msg) { fprintf(stderr, "check rank: does not fit: %s\n", msg); exit(4); }
static void lwrite(const void *p, size_t sz, size_t n) {
  if (lgbytes + (uint64_t)sz * n > lgmax) nofit("the temporary file is over CHECK_RANK_TMP_MB");
  if (n && fwrite(p, sz, n, lg) != n) die("cannot write the temporary file");
  lgbytes += (uint64_t)sz * n;
}
static void lread(void *p, size_t sz, size_t n) { if (n && fread(p, sz, n, lg) != n) die("cannot read the temporary file"); }
static uint64_t envnum(const char *name, uint64_t max) {    /* 0: not set; else a decimal 1..max */
  const char *s = getenv(name), *s0 = s; uint64_t v = 0;
  if (!s) return 0;
  for (; *s; s++) {
    if (*s < '0' || *s > '9' || v > (max - (uint64_t)(*s - '0')) / 10) break;
    v = 10 * v + (uint64_t)(*s - '0');
  }
  if (*s || !*s0 || !v) {
    fprintf(stderr, "check rank: %s is \"%.40s\", not a whole number from 1 to 2^40; unset it or set it to one\n", name, s0);
    exit(2);
  }
  return v;
}
static void lopen(void) {
  const char *d = getenv("CHECK_RANK_TMPDIR");
  if (!d || !*d) d = getenv("TMPDIR");
  if (!d || !*d) d = "/tmp";
  size_t n = strlen(d) + 32; char *path = alloc(n, 1);
  snprintf(path, n, "%s/check-rank-XXXXXX", d);
  int fd = mkstemp(path);
  if (fd < 0) die("cannot create the temporary file");
  if (unlink(path)) { close(fd); die("cannot unlink the temporary file"); }
  free(path);
  if (!(lg = fdopen(fd, "w+b"))) { close(fd); die("cannot open the temporary file"); }
  setvbuf(lg, NULL, _IOFBF, 1 << 20);
}
/* Return the rank of the dense rows M[left[r]] (column c at bit who[c]), with in *inS the echelon rows whose first
   bit is at or above sbit, and the same for the rows below pre in *rankP and *inSP.
   The memory needed is own (the sparse rows) + 20 N + 8 N BW (the block) + BASE (program, libraries, file buffer) +
   THREAD_MEM per thread (stack and arena as touched). */
#define BASE (3u << 20)
#define THREAD_MEM (64u << 10)
static uint32_t echelon(const Row *M, const uint32_t *left, const uint32_t *who, size_t sbit, int threads, uint32_t *inS,
                        uint32_t pre, uint32_t *rankP, uint32_t *inSP, uint64_t own) {
  uint64_t mb = envnum("CHECK_RANK_BLOCK_MB", 1ULL << 40), bw = envnum("CHECK_RANK_BLOCK_WORDS", 1ULL << 40),
           mem = envnum("CHECK_RANK_MEM_MB", 1ULL << 40) << 20, BW = W,
           need = own + 20 * (uint64_t)N + BASE + (uint64_t)(threads > 1 ? threads : 1) * THREAD_MEM;
  const char *by = "unmeasured";
  if (bw) BW = bw, by = "CHECK_RANK_BLOCK_WORDS";      /* for the tests */
  else if (mb) BW = N ? (mb << 20) / 8 / N : W, by = "CHECK_RANK_BLOCK_MB";
  else if (mem) {
    by = "memory";
    if (N && need + 8 * (uint64_t)N * W > mem) {
      if (need + 8 * (uint64_t)N > mem) {
        fprintf(stderr, "check rank: %zu x %zu words dense: one word needs %.1f MiB, the memory allowed "
                "(CHECK_RANK_MEM_MB) is %.1f MiB\n", N, W, (need + 8.0 * N) / 1048576, mem / 1048576.0);
        nofit("not enough memory for one word of the dense matrix");
      }
      BW = (mem - need) / 8 / N;
    }
  }
  if (BW < 1) BW = 1;
  if (BW > W) BW = W;
  uint32_t *open = alloc(N, 4), *row = alloc(N, 4), *cur = alloc(N, 4), nopen = (uint32_t)N, rank = 0;
  uint64_t *combo = alloc(N, 8);
  time_t t0 = time(NULL);
  if (BW && N > SIZE_MAX / 8 / BW) die("out of memory");
  A = alloc(N * BW, 8);
  if (BW < W) {
    uint64_t tm = envnum("CHECK_RANK_TMP_MB", 1ULL << 40);
    if (tm) lgmax = tm << 20;
    lopen();
    fprintf(stderr, "check rank: blocks of %zu of the %zu words (%.1f MiB)\n", (size_t)BW, W, 8.0 * N * BW / 1048576);
  }
  for (uint32_t i = 0; i < N; i++) open[i] = i;
  for (B0 = 0; B0 < W && nopen; B0 += BW) {
    size_t B1 = W - B0 < BW ? W : B0 + BW, hi = 64 * B1;
    /* the block from the sparse rows (their columns are sorted, so cur[r] is where block B0 starts in row r) */
    memset(A, 0, (B1 - B0) * N * 8);
    for (uint32_t r = 0; r < N; r++) {
      const Row *s = &M[left[r]]; uint32_t k = cur[r];
      for (size_t j; k < s->n && (j = who[s->c[k]]) < hi; k++) A[(j / 64 - B0) * N + r] |= 1ULL << (j % 64);
      cur[r] = k;
    }
    /* the records of the words before B0, in order */
    if (B0) {
      uint64_t got = 0, last = 0; int first = 1;
      if (fflush(lg) || fseek(lg, 0, SEEK_SET)) die("cannot read the temporary file");
      fprintf(stderr, "check rank: block at word %zu, replaying %.1f MiB, %.0f s\n", B0, lgbytes / 1048576.0, difftime(time(NULL), t0));
      while (got < lgbytes) {
        uint32_t hd[3], piv[64]; uint64_t pm[64];
        lread(hd, 4, 3);
        if (hd[0] >= B0 || (!first && hd[0] <= last) || hd[1] > 64 || !hd[1] || hd[2] > N) die("a bad record in the temporary file");
        first = 0; last = hd[0];
        lread(piv, 4, hd[1]); lread(pm, 8, hd[1]); lread(row, 4, hd[2]); lread(combo, 8, hd[2]);
        for (uint32_t j = 0; j < hd[1]; j++) if (piv[j] >= N || pm[j] >> j) die("a bad record in the temporary file");
        for (uint32_t k = 0; k < hd[2]; k++) if (row[k] >= N) die("a bad record in the temporary file");
        got += 12 + 12ULL * hd[1] + 12ULL * hd[2];
        clear_words(row, combo, hd[2], piv, pm, (int)hd[1], B0, B1, threads);
      }
      if (got != lgbytes || fseek(lg, 0, SEEK_END)) die("cannot read the temporary file");
    }
    for (size_t w = B0; w < B1 && nopen; w++) {
      if (w && !(w % 1024)) fprintf(stderr, "check rank: word %zu of %zu, %u rows left, %.0f s\n", w, W, nopen, difftime(time(NULL), t0));
      uint64_t lead[64], pm[64]; uint32_t piv[64]; int at[64], np = 0;     /* at[b]: the pivot whose lowest bit is b */
      for (int b = 0; b < 64; b++) at[b] = -1;
      size_t nr = 0; uint32_t left = 0;
      for (uint32_t k = 0; k < nopen; k++) {
        uint32_t i = open[k]; uint64_t x = A[(w - B0) * N + i], m = 0; int b;
        while (x && at[b = __builtin_ctzll(x)] >= 0) { m |= 1ULL << at[b]; x ^= lead[at[b]]; }
        if (x) {          /* a new pivot (m: its combination of the earlier ones) */
          b = __builtin_ctzll(x);
          at[b] = np; lead[np] = x; pm[np] = m; piv[np++] = i; rank++;
          *inS += 64 * w + b >= sbit;
          if (i < pre) { (*rankP)++; *inSP += 64 * w + b >= sbit; }
        } else {
          open[left++] = i;
          if (m) { row[nr] = i; combo[nr++] = m; }
        }
      }
      nopen = left;
      /* the words w+1..B1-1 of the block, split over the threads; the record, for the blocks after this one */
      clear_words(row, combo, nr, piv, pm, np, w + 1, B1, threads);
      if (B1 < W && nr) {
        uint32_t hd[3] = { (uint32_t)w, (uint32_t)np, (uint32_t)nr };
        lwrite(hd, 4, 3); lwrite(piv, 4, (size_t)np); lwrite(pm, 8, (size_t)np); lwrite(row, 4, nr); lwrite(combo, 8, nr);
      }
    }
  }
  if (lg) { fprintf(stderr, "check rank: temporary file %llu bytes\n", (unsigned long long)lgbytes); fclose(lg); lg = NULL; }
  fprintf(stderr, "check rank: mode %s by %s, %zu of %zu words a block, dense %llu bytes, held %llu bytes, memory %llu bytes,"
          " temporary file %llu bytes, clearings %llu, threaded %llu, threads started %llu, up to %d\n",
          BW < W ? "blocks" : "whole", by, (size_t)BW, W, (unsigned long long)(8 * (uint64_t)N * W), (unsigned long long)own,
          (unsigned long long)mem, (unsigned long long)lgbytes, (unsigned long long)nclear, (unsigned long long)nthreaded,
          (unsigned long long)nstarted, maxT);
  free(open); free(row); free(cur); free(combo); free(A); A = NULL;
  return rank;
}

static int rank_main(int threads, uint32_t pre) {
  time_t t0 = time(NULL);
  uint32_t h[5];
  get(h, 5);
  uint32_t nrows = h[0], ncols = h[1], nl = h[2], nops = h[3], n = h[4];
  if (nl > ncols) die("nl > ncols");
  if (pre == UINT32_MAX) pre = n;
  else if (pre > n) die("PREFIX > the pivot rows");
  Row *R = alloc(nrows, sizeof *R);
  for (uint32_t i = 0; i < nrows; i++) {
    get(&R[i].n, 1);
    if (R[i].n > ncols) die("a row longer than ncols");
    R[i].c = alloc(R[i].n, 4); get(R[i].c, R[i].n);
    qsort(R[i].c, R[i].n, 4, cmpu);
    for (uint32_t k = 0; k < R[i].n; k++)
      if (R[i].c[k] >= ncols || (k && R[i].c[k] == R[i].c[k - 1])) die("a column out of range or repeated");
  }
  for (uint32_t t = 0; t < nops; t++) {
    uint32_t o[2]; get(o, 2);
    if (o[0] >= nrows || o[1] >= nrows) die("an op out of range");
    xor_into(&R[o[0]], &R[o[1]]);
  }
  /* M; the counts and xors of ids of the columns' rows */
  Row *M = alloc(n, sizeof *M);
  uint32_t *cnt = alloc(ncols, 4), *who = alloc(ncols, 4), *stack = alloc(ncols, 4), ns = 0, large = 0, largeP = 0;
  uint8_t *inP = alloc(ncols, 1);
  for (uint32_t t = 0; t < n; t++) {
    uint32_t i; get(&i, 1);
    if (i >= nrows) die("a pivot out of range");
    M[t] = R[i];
    for (uint32_t k = 0; k < M[t].n; k++) {
      uint32_t c = M[t].c[k]; large += c < nl && !cnt[c]; cnt[c]++; who[c] ^= t;
      if (t < pre && !inP[c]) { inP[c] = 1; largeP += c < nl; }
    }
  }
  /* (1) peeling (a column in one row of all of M is in one row of the first pre rows, or in none) */
  uint8_t *gone = alloc(n, 1); uint32_t rank = 0, dimV = 0, rankP = 0, dimVP = 0;
  for (uint32_t c = 0; c < ncols; c++) if (cnt[c] == 1) stack[ns++] = c;
  while (ns) {
    uint32_t c = stack[--ns], r = who[c];
    if (cnt[c] != 1 || (c >= nl && M[r].c[0] < nl)) continue;
    gone[r] = 1; rank++; dimV += c >= nl;
    if (r < pre) { rankP++; dimVP += c >= nl; }
    for (uint32_t k = 0; k < M[r].n; k++) { uint32_t d = M[r].c[k]; who[d] ^= r; if (--cnt[d] == 1) stack[ns++] = d; }
  }
  /* (2) the rows left, over their columns, numbered again (L first) */
  uint32_t nc = 0, nL = 0, nr = 0, nrP = 0, *left = alloc(n, 4);
  for (uint32_t c = 0; c < ncols; c++) { who[c] = cnt[c] ? nc++ : 0; if (c < nl && cnt[c]) nL++; }
  for (uint32_t t = 0; t < n; t++) if (!gone[t] && M[t].n) { left[nr++] = t; nrP += t < pre; }
  fprintf(stderr, "check rank: %u rows, %u ops: %u peeled; %u x %u (%u large) dense, %.0f s\n", n, nops, rank, nr, nc, nL,
          difftime(time(NULL), t0));
  W = (nc + 63) / 64; N = nr;
  uint64_t own = 16 * ((uint64_t)nrows + n) + 13 * (uint64_t)ncols + 6 * (uint64_t)n + 4 * (uint64_t)bufcap;
  for (uint32_t i = 0; i < nrows; i++) own += 4 * (uint64_t)R[i].n + 16;      /* the sparse rows, held (+ malloc's) */
  uint32_t inS = 0, inSP = 0;
  rank += echelon(M, left, who, nL, threads, &inS, nrP, &rankP, &inSP, own);
  dimV += inS; dimVP += inSP;
  if (pre < n)
    printf("rank %u rankL %u large %u dimV %u prefix %u rank %u rankL %u large %u dimV %u\n", rank, rank - dimV, large, dimV,
           pre, rankP, rankP - dimVP, largeP, dimVP);
  else
    printf("rank %u rankL %u large %u dimV %u\n", rank, rank - dimV, large, dimV);
  fprintf(stderr, "check rank: done, %.0f s\n", difftime(time(NULL), t0));
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && !strcmp(argv[1], "rel")) return rel_main();
  if (argc >= 2 && argc <= 4 && !strcmp(argv[1], "rank")) {
    int threads = argc >= 3 ? atoi(argv[2]) : 1;
    long long pre = -1;
    if (argc == 4) {
      char *end;
      errno = 0;
      pre = strtoll(argv[3], &end, 10);
      if (errno || end == argv[3] || *end || pre < 0 || pre >= UINT32_MAX) {
        fprintf(stderr, "check rank: PREFIX must be an integer from 0 to %u\n", UINT32_MAX - 1);
        return 2;
      }
    }
    return rank_main(threads < 1 ? 1 : threads > 256 ? 256 : threads, pre < 0 ? UINT32_MAX : (uint32_t)pre);
  }
  fprintf(stderr, "usage: check rel < relations, or check rank [THREADS [PREFIX]] < rows\n");
  return 2;
}
