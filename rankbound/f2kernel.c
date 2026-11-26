/* f2kernel: combinations of rows over F_2 whose sum vanishes (the 2-descent's kernel vectors), by block Lanczos
   (Montgomery, EUROCRYPT 1995), or by Gaussian elimination when the rows are few.

   Usage:  f2kernel POOL OUT THREADS SEED PROGRESS
   POOL: f2elim's OUT.pool: uint32 the number of rows, then per row its id, its number of columns and those (distinct).
   OUT: uint32 the number of vectors, then per vector its number of rows and their ids.
   stdout: "kernel rows N cols M nnz Z empty E vectors K method lanczos|dense iterations I tries T time S status S".
   status complete: a basis of the whole kernel (dense); lanczos: some vectors of a possibly larger kernel; partial:
   every Lanczos try broke down, so OUT holds only the empty rows (the rest of the kernel is unknown, not zero).
   Exit 0 done, 2 bad input or no memory, 5 no vectors (every try broke down and no row is empty).

   An empty row is a vector by itself.  Each Lanczos vector is checked (B y = 0).  XOR sums do not depend on order,
   so the output depends on the input and SEED, not THREADS.  A breakdown retries from the next seed, at most 4
   tries.  At most DENSE_N rows (F2KERNEL_DENSE_N, default 1000; 0: never) are eliminated densely instead. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

typedef uint64_t u64;
static uint32_t N, M;           /* the rows with columns (the unknowns), the columns */
static size_t NNZ;
static size_t *rp, *cp;         /* row i: ri[rp[i] .. rp[i+1]-1]; column c: ci[cp[c] .. cp[c+1]-1] */
static uint32_t *ri, *ci;
static uint32_t *rid;           /* each row's id */
static int T = 1;
static double t0, progress, tlast;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void *chk(void *p, size_t n, const char *what) {
  if (!p && n) { fprintf(stderr, "out of memory (%s: %.0f MB)\n", what, (double)n / 1e6); exit(2); }
  return p;
}
static void *alloc0(size_t n, size_t sz, const char *what) { return chk(calloc(n ? n : 1, sz), n * sz, what); }
static u64 rng;
static u64 rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng * 0x2545F4914F6CDD1DULL; }

/* ---------- threads: f(t, lo, hi, arg) on n items, at least PAR_WORK work units (about 1 ns each) per thread, as a
   thread costs microseconds to start (a 10,000-row kernel on 4 threads: 0.35 s at 8192, 0.15 s at this).
   F2KERNEL_PAR_WORK overrides it (0: always T threads). */
#define PAR_WORK (1u << 17)
static double par_work = PAR_WORK;
typedef void (*Fn)(int, uint32_t, uint32_t, void *);
typedef struct { Fn f; void *a; int t; uint32_t lo, hi; } Job;
static void *runjob(void *p) { Job *j = p; j->f(j->t, j->lo, j->hi, j->a); return NULL; }
static int nthreads(uint32_t n, double work) {
  double w = par_work > 0 ? work / par_work : T;
  int k = w >= T ? T : w >= 2 ? (int)w : 1;
  return (uint32_t)k > n ? (n ? (int)n : 1) : k;
}
static int par_k(uint32_t n, int k, Fn f, void *a) {
  pthread_t th[256]; Job J[256]; int up[256];
  for (int t = 0; t < k; t++) J[t] = (Job){ f, a, t, (uint32_t)((u64)n * t / k), (uint32_t)((u64)n * (t + 1) / k) };
  for (int t = 1; t < k; t++) up[t] = pthread_create(&th[t], NULL, runjob, &J[t]) == 0;
  runjob(&J[0]);
  for (int t = 1; t < k; t++) { if (up[t]) pthread_join(th[t], NULL); else runjob(&J[t]); }
  return k;
}
static int par(uint32_t n, double work, Fn f, void *a) { return par_k(n, nthreads(n, work), f, a); }

/* ---------- the products: y = B x (y over the columns), z = B^T y (z over the rows) */
typedef struct { const u64 *x; u64 *y; } Pair;
static void fB(int t, uint32_t lo, uint32_t hi, void *a) {
  (void)t; const Pair *p = a;
  for (uint32_t c = lo; c < hi; c++) { u64 s = 0; for (size_t k = cp[c]; k < cp[c + 1]; k++) s ^= p->x[ci[k]]; p->y[c] = s; }
}
static void fBT(int t, uint32_t lo, uint32_t hi, void *a) {
  (void)t; const Pair *p = a;
  for (uint32_t i = lo; i < hi; i++) { u64 s = 0; for (size_t k = rp[i]; k < rp[i + 1]; k++) s ^= p->x[ri[k]]; p->y[i] = s; }
}
static void mulB(const u64 *x, u64 *y) { Pair p = { x, y }; par(M, (double)NNZ + M, fB, &p); }
static void mulBT(const u64 *x, u64 *y) { Pair p = { x, y }; par(N, (double)NNZ + N, fBT, &p); }
static u64 *tmpM;
static void mulA(const u64 *x, u64 *y) { mulB(x, tmpM); mulBT(tmpM, y); }

/* ---------- 64 x 64 matrices (row j: a uint64, bit k its entry k) and N x 64 blocks */
/* c = a b */
static void mul64(const u64 *a, const u64 *b, u64 *c) {
  u64 r[64];
  for (int i = 0; i < 64; i++) { u64 s = 0; for (u64 m = a[i]; m; m &= m - 1) s ^= b[__builtin_ctzll(m)]; r[i] = s; }
  memcpy(c, r, sizeof r);
}
/* out = x^T y (64 x 64), by tables per thread */
typedef struct { const u64 *x, *y; u64 (*tab)[8][256]; } Inner;
static void fInner(int t, uint32_t lo, uint32_t hi, void *a) {
  const Inner *p = a; u64 (*tb)[256] = p->tab[t];
  for (uint32_t i = lo; i < hi; i++) {
    u64 xi = p->x[i], yi = p->y[i];
    for (int b = 0; b < 8; b++) tb[b][(xi >> (8 * b)) & 255] ^= yi;
  }
}
static u64 (*itab)[8][256];
static void inner(const u64 *x, const u64 *y, u64 *out) {
  int k = nthreads(N, 8.0 * N);
  memset(itab, 0, (size_t)k * sizeof *itab);
  Inner p = { x, y, itab }; par_k(N, k, fInner, &p);
  memset(out, 0, 64 * 8);
  for (int t = 0; t < k; t++)
    for (int b = 0; b < 8; b++)
      for (int v = 1; v < 256; v++) {
        u64 s = itab[t][b][v];
        if (s) for (int j = 0; j < 8; j++) if (v >> j & 1) out[8 * b + j] ^= s;
      }
}
/* y[i] = (y[i] & mask) ^ x1[i] m1 ^ x2[i] m2 ^ x3[i] m3 (an x NULL: none), by tables */
typedef struct { u64 tb[3][8][256]; const u64 *x[3]; u64 *y; u64 mask; } Acc;
static void tables(const u64 *m, u64 tb[8][256]) {
  for (int b = 0; b < 8; b++) { tb[b][0] = 0; for (int v = 1; v < 256; v++) tb[b][v] = tb[b][v & (v - 1)] ^ m[8 * b + __builtin_ctz(v)]; }
}
static void fAcc(int t, uint32_t lo, uint32_t hi, void *a) {
  (void)t; Acc *p = a;
  for (uint32_t i = lo; i < hi; i++) {
    u64 s = p->y[i] & p->mask;
    for (int k = 0; k < 3; k++) if (p->x[k]) {
      u64 v = p->x[k][i];
      s ^= p->tb[k][0][v & 255] ^ p->tb[k][1][v >> 8 & 255] ^ p->tb[k][2][v >> 16 & 255] ^ p->tb[k][3][v >> 24 & 255] ^
           p->tb[k][4][v >> 32 & 255] ^ p->tb[k][5][v >> 40 & 255] ^ p->tb[k][6][v >> 48 & 255] ^ p->tb[k][7][v >> 56];
    }
    p->y[i] = s;
  }
}
static Acc *acc;
static void accum(u64 *y, u64 mask, const u64 *x1, const u64 *m1, const u64 *x2, const u64 *m2, const u64 *x3, const u64 *m3) {
  const u64 *xs[3] = { x1, x2, x3 }, *ms[3] = { m1, m2, m3 };
  for (int k = 0; k < 3; k++) { acc->x[k] = xs[k]; if (xs[k]) tables(ms[k], acc->tb[k]); }
  acc->y = y; acc->mask = mask;
  par(N, 8.0 * N * ((x1 != NULL) + (x2 != NULL) + (x3 != NULL)) + N, fAcc, acc);
}

/* Montgomery's choice of the columns S_i and W_i^-1 = (S_i^T V_i^T A V_i S_i)^-1, padded with 0, the last S's
   columns first (as msieve's find_nonsingular_sub).  Returns dim S_i (0: breakdown). */
static int nonsingular(const u64 *t, int *s, const int *last, int lastdim, u64 *w) {
  u64 Ma[64][2]; int dim = 0, cols = 64; u64 mask = 0;
  for (int i = 0; i < 64; i++) { Ma[i][0] = t[i]; Ma[i][1] = 1ULL << i; }
  for (int i = 0; i < lastdim; i++) { s[--cols] = last[i]; mask |= 1ULL << last[i]; }
  for (int i = 0; i < 64; i++) if (!(mask >> i & 1)) s[dim++] = i;
  dim = 0;
  for (int i = 0; i < 64; i++) {
    u64 bit = 1ULL << s[i], *ri_ = Ma[s[i]];
    int j;
    for (j = i; j < 64; j++) {
      u64 *rj = Ma[s[j]];
      if (rj[0] & bit) { u64 a0 = rj[0], a1 = rj[1]; rj[0] = ri_[0]; rj[1] = ri_[1]; ri_[0] = a0; ri_[1] = a1; break; }
    }
    if (j < 64) {
      for (j = 0; j < 64; j++) { u64 *rj = Ma[s[j]]; if (rj != ri_ && (rj[0] & bit)) { rj[0] ^= ri_[0]; rj[1] ^= ri_[1]; } }
      s[dim++] = s[i];
      continue;
    }
    for (j = i; j < 64; j++) {
      u64 *rj = Ma[s[j]];
      if (rj[1] & bit) { u64 a0 = rj[0], a1 = rj[1]; rj[0] = ri_[0]; rj[1] = ri_[1]; ri_[0] = a0; ri_[1] = a1; break; }
    }
    if (j == 64) return 0;
    for (j = 0; j < 64; j++) { u64 *rj = Ma[s[j]]; if (rj != ri_ && (rj[1] & bit)) { rj[0] ^= ri_[0]; rj[1] ^= ri_[1]; } }
    ri_[0] = ri_[1] = 0;
  }
  for (int i = 0; i < 64; i++) w[i] = Ma[i][1];
  return dim;
}

/* ---------- 128-bit echelon forms (lo: bits 0..63, hi: 64..127) */
typedef struct { u64 lo[128], hi[128]; char has[128]; int n; } Ech;
static int low128(u64 lo, u64 hi) { return lo ? __builtin_ctzll(lo) : 64 + __builtin_ctzll(hi); }
static void ech_add(Ech *e, u64 lo, u64 hi) {
  while (lo | hi) {
    int p = low128(lo, hi);
    if (!e->has[p]) { e->has[p] = 1; e->lo[p] = lo; e->hi[p] = hi; e->n++; return; }
    lo ^= e->lo[p]; hi ^= e->hi[p];
  }
}

/* the vectors: lists of row ids */
static uint32_t nvec, vcap; static uint32_t **vec, *vlen;
static void addvec(uint32_t *ids, uint32_t n) {
  if (nvec == vcap) { vcap = vcap ? 2 * vcap : 64; vec = chk(realloc(vec, vcap * sizeof *vec), vcap * sizeof *vec, "vectors");
                      vlen = chk(realloc(vlen, vcap * 4), vcap * 4, "vectors"); }
  vec[nvec] = chk(malloc((n ? n : 1) * 4), (size_t)n * 4, "vectors"); memcpy(vec[nvec], ids, (size_t)n * 4); vlen[nvec++] = n;
}

/* One run of block Lanczos from the seed.  Returns the number of vectors found (0: a breakdown); *it: iterations. */
static int lanczos(u64 seed, long *it) {
  u64 *x = alloc0(N, 8, "lanczos"), *rhs = alloc0(N, 8, "lanczos"), *vn = alloc0(N, 8, "lanczos"), *v[3];
  for (int k = 0; k < 3; k++) v[k] = alloc0(N, 8, "lanczos");
  rng = seed * 0x9E3779B97F4A7C15ULL + 1; for (int k = 0; k < 8; k++) rnd();
  for (uint32_t i = 0; i < N; i++) x[i] = rnd();
  mulA(x, v[0]);
  memcpy(rhs, v[0], (size_t)N * 8);
  u64 winv[3][64], vtav[2][64], vta2v[2][64], d[64], e[64], f[64], f2[64], mask0 = 0, mask1 = ~0ULL;
  memset(winv, 0, sizeof winv); memset(vtav, 0, sizeof vtav); memset(vta2v, 0, sizeof vta2v);
  int s0[64], s1[64], dim0 = 64, dim1 = 64, ok = 1;
  for (int i = 0; i < 64; i++) s1[i] = i;
  long maxit = (long)N / 32 + 100;
  for (*it = 0;; (*it)++) {
    if (*it > maxit) { ok = 0; break; }
    mulA(v[0], vn);
    inner(v[0], vn, vtav[0]);
    inner(vn, vn, vta2v[0]);
    int zero = 1; for (int i = 0; i < 64; i++) if (vtav[0][i]) zero = 0;
    if (zero) break;
    dim0 = nonsingular(vtav[0], s0, s1, dim1, winv[0]);
    if (!dim0) { ok = 0; break; }
    mask0 = 0; for (int i = 0; i < dim0; i++) mask0 |= 1ULL << s0[i];
    for (int i = 0; i < 64; i++) d[i] = (vta2v[0][i] & mask0) ^ vtav[0][i];
    mul64(winv[0], d, d);
    for (int i = 0; i < 64; i++) d[i] ^= 1ULL << i;
    mul64(winv[1], vtav[0], e);
    for (int i = 0; i < 64; i++) e[i] &= mask0;
    mul64(vtav[1], winv[1], f);
    for (int i = 0; i < 64; i++) f[i] ^= 1ULL << i;
    mul64(winv[2], f, f);
    for (int i = 0; i < 64; i++) f2[i] = ((vta2v[1][i] & mask1) ^ vtav[1][i]) & mask0;
    mul64(f, f2, f);
    accum(vn, mask0, v[0], d, v[1], e, v[2], f);          /* the next V */
    u64 g[64];
    inner(v[0], rhs, g);                                   /* x += V W^-1 V^T rhs */
    mul64(winv[0], g, g);
    accum(x, ~0ULL, v[0], g, NULL, NULL, NULL, NULL);
    u64 *tp = v[2]; v[2] = v[1]; v[1] = v[0]; v[0] = vn; vn = tp;
    memcpy(winv[2], winv[1], sizeof winv[1]); memcpy(winv[1], winv[0], sizeof winv[0]);
    memcpy(vtav[1], vtav[0], sizeof vtav[0]); memcpy(vta2v[1], vta2v[0], sizeof vta2v[0]);
    memcpy(s1, s0, sizeof s0); dim1 = dim0; mask1 = mask0;
    double tn = now();
    if (progress > 0 && tn - tlast >= progress) {
      tlast = tn;
      fprintf(stderr, "f2kernel: lanczos: iteration %ld (about %u in all), %.0f s\n", *it, N / 63 + 1, tn - t0);
    }
  }
  int found = 0;
  if (ok) {
    /* the combinations c of the columns of [x | V_m] with B [x | V_m] c = 0 */
    u64 *bx = alloc0(M, 8, "combine"), *bv = alloc0(M, 8, "combine");
    mulB(x, bx); mulB(v[0], bv);
    Ech *E = alloc0(1, sizeof *E, "combine");
    for (uint32_t r = 0; r < M && E->n < 128; r++) ech_add(E, bx[r], bv[r]);
    for (int p = 127; p >= 0; p--) if (E->has[p])
      for (int q = 0; q < p; q++) if (E->has[q] && ((p < 64 ? E->lo[q] >> p : E->hi[q] >> (p - 64)) & 1)) { E->lo[q] ^= E->lo[p]; E->hi[q] ^= E->hi[p]; }
    /* one kernel combination per free column; Cx, Cv: its parts on x and on V_m */
    u64 Cx[2][64], Cv[2][64]; int nk = 0;
    memset(Cx, 0, sizeof Cx); memset(Cv, 0, sizeof Cv);
    for (int fc = 0; fc < 128; fc++) if (!E->has[fc]) {
      int h = nk >> 6; u64 bit = 1ULL << (nk & 63);
      if (fc < 64) Cx[h][fc] |= bit; else Cv[h][fc - 64] |= bit;
      for (int q = 0; q < 128; q++) if (E->has[q] && ((fc < 64 ? E->lo[q] >> fc : E->hi[q] >> (fc - 64)) & 1)) {
        if (q < 64) Cx[h][q] |= bit; else Cv[h][q - 64] |= bit;
      }
      nk++;
    }
    /* Y = [x | V_m] C; its independent columns are the vectors */
    u64 *y[2] = { alloc0(N, 8, "combine"), alloc0(N, 8, "combine") };
    for (int h = 0; h < 2; h++) accum(y[h], 0, x, Cx[h], v[0], Cv[h], NULL, NULL);
    memset(E, 0, sizeof *E);
    for (uint32_t i = 0; i < N && E->n < nk; i++) ech_add(E, y[0][i], y[1][i]);
    u64 use[2] = { 0, 0 };
    for (int k = 0; k < 128; k++) if (E->has[k]) use[k >> 6] |= 1ULL << (k & 63);
    for (int h = 0; h < 2; h++) {
      if (!use[h]) continue;
      mulB(y[h], bx);
      for (uint32_t c = 0; c < M; c++) if (bx[c] & use[h]) { fprintf(stderr, "f2kernel: a vector does not vanish: a bug\n"); exit(2); }
    }
    uint32_t *ids = alloc0(N, 4, "combine");
    for (int k = 0; k < 128; k++) if (use[k >> 6] >> (k & 63) & 1) {
      uint32_t n = 0;
      for (uint32_t i = 0; i < N; i++) if (y[k >> 6][i] >> (k & 63) & 1) ids[n++] = rid[i];
      addvec(ids, n); found++;
    }
    free(ids); free(y[0]); free(y[1]); free(E); free(bx); free(bv);
  }
  free(x); free(rhs); free(vn); for (int k = 0; k < 3; k++) free(v[k]);
  return found;
}

/* the whole kernel of the N rows by elimination with the identity appended (few rows) */
static int dense(void) {
  size_t W = ((size_t)M + N + 63) / 64;
  u64 *A = alloc0((size_t)N * W, 8, "dense");
  int32_t *piv = chk(malloc(((size_t)M + 1) * 4), ((size_t)M + 1) * 4, "dense");
  for (uint32_t c = 0; c <= M; c++) piv[c] = -1;
  uint32_t *ids = alloc0(N, 4, "dense"); int found = 0;
  for (uint32_t i = 0; i < N; i++) {
    u64 *a = A + (size_t)i * W;
    for (size_t k = rp[i]; k < rp[i + 1]; k++) a[ri[k] / 64] ^= 1ULL << (ri[k] % 64);
    a[(M + i) / 64] ^= 1ULL << ((M + i) % 64);
    for (;;) {
      uint32_t c = M;
      for (size_t w = 0; w < (M + 63) / 64; w++) {
        u64 z = a[w] & (64 * w + 64 <= M ? ~0ULL : (1ULL << (M - 64 * w)) - 1);
        if (z) { c = (uint32_t)(64 * w) + (uint32_t)__builtin_ctzll(z); break; }
      }
      if (c == M) {                   /* no column left: the identity part is a vector */
        uint32_t n = 0;
        for (uint32_t j = 0; j <= i; j++) if (a[(M + j) / 64] >> ((M + j) % 64) & 1) ids[n++] = rid[j];
        addvec(ids, n); found++;
        break;
      }
      if (piv[c] < 0) { piv[c] = (int32_t)i; break; }
      const u64 *b = A + (size_t)piv[c] * W;
      for (size_t w = 0; w < W; w++) a[w] ^= b[w];
    }
  }
  free(A); free(piv); free(ids);
  return found;
}

static int cmpu(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  if (argc != 6) { fprintf(stderr, "usage: f2kernel POOL OUT THREADS SEED PROGRESS\n"); return 2; }
  T = atoi(argv[3]); if (T < 1) T = 1; if (T > 256) T = 256;
  if (getenv("F2KERNEL_PAR_WORK")) par_work = strtod(getenv("F2KERNEL_PAR_WORK"), NULL);
  u64 seed = strtoull(argv[4], NULL, 10);
  progress = strtod(argv[5], NULL);
  long dn = getenv("F2KERNEL_DENSE_N") ? atol(getenv("F2KERNEL_DENSE_N")) : 1000;
  t0 = tlast = now();
  FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 2; }
  uint32_t np;
  if (fread(&np, 4, 1, f) != 1) { fprintf(stderr, "bad pool file\n"); return 2; }
  /* the empty rows are vectors at once */
  rid = alloc0(np, 4, "rows"); rp = alloc0((size_t)np + 1, sizeof *rp, "rows");
  size_t cap = 1 << 20; ri = alloc0(cap, 4, "rows");
  uint32_t *buf = NULL; size_t bcap = 0;
  for (uint32_t t = 0; t < np; t++) {
    uint32_t hd[2];
    if (fread(hd, 4, 2, f) != 2) { fprintf(stderr, "short pool file\n"); return 2; }
    if (bcap < (size_t)hd[1] + 1) { bcap = 2 * ((size_t)hd[1] + 1); buf = chk(realloc(buf, bcap * 4), bcap * 4, "rows"); }
    if (hd[1] && fread(buf, 4, hd[1], f) != hd[1]) { fprintf(stderr, "short pool file\n"); return 2; }
    if (!hd[1]) { addvec(&hd[0], 1); continue; }
    if (NNZ + hd[1] > cap) { cap = 2 * (NNZ + hd[1]); ri = chk(realloc(ri, cap * 4), cap * 4, "rows"); }
    qsort(buf, hd[1], 4, cmpu);
    for (uint32_t k = 0; k < hd[1]; k++) {
      if (k && buf[k] == buf[k - 1]) { fprintf(stderr, "a repeated column in row %u\n", hd[0]); return 2; }
      ri[NNZ++] = buf[k];
    }
    rid[N] = hd[0]; rp[++N] = NNZ;
  }
  fclose(f); free(buf);
  uint32_t nempty = nvec;
  /* number the columns 0..M-1 in order of their ids, and build the transpose */
  {
    uint32_t *s = alloc0(NNZ, 4, "columns");
    memcpy(s, ri, NNZ * 4); qsort(s, NNZ, 4, cmpu);
    size_t m = 0; for (size_t k = 0; k < NNZ; k++) if (!m || s[m - 1] != s[k]) s[m++] = s[k];
    M = (uint32_t)m;
    for (size_t k = 0; k < NNZ; k++) {
      size_t lo = 0, hi = m; while (lo < hi) { size_t md = (lo + hi) / 2; if (s[md] < ri[k]) lo = md + 1; else hi = md; }
      ri[k] = (uint32_t)lo;
    }
    free(s);
    cp = alloc0((size_t)M + 1, sizeof *cp, "columns"); ci = alloc0(NNZ, 4, "columns");
    for (size_t k = 0; k < NNZ; k++) cp[ri[k] + 1]++;
    for (uint32_t c = 0; c < M; c++) cp[c + 1] += cp[c];
    size_t *pos = alloc0((size_t)M + 1, sizeof *pos, "columns"); memcpy(pos, cp, ((size_t)M + 1) * sizeof *pos);
    for (uint32_t i = 0; i < N; i++) for (size_t k = rp[i]; k < rp[i + 1]; k++) ci[pos[ri[k]]++] = i;
    free(pos);
  }
  const char *method = "dense", *status = "complete"; long it = 0; int tries = 0, found = 0;
  if (N && (long)N <= dn) found = dense();
  else if (N) {
    method = status = "lanczos";
    tmpM = alloc0(M, 8, "lanczos"); itab = alloc0((size_t)T, sizeof *itab, "lanczos"); acc = alloc0(1, sizeof *acc, "lanczos");
    for (tries = 1; tries <= 4 && !found; tries++) found = lanczos(seed + (u64)tries - 1, &it);
    tries--;
    if (!found && !nempty) { fprintf(stderr, "f2kernel: no vectors after %d tries\n", tries); return 5; }
    if (!found) { fprintf(stderr, "f2kernel: no vectors after %d tries; only the %u empty rows\n", tries, nempty); status = "partial"; }
  }
  char tmpfn[4096]; snprintf(tmpfn, sizeof tmpfn, "%s.tmp", argv[2]);
  FILE *g = fopen(tmpfn, "wb"); if (!g) { perror(tmpfn); return 2; }
  if (fwrite(&nvec, 4, 1, g) != 1) { perror(tmpfn); return 2; }
  for (uint32_t k = 0; k < nvec; k++)
    if (fwrite(&vlen[k], 4, 1, g) != 1 || fwrite(vec[k], 4, vlen[k], g) != vlen[k]) { perror(tmpfn); return 2; }
  if (fclose(g) || rename(tmpfn, argv[2])) { perror(argv[2]); return 2; }
  printf("kernel rows %u cols %u nnz %zu empty %u vectors %u method %s iterations %ld tries %d time %.1f status %s\n",
         N, M, NNZ, nempty, (uint32_t)found, method, it, tries, now() - t0, status);
  return 0;
}
