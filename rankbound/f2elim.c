/* f2elim: the bound dim Cl(K)/2 <= NS - dim V, with a record of the elimination that the checker replays.
   Phases: purge, trim (heaviest cliques of rows), merge (lightest large column first), dense (bit vectors).

   Columns 0..NS-1 are the small primes S, the others large (L); V is the subspace of rowspace(M) that vanishes on L.
   Held columns H (large) are never sparse pivots.  With characters, a row also carries K <= 64 bits C, and crank is
   the rank under C of the row combinations that vanish on L and S.  The elimination builds an echelon basis for the
   column order  sparse pivots (in the order taken) < other large < H < S < C,  each basis row's first column its own
   pivot; so dim V counts the basis rows whose first column is in S, and crank those in C.  As no H column is a sparse
   pivot, the H and S parts of the basis rows starting in H or S span V_H = {x in rowspace(M): x vanishes on L outside
   H}, which gives the 2-descent's k without another elimination.  Removing rows only makes V smaller, so the bound
   holds whatever the trim and the matching drop.

   Usage:  f2elim IN NS TARGET OUT DENSE WMAX GATE KEEP THREADS DENSE_MAX PROGRESS [WORDS [CHARS [HELD]]]
   IN: uint32 nrows, ncols, then per row its length and columns; with WORDS > 0, each row first has WORDS uint64 words
   of further columns (bit j of word w: column 64 w + j).  CHARS: lines "q r" (at most 64; q prime, r a root of F(x, 1)
   mod q) or "-"; then IN.ab holds each row's a - b alpha as int64 a, b (a free relation p: p, 0).  HELD: uint32 held
   columns (large, distinct) or "-".
   OUT.ops: uint32 pairs (dst, src), in order.  OUT.piv: the pivot rows.  OUT.vs: per basis row starting in H or S, in
   OUT.piv's order, ceil(NH/64) uint64 words of its H part, then ceil(NS/64) of its S part.  Outputs appear only
   when complete.
   TARGET < 0, or GATE = 1 and a negative excess: stop after the purge.  KEEP < 0: no trim.  DENSE_MAX: the bytes the
   dense phase may use (0: no limit), beyond which it runs in blocks.  PROGRESS: seconds between progress lines.
   stdout: "purge ...", "merge ...", "dense ...", "result rows R rank K dimV D bound B crank C pivots P ops O threaded T
   vs N time S" (crank -1: no characters, or one is 0 on a row); with blocks, "blocks words BW of W by HOW" and
   "blocks words BW of W temp MB M time S"; "nofit memory MB M" or "nofit disk" before an exit 3.
   Exit: 0 done; 3 the dense phase does not fit; 4 out of time; 2 anything else.
   Environment:
     F2ELIM_DEADLINE   Unix time to be done by: exit 4 (no outputs) once the dense estimate would end after it, or it
                       has passed; every phase polls it.
     F2ELIM_DENSE_S, F2ELIM_DENSE_MB   a dense phase estimated above this many seconds or MB is exit 3.
     F2ELIM_BLOCK_WORDS (tests; wins), F2ELIM_BLOCK_MB   the block size, in words or at most this many MiB.
     F2ELIM_TMPDIR, F2ELIM_TMP_MB   the blocks' temporary files: folder (default OUT's) and MiB (default 90% of the
                       free disk; beyond it, or a full disk, exit 3).  A bad block number (not 1..2^40) or an
                       unwritable folder is exit 2 before any work.
     F2ELIM_POOL       stop after the merge and write OUT.ops and OUT.pool for f2kernel: uint32 the number of rows, then
                       per row its id, its number of large columns and those ("pool rows R nnz Z ops O time S"). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <sys/statvfs.h>
#if defined(__GLIBC__)
#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif
/* Return the memory freed after the merge to the system, so that DENSE_MAX counts what the process holds. */
static void trim_heap(void) {
#if defined(__GLIBC__)
  malloc_trim(0);
#elif defined(__APPLE__)
  malloc_zone_pressure_relief(NULL, 0);
#endif
}

typedef struct { int n, cap; uint32_t *c; } Row;
static Row *R; static uint32_t nr; static size_t rcap;   /* columns sorted, without repeats */
static uint64_t *CHI;           /* per row: its character bits, or NULL */
static uint32_t ns, nc;
static size_t ccap;
enum { OPEN = 0, SPARSE = 1, DENSE = 2, HELD = 3 };
static uint8_t *cst;            /* per large column: OPEN, SPARSE (a sparse pivot's), DENSE or HELD (until dense) */
static uint32_t *held, nh;
static uint32_t *cval;          /* SPARSE: its pivot row; DENSE: its dense bit */
static uint32_t *cord;          /* SPARSE: its place in the column order */
static uint32_t nord;
static uint32_t *wt;            /* column weights over the pool */
static char *inpool, *lockr;
static uint32_t *pool; static size_t npool, pcap;
static uint32_t *tmp; static size_t tmpcap;
static FILE *fops;              /* OUT.ops, or NULL */
static long nops;
static uint32_t nsparse, nzero;

static int cmpu(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return (x > y) - (x < y); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static void *chk(void *p, size_t n, const char *what) {
  if (!p && n) { fprintf(stderr, "out of memory (%s: %.0f MB)\n", what, (double)n / 1e6); exit(2); }
  return p;
}
static void *grow(void *p, size_t n, size_t sz, const char *what) { return chk(realloc(p, (n ? n : 1) * sz), (n ? n : 1) * sz, what); }
static void wfail(const char *fn) { perror(fn); fprintf(stderr, "write to %s failed\n", fn); exit(2); }
/* snprintf into the array buf, exiting if the path does not fit */
#define PATHF(buf, ...) do { int n_ = snprintf(buf, sizeof buf, __VA_ARGS__); \
    if (n_ < 0 || (size_t)n_ >= sizeof buf) { fprintf(stderr, "path too long: %s...\n", buf); exit(2); } } while (0)
static double t0, progress, tlast;
static int due(void) { double t = now(); if (t - tlast < progress) return 0; tlast = t; return 1; }
static double deadline, dense_s_max, dense_mb_max;
static char opsfn[4096];
#define DENSE_SEC 3e-9          /* thread-seconds per row x word^2 of the dense phase (measured: 2.7e-9) */
static double wall(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
/* Exit 4 if s more seconds of work would end after the deadline (s = 0: if it has passed). */
static void in_time(double s) {
  if (deadline > 0 && wall() + s > deadline) {
    printf("late seconds %.0f left %.0f\n", s, deadline - wall()); fflush(stdout);
    if (*opsfn) remove(opsfn);
    exit(4);
  }
}
/* The deadline is polled every POLL_WORK units of work (a row, or one of its entries; 1 in the tests' build). */
#ifndef POLL_WORK
#define POLL_WORK (1UL << 18)
#endif
static unsigned long worked;
static void poll_time(unsigned long w) { worked += w; if (worked >= POLL_WORK) { worked = 0; in_time(0); } }

static void room_rows(size_t n) {
  if (n <= rcap) return;
  size_t old = rcap; rcap = n > 2 * rcap ? n : 2 * rcap;
  R = grow(R, rcap, sizeof *R, "rows"); memset(R + old, 0, (rcap - old) * sizeof *R);
  inpool = grow(inpool, rcap, 1, "rows"); memset(inpool + old, 0, rcap - old);
  lockr = grow(lockr, rcap, 1, "rows"); memset(lockr + old, 0, rcap - old);
  if (CHI) { CHI = grow(CHI, rcap, 8, "rows"); memset(CHI + old, 0, (rcap - old) * 8); }
}
static void room_cols(size_t m) {
  if (m <= ccap) return;
  size_t old = ccap; ccap = m > 2 * ccap ? m : 2 * ccap;
  cst = grow(cst, ccap, 1, "columns"); memset(cst + old, 0, ccap - old);
  cval = grow(cval, ccap, 4, "columns"); cord = grow(cord, ccap, 4, "columns");
  wt = grow(wt, ccap, 4, "columns"); memset(wt + old, 0, (ccap - old) * 4);
}
static void to_pool(uint32_t i) {
  if (npool == pcap) { pcap = pcap ? 2 * pcap : 1024; pool = grow(pool, pcap, 4, "pool"); }
  pool[npool++] = i; inpool[i] = 1;
}
/* Whether row i is 0 in the dense phase too. */
static int empty(uint32_t i) { return !R[i].n && !(CHI && CHI[i]); }

static void xorrow(uint32_t d, uint32_t s) {   /* logged to OUT.ops */
  Row *D = &R[d], *S = &R[s];
  size_t need = (size_t)D->n + S->n;
  if (tmpcap <= need) { tmpcap = 2 * need + 1; tmp = chk(realloc(tmp, tmpcap * sizeof *tmp), tmpcap * sizeof *tmp, "row buffer"); }
  int i = 0, j = 0, k = 0;
  while (i < D->n && j < S->n) {
    if (D->c[i] < S->c[j]) tmp[k++] = D->c[i++]; else if (D->c[i] > S->c[j]) tmp[k++] = S->c[j++]; else { i++; j++; }
  }
  while (i < D->n) tmp[k++] = D->c[i++];
  while (j < S->n) tmp[k++] = S->c[j++];
  if (D->cap < k) { D->cap = k; D->c = chk(realloc(D->c, (size_t)k * sizeof *D->c), (size_t)k * sizeof *D->c, "row"); }
  memcpy(D->c, tmp, (size_t)k * sizeof *tmp); D->n = k;
  if (CHI) CHI[d] ^= CHI[s];
  if (fops) { uint32_t pr[2] = { d, s }; if (fwrite(pr, sizeof pr, 1, fops) != 1) wfail(opsfn); }
  nops++;
}

/* Store row i from buf (any order; repeats cancel). */
static void setrow(uint32_t i, uint32_t *buf, uint32_t n) {
  for (uint32_t k = 0; k < n; k++) if (buf[k] >= nc) { room_cols((size_t)buf[k] + 1); nc = buf[k] + 1; }
  qsort(buf, n, 4, cmpu);
  int m = 0; for (uint32_t k = 0; k < n; k++) { if (m && buf[m - 1] == buf[k]) { m--; continue; } buf[m++] = buf[k]; }
  R[i].n = R[i].cap = m; R[i].c = chk(malloc((size_t)(m ? m : 1) * 4), (size_t)(m ? m : 1) * 4, "row");
  memcpy(R[i].c, buf, (size_t)m * 4);
}

static void readall(const char *fn, uint32_t words) {
  FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(2); }
  uint32_t h[2]; if (fread(h, 4, 2, f) != 2) { fprintf(stderr, "bad header\n"); exit(2); }
  if (h[1] > INT32_MAX) { fprintf(stderr, "bad header: %u columns (at most 2^31 - 1)\n", h[1]); exit(2); }
  room_rows(h[0]); room_cols((size_t)h[1] + 1); nc = h[1];
  uint32_t *buf = NULL; size_t bcap = 0;
  uint64_t *wbuf = chk(malloc((size_t)(words ? words : 1) * 8), (size_t)(words ? words : 1) * 8, "input words");
  for (uint32_t i = 0; i < h[0]; i++) {
    if (!(i & 0xffff)) { in_time(0); if (due()) fprintf(stderr, "f2elim: read: row %u of %u, %.0f s\n", i, h[0], now() - t0); }
    uint32_t nb = 0;
    if (words) {
      if (fread(wbuf, 8, words, f) != words) { fprintf(stderr, "short words at row %u\n", i); exit(2); }
      for (uint32_t w = 0; w < words; w++) nb += (uint32_t)__builtin_popcountll(wbuf[w]);
    }
    uint32_t n; if (fread(&n, 4, 1, f) != 1) { fprintf(stderr, "short file at row %u\n", i); exit(2); }
    if (n > INT32_MAX - nb) { fprintf(stderr, "row %u: %u entries (at most 2^31 - 1)\n", i, n); exit(2); }
    if (bcap < (size_t)n + nb + 1) { bcap = 2 * ((size_t)n + nb + 1); buf = chk(realloc(buf, bcap * 4), bcap * 4, "input row"); }
    if (n && fread(buf, 4, n, f) != n) { fprintf(stderr, "short row %u\n", i); exit(2); }
    for (uint32_t w = 0; w < words; w++) for (uint64_t x = wbuf[w]; x; x &= x - 1) buf[n++] = 64 * w + (uint32_t)__builtin_ctzll(x);
    for (uint32_t k = 0; k < n; k++) if (buf[k] >= h[1]) { fprintf(stderr, "column %u out of range in row %u\n", buf[k], i); exit(2); }
    setrow(i, buf, n);
  }
  nr = h[0];
  fclose(f); free(buf); free(wbuf);
}

/* ---------- the pool: purge, trim, merge (on OPEN large columns only) */

#define SCARCE 4                /* the trim keeps every row of an S column with at most this many rows */
static int scarce(uint32_t i) { for (int k = 0; k < R[i].n && R[i].c[k] < ns; k++) if (wt[R[i].c[k]] <= SCARCE) return 1; return 0; }
static int isopen(uint32_t c) { return c >= ns && cst[c] == OPEN; }
/* the trim's L columns: a held column is never a pivot, but joins rows like any L column */
static int istrim(uint32_t c) { return c >= ns && (cst[c] == OPEN || cst[c] == HELD); }
static long nnzl;      /* nonzeros in OPEN columns, from the last weights() */
static void compact(void) { size_t m = 0; for (size_t t = 0; t < npool; t++) if (inpool[pool[t]]) pool[m++] = pool[t]; npool = m; }
/* Column weights over the pool (only OPEN columns unless all). */
static void weights_of(long *nnz, uint32_t *live, uint32_t *livel, int all) {
  compact();
  memset(wt, 0, (size_t)nc * 4); *nnz = 0; *live = 0; *livel = 0; nnzl = 0;
  for (size_t t = 0; t < npool; t++) {
    uint32_t i = pool[t];
    poll_time(1 + (unsigned long)R[i].n);
    (*live)++; *nnz += R[i].n;
    if (all) for (int k = 0; k < R[i].n; k++) wt[R[i].c[k]]++;
    else for (int k = R[i].n - 1; k >= 0 && R[i].c[k] >= ns; k--) if (isopen(R[i].c[k])) wt[R[i].c[k]]++;   /* L last */
  }
  for (uint32_t c = ns; c < nc; c++) {
    if (cst[c] == OPEN && wt[c]) { (*livel)++; nnzl += wt[c]; }
    else if (cst[c] == HELD && wt[c]) (*livel)++;
  }
}
static void weights(long *nnz, uint32_t *live, uint32_t *livel) { weights_of(nnz, live, livel, 1); }
static void take(uint32_t i, uint32_t c) { inpool[i] = 0; cst[c] = SPARSE; cval[c] = i; cord[c] = nord++; nsparse++; }
/* Make each pool row with an OPEN column of weight 1 that column's pivot, until none is left; return how many.
   The rows left (the largest subset with no OPEN column of weight 1) do not depend on the order.  own[c] is the XOR
   of the ids of c's pool rows, so its row when its weight is 1; weights only fall, so the work is linear. */
static long purge(void) {
  long removed = 0;
  uint32_t *own = chk(calloc(nc ? nc : 1, 4), (size_t)(nc ? nc : 1) * 4, "purge");
  uint32_t *q = chk(malloc((size_t)(nc ? nc : 1) * 4), (size_t)(nc ? nc : 1) * 4, "purge"), nq = 0;
  for (size_t t = 0; t < npool; t++) {
    uint32_t i = pool[t]; poll_time(1); if (!inpool[i]) continue;
    poll_time((unsigned long)R[i].n);
    for (int k = R[i].n - 1; k >= 0 && R[i].c[k] >= ns; k--) own[R[i].c[k]] ^= i;
  }
  for (uint32_t c = ns; c < nc; c++) if (isopen(c) && wt[c] == 1) q[nq++] = c;
  poll_time(nc);
  while (nq) {
    uint32_t c = q[--nq]; poll_time(1);
    if (!isopen(c) || wt[c] != 1) continue;
    uint32_t i = own[c];
    poll_time((unsigned long)R[i].n);
    long single = -1;
    for (int k = R[i].n - 1; k >= 0 && R[i].c[k] >= ns; k--) if (isopen(R[i].c[k]) && wt[R[i].c[k]] == 1) { single = R[i].c[k]; break; }
    take(i, (uint32_t)single); removed++;
    for (int k = 0; k < R[i].n; k++) {
      uint32_t d = R[i].c[k];
      if (!--wt[d] || d < ns) continue;
      own[d] ^= i;
      if (wt[d] == 1 && isopen(d)) q[nq++] = d;
    }
  }
  free(own); free(q);
  return removed;
}

/* The trim, at most 20 rounds (stopping early only leaves the excess above KEEP); returns the excess. */
static uint32_t *uf;
static uint32_t root(uint32_t i) { while (uf[i] != i) i = uf[i] = uf[uf[i]]; return i; }
static uint32_t *score;
static int by_weight(const void *a, const void *b) {     /* lightest first, then by id */
  uint32_t i = *(const uint32_t *)a, j = *(const uint32_t *)b;
  return score[i] != score[j] ? (score[i] > score[j]) - (score[i] < score[j]) : (i > j) - (i < j);
}
static int by_score(const void *a, const void *b) {      /* heaviest first, then by row */
  uint32_t i = *(const uint32_t *)a, j = *(const uint32_t *)b;
  return score[i] != score[j] ? (score[i] < score[j]) - (score[i] > score[j]) : (i > j) - (i < j);
}
static long trim(long excess, long keep, long need) {
  long nnz; uint32_t live, livel;
  uf = chk(malloc((size_t)nr * 4), (size_t)nr * 4, "trim"); score = chk(calloc(nr, 4), (size_t)nr * 4, "trim");
  uint32_t *first = chk(malloc((size_t)nc * 4), (size_t)nc * 4, "trim"), *comp = chk(malloc((size_t)nr * 4), (size_t)nr * 4, "trim");
  for (int round = 0; keep >= 0 && excess > keep && round < 20; round++) {
    weights(&nnz, &live, &livel);
    in_time(0);
    for (size_t t = 0; t < npool; t++) uf[pool[t]] = pool[t], score[pool[t]] = 0;
    memset(first, 0xff, (size_t)nc * 4);
    for (size_t t = 0; t < npool; t++) {
      uint32_t i = pool[t];
      poll_time(1 + (unsigned long)R[i].n);
      for (int k = R[i].n - 1; k >= 0 && R[i].c[k] >= ns; k--) {
        uint32_t c = R[i].c[k];
        if (!istrim(c) || wt[c] != 2) continue;
        if (first[c] == UINT32_MAX) first[c] = i; else uf[root(i)] = root(first[c]);
      }
    }
    uint32_t nk = 0;
    for (size_t t = 0; t < npool; t++) {    /* score: rows plus L entries */
      uint32_t i = pool[t], r = root(i), l = 0;
      poll_time(1 + (unsigned long)R[i].n);
      for (int k = R[i].n - 1; k >= 0 && R[i].c[k] >= ns; k--) l += istrim(R[i].c[k]);
      if (!score[r]) comp[nk++] = r;
      score[r] += 1 + l;
    }
    for (size_t t = 0; t < npool; t++) {    /* keep cliques with a scarce row or a row without L columns */
      uint32_t i = pool[t]; int l = 0;
      poll_time(1 + (unsigned long)R[i].n);
      for (int k = R[i].n - 1; k >= 0 && R[i].c[k] >= ns && !l; k--) l = istrim(R[i].c[k]);
      if (!l || scarce(i)) score[root(i)] = 0;
    }
    in_time(0);
    qsort(comp, nk, 4, by_score);
    in_time(0);
    long todo = excess - keep;
    for (uint32_t k = 0; k < nk && todo > 0 && score[comp[k]]; k++) { score[comp[k]] = UINT32_MAX; todo--; }
    for (size_t t = 0; t < npool; t++) if (score[root(pool[t])] == UINT32_MAX) inpool[pool[t]] = 0;
    weights(&nnz, &live, &livel);
    purge();
    weights(&nnz, &live, &livel);
    excess = (long)live - livel - need;
    printf("trim rows %u lcols %u nnz %ld excess %ld\n", live, livel, nnz, excess);
    fflush(stdout);
  }
  free(uf); free(score); free(first); free(comp);
  return excess;
}

/* The merge; the weight buckets and column row lists may hold stale entries, which are skipped. */
typedef struct { uint32_t n, cap, *v; } Vec;
static void push(Vec *x, uint32_t e) {
  if (x->n == x->cap) { x->cap = x->cap ? 2 * x->cap : 4; x->v = grow(x->v, x->cap, 4, "merge"); }
  x->v[x->n++] = e;
}
static int has(uint32_t i, uint32_t c) {
  int lo = 0, hi = R[i].n;
  while (lo < hi) { int m = (lo + hi) / 2; if (R[i].c[m] < c) lo = m + 1; else hi = m; }
  return lo < R[i].n && R[i].c[lo] == c;
}
static long merge(uint32_t dense, int wmax) {
  long nnz, merges = 0; uint32_t live, open;
  weights_of(&nnz, &live, &open, 0);
  uint32_t open0 = open;
  Vec *rows = chk(calloc((size_t)nc, sizeof *rows), (size_t)nc * sizeof *rows, "merge"), *bk = chk(calloc((size_t)wmax + 1, sizeof *bk), ((size_t)wmax + 1) * sizeof *bk, "merge");
  for (size_t t = 0; t < npool; t++)
    for (int k = R[pool[t]].n - 1; k >= 0 && R[pool[t]].c[k] >= ns; k--) if (isopen(R[pool[t]].c[k])) push(&rows[R[pool[t]].c[k]], pool[t]);
  for (uint32_t c = ns; c < nc; c++) if (isopen(c) && wt[c] && wt[c] <= (uint32_t)wmax) push(&bk[wt[c]], c);
  uint32_t *L = NULL; size_t lcap = 0;
  /* capped at 600M nonzeros: 2.4 GB for the rows, as much again for the column lists */
  for (int w = 1; w <= wmax && open > dense && nnz <= 600000000L;) {
    if (!bk[w].n) { w++; continue; }
    uint32_t c = bk[w].v[--bk[w].n];
    if (!isopen(c) || wt[c] != (uint32_t)w) continue;
    if (lcap < rows[c].n) { lcap = rows[c].n; L = grow(L, lcap, 4, "merge"); }
    uint32_t m = 0; long piv = -1;
    for (uint32_t t = 0; t < rows[c].n; t++) {
      uint32_t i = rows[c].v[t];
      if (inpool[i] && !lockr[i] && has(i, c)) { lockr[i] = 1; L[m++] = i; if (piv < 0 || R[i].n < R[piv].n) piv = i; }
    }
    rows[c].n = 0;
    for (uint32_t t = 0; t < m; t++) {
      uint32_t i = L[t];
      lockr[i] = 0;
      if (i == (uint32_t)piv) continue;
      nnz -= R[i].n;
      for (int k = R[piv].n - 1; k >= 0 && R[piv].c[k] >= ns; k--) {
        uint32_t d = R[piv].c[k];
        if (!isopen(d)) continue;
        if (has(i, d)) { if (!--wt[d]) open--; } else { if (!wt[d]++) open++; push(&rows[d], i); }
        if (wt[d] && wt[d] <= (uint32_t)wmax) push(&bk[wt[d]], d);
      }
      xorrow(i, (uint32_t)piv);
      nnz += R[i].n;
      if (empty(i)) { inpool[i] = 0; nzero++; }
    }
    nnz -= R[piv].n;
    for (int k = R[piv].n - 1; k >= 0 && R[piv].c[k] >= ns; k--) {
      uint32_t d = R[piv].c[k];
      if (!isopen(d)) continue;
      if (!--wt[d]) open--;
      else if (wt[d] <= (uint32_t)wmax) push(&bk[wt[d]], d);
    }
    take((uint32_t)piv, c); merges++;
    w = 1;
    if (!(merges & 0xfff)) in_time(0);
    if (due())
      fprintf(stderr, "f2elim: merge: %u of %u large columns eliminated (stops at %u), nnz %ld, ops %ld, %.0f s\n",
              open0 - open, open0, dense, nnz, nops, now() - t0);
  }
  for (uint32_t c = 0; c < nc; c++) free(rows[c].v);
  for (int w = 0; w <= wmax; w++) free(bk[w].v);
  free(rows); free(bk); free(L);
  return merges;
}

/* ---------- the matching selection (a target, no characters).  Order the pool: the rows of a
   structural matching (augmenting paths capped at 32 x nnz steps), then MATCH_EXTRA others, then the rest; return
   the number of first rows.  Dropping the rest is a trim, and continuing from their basis gives the whole pool's
   bound, as dim V does not depend on the row order. */
enum { MATCH_EXTRA = 64 };   /* 64 gave the smallest witnesses */
static size_t match_rows(long extra) {
  compact();
  long nnz = 0;
  uint32_t *cw = chk(calloc(nc ? nc : 1, 4), (size_t)(nc ? nc : 1) * 4, "match");
  for (size_t t = 0; t < npool; t++) { uint32_t i = pool[t]; nnz += R[i].n; for (int k = 0; k < R[i].n; k++) cw[R[i].c[k]]++; }
  size_t *cs = chk(malloc(((size_t)nc + 1) * sizeof *cs), ((size_t)nc + 1) * sizeof *cs, "match");
  cs[0] = 0; for (uint32_t c = 0; c < nc; c++) cs[c + 1] = cs[c] + cw[c];
  uint32_t *cr = chk(malloc((size_t)(nnz ? nnz : 1) * 4), (size_t)(nnz ? nnz : 1) * 4, "match");
  size_t *fill = chk(malloc(((size_t)nc + 1) * sizeof *fill), ((size_t)nc + 1) * sizeof *fill, "match");
  memcpy(fill, cs, ((size_t)nc + 1) * sizeof *fill);
  for (size_t t = 0; t < npool; t++) { uint32_t i = pool[t]; poll_time(1 + (unsigned long)R[i].n); for (int k = 0; k < R[i].n; k++) cr[fill[R[i].c[k]]++] = i; }
  free(fill);
  uint32_t ncol = 0, *ord = chk(malloc((size_t)(nc ? nc : 1) * 4), (size_t)(nc ? nc : 1) * 4, "match");
  for (uint32_t c = 0; c < nc; c++) if (cw[c]) ord[ncol++] = c;
  score = cw;
  qsort(ord, ncol, 4, by_weight);
  uint32_t *rowof = chk(malloc((size_t)(nc ? nc : 1) * 4), (size_t)(nc ? nc : 1) * 4, "match");   /* column -> row */
  uint32_t *colof = chk(malloc((size_t)(nr ? nr : 1) * 4), (size_t)(nr ? nr : 1) * 4, "match");   /* row -> column */
  uint32_t *seen = chk(calloc(nr ? nr : 1, 4), (size_t)(nr ? nr : 1) * 4, "match");
  for (uint32_t c = 0; c < nc; c++) rowof[c] = UINT32_MAX;
  for (size_t t = 0; t < npool; t++) colof[pool[t]] = UINT32_MAX;
  uint32_t matched = 0;
  for (uint32_t k = 0; k < ncol; k++) {      /* greedy: the shortest free row, rarest columns first */
    uint32_t c = ord[k], best = UINT32_MAX;
    poll_time(1 + cw[c]);
    for (size_t e = cs[c]; e < cs[c + 1]; e++) { uint32_t i = cr[e]; if (colof[i] == UINT32_MAX && (best == UINT32_MAX || R[i].n < R[best].n)) best = i; }
    if (best != UINT32_MAX) { rowof[c] = best; colof[best] = c; matched++; }
  }
  uint32_t *stc = chk(malloc((size_t)(ncol ? ncol : 1) * 4), (size_t)(ncol ? ncol : 1) * 4, "match");
  size_t *ste = chk(malloc((size_t)(ncol ? ncol : 1) * sizeof *ste), (size_t)(ncol ? ncol : 1) * sizeof *ste, "match");
  double budget = 32.0 * (double)nnz + 1e6, steps = 0;
  for (uint32_t k = 0, stamp = 1; k < ncol && steps < budget; k++, stamp++) {
    uint32_t c0 = ord[k]; if (rowof[c0] != UINT32_MAX) continue;
    int d = 0; stc[0] = c0; ste[0] = cs[c0]; int found = 0;
    while (d >= 0 && !found && steps < budget) {
      uint32_t c = stc[d];
      if (ste[d] == cs[c + 1]) { d--; continue; }
      uint32_t i = cr[ste[d]++]; steps++; poll_time(1);
      if (seen[i] == stamp) continue;
      seen[i] = stamp;
      if (colof[i] == UINT32_MAX) {           /* a free row: flip the augmenting path */
        uint32_t ri = i;
        for (int j = d; j >= 0; j--) { uint32_t cj = stc[j], prev = rowof[cj]; rowof[cj] = ri; colof[ri] = cj; ri = prev; }
        matched++; found = 1;
      } else { d++; stc[d] = colof[i]; ste[d] = cs[stc[d]]; }
    }
  }
  uint32_t *rest = chk(malloc((npool ? npool : 1) * 4), (npool ? npool : 1) * 4, "match");
  size_t kept = 0, nrest = 0; long more = 0;
  for (size_t t = 0; t < npool; t++) {
    uint32_t i = pool[t];
    if (colof[i] != UINT32_MAX || more++ < extra) pool[kept++] = i;
    else rest[nrest++] = i;
  }
  if (nrest) memcpy(pool + kept, rest, nrest * 4);      /* pool may be NULL when empty */
  printf("match rows %zu cols %u matched %u kept %zu\n", npool, ncol, matched, kept);
  fflush(stdout);
  free(cw); free(cs); free(cr); free(ord); free(rowof); free(colof); free(seen); free(stc); free(ste); free(rest);
  score = NULL;
  return kept;
}

/* ---------- the dense phase: bit vectors over [dense L | S | C], LW + SW + CW words, whole or in blocks of BW words
   With blocks, only words B0..B1-1 of every row are in memory.  The pool's rows go to a temporary file in dense
   bits; each block is built from it, the records of all earlier panels (word, pivots with their combinations pm,
   updated rows with theirs) are replayed on it in order, and its own words are eliminated, appending their records.
   Same result as whole, by induction on the panels in run order (pass 0's words, then pass 1's): a panel's update of
   word t reads and writes word t alone, as a function of that word and the record (panel_part), so every word read
   gets the same earlier updates in the same order either way.  Examining word w reads only word w of the rows left
   and the seeds; a seed's own word comes from lead_x, its value when it joined, which no later panel changes.  So
   the pivots, lead, counts and records agree panel by panel, hence OUT.piv and OUT.vs (H and S parts are taken as
   each block finishes, including blocks after the last row left).  Records that change nothing are not written. */

static size_t LW, SW, CW, DW;   /* words of the dense large columns, S, C; DW their sum */
static uint32_t nld;            /* dense large columns (bits 0..nld-1) */
static uint32_t nb;
static uint32_t dimV, crank;
static size_t BW, B0, B1;       /* word t of row r at A[(t - B0) * nd + r] */
static int blocks;

/* The panel update on words lo..hi-1: the new pivots k0..k-1 first get their combinations pm of the earlier ones,
   then each row of up gets its combination m from 8 tables (T[g][v]: the sum of pivots 8g + j over the bits j of v). */
typedef struct { uint32_t r; uint64_t m; } Upd;
typedef struct { uint64_t *A; size_t nd, b0; const uint32_t *pr; const uint64_t *pm; int k, k0; const Upd *up; uint32_t nu; size_t lo, hi; } Part;
static void *panel_part(void *arg) {
  const Part *q = arg; uint64_t T[8][256], P[64];
  for (size_t t = q->lo; t < q->hi; t++) {
    uint64_t *a = q->A + (t - q->b0) * q->nd;
    for (int s = 0; s < q->k; s++) {
      P[s] = a[q->pr[s]];
      if (s >= q->k0) { for (uint64_t m = q->pm[s]; m; m &= m - 1) P[s] ^= P[__builtin_ctzll(m)]; a[q->pr[s]] = P[s]; }
    }
    for (int g = 0; g < 8; g++) {
      T[g][0] = 0;
      for (int v = 1; 8 * g < q->k && v < 1 << (q->k - 8 * g < 8 ? q->k - 8 * g : 8); v++) T[g][v] = T[g][v & (v - 1)] ^ P[8 * g + __builtin_ctz(v)];
    }
    for (uint32_t u = 0; u < q->nu; u++) {
      uint64_t m = q->up[u].m;
      a[q->up[u].r] ^= T[0][m & 255] ^ T[1][m >> 8 & 255] ^ T[2][m >> 16 & 255] ^ T[3][m >> 24 & 255] ^
                       T[4][m >> 32 & 255] ^ T[5][m >> 40 & 255] ^ T[6][m >> 48 & 255] ^ T[7][m >> 56];
    }
  }
  return NULL;
}
/* The update split by words over threads; return 1 if threaded. */
static int run_panel(Part P, int threads) {
  size_t words = P.hi - P.lo;
  if (threads <= 1 || (double)P.nu * words < 1e6 || words < 2) { panel_part(&P); return 0; }
  pthread_t th[256]; Part q[256]; int T = threads > 256 ? 256 : threads;
  if ((size_t)T > words) T = (int)words;
  for (int t = 0; t < T; t++) { q[t] = P; q[t].lo = P.lo + words * t / T; q[t].hi = P.lo + words * (t + 1) / T; }
  int up[256];          /* a part whose thread did not start runs here */
  for (int t = 1; t < T; t++) up[t] = pthread_create(&th[t], NULL, panel_part, &q[t]) == 0;
  panel_part(&q[0]);
  for (int t = 1; t < T; t++) { if (up[t]) pthread_join(th[t], NULL); else panel_part(&q[t]); }
  return 1;
}

/* ---------- the blocks' temporary files, the rows (fsp) and the records (frec); unlinked when created, so the
   system removes them on every exit. */
static FILE *fsp, *frec;
static uint64_t tmpbytes, tmpmax = UINT64_MAX;
static void nofit(const char *what, const char *msg) {
  fprintf(stderr, "dense phase: does not fit: %s\n", msg);
  printf("nofit %s\n", what); fflush(stdout);
  if (*opsfn) remove(opsfn);
  exit(3);
}
static void tfail(void) {      /* a full disk does not fit (exit 3); any other write error is exit 2 */
  if (errno == ENOSPC || errno == EFBIG
#ifdef EDQUOT
      || errno == EDQUOT
#endif
     ) nofit("disk", "a temporary file of the blocks: no space left");
  perror("f2elim: a temporary file of the blocks"); exit(2);
}
static void twrite(FILE *f, const void *p, size_t sz, size_t n) {
  if (tmpbytes + (uint64_t)sz * n > tmpmax) nofit("disk", "the temporary files of the blocks are over F2ELIM_TMP_MB or the free disk");
  if (n && fwrite(p, sz, n, f) != n) tfail();
  tmpbytes += (uint64_t)sz * n;
}
static void tread(FILE *f, void *p, size_t sz, size_t n) {
  if (n && fread(p, sz, n, f) != n) { fprintf(stderr, "f2elim: cannot read a temporary file of the blocks\n"); exit(2); }
}
static void trewind(FILE *f, uint64_t at) {
  if (fflush(f)) tfail();
  if (fseeko(f, (off_t)at, SEEK_SET)) { perror("f2elim: a temporary file of the blocks"); exit(2); }
}
/* 0 if not set, else a decimal 1..2^40 (anything else: exit 2). */
static uint64_t envnum(const char *name) {
  const char *s = getenv(name); uint64_t v = 0;
  if (!s) return 0;
  for (const char *c = s; *c; c++) {
    if (*c < '0' || *c > '9' || v > ((1ULL << 40) - (uint64_t)(*c - '0')) / 10) { fprintf(stderr, "%s is not a number 1..2^40\n", name); exit(2); }
    v = 10 * v + (uint64_t)(*c - '0');
  }
  if (!v) { fprintf(stderr, "%s is not a number 1..2^40\n", name); exit(2); }
  return v;
}
static uint64_t env_bw, env_bmb, env_tmb;      /* F2ELIM_BLOCK_WORDS, _BLOCK_MB, _TMP_MB */
static FILE *tmpopen(const char *dir) {
  size_t n = strlen(dir) + 32; char *path = chk(malloc(n), n, "temporary file");
  snprintf(path, n, "%s/f2elim-XXXXXX", dir);
  int fd = mkstemp(path);
  if (fd < 0) { perror(path); exit(2); }
  if (unlink(path)) { perror(path); exit(2); }
  free(path);
  FILE *f = fdopen(fd, "w+b");
  if (!f) { perror("f2elim: a temporary file of the blocks"); exit(2); }
  setvbuf(f, NULL, _IOFBF, 1 << 20);
  return f;
}

/* Build the block from fsp (per row: its number of bits, then its dense bits). */
static void build(uint64_t *A, uint32_t nd, const uint32_t *id) {
  memset(A, 0, (B1 - B0) * nd * 8);
  trewind(fsp, 0);
  uint32_t buf[1024];
  for (uint32_t t = 0; t < nd; t++) {
    if (!(t & 0xffff)) in_time(0);
    uint32_t n; tread(fsp, &n, 4, 1);
    while (n) {
      uint32_t m = n < 1024 ? n : 1024; n -= m;
      tread(fsp, buf, 4, m);
      for (uint32_t k = 0; k < m; k++) {
        size_t w = buf[k] / 64;
        if (w >= B0 && w < B1) A[(w - B0) * nd + t] ^= 1ULL << (buf[k] % 64);
      }
    }
    if (CHI && LW + SW >= B0 && LW + SW < B1) A[(LW + SW - B0) * nd + t] = CHI[id[t]];
  }
}
/* Replay the records so far on the block, in order, each on the words after its own. */
static void replay(uint64_t *A, uint32_t nd, Upd *up, int threads, uint32_t *threaded, uint64_t end) {
  trewind(frec, 0);
  uint64_t got = 0;
  uint32_t ub[2048 * 3];
  while (got < end) {
    in_time(0);
    uint32_t hd[4], pr[64]; uint64_t pm[64];
    tread(frec, hd, 4, 4);
    uint32_t w = hd[0], k = hd[1], k0 = hd[2], nu = hd[3];
    if (w >= DW || k > 64 || k0 > k || nu > nd) { fprintf(stderr, "f2elim: a bad record in a temporary file\n"); exit(2); }
    got += 16 + 12ULL * k + 12ULL * nu;
    if (w + 1 >= B1) {
      if (fseeko(frec, (off_t)(12ULL * k + 12ULL * nu), SEEK_CUR)) { perror("f2elim: a temporary file of the blocks"); exit(2); }
      continue;
    }
    tread(frec, pr, 4, k); tread(frec, pm, 8, k);
    for (uint32_t s = 0; s < k; s++) if (pr[s] >= nd || pm[s] >> s) { fprintf(stderr, "f2elim: a bad record in a temporary file\n"); exit(2); }
    for (uint32_t u = 0; u < nu;) {
      uint32_t m = nu - u < 2048 ? nu - u : 2048;
      tread(frec, ub, 4, 3 * (size_t)m);
      for (uint32_t j = 0; j < m; j++, u++) {
        up[u].r = ub[3 * j]; memcpy(&up[u].m, ub + 3 * j + 1, 8);
        if (up[u].r >= nd) { fprintf(stderr, "f2elim: a bad record in a temporary file\n"); exit(2); }
      }
    }
    Part P = { A, nd, B0, pr, pm, (int)k, (int)k0, up, nu, w + 1 > B0 ? w + 1 : B0, B1 };
    *threaded += run_panel(P, threads);
  }
  if (fseeko(frec, 0, SEEK_END)) { perror("f2elim: a temporary file of the blocks"); exit(2); }
}
static void record(uint32_t w, int k, int k0, const uint32_t *pr, const uint64_t *pm, const Upd *up, uint32_t nu) {
  int any = nu > 0;
  for (int s = k0; s < k && !any; s++) any = pm[s] != 0;
  if (!any) return;
  uint32_t hd[4] = { w, (uint32_t)k, (uint32_t)k0, nu }, ub[2048 * 3];
  twrite(frec, hd, 4, 4); twrite(frec, pr, 4, (size_t)k); twrite(frec, pm, 8, (size_t)k);
  for (uint32_t u = 0; u < nu;) {
    uint32_t m = nu - u < 2048 ? nu - u : 2048;
    for (uint32_t j = 0; j < m; j++, u++) { ub[3 * j] = up[u].r; memcpy(ub + 3 * j + 1, &up[u].m, 8); }
    twrite(frec, ub, 4, 3 * (size_t)m);
  }
}

/* A pass over the rows rem, in order, against the basis rows seed of an earlier pass (sorted by lead), which enter
   each word's panel first; so a pass over R2 after one over R1 gives the basis of one pass over R1 then R2. */
typedef struct { uint32_t *rem, nrem; const uint32_t *seed; uint32_t nseed, sk, joined; } Pass;
static uint64_t *lead_x;        /* with blocks: per row, its word at its first column when it joined */
/* Words B0..B1-1 of the pass ps (rec: append the records to frec). */
static void dense_words(uint64_t *A, uint32_t nd, Pass *ps, uint32_t *lead, Upd *up, int threads, uint32_t *threaded, int rec) {
  for (size_t w = B0; w < B1 && ps->nrem; w++) {
    if (!(w & 15)) in_time(0);
    /* the panel: the seeds of word w, then its new pivots in row order; pv[s] is word w of pivot s */
    uint64_t pv[64], pm[64]; uint32_t pr[64]; int bit[64], k = 0, k0; uint32_t nu = 0, nk = 0;
    for (int b = 0; b < 64; b++) bit[b] = -1;
    for (; ps->sk < ps->nseed && lead[ps->seed[ps->sk]] / 64 <= w; ps->sk++) {
      uint32_t t = ps->seed[ps->sk];
      pv[k] = blocks ? lead_x[t] : A[(w - B0) * nd + t]; pm[k] = 0; pr[k] = t; bit[lead[t] % 64] = k++;
    }
    k0 = k;
    for (uint32_t r = 0; r < ps->nrem; r++) {
      uint32_t t = ps->rem[r]; uint64_t x = A[(w - B0) * nd + t], m = 0;
      while (x) { int b = __builtin_ctzll(x); if (bit[b] < 0) break; x ^= pv[bit[b]]; m |= 1ULL << bit[b]; }
      if (x) {          /* a new pivot; m is applied in the update */
        int b = __builtin_ctzll(x);
        A[(w - B0) * nd + t] = x; pv[k] = x; pm[k] = m; pr[k] = t; bit[b] = k++;
        lead[t] = (uint32_t)(64 * w + b); ps->joined++;
        if (blocks) lead_x[t] = x;
        continue;
      }
      ps->rem[nk++] = t;
      if (m) { up[nu].r = t; up[nu].m = m; nu++; }
    }
    ps->nrem = nk;
    Part P = { A, nd, B0, pr, pm, k, k0, up, nu, w + 1, B1 };
    *threaded += run_panel(P, threads);
    if (rec) record((uint32_t)w, k, k0, pr, pm, up, nu);
    if (due()) {
      size_t dc = (size_t)nld + ns, cd = w < LW ? (64 * (w + 1) < nld ? 64 * (w + 1) : nld)
                                                : nld + (64 * (w + 1 - LW) < ns ? 64 * (w + 1 - LW) : ns);
      fprintf(stderr, "f2elim: dense: columns %zu of %zu done (%.1f%%), pivots %u, rows left %u, %.0f s\n", cd, dc,
              100.0 * cd / (dc ? dc : 1), ps->joined, ps->nrem, now() - t0);
    }
  }
}

static const uint32_t *lead_of;
static int by_lead(const void *a, const void *b) {
  uint32_t x = lead_of[*(const uint32_t *)a], y = lead_of[*(const uint32_t *)b];
  return (x > y) - (x < y);
}

/* The H and S parts of basis row t in this block; its words before its first column are stale and left 0. */
static uint32_t *hbitof;        /* per held column: its dense bit, or UINT32_MAX */
static void vs_part(const uint64_t *A, uint32_t nd, uint32_t t, uint32_t lead, uint64_t *rec, size_t hw) {
  for (uint32_t j = 0; j < nh; j++) {
    uint32_t b = hbitof[j];
    if (b != UINT32_MAX && b / 64 >= B0 && b / 64 < B1 && b >= lead && (A[(b / 64 - B0) * nd + t] >> (b % 64) & 1))
      rec[j / 64] |= 1ULL << (j % 64);
  }
  for (size_t u = 0; u < SW; u++)
    if (LW + u >= B0 && LW + u < B1 && LW + u >= lead / 64) rec[hw + u] = A[(LW + u - B0) * nd + t];
}

/* The dense phase; return the rows that joined the basis, or -1 if it does not fit.  With first > 0 (match_rows),
   the other rows follow only if the first rows' bound is above target. */
static uint32_t nvs;
#define BLOCK_BASE (4u << 20)   /* the blocks' buffers and records */
static long dense_batch(int threads, double dense_max, const char *pivfn, const char *vsfn,
                        uint32_t *threaded, double *mb, size_t first, long target, const char *tmpdir) {
  long nnz; uint32_t live, livel;
  weights(&nnz, &live, &livel);
  for (uint32_t c = ns; c < nc; c++) if (cst[c] == OPEN && wt[c]) { cst[c] = DENSE; cval[c] = nld++; }
  uint32_t hbit = nld;          /* held columns are the last dense large ones */
  for (uint32_t j = 0; j < nh; j++)
    if (cst[held[j]] == HELD && wt[held[j]]) { cst[held[j]] = DENSE; cval[held[j]] = nld++; }
  if (!SW) SW = (ns + 63) / 64 + 0;
  size_t need = (nld + 63) / 64;
  LW = need; DW = LW + SW + CW;
  uint32_t nd = (uint32_t)npool;
  size_t hw = (nh + 63) / 64;
  uint32_t sbit = (uint32_t)(64 * LW), cbit = (uint32_t)(64 * (LW + SW));
  double sparse = (double)rcap * (sizeof *R + 3) + (double)ccap * 14;
  for (uint32_t i = 0; i < nr; i++) sparse += (double)R[i].cap * 4;
  double dbytes = (double)nd * (DW * 8 + 12 + sizeof(Upd)) + sparse;
  *mb = (double)nd * DW * 8 / 1e6;
  double est = (double)nd * DW * DW * DENSE_SEC / (threads > 1 ? threads : 1);
  { printf("dense rows %u cols %zu (%u large + %u small) MB %.0f s %.3g\n", nd, (size_t)nld + ns, nld, ns, *mb, est); fflush(stdout); }
  int slow = dense_s_max > 0 && est > dense_s_max, big = dense_mb_max > 0 && dbytes > dense_mb_max * 1e6;
  if (!slow && !big) in_time(est);
  /* held besides the block, including the sparse rows: freed once on disk, the allocator may keep their memory */
  uint64_t fbw = env_bw, fbmb = env_bmb;
  size_t nvsmax = nd < (size_t)(nld - hbit) + ns ? nd : (size_t)(nld - hbit) + ns;
  double held_b = (double)nd * (12 + sizeof(Upd) + 16) + (double)rcap * (sizeof *R + 3 + (CHI ? 8 : 0)) + (double)ccap * 14
                  + (double)nvsmax * (hw + SW) * 8 + BLOCK_BASE + (sparse - (double)rcap * (sizeof *R + 3) - (double)ccap * 14);
  const char *by = "memory";
  BW = DW;
  if (fbw) blocks = 1, BW = fbw, by = "F2ELIM_BLOCK_WORDS";
  else if (fbmb) blocks = 1, BW = nd ? (size_t)(fbmb * 1048576 / 8 / nd) : DW, by = "F2ELIM_BLOCK_MB";
  else if (dense_max > 0 && dbytes > dense_max) {
    blocks = 1;
    BW = nd && held_b + 8.0 * nd <= dense_max ? (size_t)((dense_max - held_b) / (8.0 * nd)) : 0;
  }
  int fits = !(blocks && !BW && !fbw && !fbmb);
  if (BW < 1) BW = 1;
  if (BW > DW) BW = DW;
  uint64_t *A = NULL; uint32_t *id = NULL, *lead = NULL, *rem = NULL; Upd *up = NULL;
  if (!slow && !big && fits) {
    if (!blocks) A = malloc((size_t)(nd ? nd : 1) * DW * 8);     /* blocks: once the sparse rows are on disk */
    id = malloc((size_t)(nd ? nd : 1) * 4); lead = malloc((size_t)(nd ? nd : 1) * 4);
    rem = malloc((size_t)(nd ? nd : 1) * 4);
    up = malloc((size_t)(nd ? nd : 1) * sizeof *up);
  }
  if ((!blocks && !A) || !id || !lead || !rem || !up) {
    if (!fits) printf("nofit memory MB %.0f\n", (held_b + 8.0 * nd) / 1e6);
    if (!fits)
      fprintf(stderr, "dense phase: over the limit DENSE_MAX: one word of the dense phase needs %.0f MB (%u rows x %zu "
              "columns, %u large left after the merge; %.0f MB the whole matrix with the sparse rows)\n",
              (held_b + 8.0 * nd) / 1e6, nd, (size_t)nld + ns, nld, dbytes / 1e6);
    else
      fprintf(stderr, "dense phase: %s %.1f GB (with the sparse rows) for %u rows x %zu columns (%u large left after the merge)\n",
              slow ? "over the time limit F2ELIM_DENSE_S:" : big ? "over the size limit F2ELIM_DENSE_MB:" : "cannot allocate",
              dbytes / 1e9, nd, (size_t)nld + ns, nld);
    free(A); free(id); free(lead); free(rem); free(up);
    return -1;
  }
  uint32_t *vslot = NULL; uint64_t *VS = NULL; uint32_t nvsb = 0;
  hbitof = chk(malloc((size_t)(nh ? nh : 1) * 4), (size_t)(nh ? nh : 1) * 4, "dense");
  for (uint32_t j = 0; j < nh; j++) hbitof[j] = cst[held[j]] == DENSE ? cval[held[j]] : UINT32_MAX;
  if (blocks) {
    fsp = tmpopen(tmpdir); frec = tmpopen(tmpdir);
    uint64_t tm = env_tmb;
    struct statvfs sv;
    if (tm) tmpmax = tm << 20;
    else if (!fstatvfs(fileno(fsp), &sv)) tmpmax = (uint64_t)(0.9 * (double)sv.f_bavail * (double)sv.f_frsize);
    printf("blocks words %zu of %zu by %s\n", BW, DW, by); fflush(stdout);
    fprintf(stderr, "f2elim: dense: blocks of %zu of the %zu words (%.1f MB), temporary files in %s\n", BW, DW,
            8.0 * nd * BW / 1e6, tmpdir);
  }
  uint32_t *buf = NULL; size_t bcap = 0;
  if (!blocks) memset(A, 0, (size_t)nd * DW * 8);
  for (uint32_t t = 0; t < nd; t++) {
    uint32_t i = pool[t];
    if (blocks && bcap < (size_t)R[i].n + 1) { bcap = 2 * ((size_t)R[i].n + 1); buf = chk(realloc(buf, bcap * 4), bcap * 4, "dense"); }
    for (int k = 0; k < R[i].n; k++) {
      uint32_t c = R[i].c[k], j = c < ns ? (uint32_t)(64 * LW) + c : cval[c];
      if (blocks) buf[k] = j;
      else A[(size_t)(j / 64) * nd + t] ^= 1ULL << (j % 64);     /* size_t: the dense phase may pass 2^32 words */
    }
    if (blocks) { uint32_t n = (uint32_t)R[i].n; twrite(fsp, &n, 4, 1); twrite(fsp, buf, 4, n); }
    else if (CHI) A[(LW + SW) * nd + t] = CHI[i];
    id[t] = i; lead[t] = UINT32_MAX; rem[t] = t; inpool[i] = 0;
  }
  free(buf);
  npool = 0;
  for (uint32_t i = 0; i < nr; i++) { free(R[i].c); R[i].c = NULL; R[i].n = R[i].cap = 0; }      /* A or fsp holds them */
  trim_heap();
  if (blocks) {
    A = malloc((size_t)(nd ? nd : 1) * (BW ? BW : 1) * 8);
    lead_x = malloc((size_t)(nd ? nd : 1) * 8); vslot = malloc((size_t)(nd ? nd : 1) * 4);
    VS = calloc(nvsmax ? nvsmax * (hw + SW) : 1, 8);
    if (!A || !lead_x || !vslot || !VS) nofit("memory", "cannot allocate the block");
  }
  uint32_t n1 = first > 0 && first < nd ? (uint32_t)first : nd, *seed = NULL, joined = 0;
  double tb = now();
  for (int pass = 0; pass < 2; pass++) {
    Pass ps = { rem, n1, NULL, 0, 0, 0 };
    uint32_t tlo = 0, thi = n1, nvs0 = nvsb;
    if (pass) {
      uint32_t d1 = 0, ns1 = 0;
      for (uint32_t t = 0; t < n1; t++) d1 += lead[t] >= sbit && lead[t] < cbit;
      long b1 = (long)ns - d1;
      printf("match bound %ld target %ld %s\n", b1, target, b1 <= target ? "kept" : "all rows");
      fflush(stdout);
      if (b1 <= target) { nzero += nd - n1; break; }
      seed = chk(malloc((size_t)n1 * 4 + 4), (size_t)n1 * 4 + 4, "dense");
      for (uint32_t t = 0; t < n1; t++) if (lead[t] != UINT32_MAX) seed[ns1++] = t;
      lead_of = lead;
      qsort(seed, ns1, 4, by_lead);
      for (uint32_t t = n1; t < nd; t++) rem[t - n1] = t;
      ps = (Pass){ rem, nd - n1, seed, ns1, 0, 0 };
      tlo = n1; thi = nd;
    }
    for (B0 = 0; B0 < DW; B0 += BW) {
      B1 = DW - B0 < BW ? DW : B0 + BW;
      /* the S blocks are still needed for OUT.vs after the last row is left */
      int vsleft = blocks && nvsb > nvs0 && B0 < LW + SW;
      if (!ps.nrem && !vsleft) break;
      if (blocks) {
        if (due()) fprintf(stderr, "f2elim: dense: block at word %zu of %zu, replaying %.0f MB, %.0f s\n", B0, DW,
                           (double)(tmpbytes) / 1e6, now() - t0);
        build(A, nd, id);
        replay(A, nd, up, threads, threaded, (uint64_t)ftello(frec));
      }
      /* records are needed by later blocks, and by a second pass after pass 0 */
      dense_words(A, nd, &ps, lead, up, threads, threaded, blocks && (!pass || B1 < DW));
      if (blocks)        /* OUT.vs rows: their H and S parts in this block */
        for (uint32_t t = tlo; t < thi; t++) {
          if (lead[t] < hbit || lead[t] >= cbit || lead[t] / 64 >= B1) continue;
          if (lead[t] / 64 >= B0) vslot[t] = nvsb++;      /* joined in this block */
          vs_part(A, nd, t, lead[t], VS + (size_t)vslot[t] * (hw + SW), hw);
        }
    }
    nzero += ps.nrem;
    joined += ps.joined;
    if (n1 == nd) break;
  }
  free(seed);
  if (blocks) {
    printf("blocks words %zu of %zu temp MB %.0f time %.1f\n", BW, DW, (double)tmpbytes / 1e6, now() - tb);
    fflush(stdout);
    fclose(fsp); fclose(frec); fsp = frec = NULL;
  }
  {
    FILE *fp = fopen(pivfn, "wb"), *fv = fopen(vsfn, "wb");
    if (!fp) { perror(pivfn); exit(2); }
    if (!fv) { perror(vsfn); exit(2); }
    uint64_t *rec = chk(malloc((hw + SW) * 8 + 8), (hw + SW) * 8 + 8, "vs");
    B0 = 0; B1 = DW;
    for (uint32_t t = 0; t < nd; t++) if (lead[t] != UINT32_MAX) {
      if (fwrite(&id[t], 4, 1, fp) != 1) wfail(pivfn);
      dimV += lead[t] >= sbit && lead[t] < cbit; crank += lead[t] >= cbit; nb++;
      if (lead[t] < hbit || lead[t] >= cbit) continue;
      const uint64_t *v = rec;
      if (blocks) v = VS + (size_t)vslot[t] * (hw + SW);
      else { memset(rec, 0, (hw + SW) * 8); vs_part(A, nd, t, lead[t], rec, hw); }
      if (fwrite(v, 8, hw + SW, fv) != hw + SW) wfail(vsfn);
      nvs++;
    }
    free(rec);
    if (fclose(fp)) wfail(pivfn);
    if (fclose(fv)) wfail(vsfn);
  }
  free(A); free(id); free(lead); free(rem); free(up); free(hbitof); free(lead_x); free(vslot); free(VS);
  return joined;
}

/* ---------- the characters */

static uint64_t Q[64], RT[64]; static int K;
static int charsbad;
static uint8_t *QR[64];         /* q < QRMAX: bit x is 1 if x is a nonzero square mod q_j */
#define QRMAX (1u << 22)
/* The Jacobi symbol (a / n), n odd. */
static int jacobi(uint64_t a, uint64_t n) {
  int s = 1;
  a %= n;
  while (a) {
    while (!(a & 1)) { a >>= 1; uint64_t r = n & 7; if (r == 3 || r == 5) s = -s; }
    uint64_t t = a; a = n; n = t;
    if ((a & 3) == 3 && (n & 3) == 3) s = -s;
    a %= n;
  }
  return n == 1 ? s : 0;
}
/* The character bits of a - b alpha: bit j is ((a - b r_j) / q_j) = -1 (a 0 symbol sets charsbad). */
static uint64_t chars_of(int64_t a, int64_t b) {
  uint64_t x = 0;
  for (int j = 0; j < K; j++) {
    int64_t q = (int64_t)Q[j], am = a % q, bm = b % q;
    if (am < 0) am += q;
    if (bm < 0) bm += q;
    uint64_t t = (uint64_t)(((__int128)am - (__int128)bm * (int64_t)RT[j]) % q + q) % (uint64_t)q;
    int s = QR[j] ? (t ? ((QR[j][t >> 3] >> (t & 7)) & 1 ? 1 : -1) : 0) : jacobi(t, (uint64_t)q);
    if (!s) charsbad = 1;
    if (s < 0) x |= 1ULL << j;
  }
  return x;
}
/* Read CHARS and IN.ab into each row's character bits. */
static void read_chars(const char *cf, const char *in) {
  FILE *f = fopen(cf, "r"); if (!f) { perror(cf); exit(2); }
  unsigned long long q, r;
  while (K < 64 && fscanf(f, "%llu %llu", &q, &r) == 2) {
    if (q < 3 || q >= (1ULL << 62) || r >= q) { fprintf(stderr, "bad character %llu %llu\n", q, r); exit(2); }
    Q[K] = q; RT[K] = r;
    if (q < QRMAX) {          /* a table of squares costs about what the rows that use it save */
      QR[K] = chk(calloc(q / 8 + 1, 1), q / 8 + 1, "characters");
      for (uint64_t x = 1, y = 1; x <= (q - 1) / 2; x++) { QR[K][y >> 3] |= (uint8_t)(1u << (y & 7)); y += 2 * x + 1; y %= q; }
    }
    K++;
  }
  fclose(f);
  if (!K) return;
  char fn[4096]; PATHF(fn, "%s.ab", in);
  FILE *g = fopen(fn, "rb"); if (!g) { perror(fn); exit(2); }
  CHI = chk(calloc(rcap ? rcap : 1, 8), (rcap ? rcap : 1) * 8, "characters");
  for (uint32_t i = 0; i < nr; i++) {
    int64_t ab[2];
    if (fread(ab, 8, 2, g) != 2) { fprintf(stderr, "short %s at row %u\n", fn, i); exit(2); }
    CHI[i] = chars_of(ab[0], ab[1]);
  }
  fclose(g);
  CW = 1;
}

/* ---------- the held columns */

static void read_held(const char *fn) {
  FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(2); }
  uint32_t c, cap = 0;
  while (fread(&c, 4, 1, f) == 1) {
    if (c < ns || c >= nc || cst[c] == HELD) { fprintf(stderr, "bad held column %u\n", c); exit(2); }
    if (nh == cap) { cap = cap ? 2 * cap : 64; held = grow(held, cap, 4, "held"); }
    cst[c] = HELD; held[nh++] = c;
  }
  fclose(f);
}

static int once(int argc, char **argv) {
  long inwords = argc >= 13 ? atol(argv[12]) : 0;
  if (inwords < 0 || inwords > 1 << 20) { fprintf(stderr, "WORDS must be in 0..2^20\n"); return 2; }
  ns = (uint32_t)atol(argv[2]); long target = atol(argv[3]);
  uint32_t dense = (uint32_t)atol(argv[5]); int wmax = atoi(argv[6]);
  int gate = atoi(argv[7]); long keep = atol(argv[8]);
  int threads = atoi(argv[9]); double dense_max = strtod(argv[10], NULL);
  progress = strtod(argv[11], NULL);
  t0 = tlast = now();
  readall(argv[1], (uint32_t)inwords);
  if (argc >= 14 && strcmp(argv[13], "-")) read_chars(argv[13], argv[1]);
  if (ns > nc) { fprintf(stderr, "NS > ncols\n"); return 2; }
  if (argc == 15 && strcmp(argv[14], "-")) read_held(argv[14]);
  for (uint32_t i = 0; i < nr; i++) if (!empty(i)) to_pool(i);
  long nnz; uint32_t live, livel;
  weights(&nnz, &live, &livel);
  long pr = purge();
  in_time(0);                   /* a purge-only run (TARGET < 0) past its deadline is late too */
  weights(&nnz, &live, &livel);
  /* excess: rows beyond the L columns, less the dim V the target needs */
  long need = target > 0 ? (long)ns - target : (long)ns;
  printf("purge rows %u removed %ld lcols %u nnz %ld excess %ld time %.1f\n", live, pr, livel, nnz, (long)live - livel - need, now() - t0);
  fflush(stdout);
  if (target < 0 || (gate && (long)live - livel - need < 0)) return 0;
  trim((long)live - livel - need, keep, need);
  char pivfn[4096], vsfn[4096];
  PATHF(opsfn, "%s.ops.tmp", argv[4]); PATHF(pivfn, "%s.piv.tmp", argv[4]);
  PATHF(vsfn, "%s.vs.tmp", argv[4]);
  fops = fopen(opsfn, "wb"); if (!fops) { perror(opsfn); return 2; }
  long merges = merge(dense, wmax);
  if (fclose(fops)) wfail(opsfn);
  fops = NULL;
  free(tmp); tmp = NULL; tmpcap = 0; free(lockr); lockr = NULL;
  weights(&nnz, &live, &livel);
  printf("merge rows %u lcols %u nnz %ld merges %ld ops %ld time %.1f\n", live, livel, nnz, merges, nops, now() - t0);
  fflush(stdout);
  if (getenv("F2ELIM_POOL")) {
    char poolfn[4096], fn[4096]; long pnnz = 0;
    PATHF(poolfn, "%s.pool.tmp", argv[4]);
    FILE *fp = fopen(poolfn, "wb"); if (!fp) { perror(poolfn); return 2; }
    compact();
    uint32_t np = (uint32_t)npool;
    if (fwrite(&np, 4, 1, fp) != 1) wfail(poolfn);
    for (size_t t = 0; t < npool; t++) {
      uint32_t i = pool[t], k0 = 0;
      while (k0 < (uint32_t)R[i].n && R[i].c[k0] < ns) k0++;
      uint32_t hd[2] = { i, (uint32_t)R[i].n - k0 };
      if (fwrite(hd, 4, 2, fp) != 2 || (hd[1] && fwrite(R[i].c + k0, 4, hd[1], fp) != hd[1])) wfail(poolfn);
      pnnz += hd[1];
    }
    if (fclose(fp)) wfail(poolfn);
    PATHF(fn, "%s.ops", argv[4]); if (rename(opsfn, fn)) wfail(fn);
    PATHF(fn, "%s.pool", argv[4]); if (rename(poolfn, fn)) wfail(fn);
    printf("pool rows %u nnz %ld ops %ld time %.1f\n", np, pnnz, nops, now() - t0);
    return 0;
  }
  /* not with characters: the first rows alone would give a lower crank */
  size_t first = target > 0 && !K ? match_rows(MATCH_EXTRA) : 0;
  uint32_t threaded = 0; double mb;
  char tmpdir[4096];
  const char *td = getenv("F2ELIM_TMPDIR");
  if (td && *td) PATHF(tmpdir, "%s", td);
  else {
    PATHF(tmpdir, "%s", argv[4]);
    char *sl = strrchr(tmpdir, '/');
    if (!sl) PATHF(tmpdir, ".");
    else if (sl == tmpdir) tmpdir[1] = 0;
    else *sl = 0;
  }
  if (dense_batch(threads, dense_max, pivfn, vsfn, &threaded, &mb, first, target, tmpdir) < 0) { remove(opsfn); return 3; }
  char fn[4096];
  PATHF(fn, "%s.ops", argv[4]); if (rename(opsfn, fn)) wfail(fn);
  PATHF(fn, "%s.piv", argv[4]); if (rename(pivfn, fn)) wfail(fn);
  PATHF(fn, "%s.vs", argv[4]); if (rename(vsfn, fn)) wfail(fn);
  printf("result rows %u rank %u dimV %u bound %ld crank %d pivots %u ops %ld threaded %u vs %u time %.1f\n", nr, nb,
         dimV, (long)ns - dimV, K && !charsbad ? (int)crank : -1, nb, nops, threaded, nvs, now() - t0);
  return 0;
}

int main(int argc, char **argv) {
  if (getenv("F2ELIM_DEADLINE")) deadline = strtod(getenv("F2ELIM_DEADLINE"), NULL);
  if (getenv("F2ELIM_DENSE_S")) dense_s_max = strtod(getenv("F2ELIM_DENSE_S"), NULL);
  if (getenv("F2ELIM_DENSE_MB")) dense_mb_max = strtod(getenv("F2ELIM_DENSE_MB"), NULL);
  /* before any work, so that a bad setting costs nothing and leaves no file */
  env_bw = envnum("F2ELIM_BLOCK_WORDS"); env_bmb = envnum("F2ELIM_BLOCK_MB"); env_tmb = envnum("F2ELIM_TMP_MB");
  const char *td = getenv("F2ELIM_TMPDIR");
  if (td && *td && access(td, W_OK | X_OK)) { perror(td); fprintf(stderr, "F2ELIM_TMPDIR is not a folder f2elim can write\n"); return 2; }
  if (argc < 12 || argc > 15) {
    fprintf(stderr, "usage: f2elim IN NS TARGET OUT DENSE WMAX GATE KEEP THREADS DENSE_MAX PROGRESS "
                    "[WORDS [CHARS [HELD]]]\n");
    return 2;
  }
  return once(argc, argv);
}
