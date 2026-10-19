/* f2elim: the bound dim Cl(K)/2 <= NS - dim V, with a record of the elimination that the verifier replays.

   M is the relation matrix over F_2.  Columns 0..NS-1 are the small primes S (norm < GB), the others are large (L),
   and V is the subspace of rowspace(M) that vanishes on L.
   1. Purge: a row with an L column of weight 1 is in no combination that vanishes on L, so it is removed.
   1b. Trim: while the excess exceeds KEEP, the rows with the most L columns are removed; the rows of an S column
      with at most SCARCE rows are kept.
   2. Merge: for an L column of weight w, its shortest row is added to the other w - 1 rows (each addition logged)
      and then removed.
   3. Dense: the remaining rows, L columns first, reduced to echelon form one 64-column panel at a time.  A row only
      receives rows before it, so dim V is the number of pivots whose leading column is in S.
   Removing rows can only make V smaller, so the bound holds whatever the heuristics do.

   Usage: f2elim IN NS TARGET OUT DENSE WMAX GATE KEEP THREADS DENSE_MAX PROGRESS
   IN: uint32 nrows, ncols, then each row as its length and its columns.  OUT.ops: uint32 pairs (dst, src), in order;
   OUT.piv: the pivot rows.  Both are renamed into place only when the elimination is complete.  With TARGET < 0, or
   GATE = 1 and a negative excess, f2elim stops after the purge.  KEEP < 0: no trim.  DENSE_MAX: the bytes the dense
   phase may use (0: no limit).  PROGRESS: seconds between progress lines on stderr.
   stdout: "purge ...", "merge ...", "dense ...", "result rows R rank K dimV D bound B pivots P ops O threaded T time S".
   Exit status: 0 done; 3 the dense phase does not fit; 2 anything else. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#if defined(__GLIBC__)
#include <malloc.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif
/* return the memory freed after the merge to the system, so that DENSE_MAX counts what the process holds */
static void trim_heap(void) {
#if defined(__GLIBC__)
  malloc_trim(0);
#elif defined(__APPLE__)
  malloc_zone_pressure_relief(NULL, 0);
#endif
}

typedef struct { int n, cap; uint32_t *c; } Row;
static Row *R; static uint32_t nr, nc, ns;
static char *alive;
static uint32_t *tmp; static size_t tmpcap;
static FILE *fops;
static long nops;

static int cmpu(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return (x > y) - (x < y); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
/* P, or exit 2 if an allocation of N bytes failed */
static void *chk(void *p, size_t n, const char *what) {
  if (!p && n) { fprintf(stderr, "out of memory (%s: %.0f MB)\n", what, (double)n / 1e6); exit(2); }
  return p;
}
static void wfail(const char *fn) { perror(fn); fprintf(stderr, "write to %s failed\n", fn); exit(2); }
static char opsfn[4096];
static double t0, progress, tlast;
/* whether PROGRESS seconds have passed since the last progress line */
static int due(void) { double t = now(); if (t - tlast < progress) return 0; tlast = t; return 1; }

static void xorrow(uint32_t d, uint32_t s) {   /* R[d] ^= R[s], both sorted; logged */
  Row *D = &R[d], *S = &R[s];
  size_t need = (size_t)D->n + S->n;
  if (tmpcap < need) { tmpcap = 2 * need; tmp = chk(realloc(tmp, tmpcap * sizeof *tmp), tmpcap * sizeof *tmp, "row buffer"); }
  int i = 0, j = 0, k = 0;
  while (i < D->n && j < S->n) {
    if (D->c[i] < S->c[j]) tmp[k++] = D->c[i++]; else if (D->c[i] > S->c[j]) tmp[k++] = S->c[j++]; else { i++; j++; }
  }
  while (i < D->n) tmp[k++] = D->c[i++];
  while (j < S->n) tmp[k++] = S->c[j++];
  if (D->cap < k) { D->cap = k; D->c = chk(realloc(D->c, (size_t)k * sizeof *D->c), (size_t)k * sizeof *D->c, "row"); }
  memcpy(D->c, tmp, (size_t)k * sizeof *tmp); D->n = k;
  uint32_t pr[2] = { d, s }; if (fwrite(pr, sizeof pr, 1, fops) != 1) wfail(opsfn); nops++;
}

static void readall(const char *fn) {
  FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(2); }
  uint32_t h[2]; if (fread(h, 4, 2, f) != 2) { fprintf(stderr, "bad header\n"); exit(2); }
  nr = h[0]; nc = h[1];
  if (nc > INT32_MAX) { fprintf(stderr, "bad header: %u columns (at most 2^31 - 1)\n", nc); exit(2); }
  R = chk(calloc(nr ? nr : 1, sizeof *R), (size_t)(nr ? nr : 1) * sizeof *R, "rows");
  uint32_t *buf = NULL; size_t bcap = 0;
  for (uint32_t i = 0; i < nr; i++) {
    if (!(i & 0xffff) && due()) fprintf(stderr, "f2elim: read: row %u of %u, %.0f s\n", i, nr, now() - t0);
    uint32_t n; if (fread(&n, 4, 1, f) != 1) { fprintf(stderr, "short file at row %u\n", i); exit(2); }
    if (n > INT32_MAX) { fprintf(stderr, "row %u: %u entries (at most 2^31 - 1)\n", i, n); exit(2); }
    if (bcap < (size_t)n + 1) { bcap = 2 * ((size_t)n + 1); buf = chk(realloc(buf, bcap * 4), bcap * 4, "input row"); }
    if (n && fread(buf, 4, n, f) != n) { fprintf(stderr, "short row %u\n", i); exit(2); }
    for (uint32_t k = 0; k < n; k++) if (buf[k] >= nc) { fprintf(stderr, "column %u out of range in row %u\n", buf[k], i); exit(2); }
    qsort(buf, n, 4, cmpu);
    int m = 0; for (uint32_t k = 0; k < n; k++) { if (m && buf[m - 1] == buf[k]) { m--; continue; } buf[m++] = buf[k]; }
    R[i].n = R[i].cap = m; R[i].c = chk(malloc((size_t)(m ? m : 1) * 4), (size_t)(m ? m : 1) * 4, "row");
    memcpy(R[i].c, buf, (size_t)m * 4);
  }
  fclose(f); free(buf);
}

static uint32_t *wt;   /* column weights over alive rows */
/* the trim keeps every row of an S column with at most this many rows */
#define SCARCE 4
static int scarce(uint32_t i) { for (int k = 0; k < R[i].n && R[i].c[k] < ns; k++) if (wt[R[i].c[k]] <= SCARCE) return 1; return 0; }
static void weights(long *nnz, uint32_t *live, uint32_t *livel) {
  memset(wt, 0, (size_t)nc * 4); *nnz = 0; *live = 0; *livel = 0;
  for (uint32_t i = 0; i < nr; i++) if (alive[i]) { (*live)++; *nnz += R[i].n; for (int k = 0; k < R[i].n; k++) wt[R[i].c[k]]++; }
  for (uint32_t c = ns; c < nc; c++) if (wt[c]) (*livel)++;
}
/* remove rows with an L column of weight 1, until none is left; returns rows removed */
static long purge(void) {
  long removed = 0; int again = 1;
  while (again) {
    again = 0;
    for (uint32_t i = 0; i < nr; i++) if (alive[i]) {
      int single = 0;
      for (int k = R[i].n - 1; k >= 0 && R[i].c[k] >= ns; k--) if (wt[R[i].c[k]] == 1) { single = 1; break; }
      if (single) { alive[i] = 0; removed++; again = 1; for (int k = 0; k < R[i].n; k++) wt[R[i].c[k]]--; }
    }
  }
  return removed;
}

/* dense phase helpers: a[w..] ^= b[w..]; the panel update, split over threads */
static void xorw(uint64_t *a, const uint64_t *b, size_t w, size_t words) { for (size_t t = w; t < words; t++) a[t] ^= b[t]; }
typedef struct { uint32_t row; uint64_t m; } Upd;
typedef struct { uint64_t *A; size_t words, w; const uint32_t *prow; const Upd *up; uint32_t nu; } Panel;
typedef struct { const Panel *P; uint32_t lo, hi; } Part;
static void *panel_part(void *arg) {
  const Part *q = arg; const Panel *P = q->P;
  for (uint32_t u = q->lo; u < q->hi; u++) {
    uint64_t *a = P->A + (size_t)P->up[u].row * P->words;
    for (uint64_t m = P->up[u].m; m; m &= m - 1) xorw(a, P->A + (size_t)P->prow[__builtin_ctzll(m)] * P->words, P->w, P->words);
  }
  return NULL;
}
/* returns 1 if the update ran on threads */
static int run_panel(const Panel *P, int threads) {
  if (threads <= 1 || (double)P->nu * (P->words - P->w) < 1e6) { Part q = { P, 0, P->nu }; panel_part(&q); return 0; }
  pthread_t th[256]; Part q[256]; int T = threads > 256 ? 256 : threads;
  for (int t = 0; t < T; t++) { q[t].P = P; q[t].lo = (uint32_t)((uint64_t)P->nu * t / T); q[t].hi = (uint32_t)((uint64_t)P->nu * (t + 1) / T); }
  int up[256];          /* a part without a thread runs here */
  for (int t = 1; t < T; t++) up[t] = pthread_create(&th[t], NULL, panel_part, &q[t]) == 0;
  panel_part(&q[0]);
  for (int t = 1; t < T; t++) { if (up[t]) pthread_join(th[t], NULL); else panel_part(&q[t]); }
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 12) { fprintf(stderr, "usage: f2elim IN NS TARGET OUT DENSE WMAX GATE KEEP THREADS DENSE_MAX PROGRESS\n"); return 2; }
  ns = (uint32_t)atol(argv[2]); long target = atol(argv[3]);
  uint32_t dense = (uint32_t)atol(argv[5]); int wmax = atoi(argv[6]);
  int gate = atoi(argv[7]); long keep = atol(argv[8]);
  int threads = atoi(argv[9]); double dense_max = strtod(argv[10], NULL);
  progress = strtod(argv[11], NULL);
  t0 = tlast = now();
  readall(argv[1]);
  if (ns > nc) { fprintf(stderr, "NS > ncols\n"); return 2; }
  alive = chk(calloc(nr ? nr : 1, 1), nr ? nr : 1, "rows"); for (uint32_t i = 0; i < nr; i++) alive[i] = R[i].n > 0;
  wt = chk(calloc(nc ? nc : 1, 4), (size_t)(nc ? nc : 1) * 4, "column weights");
  long nnz; uint32_t live, livel;
  weights(&nnz, &live, &livel);
  long pr = purge();
  weights(&nnz, &live, &livel);
  /* excess: rows beyond the L columns, compared with the dimension of V that the target needs */
  long need = target > 0 ? (long)ns - target : (long)ns;
  printf("purge rows %u removed %ld lcols %u nnz %ld excess %ld time %.1f\n", live, pr, livel, nnz, (long)live - livel - need, now() - t0);
  fflush(stdout);
  if (target < 0 || (gate && (long)live - livel - need < 0)) return 0;
  long excess = (long)live - livel - need;
  /* at most 20 rounds; stopping early only leaves the excess above KEEP */
  for (int round = 0; keep >= 0 && excess > keep && round < 20; round++) {
    /* remove the rows with the most L columns, then purge */
    int maxw = 0; for (uint32_t i = 0; i < nr; i++) if (alive[i]) { int l = 0; for (int k = R[i].n - 1; k >= 0 && R[i].c[k] >= ns; k--) l++; if (l > maxw) maxw = l; }
    long todo = excess - keep;
    for (int w = maxw; w >= 1 && todo > 0; w--)    /* a row without L columns stays */
      for (uint32_t i = 0; i < nr && todo > 0; i++) if (alive[i]) {
        int l = 0; for (int k = R[i].n - 1; k >= 0 && R[i].c[k] >= ns; k--) l++;
        if (l == w && !scarce(i)) { alive[i] = 0; todo--; for (int k = 0; k < R[i].n; k++) wt[R[i].c[k]]--; }
      }
    purge();
    weights(&nnz, &live, &livel);
    excess = (long)live - livel - need;
    printf("trim rows %u lcols %u nnz %ld excess %ld\n", live, livel, nnz, excess);
    fflush(stdout);
  }
  char pivfn[4096];
  snprintf(opsfn, sizeof opsfn, "%s.ops.tmp", argv[4]); snprintf(pivfn, sizeof pivfn, "%s.piv.tmp", argv[4]);
  fops = fopen(opsfn, "wb"); if (!fops) { perror(opsfn); return 2; }
  uint32_t *start = chk(calloc((size_t)nc + 1, 4), ((size_t)nc + 1) * 4, "merge"), *lst = NULL; size_t lcap = 0;
  char *lock = chk(calloc(nr ? nr : 1, 1), nr ? nr : 1, "merge"), *dirty = chk(calloc(nc ? nc : 1, 1), nc ? nc : 1, "merge");
  int W = 2; long merges = 0; uint32_t lc0 = livel;
  for (int pass = 0;; pass++) {
    weights(&nnz, &live, &livel);
    purge();
    weights(&nnz, &live, &livel);
    if (due()) fprintf(stderr, "f2elim: merge: %ld of %u large columns eliminated (pass %d, W %d; stops at %u), rows %u, "
                       "nnz %ld, ops %ld, %.0f s\n", (long)lc0 - livel, lc0, pass, W, dense, live, nnz, nops, now() - t0);
    /* the merge's memory is capped at 600M nonzeros: 2.4 GB for the rows, as much for the column lists */
    if (livel <= dense || nnz > 600000000L) break;
    if (lcap < (size_t)nnz) { lcap = nnz; lst = chk(realloc(lst, lcap * 4), lcap * 4, "merge"); }
    start[0] = 0; for (uint32_t c = 0; c < nc; c++) start[c + 1] = start[c] + wt[c];
    uint32_t *fill = chk(malloc(((size_t)nc + 1) * 4), ((size_t)nc + 1) * 4, "merge"); memcpy(fill, start, ((size_t)nc + 1) * 4);
    for (uint32_t i = 0; i < nr; i++) if (alive[i]) for (int k = 0; k < R[i].n; k++) lst[fill[R[i].c[k]]++] = i;
    free(fill);
    memset(lock, 0, nr); memset(dirty, 0, nc); long done = 0;
    for (int w = 2; w <= W; w++) for (uint32_t c = ns; c < nc; c++) {
      if (!((c - ns) & 0xffff) && due())       /* a long pass */
        fprintf(stderr, "f2elim: merge: pass %d, weight %d of %d: column %u of %u, %ld merges in this pass, %.0f s\n",
                pass, w, W, c - ns, nc - ns, done, now() - t0);
      if (wt[c] != (uint32_t)w) continue;
      uint32_t *L = lst + start[c]; int bad = 0; long piv = -1;
      if (dirty[c]) continue;
      for (int t = 0; t < w; t++) if (lock[L[t]]) { bad = 1; break; }
      if (bad) continue;
      for (int t = 0; t < w; t++) if (piv < 0 || R[L[t]].n < R[piv].n) piv = L[t];
      for (int t = 0; t < w; t++) { lock[L[t]] = 1; if (L[t] != (uint32_t)piv) { xorrow(L[t], (uint32_t)piv); if (!R[L[t]].n) alive[L[t]] = 0; } }
      for (int k = 0; k < R[piv].n; k++) dirty[R[piv].c[k]] = 1;
      alive[piv] = 0; done++;
    }
    merges += done;
    if (!done) { if (++W > wmax) break; }
  }
  if (fclose(fops)) wfail(opsfn);
  free(lst); free(lock); free(dirty); free(start); free(tmp); tmp = NULL; tmpcap = 0;     /* the merge's own memory */
  weights(&nnz, &live, &livel);
  printf("merge rows %u lcols %u nnz %ld merges %ld ops %ld time %.1f\n", live, livel, nnz, merges, nops, now() - t0);
  fflush(stdout);

  /* dense phase: L columns first (bits 0..nl-1), then S (nl..nl+ns-1); A = the alive rows, in row order */
  uint32_t *map = chk(malloc((size_t)(nc ? nc : 1) * 4), (size_t)(nc ? nc : 1) * 4, "column map"), nl = 0;
  for (uint32_t c = ns; c < nc; c++) map[c] = wt[c] ? nl++ : UINT32_MAX;
  for (uint32_t c = 0; c < ns; c++) map[c] = nl + c;
  free(wt); wt = NULL;
  /* the memory besides A: the live rows, R, alive and map */
  double sparse = (double)nr * (sizeof *R + 1) + (double)nc * 4;
  for (uint32_t i = 0; i < nr; i++) {
    if (alive[i]) sparse += (double)R[i].cap * 4;
    else { free(R[i].c); R[i].c = NULL; R[i].n = R[i].cap = 0; }
  }
  trim_heap();
  uint32_t dc = nl + ns; size_t words = (dc + 63) / 64;
  double dbytes = (double)live * (words * 8 + 12 + sizeof(Upd)) + sparse;      /* A, id, lead, rem, up; the sparse rows */
  printf("dense rows %u cols %u (%u large + %u small) MB %.0f\n", live, dc, nl, ns, (double)live * words * 8 / 1e6);
  fflush(stdout);
  uint64_t *A = NULL; uint32_t *id = NULL, *lead = NULL, *rem = NULL; Upd *up = NULL;
  if (!(dense_max > 0 && dbytes > dense_max)) {
    A = malloc((size_t)(live ? live : 1) * words * 8);
    id = malloc((size_t)(live ? live : 1) * 4); lead = malloc((size_t)(live ? live : 1) * 4);
    rem = malloc((size_t)(live ? live : 1) * 4);
    up = malloc((size_t)(live ? live : 1) * sizeof *up);
  }
  if (!A || !id || !lead || !rem || !up) {   /* too dense: no bound from this matrix */
    fprintf(stderr, "dense phase: %s %.1f GB (with the sparse rows) for %u rows x %u columns (%u large left after the merge)\n",
            dense_max > 0 && dbytes > dense_max ? "over the limit DENSE_MAX:" : "cannot allocate", dbytes / 1e9, live, dc, nl);
    remove(opsfn);
    return 3;
  }
  memset(A, 0, (size_t)live * words * 8);
  uint32_t nd = 0;
  for (uint32_t i = 0; i < nr; i++) if (alive[i]) {
    uint64_t *a = A + (size_t)nd * words;
    for (int k = 0; k < R[i].n; k++) { uint32_t j = map[R[i].c[k]]; a[j / 64] ^= 1ULL << (j % 64); }
    id[nd] = i; lead[nd] = UINT32_MAX; rem[nd] = nd; nd++;
  }
  for (uint32_t i = 0; i < nr; i++) free(R[i].c);      /* A holds the rows now */
  free(R); R = NULL; free(alive); alive = NULL; free(map);
  uint32_t nrem = nd, rank = 0, srank = 0, threaded = 0;
  for (size_t w = 0; w < words && nrem; w++) {
    /* the panel: rows in order; pv[s] = word w of the panel's pivot s (lowest bit: its leading column) */
    uint64_t pv[64]; uint32_t prow[64]; int bit[64], k = 0; uint32_t nu = 0, nk = 0;
    for (int b = 0; b < 64; b++) bit[b] = -1;
    for (uint32_t r = 0; r < nrem; r++) {
      uint32_t i = rem[r]; uint64_t *a = A + (size_t)i * words, x = a[w], m = 0;
      while (x) { int b = __builtin_ctzll(x); if (bit[b] < 0) break; x ^= pv[bit[b]]; m |= 1ULL << bit[b]; }
      if (x) {          /* a new pivot: its combination is applied now */
        for (uint64_t mm = m; mm; mm &= mm - 1) xorw(a, A + (size_t)prow[__builtin_ctzll(mm)] * words, w, words);
        int b = __builtin_ctzll(x);
        pv[k] = x; prow[k] = i; bit[b] = k++;
        lead[i] = (uint32_t)(64 * w + b); rank++; srank += lead[i] >= nl;
        continue;
      }
      rem[nk++] = i;
      if (m) { up[nu].row = i; up[nu].m = m; nu++; }
    }
    nrem = nk;
    /* every other row is cleared on the panel by its combination of pivots */
    Panel P = { A, words, w, prow, up, nu };
    threaded += run_panel(&P, threads);
    if (due()) {
      size_t cd = 64 * (w + 1) < dc ? 64 * (w + 1) : dc;
      fprintf(stderr, "f2elim: dense: columns %zu of %u done (%.1f%%), pivots %u, rows left %u, %.0f s\n", cd, dc,
              100.0 * cd / dc, rank, nrem, now() - t0);
    }
  }
  free(A);
  FILE *fp = fopen(pivfn, "wb"); if (!fp) { perror(pivfn); return 2; }
  uint32_t kept = 0;
  for (uint32_t i = 0; i < nd; i++) if (lead[i] != UINT32_MAX) { if (fwrite(&id[i], 4, 1, fp) != 1) wfail(pivfn); kept++; }
  if (fclose(fp)) wfail(pivfn);
  char fn[4096];
  snprintf(fn, sizeof fn, "%s.ops", argv[4]); if (rename(opsfn, fn)) wfail(fn);
  snprintf(fn, sizeof fn, "%s.piv", argv[4]); if (rename(pivfn, fn)) wfail(fn);
  printf("result rows %u rank %u dimV %u bound %ld pivots %u ops %ld threaded %u time %.1f\n", nr, rank, srank,
         (long)ns - srank, kept, nops, threaded, now() - t0);
  return 0;
}
