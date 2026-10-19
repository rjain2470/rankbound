/* f2rank: the rank over F_2 of a 0/1 matrix, and dimV, the number of pivots whose lowest column is >= nl: the
   dimension of the row space intersected with the vectors that vanish on columns 0..nl-1.
   stdin (uint32, native byte order): nrows, ncols, nl, then each row as its length and its distinct columns.
   argv[1]: threads (default 1).  stdout: "rank R dimV D".
   Gaussian elimination, one 64-column word at a time.  A row only receives rows before it, so the pivots form an
   echelon basis with distinct lowest columns. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>

static uint64_t *M;          /* the rows, W words each */
static size_t W;

typedef struct { uint32_t row; uint64_t combo; } Todo;
typedef struct { const Todo *todo; size_t lo, hi, w; const uint32_t *pivot_row; } Slice;

static void add_row(uint32_t dst, uint32_t src, size_t from) {
  uint64_t *d = M + (size_t)dst * W; const uint64_t *s = M + (size_t)src * W;
  for (size_t t = from; t < W; t++) d[t] ^= s[t];
}

static void *apply(void *arg) {
  const Slice *sl = arg;
  for (size_t k = sl->lo; k < sl->hi; k++)
    for (uint64_t c = sl->todo[k].combo; c; c &= c - 1)
      add_row(sl->todo[k].row, sl->pivot_row[__builtin_ctzll(c)], sl->w);
  return NULL;
}

int main(int argc, char **argv) {
  int threads = argc > 1 ? atoi(argv[1]) : 1;
  if (threads < 1) threads = 1;
  if (threads > 256) threads = 256;
  uint32_t h[3];
  if (fread(h, 4, 3, stdin) != 3) { fprintf(stderr, "bad header\n"); return 2; }
  uint32_t nrows = h[0], ncols = h[1], nl = h[2];
  W = (ncols + 63) / 64;
  M = calloc((size_t)(nrows ? nrows : 1) * (W ? W : 1), 8);
  uint32_t *open_rows = malloc((size_t)(nrows ? nrows : 1) * 4);
  Todo *todo = malloc((size_t)(nrows ? nrows : 1) * sizeof *todo);
  uint32_t *col = NULL; size_t ccap = 0;
  if (!M || !open_rows || !todo) { fprintf(stderr, "out of memory (%u rows x %u columns)\n", nrows, ncols); return 2; }
  for (uint32_t i = 0; i < nrows; i++) {
    uint32_t n;
    if (fread(&n, 4, 1, stdin) != 1) { fprintf(stderr, "short row %u\n", i); return 2; }
    if (ccap < n) {
      uint32_t *c2 = realloc(col, 2 * (size_t)n * 4);
      if (!c2) { fprintf(stderr, "out of memory (row %u: %u columns)\n", i, n); return 2; }
      col = c2; ccap = 2 * (size_t)n;
    }
    if (n && fread(col, 4, n, stdin) != n) { fprintf(stderr, "short row %u\n", i); return 2; }
    uint64_t *r = M + (size_t)i * W;
    for (uint32_t k = 0; k < n; k++) {
      if (col[k] >= ncols) { fprintf(stderr, "column %u out of range in row %u\n", col[k], i); return 2; }
      r[col[k] / 64] ^= 1ULL << (col[k] % 64);
    }
    open_rows[i] = i;
  }
  uint32_t nopen = nrows, rank = 0, dimV = 0;
  for (size_t w = 0; w < W && nopen; w++) {
    uint64_t word_of[64]; uint32_t pivot_row[64]; int at_bit[64], npiv = 0;
    for (int b = 0; b < 64; b++) at_bit[b] = -1;
    size_t ntodo = 0; uint32_t still = 0;
    for (uint32_t k = 0; k < nopen; k++) {
      uint32_t i = open_rows[k];
      uint64_t x = M[(size_t)i * W + w], combo = 0;
      int b;
      while (x && at_bit[b = __builtin_ctzll(x)] >= 0) { combo |= 1ULL << at_bit[b]; x ^= word_of[at_bit[b]]; }
      if (x) {
        for (uint64_t c = combo; c; c &= c - 1) add_row(i, pivot_row[__builtin_ctzll(c)], w);
        b = __builtin_ctzll(x);
        at_bit[b] = npiv; word_of[npiv] = x; pivot_row[npiv] = i; npiv++;
        rank++;
        if (64 * w + (size_t)b >= nl) dimV++;
      } else {
        open_rows[still++] = i;
        if (combo) { todo[ntodo].row = i; todo[ntodo].combo = combo; ntodo++; }
      }
    }
    nopen = still;
    int T = (ntodo * (W - w) < 1000000) ? 1 : threads;
    Slice sl[256]; pthread_t th[256];
    for (int t = 0; t < T; t++) {
      sl[t].todo = todo; sl[t].lo = ntodo * t / T; sl[t].hi = ntodo * (t + 1) / T; sl[t].w = w; sl[t].pivot_row = pivot_row;
    }
    int up[256];        /* a slice without a thread runs here */
    for (int t = 1; t < T; t++) up[t] = pthread_create(&th[t], NULL, apply, &sl[t]) == 0;
    apply(&sl[0]);
    for (int t = 1; t < T; t++) { if (up[t]) pthread_join(th[t], NULL); else apply(&sl[t]); }
  }
  printf("rank %u dimV %u\n", rank, dimV);
  return 0;
}
