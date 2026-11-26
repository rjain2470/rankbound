"""check rank in column blocks: with CHECK_RANK_BLOCK_WORDS / CHECK_RANK_BLOCK_MB the dense
matrix is eliminated a block of words at a time, the earlier words' clearings replayed from a temporary file.  Its
stdout must be byte for byte that of the default (one block), on seeded random matrices with ops, repeated rows and
pivots, prefixes and peeling, and both must equal a plain echelon in Python; also with every word cleared on threads
(PAR_WORK=0).  The temporary file is gone after every run, bad settings exit 2, a file over its budget exits 4 (does not fit)."""
import array, os, random, shutil, subprocess, tempfile, unittest
from rankbound.check import selmer as vsel

N_RANDOM = 60 if os.environ.get('RANKBOUND_TEST_FAST') else 160


def rank(ints):
    piv = {}
    for x in ints:
        while x:
            h = x.bit_length() - 1
            if h not in piv:
                piv[h] = x
                break
            x ^= piv[h]
    return len(piv)


def counts(R, nl, piv):
    """(rank, rankL, large, dimV) of the rows R[i], i in piv."""
    M = [sum(1 << c for c in R[i]) for i in piv]
    rM, rL = rank(M), rank([x & ((1 << nl) - 1) for x in M])
    large = len({c for i in piv for c in R[i] if c < nl})
    return rM, rL, large, rM - rL


def expect(rows, nl, ops, piv, pre):
    R = [set(r) for r in rows]
    for d, s in ops:
        R[d] = R[d] ^ R[s]
    a = counts(R, nl, piv)
    line = 'rank %d rankL %d large %d dimV %d' % a
    if pre is not None and pre < len(piv):
        line += ' prefix %d rank %d rankL %d large %d dimV %d' % ((pre,) + counts(R, nl, piv[:pre]))
    return (line + '\n').encode()


def random_case(rnd):
    """Rows over 64..700 columns (several words), dense enough that much survives peeling, with some sparse columns,
    repeated rows and sums of rows; ops (dst = src too), pivots with repeats, a prefix."""
    ncols = rnd.randint(1, 700)
    nrows = rnd.randint(1, 160)
    dens = rnd.choice([0.02, 0.1, 0.3, 0.5])
    rows = [set(c for c in range(ncols) if rnd.random() < dens) for _ in range(nrows)]
    for _ in range(rnd.randint(0, nrows // 2)):     # sparse columns for peeling: one row each
        rows[rnd.randrange(nrows)].add(rnd.randrange(ncols))
    if rnd.random() < 0.5:                           # repeated rows and sums (rank deficiency)
        for _ in range(rnd.randint(1, nrows)):
            rows[rnd.randrange(nrows)] = rows[rnd.randrange(nrows)] ^ rows[rnd.randrange(nrows)]
    if rnd.random() < 0.2:
        rows[rnd.randrange(nrows)] = set()
    ops = [(rnd.randrange(nrows), rnd.randrange(nrows)) for _ in range(rnd.randint(0, 2 * nrows))]
    piv = [rnd.randrange(nrows) for _ in range(rnd.randint(0, nrows + 5))]
    if piv and rnd.random() < 0.3:
        piv += piv[:rnd.randint(1, len(piv))]
    nl = rnd.randint(0, ncols)
    pre = None if rnd.random() < 0.3 else rnd.randint(0, len(piv))
    return rows, ncols, nl, ops, piv, pre


def data(rows, ncols, nl, ops, piv):
    d, rnd = array.array('I', [len(rows), ncols, nl, len(ops), len(piv)]), random.Random(len(rows))
    for r in rows:
        r = list(r); rnd.shuffle(r)                  # unsorted columns
        d.append(len(r)); d.extend(r)
    for o in ops:
        d.extend(o)
    d.extend(piv)
    return d.tobytes()


class TestColumnBlocks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.par0 = os.path.join(cls.tmp, 'check_par0')       # every word's clearing on the threads
        src = os.path.join(os.path.dirname(vsel.CHECK), 'check.c')
        subprocess.run([os.environ.get('CC', 'cc'), '-O1', '-DPAR_WORK=0', '-pthread', '-o', cls.par0, src, '-lm'],
                       check=True)
        cls.logdir = os.path.join(cls.tmp, 'logs')
        os.mkdir(cls.logdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def run_rank(self, inp, threads=1, pre=None, exe=None, **env):
        e = dict(os.environ, CHECK_RANK_TMPDIR=self.logdir)
        for k in ('CHECK_RANK_BLOCK_MB', 'CHECK_RANK_BLOCK_WORDS', 'CHECK_RANK_TMP_MB', 'CHECK_RANK_MEM_MB'):
            e.pop(k, None)
        e.update({k: str(v) for k, v in env.items()})
        p = subprocess.run([exe or vsel.CHECK, 'rank', str(threads)] + ([] if pre is None else [str(pre)]),
                           input=inp, capture_output=True, env=e, timeout=120)
        self.assertEqual(os.listdir(self.logdir), [])         # the temporary file is gone
        return p

    def test_random_blocks(self):
        """Every block size 1..4 words and one at random, threads 1..4, against the default and the Python echelon."""
        rnd = random.Random(4545)
        blocked = 0
        for trial in range(N_RANDOM):
            rows, ncols, nl, ops, piv, pre = random_case(rnd)
            inp, want = data(rows, ncols, nl, ops, piv), expect(rows, nl, ops, piv, pre)
            full = self.run_rank(inp, pre=pre)
            self.assertEqual((full.returncode, full.stdout), (0, want), trial)
            W = (ncols + 63) // 64
            for bw in sorted({1, 2, 3, 4, rnd.randint(1, max(1, W))}):
                th = rnd.randint(1, 4)
                exe = self.par0 if rnd.random() < 0.5 else None
                p = self.run_rank(inp, th, pre, exe, CHECK_RANK_BLOCK_WORDS=bw)
                self.assertEqual((p.returncode, p.stdout), (0, want), (trial, bw, th, exe))
                blocked += b'replaying' in p.stderr
        self.assertGreater(blocked, N_RANDOM)          # most cases really had several blocks

    def test_block_mb(self):
        """CHECK_RANK_BLOCK_MB: 1 MiB over about 9000 dense rows of 900 columns is 14 words: two blocks, as the default."""
        rnd = random.Random(7)
        rows = [set(c for c in range(900) if rnd.random() < 0.05) for _ in range(9000)]
        for _ in range(2000):
            rows[rnd.randrange(9000)] = rows[rnd.randrange(9000)] ^ rows[rnd.randrange(9000)]
        inp = data(rows, 900, 300, [], list(range(9000)))
        full = self.run_rank(inp, 1, 5000)
        self.assertEqual(full.returncode, 0)
        p = self.run_rank(inp, 2, 5000, CHECK_RANK_BLOCK_MB=1)
        self.assertEqual((p.returncode, p.stdout), (0, full.stdout))
        self.assertIn(b'check rank: blocks of ', p.stderr)
        self.assertIn(b'replaying', p.stderr)

    def test_settings_and_budget(self):
        """Bad settings (exit 2) and a temporary file over its budget (exit 4): nothing on stdout, no file left."""
        rnd = random.Random(11)
        rows = [set(rnd.sample(range(640), 250)) for _ in range(16000)]
        inp = data(rows, 640, 100, [], list(range(16000)))
        for env in ({'CHECK_RANK_BLOCK_MB': ''}, {'CHECK_RANK_BLOCK_MB': '0'}, {'CHECK_RANK_BLOCK_MB': '-1'},
                    {'CHECK_RANK_BLOCK_WORDS': '1x'}, {'CHECK_RANK_BLOCK_WORDS': '99999999999999999999'},
                    {'CHECK_RANK_MEM_MB': 'x'}, {'CHECK_RANK_BLOCK_WORDS': 1, 'CHECK_RANK_TMP_MB': 1}):
            p = self.run_rank(inp, **env)
            self.assertEqual((p.returncode, p.stdout), (4 if 'CHECK_RANK_TMP_MB' in env else 2, b''), env)
            self.assertIn(b'check rank: ', p.stderr)
        p = self.run_rank(inp, CHECK_RANK_BLOCK_WORDS=1, CHECK_RANK_TMPDIR='/nonexistent/dir')
        self.assertEqual((p.returncode, p.stdout), (2, b''))
        self.assertIn(b'temporary file', p.stderr)
        ok = self.run_rank(inp, CHECK_RANK_BLOCK_WORDS=5, CHECK_RANK_TMP_MB=100)
        self.assertEqual((ok.returncode, ok.stdout), (0, self.run_rank(inp).stdout))


if __name__ == '__main__':
    unittest.main()
