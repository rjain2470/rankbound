"""f2elim's dense phase in blocks of words: with blocks (F2ELIM_BLOCK_WORDS, F2ELIM_BLOCK_MB, or a DENSE_MAX below the
whole matrix) its output is the whole matrix's, byte for byte (OUT.ops, .piv, .vs and the result line but its time
and threads), on random matrices with every feature of the dense phase (held columns, characters, the matching's
second pass, leading words, trims, rank deficiency), blocks of one word, of a few, of the whole matrix, an empty
remainder, and real relation sets; a dense phase that does not fit (not one word in DENSE_MAX, the temporary files
over F2ELIM_TMP_MB, F2ELIM_DENSE_MB) exits 3 without output files; bad values exit 2; and a witness written with
blocks is accepted by the checker with the same G and S as one without.  `make test-asan` runs these under ASan."""
import array, json, os, random, shutil, subprocess, sys, tempfile, unittest
from unittest import mock
from rankbound import linalg
from helpers import SRC

Q = [q for q in range(1001, 3000, 2) if all(q % p for p in range(3, int(q ** .5) + 1, 2))]


def random_rows(rnd, nrows, ncols):
    """Random rows (sets of columns < ncols) of one of several shapes, with repeats, sums, a dense column, empty rows."""
    shape = rnd.randrange(7)
    rows = []
    for _ in range(nrows):
        k = rnd.randint(3, 25) if shape == 6 else {0: 3, 1: 8, 2: max(1, ncols // 2), 3: 2, 4: 5, 5: 1}[shape]
        rows.append(set(rnd.sample(range(ncols), min(ncols, rnd.randint(0, k)))))
    if rows and rnd.random() < 0.3:          # sums of rows: rank deficiency
        for _ in range(rnd.randint(1, nrows)):
            rows[rnd.randrange(nrows)] = rows[rnd.randrange(nrows)] ^ rows[rnd.randrange(nrows)]
    if rows and rnd.random() < 0.3:          # a dense column
        c = rnd.randrange(ncols)
        for r in rows:
            if rnd.random() < 0.9:
                r.add(c)
    if rows and rnd.random() < 0.2:          # empty rows
        for _ in range(rnd.randint(1, 3)):
            rows[rnd.randrange(nrows)] = set()
    return rows


def write_input(d, rnd, rows, ns, ncols, chars=False, held=(), words=0):
    """f2elim's input files in d: m.bin (with WORDS leading words), chars.txt and m.bin.ab, held.bin; returns the
    arguments CHARS, HELD."""
    a = array.array('I', [len(rows), ncols])
    for r in rows:
        rr = sorted(r)
        if words:
            a.frombytes(sum(1 << c for c in rr if c < 64 * words).to_bytes(8 * words, 'little'))
            rr = [c for c in rr if c >= 64 * words]
        a.append(len(rr)); a.extend(rr)
    with open(os.path.join(d, 'm.bin'), 'wb') as f:
        a.tofile(f)
    cf = hf = '-'
    if chars:
        cf = os.path.join(d, 'chars.txt')
        with open(cf, 'w') as f:
            for q in rnd.sample(Q, rnd.randint(1, 64)):
                f.write('%d %d\n' % (q, rnd.randrange(q)))
        with open(os.path.join(d, 'm.bin.ab'), 'wb') as f:
            array.array('q', [x for _ in rows for x in (rnd.randint(-10 ** 9, 10 ** 9), rnd.randint(1, 10 ** 6))]).tofile(f)
    if held:
        hf = os.path.join(d, 'held.bin')
        with open(hf, 'wb') as f:
            array.array('I', sorted(held)).tofile(f)
    return cf, hf


def result(stdout):
    """f2elim's stdout without its times, threads and the lines of the blocks."""
    out = []
    for l in stdout.splitlines():
        t = l.split()
        if not t or t[0] == 'blocks':
            continue
        if t[0] in ('purge', 'merge', 'trim', 'result'):
            kv = [(t[i], t[i + 1]) for i in range(1, len(t) - 1, 2) if t[i] not in ('time', 'threaded')]
            l = ' '.join([t[0]] + ['%s %s' % x for x in kv])
        out.append(l)
    return '\n'.join(out)


def run(d, args, env=None, dense_max=0):
    """Run f2elim on d/m.bin with args (NS TARGET DENSE KEEP THREADS WORDS CHARS HELD) and env; return (exit status,
    result(stdout), {extension: bytes or None}, stdout, stderr)."""
    ns, target, dense, keep, threads, words, cf, hf = args
    out = os.path.join(d, 'o')
    for e in ('.ops', '.piv', '.vs'):
        if os.path.exists(out + e):
            os.remove(out + e)
    p = subprocess.run([linalg.ENGINE, os.path.join(d, 'm.bin'), str(ns), str(target), out, str(dense), '64', '0',
                        str(keep), str(threads), str(dense_max), '1000', str(words), cf, hf],
                       capture_output=True, text=True, timeout=600,
                       env=dict({k: v for k, v in os.environ.items() if not k.startswith('F2ELIM_')}, **(env or {})))
    files = {e: open(out + e, 'rb').read() if os.path.exists(out + e) else None for e in ('.ops', '.piv', '.vs')}
    return p.returncode, result(p.stdout), files, p.stdout, p.stderr


def words_of(stdout):
    """(BW, W) of f2elim's 'blocks words BW of W' line, or None (the whole matrix)."""
    l = next((l.split() for l in stdout.splitlines() if l.startswith('blocks words ')), None)
    return (int(l[2]), int(l[4])) if l else None


@unittest.skipUnless(os.path.exists(linalg.ENGINE), 'f2elim not built')
class TestBlocks(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='f2blocks')

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def same(self, args, env, msg):
        """f2elim with env (blocks) gives the whole matrix's exit status, result and files; returns (whole, blocks)."""
        w, b = run(self.d, args), run(self.d, args, env)
        self.assertEqual(w[0], 0, (msg, w[4][-800:]))
        self.assertEqual(b[:3], w[:3], (msg, env, b[4][-800:]))
        return w, b

    def test_random(self):
        """Blocks of 1, 2, 3, a few, or more words than the matrix has, on 400 random matrices (characters, held
        columns, targets that keep the matching's rows or go on with all, leading words, trims, threads), and blocks
        of an MiB size: the whole matrix's output.  Some of them have the second pass, an empty remainder."""
        rnd = random.Random(53)
        seen = {'second pass': 0, 'blocks': 0, 'empty': 0, 'chars': 0}
        for trial in range(400):
            big = trial % 10 == 0
            ns, nl = (rnd.randint(64, 500), rnd.randint(50, 800)) if big else (rnd.randint(0, 150), rnd.randint(1, 300))
            ncols = ns + nl
            rows = random_rows(rnd, rnd.randint(0, ncols + 60), ncols)
            chars = rnd.random() < 0.3
            held = rnd.sample(range(ns, ncols), rnd.randint(1, min(nl, 70))) if rnd.random() < 0.5 else ()
            words = rnd.randint(1, ns // 64) if ns >= 64 and rnd.random() < 0.2 else 0
            cf, hf = write_input(self.d, rnd, rows, ns, ncols, chars, held, words)
            args = (ns, rnd.choice([0, 1, 1, rnd.randint(1, max(1, ns)), ns + 5]), rnd.choice([0, 5, 1000, 100000]),
                    rnd.choice([-1, -1, rnd.randint(0, 20)]), rnd.choice([1, 3]), words, cf, hf)
            env = {'F2ELIM_BLOCK_WORDS': str(rnd.choice([1, 1, 2, 3, rnd.randint(1, 40), 10 ** 6]))}
            if trial % 7 == 3:
                env = {'F2ELIM_BLOCK_MB': '1'}
            w, b = self.same(args, env, trial)
            seen['second pass'] += 'all rows' in w[1]
            seen['chars'] += chars and ' crank -1 ' not in w[1]
            seen['empty'] += 'dense rows 0 ' in w[1]
            bw = words_of(b[3])
            seen['blocks'] += bw is not None and bw[0] < bw[1]
        for k, v in seen.items():
            self.assertGreater(v, 3, (k, seen))

    def test_dense(self):
        """Hundreds of rows per word (the four Russians' tables, threads), held and small columns over several words:
        blocks of 1 and 2 words and the whole matrix agree."""
        rnd = random.Random(59)
        for trial in range(12):
            ns, nl = rnd.randint(64, 700), rnd.randint(20, 300)
            rows = [set(rnd.sample(range(ns + nl), rnd.randint(5, 30))) for _ in range(rnd.randint(300, 1200))]
            rows += rnd.sample(rows, 30)
            held = rnd.sample(range(ns, ns + nl), rnd.randint(0, min(nl, 100)))
            cf, hf = write_input(self.d, rnd, rows, ns, ns + nl, trial % 3 == 0, held)
            args = (ns, rnd.choice([0, 1, ns // 2]), rnd.choice([50, 10 ** 5]), -1, rnd.choice([1, 4]), 0, cf, hf)
            for bw in (1, 2):
                self.same(args, {'F2ELIM_BLOCK_WORDS': str(bw)}, (trial, bw))

    def test_memory_choice(self):
        """A DENSE_MAX below the whole matrix: blocks by memory, as large as fit, with the whole matrix's output; below
        one word of it: exit 3, 'nofit memory MB', no output files."""
        rnd = random.Random(61)
        ns, nl = 1500, 10000                # a dense phase of about 17 MB: several times f2elim's base of 4 MiB
        rows = [set(rnd.sample(range(ns + nl), rnd.randint(5, 30))) for _ in range(12000)]
        cf, hf = write_input(self.d, rnd, rows, ns, ns + nl, False, rnd.sample(range(ns, ns + nl), 40))
        args = (ns, 0, 10 ** 5, -1, 1, 0, cf, hf)
        w = run(self.d, args)
        t = next(l.split() for l in w[3].splitlines() if l.startswith('dense '))
        mb = float(t[t.index('MB') + 1])
        self.assertGreater(mb, 15, w[3])
        b = run(self.d, args, dense_max=int(mb * 1e6 / 2 + 5.5e6))
        self.assertEqual(b[:3], w[:3], b[4][-800:])
        bw = words_of(b[3])
        self.assertTrue(bw is not None and 1 <= bw[0] < bw[1] and 'by memory' in b[3], b[3])
        n = run(self.d, args, dense_max=1000)
        self.assertEqual(n[0], 3, n[4])
        self.assertIn('nofit memory MB ', n[3])
        self.assertEqual(n[2], {'.ops': None, '.piv': None, '.vs': None})

    def test_limits(self):
        """The temporary files over F2ELIM_TMP_MB: exit 3, 'nofit disk', no output files; F2ELIM_DENSE_MB below the
        whole matrix: exit 3; F2ELIM_TMPDIR the folder of the temporary files (removed, as unlinked); a value that is not
        a number 1..2^40, or a missing F2ELIM_TMPDIR: exit 2 at the start."""
        rnd = random.Random(67)
        ns, nl = 640, 400
        rows = [set(rnd.sample(range(ns + nl), rnd.randint(40, 80))) for _ in range(6000)]
        cf, hf = write_input(self.d, rnd, rows, ns, ns + nl)
        args = (ns, 0, 50, -1, 1, 0, cf, hf)
        b = run(self.d, args, {'F2ELIM_BLOCK_WORDS': '1'})
        self.assertGreater(float(b[3].split(' temp MB ')[1].split()[0]), 2, b[3])     # well over 1 MiB
        t = run(self.d, args, {'F2ELIM_BLOCK_WORDS': '1', 'F2ELIM_TMP_MB': '1'})
        self.assertEqual(t[0], 3, t[4][-500:])
        self.assertIn('nofit disk', t[3])
        self.assertEqual(t[2], {'.ops': None, '.piv': None, '.vs': None})
        c = run(self.d, args, {'F2ELIM_DENSE_MB': '1'})
        self.assertEqual(c[0], 3, c[4][-500:])
        self.assertIn('F2ELIM_DENSE_MB', c[4])
        tmp = os.path.join(self.d, 'tmp')
        os.mkdir(tmp)
        self.same(args, {'F2ELIM_BLOCK_WORDS': '3', 'F2ELIM_TMPDIR': tmp}, 'tmpdir')
        self.assertEqual(os.listdir(tmp), [])
        for var, v in (('F2ELIM_BLOCK_WORDS', '0'), ('F2ELIM_BLOCK_WORDS', 'x'), ('F2ELIM_BLOCK_WORDS', ''),
                       ('F2ELIM_BLOCK_WORDS', '2000000000000'), ('F2ELIM_BLOCK_MB', '1.5'), ('F2ELIM_TMP_MB', '-1'),
                       ('F2ELIM_TMPDIR', os.path.join(self.d, 'none'))):
            p = run(self.d, args, {var: v})          # at the start: no work, no file (not even OUT.ops.tmp)
            self.assertEqual(p[0], 2, (var, v, p[4]))
            self.assertNotIn('purge', p[3], (var, v))
            self.assertEqual(sorted(f for f in os.listdir(self.d) if f.startswith('o.')), [], (var, v))

    def test_relations(self):
        """Real relation sets (line-sieved relations of small fields, with their free relations): solve untrimmed with
        characters, trimmed, and with the target of a witness, with blocks of 1 and 2 words; the same bound, files and
        k (selmer.k_of, from OUT.vs)."""
        from test_reference import SMALL, field_of, line_relations, write_roots
        from rankbound import selmer
        for ai, _ in SMALL[:6]:
            F, small = field_of(ai)
            lpb = 12
            M = linalg.Matrix(F, small)
            fb = os.path.join(self.d, 'roots.gz')
            write_roots(F['poly'], 2 ** lpb, fb)
            for p, labs in linalg.free_rows(F, fb, 2 ** lpb):
                M.add_free(p, labs)
            for a, b, H in line_relations(F, lpb, 40):
                M.add_relation('%d,%d:%s' % (a, b, ','.join('%x' % h for h in H)))
            chars = linalg.characters([int(x) for x in F['poly']], max(2 ** lpb, M.maxp))
            scols = M.col.with_prime({2})
            for keep, target, ch in ((-1, 0, chars), (M.ns, 0, ()), (M.ns, max(1, M.ns // 2), ())):
                got = []
                for env in ({}, {'F2ELIM_BLOCK_WORDS': '1'}, {'F2ELIM_BLOCK_WORDS': '2'}):
                    w = os.path.join(self.d, 'la%d' % len(got))
                    with mock.patch.dict(os.environ, env):
                        r = linalg.solve(M, w, target, keep=keep, chars=ch, dense=50)
                    files = [open(x, 'rb').read() for x in (r.ops, r.piv, r.vs)]
                    got.append(((r.bound, r.dimV, r.rank, r.crank), files, selmer.k_of(r, M.ns, scols)))
                    shutil.rmtree(w)
                self.assertEqual(got[1], got[0], (ai, keep))
                self.assertEqual(got[2], got[0], (ai, keep))


@unittest.skipUnless(os.path.exists(linalg.ENGINE) and shutil.which('gp'), 'f2elim not built, or no gp')
class TestWitness(unittest.TestCase):
    def test_witness(self):
        """rankbound bound on a curve with every elimination in blocks of one word, and without: both witnesses accepted
        by the checker, with the same G and S."""
        import gzip
        from rankbound import check as verify
        env = dict({k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'F2ELIM_'))},
                   PYTHONPATH=SRC)
        out = []
        with tempfile.TemporaryDirectory() as d:
            for k, more in enumerate(({}, {'F2ELIM_BLOCK_WORDS': '1'})):
                w = os.path.join(d, 'w%d' % k)
                r = subprocess.run([sys.executable, '-m', 'rankbound', 'bound', '0,-1,1,-929,-10595', '--dir', w,
                                    '--compute', '--cores', '1'], env=dict(env, **more), capture_output=True,
                                   text=True, timeout=600)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
                wit = [os.path.join(p, f) for p, _, fs in os.walk(w) for f in fs if '.witness.json' in f]
                self.assertEqual(len(wit), 1, wit)
                self.assertEqual(verify.check(wit[0])['verdict'], 'accepted')
                with (gzip.open if wit[0].endswith('.gz') else open)(wit[0], 'rt') as f:
                    c = json.load(f)['claims']
                logs = ''.join(open(os.path.join(p, f)).read() for p, _, fs in os.walk(w) for f in fs
                               if os.path.basename(p) == 'la' and f.endswith('.json'))
                self.assertEqual('blocks words 1 of' in logs, bool(more), k)
                out.append((c['G'], c['S']))
        self.assertEqual(out[0], out[1])


if __name__ == '__main__':
    unittest.main()
