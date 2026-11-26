"""Memory and reuse of the relations' linear algebra: f2elim's queue-driven singleton purge (the same rows left as
the rescanning purge, linear on a chain); linalg.Matrix's packed rows and exact deduplication keys (the same rows,
ids and matrix.bin bytes); sieve.finish_slice streams the slice into gzip (the same relations and statistics, a
fraction of the memory); a reuse of an elimination or kernel checks its manifest of digests (a changed output of the
same size is redone); f2kernel's threads by work (the same vectors for any threads and threshold); the splitting
types shared with the BDF search; and Field's fixed multiplication table, omat and norm (the matching selection:
test_kernels)."""
import array, gzip, json, os, random, shutil, subprocess, sys, tempfile, time, tracemalloc, unittest
from unittest import mock
from fractions import Fraction
from rankbound import field, linalg, runtime, sieve
from helpers import SRC


def write_matrix(path, rows, nc):
    with open(path, 'wb') as f:
        array.array('I', [len(rows), nc]).tofile(f)
        for r in rows:
            array.array('I', [len(r)] + list(r)).tofile(f)


def rescanning_purge(rows, ns, held):
    """The rescanning purge, a reference implementation: passes over the pool in order until no row has an OPEN large column
    of weight 1; returns (live rows, removed, live L columns, nonzeros) as f2elim's purge line."""
    rows = [sorted(set(r)) for r in rows]      # repeats are distinct here
    pool = [i for i, r in enumerate(rows) if r]
    inpool = {i: True for i in pool}
    wt = {}
    for i in pool:
        for c in rows[i]:
            wt[c] = wt.get(c, 0) + 1
    sparse, removed, again = set(), 0, True
    isopen = lambda c: c >= ns and c not in sparse and c not in held
    while again:
        again = False
        for i in pool:
            if not inpool[i]:
                continue
            single = next((c for c in reversed(rows[i]) if isopen(c) and wt[c] == 1), None)
            if single is not None:
                inpool[i] = False; sparse.add(single); removed += 1; again = True
                for c in rows[i]:
                    wt[c] -= 1
    live = [i for i in pool if inpool[i]]
    lcols = {c for i in live for c in rows[i] if c >= ns and (isopen(c) or c in held)}
    return len(live), removed, len(lcols), sum(len(rows[i]) for i in live)


class TestPurge(unittest.TestCase):
    def purge_line(self, d, rows, ns, nc, held=()):
        inp = os.path.join(d, 'm.bin')
        write_matrix(inp, rows, nc)
        hf = '-'
        if held:
            hf = os.path.join(d, 'held.bin')
            with open(hf, 'wb') as f:
                array.array('I', sorted(held)).tofile(f)
        rc, out, err = linalg.run_engine([inp, str(ns), '-1', os.path.join(d, 'p'), '1000', '64', '0', '-1', '1', '0',
                                          '300', '0', '-', hf], None)
        self.assertEqual(rc, 0, err)
        p = linalg._parse(out)[0]
        return int(p['rows']), int(p['removed']), int(p['lcols']), int(p['nnz'])

    def test_same_rows_left(self):
        """The rows left (their number, L columns and nonzeros) and the number removed equal the rescanning purge's
        on random matrices with chains, repeats and held columns (the order of removals differs)."""
        d = tempfile.mkdtemp(prefix='o23')
        try:
            for seed in range(150):
                R = random.Random(seed)
                ns, nl = R.randint(1, 20), R.randint(1, 120)
                nc = ns + nl
                rows = [sorted(set(R.randrange(nc) if R.random() < .3 else R.randrange(ns, nc)
                                   for _ in range(R.randint(0, 6)))) for _ in range(R.randint(1, 200))]
                L = R.randint(1, nl)
                rows += [[ns]] + [[ns + i - 1, ns + i] for i in range(1, L)]
                rows += [rows[R.randrange(len(rows))] for _ in range(R.randint(0, 10))]
                R.shuffle(rows)
                held = set(R.sample(range(ns, nc), R.randint(0, min(5, nl))))
                self.assertEqual(self.purge_line(d, rows, ns, nc, held), rescanning_purge(rows, ns, held), seed)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_chain_linear(self):
        """The chain R_0 = {0}, R_i = {i-1, i} (all large), in this order: the rescanning purge took one pass per row
        (40,000 rows: 3.3-4.6 s); the queue removes every row at once."""
        d = tempfile.mkdtemp(prefix='o23')
        try:
            n = 40000
            inp = os.path.join(d, 'chain.bin')
            with open(inp, 'wb') as f:
                a = array.array('I', [n, n + 1, 1, 1])
                for i in range(1, n):
                    a.extend((2, i, i + 1))
                a.tofile(f)
            t = time.time()
            rc, out, err = linalg.run_engine([inp, '1', '-1', os.path.join(d, 'p'), '1000', '64', '0', '-1', '1', '0',
                                              '300'], runtime.Budget(seconds=30), deadline=True)
            self.assertEqual(rc, 0, err)
            self.assertLess(time.time() - t, 1.0)
            p = linalg._parse(out)[0]
            self.assertEqual((p['rows'], p['removed']), (0, n))
        finally:
            shutil.rmtree(d, ignore_errors=True)


def _read(path):
    with open(path, 'rb') as f:
        return f.read()


def old_write(rows, ncols, path):
    """matrix.bin as Matrix.write wrote it from a list of arrays: the header, then per row its length and ids."""
    with open(path, 'wb') as f:
        array.array('I', [len(rows), ncols]).tofile(f)
        for r in rows:
            array.array('I', [len(r)]).tofile(f)
            array.array('I', r).tofile(f)


class TestRows(unittest.TestCase):
    def test_rows_sequence(self):
        R = random.Random(5)
        ref = [[R.randrange(2 ** 32) for _ in range(R.randint(0, 9))] for _ in range(300)]
        rows = linalg.Rows(ref)
        self.assertEqual(len(rows), 300)
        self.assertEqual([list(r) for r in rows], ref)
        self.assertEqual([list(rows[i]) for i in range(-300, 300)], ref + ref)
        for i, j in ((0, 0), (0, 17), (5, 290), (100, 50), (-20, None), (0, None)):
            self.assertEqual([list(r) for r in rows[i:j]], ref[i:j])
        self.assertEqual([list(r) for r in rows[1:200:7]], ref[1:200:7])
        with self.assertRaises(IndexError):
            rows[300]
        p = rows[:120]                      # a prefix shares the arrays and stays as it was
        rows.append([7, 8])
        self.assertEqual((len(p), [list(r) for r in p]), (120, ref[:120]))
        with self.assertRaises(TypeError):
            p.append([1])
        self.assertEqual(list(rows.pop()), [7, 8])
        self.assertEqual([list(r) for r in rows], ref)

    def test_same_matrix_bytes(self):
        """Relations with repeats, big elements and a J label: the same rows (ids, order), sources, elements and
        matrix.bin bytes as the list of arrays; an append writes what a full write does; a prefix writes its rows."""
        d = tempfile.mkdtemp(prefix='o24')
        try:
            R = random.Random(11)
            ps = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
            M = linalg.Matrix({'J': [(2, 1)]}, [(p, 0) for p in ps[:6]])
            ref, lines = [], []
            for t in range(2000):
                a, b = R.randint(-10 ** 6, 10 ** 6), R.randint(1, 10 ** 4)
                if t % 97 == 0:
                    a, b = R.choice([(2 ** 70 + t, 3), (-(2 ** 63) - t, 2 ** 63 + t), (5, -(2 ** 64))])
                if t % 13 == 0 and lines:
                    l = R.choice(lines)          # a repeat
                else:
                    l = '%d,%d:%s' % (a, b, ','.join('%x' % R.choice(ps) for _ in range(R.randint(0, 6))))
                    lines.append(l)
                n = len(M.rows)
                if M.add_relation(l):
                    ref.append(list(M.rows[n]))
                if t == 1000:
                    M.write(os.path.join(d, 'm.bin'))
                    M.add_free(101, {(101, 1), (101, 7)}); ref.append(list(M.rows[-1]))
            keys = set()
            for l in lines:
                a, b = map(int, l.split(':')[0].split(','))
                keys.add((a, b))
            self.assertEqual(len(M.rows), len(keys) + 1)
            self.assertEqual(len(M.seen), len(keys))
            M.write(os.path.join(d, 'm.bin'))                        # appended
            old_write(ref, linalg.ncols(M), os.path.join(d, 'old.bin'))
            full = linalg.Matrix({'J': []}, [])
            full.rows, full.col = M.rows, M.col
            full.write(os.path.join(d, 'full.bin'))
            data = _read(os.path.join(d, 'old.bin'))
            self.assertEqual(_read(os.path.join(d, 'm.bin')), data)
            self.assertEqual(_read(os.path.join(d, 'full.bin')), data)
            P = linalg.Matrix({'J': []}, [])
            P.rows, P.col = M.rows[:777], M.col
            P.write(os.path.join(d, 'p.bin'))
            old_write(ref[:777], linalg.ncols(M), os.path.join(d, 'oldp.bin'))
            self.assertEqual(_read(os.path.join(d, 'p.bin')), _read(os.path.join(d, 'oldp.bin')))
            self.assertEqual(len(M.src), len(M.rows))
            self.assertEqual(len(M.elt), 2 * len(M.rows))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_columns_and_seen(self):
        """Columns gives a dict's setdefault(label, len) ids, get, in, items and labels, for packed labels (p, r) and
        the others (huge or negative entries, other types), through its growth; Seen deduplicates as a set of exact keys,
        big elements too; FileRows and FileLines read back what was appended, popped, and a prefix."""
        R = random.Random(13)
        labs = [(R.randrange(2 ** 34) if R.random() < 0.1 else R.randrange(1000), R.randrange(50)) for _ in range(30000)]
        labs += [('J', 'x%d' % i) for i in range(20)] + [(-3, 1), (2 ** 32 - 1, 0), (2 ** 32 - 2, 2 ** 32 - 1), (5, -1)]
        R.shuffle(labs)
        C, ref = linalg.Columns(), {}
        for l in labs:
            self.assertEqual(C.id(l), ref.setdefault(l, len(ref)))
        self.assertEqual(len(C), len(ref))
        self.assertEqual(list(C.items()), list(ref.items()))
        for l in labs[:2000] + [(7, 10 ** 6), ('J', 'y'), (2 ** 40, 1)]:
            self.assertEqual((C.get(l), l in C), (ref.get(l), l in ref))
        self.assertEqual(C.with_prime({3, 'J'}), [c for l, c in ref.items() if l[0] in (3, 'J')])
        elt = array.array('q')
        S, keys = linalg.Seen(elt), set()
        for t in range(20000):
            a, b = R.choice([(R.randrange(-50, 50), R.randrange(1, 40)), (R.randrange(2 ** 70), 1)])
            new = linalg.seen_key(a, b) not in keys
            keys.add(linalg.seen_key(a, b))
            self.assertEqual(S.add(a, b, len(elt) // 2), new)
            if new:
                big = abs(a) >= linalg.INT62 or abs(b) >= linalg.INT62
                elt.extend((0, 0) if big else (a, b))
        self.assertEqual(len(S), len(keys))
        rows, lines = linalg.FileRows(), linalg.FileLines()
        ref_r = [[R.randrange(2 ** 32) for _ in range(R.randint(0, 9))] for _ in range(5000)]
        for r in ref_r:
            rows.append(r); lines.append(','.join(map(str, r)))
        self.assertEqual(list(rows.pop()), ref_r.pop()); lines.pop()
        p = rows[:1234]
        self.assertEqual(([list(r) for r in rows], [list(r) for r in p], len(p)), (ref_r, ref_r[:1234], 1234))
        self.assertEqual(list(lines), [','.join(map(str, r)) for r in ref_r])
        self.assertEqual((lines[17], list(rows[-1]), [list(r) for r in rows[3:9]]),
                         (','.join(map(str, ref_r[17])), ref_r[-1], ref_r[3:9]))

    def test_seen_key_exact(self):
        """seen_key is injective, beyond int64 too (the tuple is the fallback)."""
        vals = [0, 1, -1, 2 ** 62, -(2 ** 62), 2 ** 63 - 1, -(2 ** 63), 2 ** 63, -(2 ** 63) - 1, 2 ** 64, -(2 ** 64),
                2 ** 64 - 1, 2 ** 100, -(2 ** 100) + 3]
        pairs = [(a, b) for a in vals for b in vals]
        self.assertEqual(len({linalg.seen_key(a, b) for a, b in pairs}), len(pairs))


PARENT = r"""
import os, random, resource, sys, tracemalloc
from rankbound import linalg
n, d = int(sys.argv[1]), sys.argv[2]
rnd = random.Random(1)
small = [(p, r) for p in range(3, 1803, 3) for r in (1, 2, 3)]
large = [(10 ** 7 + 7 * c, c) for c in range(100000)]
tracemalloc.start()
t0 = tracemalloc.get_traced_memory()[0]
M = linalg.Matrix({'J': []}, small[:600], folder=d)
for i in range(n):
    a, b = rnd.randint(-2 ** 40, 2 ** 40), rnd.randrange(1, 2 ** 20, 2)
    labels = set(rnd.sample(small, 8) + rnd.sample(large, 3))
    if M.seen.add(a, b, len(M.rows)):
        M._add(labels, '%d,%d:%s' % (a, b, ','.join('%x' % z[0] for z in labels)), a, b)
res = linalg.solve(M, os.path.join(d, 'la'), 0, keep=M.ns, dense=50)
for i in (0, len(M.rows) // 2, len(M.rows) - 1):
    assert list(M.rows[i]) and M.src[i]
held = tracemalloc.get_traced_memory()[0] - t0
print(len(M.rows), len(M.col), held, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, res.bound)
"""


class TestParent(unittest.TestCase):
    def test_flat(self):
        """The parent's memory over adding N relations (each a row of 8 of 1,800 small labels and 3 of 100,000 large
        ones) and a check (linalg.solve), from 50,000 to 200,000 relations: what Python holds (tracemalloc) grows by
        less than 80 bytes a row (measured 46), its peak (getrusage) by less than 300 (measured 91-207 on macOS: the
        peak is noisy): the rows and their lines are on disk, the column ids and the deduplication in compact tables
        (lists, a set and a dict of tuples take over 400 bytes a row)."""
        got = []
        for n in (50000, 200000):
            with tempfile.TemporaryDirectory() as d:
                p = subprocess.run([sys.executable, '-c', PARENT, str(n), d], capture_output=True, text=True,
                                   timeout=900, env=dict(os.environ, PYTHONPATH=SRC))
                self.assertEqual(p.returncode, 0, p.stderr[-2000:])
                rows, cols, held, peak = (int(x) for x in p.stdout.split()[:4])
                got.append((rows, held, peak * (1 if sys.platform == 'darwin' else 1024)))
        dr = got[1][0] - got[0][0]
        self.assertLess((got[1][1] - got[0][1]) / dr, 80, got)
        self.assertLess((got[1][2] - got[0][2]) / dr, 300, got)


def old_finish_slice(sdir, name, meta, lines):
    """A reference finish_slice that keeps the normalized lines in a list, then writes them."""
    n, com, out = 0, [], []
    for l in lines:
        if l[0] == '#':
            com.append(l)
            continue
        t = l.split(':', 2)
        if len(t) < 2:
            continue
        out.append('%s:%s\n' % (t[0], t[1].rstrip('\n'))); n += 1
    g = sieve.stats_of(''.join(com))
    with gzip.open(os.path.join(sdir, name + '.rels.gz'), 'wt', compresslevel=1) as f:
        f.writelines(out)
    with open(os.path.join(sdir, name + '.json'), 'w') as f:
        json.dump(dict(meta, cpu=g['cpu'], relations=n, special_q=g['nq']), f)
    return g['cpu'] or 0.0, n


class TestSlice(unittest.TestCase):
    def slice_lines(self, n):
        R = random.Random(3)
        out = ['# las output\n']
        for i in range(n):
            out.append('%d,%d:%s:%s\n' % (R.randint(-10 ** 9, 10 ** 9), R.randint(1, 10 ** 6),
                                         ','.join('%x' % R.randrange(2 ** 30) for _ in range(9)),
                                         ','.join('%x' % R.randrange(2 ** 20) for _ in range(3))))
            if i % 5000 == 0:
                out.append('# a comment\n'); out.append('junk without colon\n')
        out.append('# Total cpu time 12.5s [norm 0.1+0.2, sieving 3.0]\n')
        out.append('# Total %d reports [0.001s/r, 2.5r/sq]\n' % n)
        return out

    def test_same_slice(self):
        d = tempfile.mkdtemp(prefix='o20')
        try:
            lines = self.slice_lines(60000)
            os.mkdir(os.path.join(d, 'old')); os.mkdir(os.path.join(d, 'new'))
            tracemalloc.start()
            r0 = old_finish_slice(os.path.join(d, 'old'), 'F_1', {'poly': 'F'}, iter(lines))
            old_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.reset_peak()
            r1 = sieve.finish_slice(os.path.join(d, 'new'), 'F_1', {'poly': 'F'}, iter(lines))
            new_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            self.assertEqual(r0, r1)
            self.assertEqual(r1[1], 60000)
            for e in ('.rels.gz', '.json'):
                rd = (lambda p: gzip.decompress(_read(p))) if e.endswith('.gz') else (lambda p: json.loads(_read(p)))
                self.assertEqual(rd(os.path.join(d, 'new', 'F_1' + e)), rd(os.path.join(d, 'old', 'F_1' + e)), e)
            self.assertEqual(sorted(os.listdir(os.path.join(d, 'new'))), ['F_1.json', 'F_1.manifest.json', 'F_1.rels.gz'])
            self.assertLess(new_peak, old_peak / 4, (new_peak, old_peak))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_failed_read(self):
        """A reader that fails part way: no committed slice, no temporary file left, the error raised."""
        d = tempfile.mkdtemp(prefix='o20')
        try:
            def lines():
                yield '1,2:3,5\n'
                raise OSError('reader died')
            self.assertRaises(OSError, sieve.finish_slice, d, 'F_1', {'poly': 'F'}, lines())
            self.assertEqual(os.listdir(d), [])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestManifest(unittest.TestCase):
    def flip(self, path):
        with open(path, 'r+b') as f:
            b = f.read(1)
            f.seek(0)
            f.write(bytes([b[0] ^ 1]))

    def test_solve_reuse(self):
        d = tempfile.mkdtemp(prefix='o27')
        try:
            R = random.Random(2)
            M = linalg.Matrix({'J': []}, [(i, 0) for i in range(12)])
            for t in range(60):
                M._add([(c, 0) if c < 12 else ('L', c) for c in R.sample(range(30), 4)], 'x', t + 1, 1)
            r0 = linalg.solve(M, d, 0, keep=-1, dense=0)
            with mock.patch.object(linalg, 'run_engine', wraps=linalg.run_engine) as run:
                self.assertEqual(linalg.solve(M, d, 0, keep=-1, dense=0).bound, r0.bound)
                self.assertEqual(run.call_count, 0)                  # reused
                for e in ('.ops', '.piv', '.vs'):
                    self.flip(os.path.join(d, 'elim' + e))           # same size, other bytes
                    self.assertEqual(linalg.solve(M, d, 0, keep=-1, dense=0).bound, r0.bound)
                self.assertEqual(run.call_count, 3)                  # each redone
            self.assertIn('digests', json.loads(_read(os.path.join(d, 'elim.json'))))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_kernel_reuse(self):
        d = tempfile.mkdtemp(prefix='o27')
        try:
            R = random.Random(4)
            rows = [sorted(R.sample(range(40), 3)) for _ in range(80)]
            k0 = linalg.kernel(rows, 40, 5, work=d)
            with mock.patch.object(linalg, 'run_engine', wraps=linalg.run_engine) as run:
                self.assertEqual(linalg.kernel(rows, 40, 5, work=d).vectors, k0.vectors)
                self.assertEqual(run.call_count, 0)
                self.flip(os.path.join(d, 'kernel.pool'))           # f2elim and f2kernel again
                self.assertEqual(linalg.kernel(rows, 40, 5, work=d).vectors, k0.vectors)
                self.assertEqual(run.call_count, 2)
                self.flip(os.path.join(d, 'kernel.vec1'))           # f2kernel again
                self.assertEqual(linalg.kernel(rows, 40, 5, work=d).vectors, k0.vectors)
                self.assertEqual(run.call_count, 3)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestThreads(unittest.TestCase):
    def test_same_vectors(self):
        """Random 8-entry rows: the vector file is the same on 1, 2, 5 threads, with the default work per thread and
        with F2KERNEL_PAR_WORK 0 (every product on all threads, as before) or 1."""
        d = tempfile.mkdtemp(prefix='o26')
        try:
            for n in (3000, 20000):
                R = random.Random(n)
                a = array.array('I', [n])
                for i in range(n):
                    a.extend([i, 8] + R.sample(range(n - 100), 8))
                pool = os.path.join(d, 'pool')
                with open(pool, 'wb') as f:
                    a.tofile(f)
                outs = set()
                for t in (1, 2, 5):
                    for w in (None, '0', '1'):
                        env = dict(os.environ, F2KERNEL_DENSE_N='0')
                        if w is not None:
                            env['F2KERNEL_PAR_WORK'] = w
                        v = os.path.join(d, 'v')
                        p = subprocess.run([linalg.KERNEL, pool, v, str(t), '1', '300'], capture_output=True, text=True,
                                           env=env)
                        self.assertEqual(p.returncode, 0, p.stderr)
                        outs.add((_read(v), p.stdout.split(' time ')[0]))
                self.assertEqual(len(outs), 1, n)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestField(unittest.TestCase):
    def test_table_and_norm(self):
        """On random forms: T equals the table pmul and ocoords find; omat's columns are omul by the basis; the norm
        equals the norm of the power-basis element (the resultant of f with it, by its Fraction multiplication
        matrix) for ints and Fractions."""
        R = random.Random(13)
        for _ in range(300):
            f = [R.randint(-50, 50) for _ in range(3)] + [R.choice([-7, -3, -1, 1, 2, 5, 12])]
            K = field.Field(f)
            B = K.B
            T = [[K.ocoords(field.pmul(K.f, B[i], B[j])) for j in range(3)] for i in range(3)]
            self.assertEqual(K.T, [[tuple(int(x) for x in v) for v in r] for r in T], f)
            for _ in range(5):
                x = tuple(R.choice([R.randint(-30, 30), Fraction(R.randint(-30, 30), R.randint(1, 9))]) for _ in range(3))
                cols = [K.omul(x, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
                self.assertEqual(K.omat(x), [[cols[j][i] for j in range(3)] for i in range(3)])
                y = K.power(x)          # N(y) = det of multiplication by y on 1, alpha, alpha^2
                c = [field.pmul(K.f, y, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
                (a, b, cc), (d, e, g), (h, i, j) = [[c[k][l] for k in range(3)] for l in range(3)]
                self.assertEqual(K.norm(x), a * (e * j - g * i) - b * (d * j - g * h) + cc * (d * i - e * h))
        self.assertRaises(ValueError, field.Field, [1, 2, 3, 0])


class TestTypes(unittest.TestCase):
    def test_bdf_shared(self):
        """bdf with a shared types dict: the same T, and the dict holds ftypes of every prime below it."""
        f = [-17, 3, 0, 1]
        dK = field.disc((f[3], f[2], f[1], f[0]))
        types = {}
        T = field.bdf(f, dK, 1, 10 ** 6, types)
        self.assertEqual(T, field.bdf(f, dK, 1, 10 ** 6))
        from rankbound import arith
        self.assertTrue(all(types[p] == field.ftypes(f, p) for p in arith.primes(2, T)))


if __name__ == '__main__':
    unittest.main()
