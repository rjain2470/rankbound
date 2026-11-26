"""The checker within its limits: check.gp's tdzp without recursion (up to the input limit); check rank whole or in
column blocks by default, chosen from the free memory and temporary disk (the CHECK_RANK_ variables override; exit 4
/ 'limit' when neither fits), with a process-tree memory test; check rank's threads by the work, with a guard."""
import array, json, os, random, subprocess, sys, tempfile, unittest
from rankbound import check
from rankbound.check import selmer as vsel
from helpers import SRC

# check.gp's earlier tdzp and tdz2 (recursive), the reference for the iterative tdzp: the same answers
OLD_TDZP = r'''
oldzp(H, p) =
{
  my(c = valuation(content(H), p), R);
  if(c >= 2, return(oldzp(H / p^(2 * (c \ 2)), p)));
  if(c == 1, R = tdroots(H / p, p),
    if(p == 2, return(oldz2(H)));
    if(tdsquare(H, p), return(1));
    R = tdroots(H, p));
  for(i = 1, #R, if(oldzp(subst(H, x, R[i] + p * x), p), return(1)));
  0;
}
oldz2(H) =
{
  for(s0 = 0, 7, my(h = subst(H, x, s0));
    if(h == 0, return(1));
    if(h % 2, if(h % 8 == 1, return(1)), if(oldzp(subst(H, x, s0 + 8 * x), 2), return(1))));
  0;
}
'''

COMPARE = r'''{
setrand(7); n = 0; bad = 0; ones = 0;
forprime(p = 2, 101, for(t = 1, %d,
  my(g = vector(5, i, (random(2001) - 1000) * p^random(4)), H);
  if(g[1] && g[5], H = Pol(g);
    if(poldisc(H), n++; my(a = tdzp(H, p), b = oldzp(H, p)); ones += a; if(a != b, bad++)))));
for(e = 1, 60, foreach([2, 3, 5], p, forvec(v = [[-3, 3], [-3, 3]],
  my(A = v[1], B = v[2] * p^e, d = p^(e %% 2) * (1 + 2 * (e %% 3)), g, a, b);
  if(B && A^2 != 4 * B, n++; g = [d^3, 0, A * d^2, 0, B * d]; a = tdquartic(g, p);
    b = oldzp(Pol(g), p) || oldzp(subst(Polrev(g), x, p * x), p); ones += a; if(a != b, bad++)))));
print(n, " ", ones, " ", bad);
}
'''


class TestDeepLocal(unittest.TestCase):
    """check.gp's tdzp recursed once per refinement level: tdqp(2, 3^e, 3, 3) (3 x^4 + 2 x^2 + 3^(e-1)
    after the content) failed with "deep recursion" from about e = 7300 (3,480 digits), inside MAX_DIGITS = 4000, so a
    valid witness within the input limits could not be checked.  Now an explicit stack: the same answers."""

    def test_input_limit(self):
        e = max(e for e in range(8000, 9000) if len(str(3 ** e)) <= check.MAX_DIGITS)       # 8380: 3,999 digits
        out = check.run_gp('print(tdqp(2, 3^%d, 3, 3), " ", tdqp(2, 3^%d, 3, 3), " ", tdqp(2, 3^7320, 3, 3));'
                           % (e, e - 1))
        self.assertEqual(out.split(), ['0' if e % 2 == 0 else '1', '1' if e % 2 == 0 else '0', '0'])

    def test_same_as_recursion(self):
        """Random separable quartics at p <= 101 and the family d^3 u^4 + A d^2 u^2 v^2 + B d v^4 at p = 2, 3, 5 with
        B up to p^60: tdzp equals the earlier recursive tdzp/tdz2 on every one."""
        trials = 40 if os.environ.get('RANKBOUND_TEST_FAST') else 300
        n, ones, bad = map(int, check.run_gp(OLD_TDZP + COMPARE % trials).split())
        self.assertGreater(n, 5000)
        self.assertTrue(0 < ones < n)
        self.assertEqual(bad, 0)


def matrix(n, ncols, per_row, seed):
    """n sparse rows of per_row random columns of ncols (few columns in one row: little peeling, nearly full rank)."""
    rnd = random.Random(seed)
    return [set(rnd.sample(range(ncols), per_row)) for _ in range(n)]


def rank_input(rows, ncols, nl):
    d = array.array('I', [len(rows), ncols, nl, 0, len(rows)])
    for r in rows:
        d.append(len(r)); d.extend(sorted(r))
    d.extend(range(len(rows)))
    return d.tobytes()


CLEAN = ('CHECK_RANK_BLOCK_MB', 'CHECK_RANK_BLOCK_WORDS', 'CHECK_RANK_TMP_MB', 'CHECK_RANK_MEM_MB', 'CHECK_RANK_TMPDIR')


class TestChoice(unittest.TestCase):
    """The choice in check rank (check.c, echelon) from CHECK_RANK_MEM_MB, and its overrides: the same stdout in
    every mode; the mode line; exit 4 when neither the memory nor the temporary budget holds it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.rows = matrix(6000, 6000, 40, 1)      # dense 6000 x 94 words: 4.3 MiB; sparse 1.1 MiB
        cls.inp = rank_input(cls.rows, 6000, 2000)

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmp)

    def run_rank(self, threads=1, **env):
        e = {k: v for k, v in os.environ.items() if k not in CLEAN}
        e.update(CHECK_RANK_TMPDIR=self.tmp, **{k: str(v) for k, v in env.items()})
        p = subprocess.run([vsel.CHECK, 'rank', str(threads)], input=self.inp, capture_output=True, env=e, timeout=300)
        self.assertEqual(os.listdir(self.tmp), [])
        return p, vsel.rank_mode(p.stderr)

    def test_choice(self):
        whole, m = self.run_rank()
        self.assertEqual(whole.returncode, 0)
        self.assertEqual((m['mode'], m['by'], m['block_words'], m['words']), ('whole', 'unmeasured', 94, 94))
        N, W, held = 6000, 94, m['held_mib'] * 2 ** 20
        self.assertGreater(held, 6000 * 40 * 4)
        need = lambda bw: held + 20 * N + 8 * N * bw + 3 * 2 ** 20 + 2 ** 16    # blocks of bw words (BASE 3 MiB, 1 thread 64 KiB)
        for mem_mb, mode in ((64, 'whole'), (9, 'whole'), (8, 'blocks'), (6, 'blocks'), (5, 'blocks')):
            p, m = self.run_rank(CHECK_RANK_MEM_MB=mem_mb)
            self.assertEqual((p.returncode, p.stdout), (0, whole.stdout), mem_mb)
            self.assertEqual((m['mode'], m['by'], m['memory_mib']), (mode, 'memory', mem_mb), mem_mb)
            bw = m['block_words']
            self.assertLessEqual(abs(need(bw) - need(W if mode == 'whole' else bw)), 0)
            self.assertLessEqual(need(bw), mem_mb * 2 ** 20 + 2 ** 16)       # within the memory (rounding of held)
            if mode == 'blocks':
                self.assertGreater(need(bw + 1), mem_mb * 2 ** 20 - 2 ** 16)  # the largest blocks that fit
                self.assertGreater(m['temp_mib'], 0)
        # not one word fits: exit 4, nothing decided
        p, m = self.run_rank(CHECK_RANK_MEM_MB=4)
        self.assertEqual((p.returncode, p.stdout, m), (4, b'', None))
        self.assertIn(b'does not fit: not enough memory', p.stderr)
        self.assertIn(b'the memory allowed (CHECK_RANK_MEM_MB) is 4.0 MiB', p.stderr)
        # a value that is not a whole number 1..2^40: exit 2, the variable and its value named (as the verdict's reason)
        for bad in ('abc', '0', '-5', '5x', str(2 ** 40 + 1)):
            p, m = self.run_rank(CHECK_RANK_MEM_MB=bad)
            self.assertEqual((p.returncode, p.stdout), (2, b''), bad)
            self.assertEqual(p.stderr.decode().splitlines()[-1], 'check rank: CHECK_RANK_MEM_MB is "%s", not a whole '
                             'number from 1 to 2^40; unset it or set it to one' % bad)
        # blocks, and the temporary file over its budget: exit 4
        p, m = self.run_rank(CHECK_RANK_MEM_MB=5, CHECK_RANK_TMP_MB=1)
        self.assertEqual((p.returncode, p.stdout, m), (4, b'', None))
        self.assertIn(b'does not fit: the temporary file', p.stderr)
        # the overrides come first: CHECK_RANK_BLOCK_WORDS, then CHECK_RANK_BLOCK_MB, over the memory's choice
        p, m = self.run_rank(CHECK_RANK_MEM_MB=64, CHECK_RANK_BLOCK_WORDS=10, CHECK_RANK_BLOCK_MB=1)
        self.assertEqual((p.stdout, m['mode'], m['by'], m['block_words']), (whole.stdout, 'blocks',
                                                                            'CHECK_RANK_BLOCK_WORDS', 10))
        p, m = self.run_rank(CHECK_RANK_MEM_MB=1, CHECK_RANK_BLOCK_MB=1)
        self.assertEqual((p.stdout, m['mode'], m['by'], m['block_words']), (whole.stdout, 'blocks',
                                                                            'CHECK_RANK_BLOCK_MB', 2 ** 20 // 8 // N))

    def test_rank_env(self):
        """The checker's side: CHECK_RANK_MEM_MB and CHECK_RANK_TMP_MB from the free memory and disk unless set, the
        folder passed on; rank_V raises LimitExceeded (a 'limit' verdict) on exit 4 and reports the mode."""
        e = vsel.rank_env({'TMPDIR': self.tmp})
        self.assertEqual(e['CHECK_RANK_TMPDIR'], self.tmp)
        self.assertGreater(int(e['CHECK_RANK_MEM_MB']), 100)
        self.assertGreater(int(e['CHECK_RANK_TMP_MB']), 0)
        e = vsel.rank_env({'CHECK_RANK_MEM_MB': '7', 'CHECK_RANK_TMP_MB': '9', 'CHECK_RANK_TMPDIR': self.tmp,
                           'TMPDIR': '/nonexistent'})
        self.assertEqual((e['CHECK_RANK_MEM_MB'], e['CHECK_RANK_TMP_MB'], e['CHECK_RANK_TMPDIR']), ('7', '9', self.tmp))
        self.assertNotIn('CHECK_RANK_TMP_MB', vsel.rank_env({'CHECK_RANK_TMPDIR': '/nonexistent/dir'}))
        names = ['%d:1' % (i + 2) for i in range(6000)]
        saved = {k: os.environ.pop(k) for k in CLEAN if k in os.environ}
        try:
            os.environ['CHECK_RANK_TMPDIR'] = self.tmp
            info = []
            got = vsel.rank_V(self.rows[:3000], names, [], list(range(3000)), 4000, 2, info=info)
            os.environ['CHECK_RANK_MEM_MB'] = '4'
            with self.assertRaises(check.LimitExceeded) as cm:
                vsel.rank_V(self.rows, names, [], list(range(6000)), 4000, 2, info=info)
            self.assertIn('does not fit', str(cm.exception))
            os.environ['CHECK_RANK_MEM_MB'] = 'abc'         # check rank's last line, not its progress, is the reason
            with self.assertRaisesRegex(RuntimeError, r'^check rank failed: check rank: CHECK_RANK_MEM_MB is "abc", '):
                vsel.rank_V(self.rows, names, [], list(range(6000)), 4000, 2)
        finally:
            for k in CLEAN:
                os.environ.pop(k, None)
            os.environ.update(saved)
        self.assertEqual(len(info), 1)
        self.assertEqual((info[0]['mode'], info[0]['by']), ('whole', 'memory'))
        self.assertEqual(got[2], 3000)

    def test_memory_free(self):
        """MemAvailable and the cgroups (v2 with a limit on a parent, v1), from a stand-in /proc and /sys."""
        def tree(files):
            root = tempfile.mkdtemp(dir=self.tmp + '/..')
            for path, text in files.items():
                os.makedirs(os.path.dirname(os.path.join(root, path)), exist_ok=True)
                with open(os.path.join(root, path), 'w') as f:
                    f.write(text)
            return root
        G = 2 ** 30
        mi = 'MemTotal: 100 kB\nMemAvailable:   %d kB\n' % (64 * G // 1024)
        v2 = tree({'proc/meminfo': mi, 'proc/self/cgroup': '0::/slurm/job_1/step_0\n',
                   'sys/fs/cgroup/slurm/job_1/memory.max': '%d\n' % (10 * G),
                   'sys/fs/cgroup/slurm/job_1/memory.current': '%d\n' % (7 * G),
                   'sys/fs/cgroup/slurm/job_1/memory.stat': 'anon 1\ninactive_file %d\n' % G,
                   'sys/fs/cgroup/slurm/job_1/step_0/memory.max': 'max\n',
                   'sys/fs/cgroup/slurm/job_1/step_0/memory.current': '%d\n' % G})
        self.assertEqual(vsel.memory_free(v2 + '/'), 4 * G)
        v1 = tree({'proc/meminfo': mi, 'proc/self/cgroup': '5:cpuacct,cpu:/x\n4:memory:/slurm/uid_1/job_2\n',
                   'sys/fs/cgroup/memory/slurm/uid_1/job_2/memory.limit_in_bytes': '%d\n' % (8 * G),
                   'sys/fs/cgroup/memory/slurm/uid_1/job_2/memory.usage_in_bytes': '%d\n' % (5 * G),
                   'sys/fs/cgroup/memory/slurm/uid_1/job_2/memory.stat': 'cache 9\ntotal_inactive_file 0\n',
                   'sys/fs/cgroup/memory/memory.limit_in_bytes': '9223372036854771712\n',
                   'sys/fs/cgroup/memory/memory.usage_in_bytes': '%d\n' % (50 * G)})
        self.assertEqual(vsel.memory_free(v1 + '/'), 3 * G)
        self.assertEqual(vsel.memory_free(tree({'proc/meminfo': mi}) + '/'), 64 * G)
        self.assertIsNone(vsel.memory_free(tree({'x': ''}) + '/'))
        self.assertGreater(vsel.memory_free(), 0)


# a fresh process per run, so that its children's peak is check rank's alone.  On Linux a child's ru_maxrss keeps the
# peak of the forked Python before its exec (40 MiB here), so there check rank's VmHWM (its own memory's high-water
# mark) is read from /proc while it runs, every 5 ms (the block is allocated at the start, the rows before it).
PEAK = r"""
import json, os, random, resource, subprocess, sys, threading, time
from rankbound.check import selmer as vsel
hwm = [0]
if sys.platform.startswith('linux'):
    run = subprocess.run
    def watched(cmd, **kw):
        stdin = kw.pop('input'); kw.pop('capture_output'); kw.pop('timeout', None)
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
        def poll():
            while p.poll() is None:
                try:
                    with open('/proc/%%d/status' %% p.pid) as f:
                        for line in f:
                            if line.startswith('VmHWM:'):
                                hwm[0] = max(hwm[0], int(line.split()[1]) * 1024)
                except (OSError, ValueError):
                    pass
                time.sleep(0.005)
        t = threading.Thread(target=poll); t.start()
        out, err = p.communicate(stdin); t.join()
        return subprocess.CompletedProcess(cmd, p.returncode, out, err)
    vsel.subprocess.run = watched
rnd = random.Random(5)
n, nc = %d, %d
rows = [set(rnd.sample(range(nc), 12)) for _ in range(n)]
info = []
got = vsel.rank_V(rows, ['%%d:1' %% (i + 2) for i in range(nc)], [], list(range(n)), nc // 3, %d, info=info)
k = 1 if sys.platform == 'darwin' else 1024
print(json.dumps([got, info, hwm[0] or resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * k,
                  resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * k]))
"""


@unittest.skipIf('ASAN_OPTIONS' in os.environ, 'AddressSanitizer changes the memory measured')
class TestPeak(unittest.TestCase):
    """The process tree's peak memory, check rank apart from Python: 16,000 x 16,000 (dense 250 words, 30.5 MiB)
    whole (the free memory) and in blocks (CHECK_RANK_MEM_MB = 10): the same counts; whole holds the dense matrix,
    the blocks stay within the 10 MiB allowed."""

    def peak(self, **env):
        e = {k: v for k, v in os.environ.items() if k not in CLEAN}
        e.update(PYTHONPATH=SRC, **env)
        with tempfile.TemporaryDirectory() as d:
            e['CHECK_RANK_TMPDIR'] = d
            p = subprocess.run([sys.executable, '-c', PEAK % (16000, 16000, 2)], capture_output=True, text=True,
                               env=e, timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        return json.loads(p.stdout)

    def test_peak(self):
        got, info, child, py = self.peak()
        got2, info2, child2, py2 = self.peak(CHECK_RANK_MEM_MB='10')
        MiB = 2 ** 20
        self.assertEqual(got, got2)
        self.assertEqual((info[0]['mode'], info[0]['dense_mib']), ('whole', 30.5))
        self.assertEqual((info2[0]['mode'], info2[0]['by'], info2[0]['memory_mib']), ('blocks', 'memory', 10.0))
        self.assertGreater(child, 30 * MiB)
        self.assertLess(child2, 10.5 * MiB)            # the 10 MiB allowed
        print('\ncheck rank peak: whole %.1f MiB, blocks %.1f MiB (Python %.1f, %.1f MiB)' %
              (child / MiB, child2 / MiB, py / MiB, py2 / MiB), file=sys.stderr)


class TestThreads(unittest.TestCase):
    """Guard: check rank's threads by the work (PAR_WORK = 2^17 table reads a thread), not all threads once
    rows x words >= 10^6 (the earlier PAR_MIN).  At 64 threads on 12,000 x 12,000 (188 words), the largest clearing
    (about 12,000 rows on 187 words, 2.6 x 10^6 units) gets about 20 threads, not 64; a 300-row matrix one."""

    def mode(self, n, threads):
        rows = matrix(n, n, 12, 3)
        p = subprocess.run([vsel.CHECK, 'rank', str(threads)], input=rank_input(rows, n, n // 3), capture_output=True,
                           env={k: v for k, v in os.environ.items() if k not in CLEAN}, timeout=300)
        self.assertEqual(p.returncode, 0)
        return p.stdout, vsel.rank_mode(p.stderr)

    def test_threads(self):
        out1, m1 = self.mode(12000, 1)
        out64, m = self.mode(12000, 64)
        self.assertEqual(out1, out64)
        self.assertEqual((m1['threads_max'], m1['threaded']), (1, 0))
        W = m['words']
        self.assertEqual(W, 188)
        self.assertLessEqual(m['threads_max'], (W - 1) * (12000 + 2048) // 2 ** 17)     # 20; before: 64
        self.assertGreaterEqual(m['threads_max'], 8)
        self.assertGreater(m['threaded'], 0)
        self.assertLess(m['threaded'], m['clearings'])     # the last words' small clearings on one thread
        _, small = self.mode(300, 64)
        self.assertEqual(small['threads_max'], 1)


if __name__ == '__main__':
    unittest.main()
