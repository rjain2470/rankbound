"""Release behaviour: the trial sieve samples production's special-q population; the checker's integer XOR of bit
strings; the witnesses.tsv lookup index; one curve analysis per request; and the operational workflows: the
curve-only call's result fields, the versioned --json schema on stdout only, preflight, the resume command, Ctrl-C,
timeout, child failure, full disk, interrupted publication, a received witness checked with --curve and a mixed
batch with --json."""
import contextlib, errno, io, json, os, random, shutil, subprocess, sys, tempfile, unittest
from unittest import mock
import rankbound
from rankbound import core, curve as cv, runner, runtime, sieve
from rankbound.check import selmer as csel
from helpers import PKG, SRC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIEVE = os.path.join(PKG, 'sieve')
HAVE_GP = shutil.which('gp') is not None
E25 = [0, 0, 0, -25, 0]                 # 2-torsion: the descent, rank <= 1 unconditionally, seconds
E1 = [0, 0, 0, -1, 0]                   # 2-torsion: rank 0


def tmpdir(t):
    d = tempfile.mkdtemp(prefix='rb_release')
    t.addCleanup(shutil.rmtree, d, True)
    return d


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** .5) + 1))


class TestSampling(unittest.TestCase):
    """X^3 - x - 1 on [8192, 16384): production sieves 877 pairs (q, r), 441 of them
    (50.3%) from primes with three roots; the old sample (the next prime after a random point, one root) had 25.5%
    from such primes.  The sample is now uniform over production's pairs, without repeats."""

    ARGS = [SIEVE, 'sieve', '-poly', '-1,-1,0,1', '-lim', '4096', '-lpb', '16', '-mfb', '16', '-N', '16',
            '-q0', '8192', '-q1', '16384']
    done = {}

    def sample(self, n, seed):
        if (n, seed) not in self.done:
            r = subprocess.run(self.ARGS + ['-sample', str(n), '-seed', str(seed)], capture_output=True, text=True,
                               check=True)
            m = sieve.SAMPLE.search(r.stdout)
            self.assertIsNotNone(m, r.stdout[-500:])
            self.done[n, seed] = list(map(int, m.groups())), sieve.stats_of(r.stdout)
        return self.done[n, seed]

    def test_population(self):
        P = [(q, sum((x ** 3 - x - 1) % q == 0 for x in range(q))) for q in range(8192, 16384) if is_prime(q)]
        pairs = sum(n for _, n in P)
        three = sum(n for _, n in P if n == 3)
        self.assertEqual((pairs, three), (877, 441))
        got, tot = [0, 0, 0], 0
        for seed in (1, 2, 3):
            (n, r1, r2, r3, rep, draws), g = self.sample(300, seed)
            self.assertEqual((n, r1 + r2 + r3), (300, 300))
            self.assertEqual(g['nq'], 300)              # the special-q sieved are the pairs sampled
            got = [got[0] + r1, got[1] + r2, got[2] + r3]
            tot += n
        frac = got[2] / tot
        self.assertTrue(0.44 < frac < 0.56, frac)       # production 0.503; the old sampler 0.255

    def test_yield(self):
        """The sample's relations per special-q are production's: the full range gives 1,089,515 relations from
        877 pairs (1242 per q); three samples of 300 gave 1240, 1227 and 1210 per q (1226 pooled, 1.3% below)."""
        r = subprocess.run(self.ARGS, capture_output=True, text=True, check=True)
        full = sieve.stats_of(r.stdout)
        self.assertEqual(full['nq'], 877)
        rels = nq = 0
        for seed in (1, 2, 3):
            _, g = self.sample(300, seed)
            rels, nq = rels + g['reports'], nq + g['nq']
        ratio = (rels / nq) / (full['reports'] / full['nq'])
        self.assertTrue(0.95 < ratio < 1.05, ratio)     # within 5%

    def test_small_range_without_repeats(self):
        """A range of 3 pairs ([8192, 8250) has primes 8209 8219 8221 8231 8233 8237 8243 with their roots):
        asked for 50, the sample is the whole population once."""
        pop = sum(sum((x ** 3 - x - 1) % q == 0 for x in range(q)) for q in range(8192, 8250) if is_prime(q))
        r = subprocess.run([SIEVE, 'sieve', '-poly', '-1,-1,0,1', '-lim', '4096', '-lpb', '16', '-mfb', '16', '-N', '16',
                            '-q0', '8192', '-q1', '8250', '-sample', '50', '-seed', '7'],
                           capture_output=True, text=True, check=True)
        n = int(sieve.SAMPLE.search(r.stdout).group(1))
        self.assertEqual(n, pop)


def setuptools_version():
    """The installed setuptools' (major, minor), or None."""
    r = subprocess.run([sys.executable, '-c', 'import setuptools; print(setuptools.__version__)'],
                       capture_output=True, text=True)
    try:
        return tuple(int(x) for x in r.stdout.strip().split('.')[:2])
    except ValueError:
        return None


class TestWheel(unittest.TestCase):
    """`pip wheel .` builds a platform wheel (py3-none-PLATFORM, not py3-none-any) that holds the compiled
    programs.  Offline with an installed setuptools >= 77; otherwise pip fetches one, and without a network the
    test is skipped."""

    def test_platform_wheel(self):
        import zipfile
        if subprocess.run([sys.executable, '-m', 'pip', '--version'], capture_output=True).returncode:
            self.skipTest('needs pip')
        offline = (setuptools_version() or (0,)) >= (77,)
        d = tmpdir(self)
        src = os.path.join(d, 'src')
        shutil.copytree(ROOT, src, symlinks=True, ignore=shutil.ignore_patterns(
            '.git', 'build', '*.egg-info', '__pycache__', 'rankbound_work', 'witnesses', 'logs', 'out'))
        cmd = [sys.executable, '-m', 'pip', 'wheel', src, '--no-deps', '-q', '--retries', '0', '--timeout', '10',
               '-w', os.path.join(d, 'wheel')]
        r = subprocess.run(cmd + (['--no-build-isolation'] if offline else []), capture_output=True, text=True)
        if r.returncode and not offline:
            self.skipTest('the wheel build needs setuptools >= 77 from the network: ' + r.stderr.strip()[-200:])
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        wheels = os.listdir(os.path.join(d, 'wheel'))
        self.assertEqual(len(wheels), 1, wheels)
        name = wheels[0]
        if name.startswith('UNKNOWN-') and not offline:     # pip's isolated build fell back to the old setuptools
            self.skipTest('the isolated build did not use setuptools >= 77: ' + name)
        self.assertTrue(name.startswith('rankbound-') and name.endswith('.whl'), name)
        self.assertNotIn('-none-any', name)
        self.assertIn('-py3-none-', name)
        with zipfile.ZipFile(os.path.join(d, 'wheel', name)) as z:
            held = set(z.namelist())
        for prog in ('rankbound/sieve', 'rankbound/f2elim', 'rankbound/f2kernel', 'rankbound/check/check'):
            self.assertIn(prog, held)


class TestSetup(unittest.TestCase):
    def setup_fn(self, name):
        """The function name of setup.py (which runs setup() when imported), and the modules it needs."""
        import ast
        src = open(os.path.join(ROOT, 'setup.py')).read()
        fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
        ns = {}
        exec('import os, platform, re, sys\n' + ast.get_source_segment(src, fn), ns)
        return ns[name]

    def test_machine_tag(self):
        """The wheel is tagged for the architecture its programs were compiled for, not universal2."""
        tag = self.setup_fn('machine_tag')
        self.assertEqual(tag('macosx_10_9_universal2', 'arm64'), 'macosx_11_0_arm64')
        self.assertEqual(tag('macosx_10_9_universal2', 'x86_64'), 'macosx_10_9_x86_64')
        self.assertEqual(tag('macosx_12_0_universal2', 'arm64'), 'macosx_12_0_arm64')
        self.assertEqual(tag('linux_x86_64'), 'linux_x86_64')

    def test_license(self):
        """pyproject.toml: the license as an SPDX string, with the setuptools that reads it."""
        with open(os.path.join(ROOT, 'pyproject.toml')) as f:
            t = f.read()
        self.assertIn('license = "GPL-2.0-or-later"', t)
        self.assertIn('requires = ["setuptools>=77"]', t)


class TestXor(unittest.TestCase):
    @staticmethod
    def old(strs):
        strs = list(strs)
        n = len(strs[0]) if strs else 0
        acc = [0] * n
        for s in strs:
            for i, ch in enumerate(s):
                acc[i] ^= ch == '1'
        return ''.join('1' if b else '0' for b in acc)

    def test_equal(self):
        rnd = random.Random(39)
        for w in (0, 1, 63, 64, 65, 1000):
            for k in (0, 1, 2, 7):
                L = [''.join(rnd.choice('01') for _ in range(w)) for _ in range(k)]
                self.assertEqual(csel._xor_bits(iter(L)), self.old(L), (w, k))
            self.assertEqual(csel._xor_bits(['0' * w] * 2), '0' * w)     # leading zeroes kept
        self.assertEqual(csel._xor_bits([]), '')

    def test_bits_only(self):
        """Classes are validated as bits before the XOR (int(s, 2) would take '0_1' or ' 01')."""
        for s in ('0_1', ' 01', '012', '+1'):
            self.assertIsNone(csel._BITS.fullmatch(s), s)
        self.assertIsNotNone(csel._BITS.fullmatch('0101'))


class TestIndex(unittest.TestCase):
    @staticmethod
    def brute(mm, path):
        with open(path) as f:
            head = f.readline().rstrip('\n').split('\t')
            rows = [dict(zip(head, l.rstrip('\n').split('\t'))) for l in f if l.strip()]
        key = ','.join(map(str, mm))
        hit = sorted((r for r in rows if r.get(cv.COL_MM) == key and r.get('file') and r.get(cv.COL_EQ) in cv.ROW_ORDER),
                     key=lambda r: (int(r[cv.COL_UB]), cv.ROW_ORDER[r[cv.COL_EQ]], r['name']))
        return dict(hit[0]) if hit else None

    def test_same_winners_copies_invalidation(self):
        with open(cv.WITNESSES) as f:
            mms = sorted({l.rstrip('\n').split('\t')[-1] for l in list(f)[1:]})
        for m in random.Random(11).sample(mms, 60) + ['1,2,3,4,5']:
            mm = [int(x) for x in m.split(',')]
            self.assertEqual(cv.table_row(mm), self.brute(mm, cv.WITNESSES), m)
        row = cv.table_row([0, -1, 1, 0, 0])
        row['rank upper bound'] = '99'                  # a caller's change is its copy's
        self.assertEqual(cv.table_row([0, -1, 1, 0, 0])['rank upper bound'], '0')
        d = tmpdir(self)
        p = os.path.join(d, 'idx.tsv')
        with open(cv.WITNESSES) as f:
            head = f.readline()
        line = lambda ub: '\t'.join(['x', '0', 'x.json', '0', ub, 'no', '', '', '[]', 'yes', '0,-1,1,0,0']) + '\n'
        with open(p, 'w') as f:
            f.write(head + line('5'))
        self.assertEqual(cv.table_row([0, -1, 1, 0, 0], p)['rank upper bound'], '5')
        q = p + '.new'                                  # replaced (another inode): read again
        with open(q, 'w') as f:
            f.write(head + line('3'))
        os.replace(q, p)
        self.assertEqual(cv.table_row([0, -1, 1, 0, 0], p)['rank upper bound'], '3')

    def test_table_hit_is_no_new_proof(self):
        res = rankbound.rankbound([0, -1, 1, 0, 0])
        j = res.to_json()
        self.assertEqual((res.known, res.origin, res.outcome, res.verification), (True, 'table', 'table', 'unavailable'))
        self.assertEqual((j['schema'], j['origin'], j['checked']), (core.RESULT_SCHEMA, 'table', None))


@unittest.skipUnless(HAVE_GP, 'needs PARI/GP')
class TestAnalysis(unittest.TestCase):
    def test_torsion_once(self):
        """#1501's profile computed the torsion twice (the triage and the descent): now once per request."""
        calls = []
        real = cv._torsion_group
        with mock.patch.object(cv, '_torsion_group', side_effect=lambda a: calls.append(list(a)) or real(a)):
            res = rankbound.rankbound(E25, compute=True, workdir=os.path.join(tmpdir(self), 'w'))
        self.assertEqual(res.verification, 'accepted')
        self.assertEqual(calls, [E25])
        # outside a request nothing is kept; a cached answer is a copy
        with cv.analysis():
            g = cv.torsion_group(E25)
            g[0].append(7)
            self.assertEqual(cv.torsion_group(E25)[0], [2, 2])


class CLI:
    def cli(self, *argv):
        """(exit status, stdout's lines as JSON, stderr) of `rankbound bound ARGV`; stdout must be JSON only."""
        from rankbound import cli
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = cli.main(['bound'] + list(argv))
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
                err.write(str(e.code))
        lines = [l for l in out.getvalue().splitlines() if l.strip()]
        js = [json.loads(l) for l in lines]             # a non-JSON line on stdout fails here
        for j in js:
            self.assertEqual(j['schema'], core.RESULT_SCHEMA)
        return rc, js, err.getvalue()


@unittest.skipUnless(HAVE_GP, 'needs PARI/GP')
class TestWorkflows(CLI, unittest.TestCase):
    def setUp(self):
        self.d = tmpdir(self)

    def test_fresh_proof_fields(self):
        """rankbound(E): the bound, its hypothesis, the verification, the witness; the lower bound apart."""
        res = rankbound.rankbound(E25, compute=True, workdir=os.path.join(self.d, 'w'))
        self.assertEqual((res.rank_bound, res.hypothesis, res.verification, res.lower_bound), (1, 'none', 'accepted', None))
        self.assertTrue(os.path.isfile(res.witness))
        rc, js, err = self.cli(','.join(map(str, E25)), '--compute', '--json', '--dir', os.path.join(self.d, 'c'))
        self.assertEqual((rc, len(js)), (0, 1))
        j = js[0]
        self.assertEqual((j['outcome'], j['rank_bound'], j['hypothesis'], j['verification'], j['curve']),
                         ('completed', 1, 'none', 'accepted', ['0', '0', '0', '-25', '0']))
        self.assertIn('Witness:', err)                  # progress on stderr only

    def test_received_witness_checked_with_curve(self):
        res = rankbound.rankbound(E25, compute=True, workdir=os.path.join(self.d, 'w'))
        got = os.path.join(self.d, 'received.witness.json')
        shutil.copyfile(res.witness, got)
        env = dict(os.environ, PYTHONPATH=SRC)
        ok = subprocess.run([sys.executable, '-m', 'rankbound', 'check', '--json', '--curve', '0,0,0,-25,0', got],
                            capture_output=True, text=True, env=env)
        self.assertEqual(json.loads(ok.stdout.splitlines()[-1])['verdict'], 'accepted', ok.stderr[-500:])
        bad = subprocess.run([sys.executable, '-m', 'rankbound', 'check', '--json', '--curve', '0,0,0,-1,0', got],
                             capture_output=True, text=True, env=env)
        self.assertNotEqual(json.loads(bad.stdout.splitlines()[-1])['verdict'], 'accepted')
        self.assertNotEqual(bad.returncode, 0)

    def test_timeout_prints_resume(self):
        w = os.path.join(self.d, 'c')
        argv = ['0,0,0,-25,0', '--compute', '--json', '--dir', w, '--max-hours', '0']
        rc, js, err = self.cli(*argv)
        self.assertEqual(rc, core.EXIT_STOPPED)
        self.assertEqual((js[-1]['outcome'], js[-1]['resume']), ('stopped', 'rankbound bound ' + ' '.join(argv)))
        self.assertIn('To resume: rankbound bound 0,0,0,-25,0', err)
        rc, js, _ = self.cli(*argv[:-2])
        self.assertEqual((rc, js[-1]['outcome'], js[-1]['verification']), (0, 'completed', 'accepted'))

    def test_ctrl_c_then_resume(self):
        from rankbound import descent
        w = os.path.join(self.d, 'w')
        with mock.patch.object(descent.Descent, 'prove', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                rankbound.rankbound(E25, compute=True, workdir=w)
        self.assertEqual(runtime._CHILDREN, set())
        self.assertTrue(rankbound.rankbound(E25, compute=True, workdir=w).checked)

    def test_child_failure(self):
        from rankbound import descent
        w = os.path.join(self.d, 'c')
        with mock.patch.object(descent.Descent, 'prove', side_effect=RuntimeError('gp failed (exit 3).')):
            rc, js, err = self.cli('0,0,0,-25,0', '--compute', '--json', '--dir', w)
        self.assertEqual((rc, js[-1]['outcome'], js[-1]['error']), (1, 'failed', 'gp failed (exit 3).'))
        rc, js, _ = self.cli('0,0,0,-25,0', '--compute', '--json', '--dir', w)
        self.assertEqual((rc, js[-1]['verification']), (0, 'accepted'))

    def test_full_disk(self):
        """ENOSPC while committing the proof: named with its remedy; what was saved is kept; the same command then
        finishes."""
        w = os.path.join(self.d, 'c')
        full = OSError(errno.ENOSPC, 'No space left on device', os.path.join(w, 'proof.json'))
        with mock.patch.object(runner, 'save_proof', side_effect=full):
            rc, js, err = self.cli('0,0,0,-25,0', '--compute', '--json', '--dir', w)
        self.assertEqual((rc, js[-1]['outcome']), (1, 'failed'))
        self.assertIn('No space left on the device', js[-1]['error'])
        self.assertIn('run the same command again', err)
        rc, js, _ = self.cli('0,0,0,-25,0', '--compute', '--json', '--dir', w)
        self.assertEqual((rc, js[-1]['verification']), (0, 'accepted'))

    def test_interrupted_publication(self):
        """Killed while publishing the verdict: the committed witness is checked again, not proved again."""
        from rankbound import descent
        w = os.path.join(self.d, 'w')
        with mock.patch.object(runtime.RunStore, 'publish', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                rankbound.rankbound(E25, compute=True, workdir=w)
        with mock.patch.object(descent.Descent, 'prove', side_effect=AssertionError('proved again')):
            res = rankbound.rankbound(E25, compute=True, workdir=w)
        self.assertEqual((res.verification, res.origin), ('accepted', 'resumed'))

    def test_mixed_batch_json(self):
        from rankbound import descent
        lst = os.path.join(self.d, 'curves.json')
        with open(lst, 'w') as f:
            json.dump({'curves': [{'id': 'good', 'ainvs': E25}, {'id': 'bad', 'ainvs': E1}]}, f)
        real = descent.Descent.prove

        def prove(self_, curve, *a, **k):
            if curve.name.endswith('bad'):
                raise RuntimeError('a child failed.')
            return real(self_, curve, *a, **k)
        with mock.patch.object(descent.Descent, 'prove', prove):
            rc, js, err = self.cli('--curves', lst, '--compute', '--json', '--dir', os.path.join(self.d, 'b'))
        self.assertEqual(rc, 1)
        by = {j['name']: j for j in js}                 # (a list's names: curve + id)
        self.assertEqual((by['curvegood']['outcome'], by['curvegood']['verification']), ('completed', 'accepted'))
        self.assertEqual((by['curvebad']['outcome'], by['curvebad']['error']), ('failed', 'a child failed.'))


class TestPreflight(CLI, unittest.TestCase):
    def test_missing_native(self):
        d = tmpdir(self)
        with mock.patch.object(runner, 'NATIVE', ('no_such_program',)), \
                mock.patch.object(cv, 'from_ainvs', side_effect=AssertionError('work before preflight')):
            with self.assertRaises(core.Unprepared) as e:
                rankbound.rankbound(E25, compute=True, workdir=os.path.join(d, 'w'))
            self.assertIn('make build', str(e.exception))
            rc, js, err = self.cli('0,0,0,-25,0', '--compute', '--json', '--dir', d)
        self.assertEqual((rc, js[-1]['outcome']), (1, 'failed'))
        self.assertIn('make build', js[-1]['error'])

    @unittest.skipIf(os.geteuid() == 0, 'root writes anywhere')
    def test_unwritable_folder(self):
        d = tmpdir(self)
        ro = os.path.join(d, 'ro')
        os.mkdir(ro, 0o500)
        self.addCleanup(os.chmod, ro, 0o700)
        with mock.patch.object(cv, 'from_ainvs', side_effect=AssertionError('work before preflight')):
            with self.assertRaises(core.Unprepared) as e:
                rankbound.rankbound(E25, compute=True, workdir=os.path.join(ro, 'w'))
        self.assertIn('Choose another with workdir=.', str(e.exception))      # the API's word

    def test_no_gp_said(self):
        d = tmpdir(self)
        with mock.patch.object(runner.shutil, 'which', return_value=None), self.assertLogs('rankbound', 'INFO') as lg:
            runner.preflight(d)
        self.assertIn('PARI/GP (gp) is not on the PATH', lg.output[-1])


if __name__ == '__main__':
    unittest.main()
