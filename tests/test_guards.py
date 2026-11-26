"""Estimates, provenance, speed guards and interrupts: honest estimates (the descent's cost is not estimated, and its
time is not mentioned; a list's total sums the known estimates only, never an unknown as 0), the provenance record of
a published result (versions, effective parameters, seeds, artifact ids), a guard per speedup that fails if it is
reverted (below), and real interrupts of a live run with its sieve processes: SIGINT (Ctrl-C) and SIGTERM leave no
process behind and the run resumes to the same checked bound; a full disk (RLIMIT_FSIZE, writes failing with EFBIG as
with ENOSPC) fails closed, with no result published, and the run resumes."""
import array, collections, contextlib, glob, hashlib, inspect, io, json, os, random, resource, shutil, signal
import subprocess, sys, tempfile, time, tracemalloc, unittest
from fractions import Fraction
from unittest import mock

import rankbound
from rankbound import cli, machine, runner, runtime
from rankbound import arith as A, curve as C, descent as D, field as fld, linalg, selmer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_math_speedups as M                       # the old implementations (the speedups' controls), kept there
from helpers import PKG, SRC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = [sys.executable, '-m', 'rankbound']
ENV = dict({k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}, PYTHONPATH=SRC)
E25 = [0, 0, 0, -25, 0]                         # rational 2-torsion: the descent, rank 0 (unconditional)
E571 = [0, -1, 1, -929, -10595]                 # 571a1: the sieve, rank 0 (GRH); a few seconds
N571 = '0_-1_1_-929_-10595'
HAS_GP = shutil.which('gp') is not None


def run_cli(*args, timeout=300, **kw):
    return subprocess.run(CLI + list(args), env=kw.pop('env', ENV), capture_output=True, text=True, timeout=timeout,
                          **kw)


# ---------- honest estimates

class TestEstimates(unittest.TestCase):
    def test_descent_unknown(self):
        """The descent's estimate is None (not 0, not 'seconds'), and its time is not mentioned at all."""
        with tempfile.TemporaryDirectory() as d:
            r = run_cli('bound', '--estimate', '0,0,0,-1,0', '--dir', d)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('has rational 2-torsion, so a 2-isogeny descent bounds its rank.\n', r.stdout)
        self.assertNotIn('seconds', r.stdout)
        self.assertNotIn('estimated', r.stdout)
        from rankbound.descent import Descent
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(Descent(d).estimate(None, quiet=True))

    def total(self, estimates):
        """The last line of `bound --estimate --curves` with these estimates (runner.each a stand-in)."""
        items = [runner.Item('c%d' % i, [0, 0, 0, 0, i], alone=False, estimate=e) for i, e in enumerate(estimates)]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'list.json')
            with open(path, 'w') as f:
                json.dump({'curves': [{'id': 1, 'ainvs': [0, 1]}]}, f)       # (read before runner.each)
            out = io.StringIO()
            with mock.patch.object(runner, 'each', lambda *x, **k: iter(items)), \
                    mock.patch.object(cli, 'say', lambda key, **kw: out.write(runtime.text(key, **kw) + '\n')):
                self.assertEqual(cli.main(['bound', '--estimate', '--curves', path, '--dir', d]), 0)
        return out.getvalue().strip().splitlines()[-1]

    def test_list_total(self):
        """A list's total is the sum of the known estimates (never an unknown as 0); an unknown time is not
        mentioned at all."""
        self.assertEqual(self.total([0.5, 1.0]), '2 curves, estimated 1.5 core-hours in all.')
        self.assertEqual(self.total([0.5, None, 1.0]), '3 curves: 2 estimated at 1.5 core-hours in all.')
        self.assertEqual(self.total([None, None]), '2 curves.')
        self.assertEqual(self.total([None] * 7 + [2.0]), '8 curves: 1 estimated at 2.0 core-hours in all.')


# ---------- the provenance record

@unittest.skipUnless(HAS_GP, 'needs PARI/GP')
class TestProvenance(unittest.TestCase):
    def test_allocated_cores(self):
        """More cores than the job is allocated (SLURM_CPUS_ON_NODE) are clamped to the allocation, said once, and the
        provenance records the cores used."""
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {'SLURM_CPUS_ON_NODE': '1'}), \
                self.assertLogs('rankbound', 'INFO') as logs:
            res = rankbound.rankbound(E25, compute=True, workdir=os.path.join(d, 'w'), cores=4)
            self.assertEqual(machine.allowed_cores(), 1)
        self.assertEqual(res.to_json()['provenance']['parameters']['cores'], 1)
        said = [r.getMessage() for r in logs.records if getattr(r, 'event', None) == 'cores_clamped']
        self.assertEqual(said, ['More cores were asked for (4) than this machine or job allows; it runs on 1.'])

    def test_default_cores(self):
        """Without cores (the API's default, the CLI without --cores): RANKBOUND_CORES, else the machine's cores
        (machine.detect, stubbed here), said first, before the work folder's line, and recorded with their source in
        the provenance; given cores are not said again; a RANKBOUND_CORES that is not a positive integer is a usage
        error."""
        env = {k: v for k, v in os.environ.items() if k != 'RANKBOUND_CORES' and not k.startswith('SLURM_')}
        for given, setting, want, said in (
                (None, None, (2, 'performance'), ['Using 2 cores (the performance cores of 8). --cores N or '
                                                  'RANKBOUND_CORES=N to change.']),
                (None, '1', (1, 'environment'), []), (1, '3', (1, 'option'), [])):
            with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(machine, 'detect', lambda env=None: machine.Cores(2, 'performance', 8)), \
                    self.assertLogs('rankbound', 'INFO') as logs:
                if setting is not None:
                    os.environ['RANKBOUND_CORES'] = setting
                res = rankbound.rankbound(E25, compute=True, workdir=os.path.join(d, 'w'), cores=given)
            q = res.to_json()['provenance']['parameters']
            self.assertEqual((q['cores'], q['cores_from']), want)
            lines = [r.getMessage() for r in logs.records]
            self.assertEqual([x for x in lines if x.startswith('Using')], said)
            first = lines.index(next(x for x in lines if x.startswith('Work folder')))
            self.assertEqual(lines[:first], said)
            self.assertIn(', %d core' % want[0], lines[first])
        for bad in ('0', 'x', '-2'):
            with tempfile.TemporaryDirectory() as d:
                with mock.patch.dict(os.environ, {'RANKBOUND_CORES': bad}), self.assertRaises(ValueError) as e:
                    rankbound.rankbound(E25, compute=True, workdir=os.path.join(d, 'w'))
                self.assertEqual(str(e.exception), 'RANKBOUND_CORES=%s is not a positive integer.' % bad)
                r = run_cli('bound', ','.join(map(str, E25)), '--compute', '--dir', d, env=dict(ENV, RANKBOUND_CORES=bad))
                self.assertEqual((r.returncode, r.stderr.strip()), (2, 'RANKBOUND_CORES=%s is not a positive integer.' % bad))
                self.assertEqual(os.listdir(d), [])         # refused before any work

    def test_record(self):
        """A computed result carries its provenance in run.json and --json (schema still rankbound-result-1): the
        versions (the native programs' sha256 as on disk, the checker's PARI/GP), the effective parameters (cores,
        the environment's RANKBOUND_* settings), the seeds, and each artifact's id (the proof's witness by the sha256
        the checker read).  A resume returns the same record; a table row has none."""
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {'RANKBOUND_PROVENANCE_TEST': '1'}):
            w = os.path.join(d, 'w')
            res = rankbound.rankbound(E25, compute=True, workdir=w, cores=2)
            j = res.to_json()
            self.assertEqual((j['schema'], j['verification']), ('rankbound-result-1', 'accepted'))
            p = j['provenance']
            self.assertEqual(p['format'], 'rankbound-provenance-1')
            v = p['versions']
            self.assertEqual(v['rankbound'], rankbound.__version__)
            self.assertRegex(v['gp'], r'^\d+\.\d+')
            self.assertEqual(v['checker'], res.details['checker'])
            for prog in runner.NATIVE:
                with open(os.path.join(PKG, prog), 'rb') as f:
                    self.assertEqual(v['native'][prog]['sha256'], hashlib.sha256(f.read()).hexdigest(), prog)
            self.assertEqual(len(v['prover_sha256']), 64)
            q = p['parameters']
            self.assertEqual((q['cores'], q['cores_from'], q['hypothesis'],
                              q['environment'].get('RANKBOUND_PROVENANCE_TEST')), (2, 'option', 'none', '1'))
            self.assertIn('f2kernel', p['seeds'])
            proof = p['artifacts']['proof']
            self.assertEqual(proof['schema'], runner.PROOF_SCHEMA)
            self.assertEqual(proof['files'][os.path.basename(res.witness)], res.witness_sha256)
            self.assertEqual(runtime.read_json(os.path.join(w, 'run.json'))['result']['provenance'], p)
            again = rankbound.rankbound(E25, compute=True, workdir=w, cores=2)
            self.assertEqual((again.origin, again.to_json()['provenance']), ('resumed', p))
        self.assertIsNone(runner.lookup(runner.model([0, -1, 1, 0, 0])[1]).to_json()['provenance'])


# ---------- real interrupts

def parents():
    """{pid: its parent's pid} of every process (Linux: /proc; else ps), or None if they cannot be listed here."""
    if os.path.isdir('/proc/self'):
        out = {}
        for d in os.listdir('/proc'):
            if d.isdigit():
                try:
                    with open('/proc/%s/stat' % d) as f:
                        out[int(d)] = int(f.read().rsplit(')', 1)[1].split()[1])
                except (OSError, IndexError, ValueError):
                    pass
        return out
    try:
        ps = subprocess.run(['ps', '-A', '-o', 'pid=,ppid='], capture_output=True, text=True).stdout
    except OSError:                 # a sandbox (macOS) that does not let ps run
        return None
    return {int(a): int(b) for a, b in (l.split() for l in ps.splitlines())}


def descendants(pid):
    """The pids of the live descendants of pid (children in their own sessions included)."""
    kids = {}
    for a, b in parents().items():
        kids.setdefault(b, []).append(a)
    todo, found = [pid], []
    while todo:
        for c in kids.get(todo.pop(), []):
            found.append(c)
            todo.append(c)
    return found


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open('/proc/%d/stat' % pid) as f:
            return f.read().rsplit(')', 1)[1].split()[0] != 'Z'        # a zombie is gone but for its parent's wait
    except FileNotFoundError:
        return False
    except OSError:
        st = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
        return bool(st) and not st.startswith('Z')


@unittest.skipUnless(HAS_GP, 'needs PARI/GP')
class TestInterrupts(unittest.TestCase):
    """571a1's folder after prepare and tune (setUpClass runs it once), copied for each test."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.res = rankbound.rankbound(E571, cores=1, compute=True, workdir=os.path.join(cls.tmp, 'ref', N571))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def fresh(self, name, keep=()):
        """A copy of the run's folder at tmp/name/N571 after prepare and tune (no slice, check or result unless
        kept); return it."""
        st = os.path.join(self.tmp, name, N571)
        drop = {'slices', 'la', 'history.json', 'history.manifest.json', 'STOP', 'check.json', 'check.log', 'selmer.json',
                'selmer.json.gz', 'selmer.manifest.json', 'proof.json', 'proof.manifest.json', 'checking.json'} - set(keep)
        shutil.copytree(self.res.workdir, st, ignore=lambda d, fs: [f for f in fs if f in drop or f.endswith(
            ('.witness.json', '.witness.json.gz', '.lock'))])
        p = os.path.join(st, 'run.json')
        runtime.write_json(p, dict(runtime.read_json(p), result=None, final=None, stage='not started'))
        return st

    def bound(self, B, *extra, **kw):
        return ['bound', ','.join(map(str, E571)), '--compute', '--json', '--dir', os.path.dirname(B)] + list(extra)

    def finish(self, B, why):
        """Resume the folder B to its end: the same checked bound as the reference run."""
        r = run_cli(*self.bound(B, '--cores', '2'), cwd=self.tmp)
        self.assertEqual(r.returncode, 0, why + ': ' + r.stdout + r.stderr)
        j = json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1])
        self.assertEqual((j['outcome'], j['verification'], j['rank_bound']), ('completed', 'accepted',
                                                                              self.res.rank_bound), why)
        self.assertEqual(vf_verdict(j['witness']), 'accepted')

    def test_signals(self):
        """SIGINT to the run's process group (Ctrl-C) and SIGTERM to the run (scancel), each while it sieves with
        2 sieve processes (stopped by SIGSTOP first, so that only the run can end them): it exits within seconds,
        every process it started is gone, no partial slice is left, no result is published; the same command then
        finishes with the reference's checked bound."""
        if parents() is None:
            self.skipTest('processes cannot be listed here')
        for sig in (signal.SIGINT, signal.SIGTERM):
            B = self.fresh('sig%d' % sig)
            prm_path = os.path.join(B, 'params.json')
            with open(prm_path) as f:
                saved = f.read()
            prm = json.loads(saved)
            prm['polys']['F']['sec_per_q'] = 1e3          # a slice is one special-q: a run that sieves for a while
            runtime.write_json(prm_path, prm)
            p = subprocess.Popen(CLI + self.bound(B, '--cores', '2'), cwd=self.tmp, env=ENV, start_new_session=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.addCleanup(lambda p=p: p.poll() is None and (p.kill(), p.communicate()))
            t = time.time()
            kids = []
            while p.poll() is None and time.time() - t < 60:
                if glob.glob(os.path.join(B, 'slices', '*.rels.gz')):
                    kids = descendants(p.pid)
                    if kids:
                        break
                time.sleep(0.05)
            self.assertIsNone(p.poll(), 'it exited before it sieved')
            self.assertTrue(kids, 'no live child to interrupt')
            for k in kids:                  # stopped, so that a child left behind cannot die of SIGPIPE at its next
                with contextlib.suppress(ProcessLookupError):       # write to the dead run: only the run's own
                    os.kill(k, signal.SIGSTOP)                       # cleanup (SIGTERM, then SIGKILL) ends it
            self.addCleanup(lambda kids=kids: [os.kill(k, signal.SIGKILL) for k in kids if alive(k)])
            if sig == signal.SIGINT:
                os.killpg(p.pid, sig)                       # Ctrl-C: the terminal's foreground group
            else:
                p.send_signal(sig)
            t = time.time()
            out = p.communicate(timeout=60)[0]
            self.assertLess(time.time() - t, 15, out)
            self.assertNotEqual(p.returncode, 0, out)
            # the control, SIGKILL in place of these signals, fails here (the stopped children are left behind)
            self.assertEqual([k for k in kids if alive(k)], [], 'processes left behind after %s' % sig)
            self.assertEqual(glob.glob(os.path.join(B, 'slices', '*.part')), [])
            self.assertIsNone(runtime.read_json(os.path.join(B, 'run.json')).get('result'))
            with open(prm_path, 'w') as f:
                f.write(saved)
            self.finish(B, 'resumed after signal %d' % sig)

    def test_full_disk(self):
        """Writes past RLIMIT_FSIZE fail (EFBIG, SIGXFSZ ignored: as a full disk's ENOSPC) in the run and in every
        process it starts.  At 200 KB the run fails in its sieve, at 600 KB in its check (the matrix): each time it
        exits 1 with an error, publishes no result and leaves every artifact valid or absent (never damaged); the
        same command without the limit then finishes with the reference's checked bound."""
        B = self.fresh('disk')
        for kb in (200, 600):
            def limit(kb=kb):
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
                resource.setrlimit(resource.RLIMIT_FSIZE, (kb << 10, kb << 10))
            r = run_cli(*self.bound(B), cwd=self.tmp, preexec_fn=limit)
            self.assertEqual(r.returncode, 1, '%d KB: %s' % (kb, r.stdout + r.stderr))
            j = json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1])
            self.assertEqual(j['outcome'], 'failed', j)
            self.assertIsNone(runtime.read_json(os.path.join(B, 'run.json')).get('result'))
            for m in glob.glob(os.path.join(B, '**', '*.manifest.json'), recursive=True):
                name = os.path.basename(m)[:-len('.manifest.json')]
                self.assertIn(runtime.inspect(os.path.dirname(m), name)[0], ('valid', 'stale'), (kb, m))
        self.assertTrue(glob.glob(os.path.join(B, 'slices', '*.rels.gz')))    # the 600 KB run got past the sieve
        self.finish(B, 'resumed after a full disk')


def vf_verdict(path):
    from rankbound import check
    return check.check(path)['verdict']


# ---------- guards for the speedups that had only equivalence tests:
# each test counts the operations the optimization saves (or times it against the behaviour before it) and fails if
# the change is reverted.  Each also runs the control: the old code (test_math_speedups's old_* functions, a
# pre-optimization copy of the function's source, or the old setting) under the same counter, and asserts that it
# would fail the guard.
# Not guarded: Horner in _sign_at: measured 1.28x on large inputs (real_roots_floor to 3000 bits), within timing
# noise, and it saves no countable call.


def counting(obj, name, log, key=lambda *a: a):
    """Patch obj.name with a wrapper that appends key(*args) to log."""
    orig = getattr(obj, name)

    def w(*a, **k):
        log.append(key(*a))
        return orig(*a, **k)
    return mock.patch.object(obj, name, w)


def reverted(fn, old, new, module):
    """fn's source with old replaced by new, compiled in a copy of module's namespace (the pre-optimization code)."""
    src = inspect.getsource(fn)
    assert old in src, (fn.__name__, old)
    ns = dict(vars(module))
    exec(src.replace(old, new), ns)
    return ns


class TestArithGuards(unittest.TestCase):
    def test_hensel_doubling(self):
        """A simple root lifted by Newton with doubling moduli reads about 4 full-precision moduli in all (the
        sum of the moduli's bits over peval calls); the old fixed-modulus Newton reads one per step (48 here)."""
        k = 2000
        full = (7 ** k).bit_length()

        def bits(fn):
            log = []
            with counting(A, 'peval', log, lambda f, x, m=None: m.bit_length() if m is not None and m > 7 else 0):
                fn([-2, 0, 1], 7, k)
            return sum(log) / full
        self.assertLess(bits(A.padic_roots), 16)            # measured 8.1
        self.assertGreater(bits(M.old_padic_roots), 16)     # control: 48

    def test_ecm_table(self):
        """ECM's stage 1 builds its prime table only to B1 (stage 2 has its own wheel), not to stage 2's 50 B1."""
        n = A.next_prime(10 ** 10) * A.next_prime(10 ** 12)
        his, primes = [], A.primes

        def run(ns):
            his.clear()
            with mock.patch.dict(ns, {'primes': lambda lo, hi: his.append(hi) or primes(lo, hi)}):
                A.random.seed(1)
                ns['_ecm'](n, 2000, 2)
            return max(his)
        self.assertLessEqual(run(vars(A)), 2001)
        old = reverted(A._ecm, 'primes(2, B1 + 1)', 'primes(2, 50 * B1 + 1)', A)
        self.assertGreater(run(old), 2001)                  # control: 100001

    def test_fraction_free(self):
        """Solve3 is Cramer's rule on an integer system (4 det3, no Fraction elimination); _inverse is one det3
        and the cofactors (no solve3)."""
        rng = random.Random(1)
        Mx = [[M.rfrac(rng) for _ in range(3)] for _ in range(3)]
        dets, solves = [], []
        with counting(A, 'det3', dets):
            A.solve3(Mx, [1, 2, 3])
        self.assertEqual(len(dets), 4)
        dets.clear()
        with counting(A, 'det3', dets):
            M.old_solve3(Mx, [1, 2, 3])
        self.assertEqual(len(dets), 0)                      # control: the elimination needs none
        with counting(A, 'solve3', solves):
            A._inverse(Mx)
        self.assertEqual(len(solves), 0)                    # the old _inverse: three solves

    def test_prime_exponents(self):
        """_split tries exact k-th roots for prime k only (62 primes below 310 bits, not 308 exponents);
        power_base extracts prime roots (78 calls for a 400-bit non-power, where the old one tried 399 exponents)."""
        n = 1009 * A.next_prime(1 << 300)
        log = []
        with counting(A, 'iroot', log):
            self.assertIn(A._split(n, 0), (1009, n // 1009))
        self.assertLessEqual(len(log), len(A.primes(2, n.bit_length())) + 1)
        self.assertLess(len(log), (n.bit_length() - 2) // 2)  # control: every exponent, 308
        m = A.next_prime(1 << 399) * 3
        for fn, ok in ((C.power_base, True), (M.old_power_base, False)):
            log = []
            with counting(C, 'root', log):
                self.assertEqual(fn(m), (m, 1))
            self.assertEqual(len(log) <= 100, ok, (fn.__name__, len(log)))     # measured 78 and 400


class TestCurveGuards(unittest.TestCase):
    def test_rkproof(self):
        """ICARM #8 (r = 22) with its 22 primes: the cubic's root once per prime (22 roots_mod, not one per point
        and prime: 484), and the primes below pmax never sieved (the old code sieved to pmax = 10^6 first)."""
        c = M.REF['icarm8']
        rl = c['rank_lower_bound']
        a, P = C._ainvs5(c['curve']['ainvs']), [int(p) for p in c['curve']['bad_primes']]
        pts = [[Fraction(x) for x in p] for p in rl['points']]
        PR = [int(p) for p in rl['primes']]
        for fn, ok in ((C.rkproof, True), (M.old_rkproof, False)):
            roots, his = [], []
            with counting(A, 'roots_mod', roots), counting(A, 'primes', his, lambda lo, hi: hi):
                fn(a, a, P, 22, pts, PR, 10 ** 6)
            self.assertEqual(len(roots) <= 2 * len(PR) and max(his or [0]) < 10 ** 6, ok,
                             (fn.__name__, len(roots), his))


class TestDescentGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):                            # char_primes' calls in three real descents
        cls.calls = []
        orig = D.char_primes
        with mock.patch.object(D, 'char_primes', lambda *a: cls.calls.append(a) or orig(*a)):
            for e, P, pr in M.curves(3):
                D.descent(e, P, proof=pr)

    def test_short_precision(self):
        """Lclass works at 4 (6 at 2) relative digits and goes to 100 (99) only on a cancellation: in three
        descents 5 of 78 lclass square roots were at full precision (the old lclass: all 78)."""
        def shares(lclass):
            rels = []
            osq = D.sqrt_padic
            with mock.patch.object(D, 'sqrt_padic', lambda w, p, rel: rels.append(rel) or osq(w, p, rel)), \
                    mock.patch.object(D, 'lclass', lclass):
                for e, P, pr in M.curves(3):
                    D.descent(e, P, proof=pr)
            n = collections.Counter(rels)
            return (n[99] + n[100]) / max(1, n[4] + n[6] + n[99] + n[100])
        self.assertLess(shares(D.lclass), 0.25)             # measured 0.06
        self.assertGreater(shares(M.old_lclass), 0.25)      # control: 1.0

    def test_lazy_primes(self):
        """Char_primes sieves the primes below pmax only as far as it reads them (to ~1,000 here, not
        pmax = 10^5), and grows an echelon basis (no rank2 recomputed per prime)."""
        self.assertTrue(self.calls)
        for mm, T, R, PR, pmax in self.calls:
            for fn, ok in ((D.char_primes, True), (M.old_char_primes, False)):
                his, rk = [], []
                with counting(A, 'primes', his, lambda lo, hi: hi), counting(A, 'rank2', rk):
                    self.assertIsNotNone(fn(mm, T, R, [], pmax))
                self.assertEqual(max(his) <= 1 << 16 and not rk, ok, (fn.__name__, max(his), len(rk)))


class TestFieldGuards(unittest.TestCase):
    def test_types_shared(self):
        """Field_data computes each prime's splitting type once, shared by the BDF search, the small primes,
        Brumer-Kramer and the ramified primes, and builds no Field.dec prime ideals (389a1: 7 primes, each once;
        with bdf's own types the small primes are computed twice)."""
        ob = fld.bdf
        for bdf, ok in ((ob, True), (lambda f, dK, r1, bach, types=None: ob(f, dK, r1, bach), False)):
            ft, dec = [], []
            with counting(fld, 'ftypes', ft, lambda f, p: p), counting(fld.Field, 'dec', dec), \
                    mock.patch.object(fld, 'bdf', bdf):
                fld.field_data([389], [0, 1, 1, -2, 0])
            self.assertEqual(max(collections.Counter(ft).values()) == 1 and not dec, ok)

    def test_fixed_table(self):
        """Field's multiplication table, omat and norm are fixed formulas: no pmul, ocoords or omul (the old
        table took 9 of each pmul and ocoords, in Fractions)."""
        pm, oc, om = [], [], []
        with counting(fld, 'pmul', pm), counting(fld.Field, 'ocoords', oc), counting(fld.Field, 'omul', om):
            K = fld.Field([-17, 3, 0, 5])
            K.omat((1, 2, 3))
            K.norm((1, Fraction(1, 2), 3))
            self.assertEqual((len(pm), len(oc), len(om)), (0, 0, 0))
            [[K.ocoords(fld.pmul(K.f, K.B[i], K.B[j])) for j in range(3)] for i in range(3)]   # control: the old table
        self.assertEqual((len(pm), len(oc)), (9, 9))


class TestSelmerGuards(unittest.TestCase):
    def test_theta_reused(self):
        """Selmer.theta with the curve takes field.field_of's exact theta: no numerical roots (without it, the
        embeddings: two A.roots calls)."""
        for a, P in [([0, 1, 1, -2, 0], [389]), ([0, 0, 1, -7, 6], [5077])]:
            mm, _ = fld.check_bad_primes(a, P)
            f, _ = fld.field_of(mm, [2, 3] + P)
            b2, b4, b6, _ = C.binv(mm)
            W = [16 * b6, 8 * b4, b2, 1]
            k = selmer.kfield(f, P)
            r1, r2 = [], []
            with counting(A, 'roots', r1):
                selmer.theta(k, W, mm, P)
            with counting(A, 'roots', r2):
                selmer.theta(k, W)
            self.assertEqual((len(r1), len(r2)), (0, 2))

    def test_classes_once(self):
        """Selmer_part XORs each element's local classes once (3 XORs an element: characters, classes, norms),
        not once per place group of S (the old loop: 2 + #S an element).  389a1 under the GRH, with S = {2, 389}."""
        old = reverted(selmer.selmer_part, '(ecls[i] >> o)', '(xor(cls, els[i]) >> o)', selmer)['selmer_part']
        for part, ok in ((selmer.selmer_part, True), (old, False)):
            caps, xs = [], []

            def sp(d, rels, els, chars, F, S, *r, **kw):
                n0 = len(xs)
                out = part(d, rels, els, chars, F, S, *r, **kw)
                caps.append((len(els), len(S), len(xs) - n0))
                return out
            with counting(selmer, 'xor', xs), mock.patch.object(selmer, 'selmer_part', sp), \
                    mock.patch.object(selmer, 'REDUCE_S', False), \
                    mock.patch.dict(old.__globals__, {'xor': lambda *a: selmer.xor(*a)}):   # the copy's xor counted
                selmer.prove(C.from_ainvs([0, 1, 1, -2, 0], name='g28'), hypothesis='GRH', log=lambda *a: None)
            v, s, n = caps[-1]
            self.assertGreaterEqual(s, 2)
            self.assertEqual(n <= 3 * v, ok, (v, s, n))


class TestLinalgGuards(unittest.TestCase):
    def test_packed_rows(self):
        """Linalg.Rows holds a 6-entry row in about 38 bytes (12 + 4 an entry, packed); a list of arrays takes
        about 113."""
        R = random.Random(1)
        ref = [[R.randrange(1 << 20) for _ in range(6)] for _ in range(20000)]
        tracemalloc.start()
        try:
            b0 = tracemalloc.get_traced_memory()[0]
            rows = linalg.Rows()
            for r in ref:
                rows.append(r)
            b1 = tracemalloc.get_traced_memory()[0]
            old = [array.array('I', r) for r in ref]
            b2 = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
        self.assertEqual(len(rows), len(old))
        self.assertLess((b1 - b0) / len(ref), 60)
        self.assertGreater((b2 - b1) / len(ref), 60)        # control

    def test_kernel_threads(self):
        """F2kernel on a 10,000-row pool with 64 threads starts threads by the work (laptop: 0.07 s), where one
        thread per product on all 64 (F2KERNEL_PAR_WORK=0, as before) takes 1.4 s; the same vectors."""
        d = tempfile.mkdtemp(prefix='g26')
        try:
            n = 10000
            R = random.Random(n)
            a = array.array('I', [n])
            for i in range(n):
                a.extend([i, 8] + R.sample(range(n - 100), 8))
            pool = os.path.join(d, 'pool')
            with open(pool, 'wb') as f:
                a.tofile(f)

            def run(work):
                env = dict(os.environ, F2KERNEL_DENSE_N='0')
                env.pop('F2KERNEL_PAR_WORK', None)
                if work is not None:
                    env['F2KERNEL_PAR_WORK'] = work
                best, out = None, None
                for _ in range(3):
                    t = time.perf_counter()
                    p = subprocess.run([linalg.KERNEL, pool, os.path.join(d, 'v'), '64', '1', '300'],
                                       capture_output=True, env=env)
                    best = min(best or 1e9, time.perf_counter() - t)
                    self.assertEqual(p.returncode, 0, p.stderr)
                    with open(os.path.join(d, 'v'), 'rb') as f:
                        out = f.read()
                return best, out
            (t_new, v_new), (t_old, v_old) = run(None), run('0')
            self.assertEqual(v_new, v_old)
            self.assertLess(t_new, 0.25 * t_old, (t_new, t_old))   # measured 0.07 against 1.43 s
        finally:
            shutil.rmtree(d, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()
