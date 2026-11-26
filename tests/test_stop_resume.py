"""Stopping and resuming a request: the request budget in CPU time, stops by signals, the slices in progress at a
stop, sievers that die with their parent, a work folder's relations shared by requests that differ only in r, a
shard's end, and the count a pending check names."""
import io, json, os, shutil, signal, subprocess, sys, tempfile, time, unittest
from contextlib import redirect_stdout

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
from rankbound import cli, runtime, sieve as jobs  # noqa: E402
from rankbound.messages import text  # noqa: E402
from rankbound.sieve import State  # noqa: E402
from test_stopping import E571, N571, PRM, R, FakeRun  # noqa: E402

BUSY = 'import time\nt = time.process_time()\nwhile time.process_time() - t < %s: pass'


def tmpdir(test):
    d = tempfile.mkdtemp(prefix='rb-fix-')
    test.addCleanup(shutil.rmtree, d, True)
    return d


def wait_file(path, seconds=60):
    t = time.monotonic()
    while not os.path.exists(path):
        if time.monotonic() - t > seconds:
            raise AssertionError('no %s after %d s' % (path, seconds))
        time.sleep(0.02)
    with open(path) as f:
        return f.read()


def gone(pid, seconds=10):
    """Whether the process pid has ended within seconds (runtime.terminate's grace is 5 s)."""
    t = time.monotonic()
    while time.monotonic() - t < seconds:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    return False


class TestBudgetCPU(unittest.TestCase):
    def test_meter_charges_cpu(self):
        """The Meter charges the CPU time of this process and its children (the unit of the sieve's estimate), not
        cores x wall time: an idle second costs nothing; a child's CPU counts while it runs (the 10-minute writes)
        and once it is reaped, once."""
        st = runtime.RunStore(tmpdir(self))
        st.begin({'k': 1})
        m = runtime.Meter(st)
        time.sleep(0.3)
        self.assertLess(m.core_hours() * 3600, 0.1)                 # idle: before, 0.3 s x the cores
        p = runtime.start([sys.executable, '-c', BUSY % 1.0])
        time.sleep(0.8)
        live = m.core_hours() * 3600
        p.wait()
        runtime.reap(p)
        m.close()
        sp = st.spent()
        self.assertGreater(live, 0.3, live)                         # counted while it runs
        self.assertTrue(0.9 < sp['core_hours'] * 3600 < 2.0, sp)    # its second, once
        self.assertGreater(sp['wall_s'], 1.0)
        self.assertEqual(sp['runs'], 1)


class TestFirstCheck(FakeRun, unittest.TestCase):
    def test_budget_waits_for_first_check(self):
        """A request budget spent before the run (here at its start) ends the run only after its first check has
        finished; the stop does not say the sieve is
        finished."""
        st = runtime.RunStore(self.d)
        st.begin({'k': 1})
        st.set_budget(0.0, 1.0)
        st.add_spent(core_hours=2.0)
        self.nxt = 2 * R
        meter = runtime.Meter(st)
        try:
            with self.assertLogs('rankbound', 'INFO') as out:
                oc = jobs.sieve_run(self.d, jobs=1, budget=runtime.Budget(seconds=60), meter=meter)[0]
        finally:
            meter.close()
        self.assertEqual((oc, self.checks), ('budget', [(2 * R, False)]))
        self.assertFalse(any('Sieving is finished' in l for l in out.output), out.output)
        self.assertTrue(any(l.startswith('INFO:rankbound:Sieving stopped with ') for l in out.output), out.output)


CHECKER = r'''
import os, sys, time
sys.path.insert(0, %r)
from rankbound.runner import bound
d, pidf = sys.argv[1], sys.argv[2]
checker = [sys.executable, '-c', 'import os, time\nopen(%%r, "w").write(str(os.getpid()))\nos.replace(%%r, %%r)\n'
           'time.sleep(60)' %% (pidf + '.tmp', pidf + '.tmp', pidf)]
try:
    bound([0, -1, 1, -929, -10595], compute=True, workdir=d, checker=checker)
except KeyboardInterrupt as e:
    sys.exit(128 + getattr(e, 'signum', 2))
''' % SRC


class TestSignals(unittest.TestCase):
    """A signal to rankbound while the checker runs (in its own session: the signal does not reach it), and the CLI's
    words after one (tests/test_hpc_workflow.py: a signal while a program of runtime.run runs)."""

    def test_sigterm_during_the_checker(self):
        """SIGTERM while the checker runs: rankbound ends after its cleanup (exit 128 + 15), the checker is stopped,
        the private copy of the witness (.verify.*) removed, and run.json's spent written."""
        d = tmpdir(self)
        w, pidf = os.path.join(d, 'w'), os.path.join(d, 'checker.pid')
        p = subprocess.Popen([sys.executable, '-c', CHECKER, w, pidf], stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, text=True)
        try:
            kid = int(wait_file(pidf, 300))
        except AssertionError:
            p.kill()
            raise AssertionError(p.communicate()[1][-2000:])
        os.kill(p.pid, signal.SIGTERM)
        err = p.communicate(timeout=60)[1]
        self.assertEqual(p.returncode, 128 + signal.SIGTERM, err[-2000:])
        self.assertTrue(gone(kid))
        self.assertEqual([f for f in os.listdir(w) if f.startswith('.verify.')], [])
        with open(os.path.join(w, 'run.json')) as f:
            self.assertEqual(json.load(f)['spent']['runs'], 1)

    def test_cli_says_how_to_resume(self):
        """Ctrl-C or SIGTERM in `rankbound bound`: no traceback, the resume line and the stop's message; the exit
        status 128 + the signal."""
        real = cli.cmd_bound
        self.addCleanup(setattr, cli, 'cmd_bound', real)
        d = tmpdir(self)
        for exc, rc in ((KeyboardInterrupt(), 130), (runtime.Interrupted(), 143)):
            def interrupted(a):
                raise exc
            cli.cmd_bound = interrupted
            with self.assertLogs('rankbound', 'INFO') as out, redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as e:
                    cli.main(['bound', '0,-1,1,-929,-10595', '--dir', d])
            self.assertEqual(e.exception.code, rc)
            self.assertEqual([l.split(':', 2)[2] for l in out.output],
                             ['To resume: rankbound bound 0,-1,1,-929,-10595 --dir ' + d, text('interrupted')])

    def test_sigterm_raises_once(self):
        """runtime.interruptible: SIGTERM raises Interrupted (a KeyboardInterrupt) once; a second one during the
        cleanup is ignored; outside, the old handler is back."""
        old = signal.getsignal(signal.SIGTERM)
        seen = []
        with self.assertRaises(runtime.Interrupted):
            with runtime.interruptible():
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                    time.sleep(1)
                finally:
                    os.kill(os.getpid(), signal.SIGTERM)      # ignored
                    time.sleep(0.05)
                    seen.append(True)
        self.assertEqual((seen, signal.getsignal(signal.SIGTERM)), ([True], old))


class TestSlicesKept(FakeRun, unittest.TestCase):
    """The slices in progress at a stop that leaves the sieve unfinished are committed with what they have."""

    def setUp(self):
        super().setUp()
        self.delay = 3600.0                     # slices that never end by themselves ...
        stops = self.stops = []
        Slice = jobs.siever(PRM).start_slice(None, None, None, None, None).__class__

        def stop(h):                            # ... but answer a stop at once, at their next special-q
            stops.append(h)
            h.t = -1e9
        Slice.stop = stop
        Slice.end = lambda h, q1: q1 - 1

    def test_signal(self):
        """SIGTERM or Ctrl-C while slices are in progress: each is committed (as the special-q it covered) and the
        interruption goes on; before, they were killed and their work lost."""
        real = jobs.siever
        calls = []

        def siever(prm):
            sv = real(prm)
            def wait():
                calls.append(1)
                if len(calls) > 3:
                    raise runtime.Interrupted()
                time.sleep(0.01)
            sv.wait = wait
            return sv
        jobs.siever = siever
        with self.assertRaises(KeyboardInterrupt):
            jobs.sieve_run(self.d, jobs=3, budget=runtime.Budget(seconds=60))
        S = State(self.d)
        self.assertEqual((len(self.stops), len(S.finished()), S.slice_stats()[1]), (3, 3, 3 * R))
        sched = {jobs.slice_name(p, q0): q1 for _, p, q0, q1 in jobs.schedule(PRM)}
        for n in S.finished():
            with open(os.path.join(S.sdir, n + '.json')) as f:
                self.assertEqual(json.load(f)['q1'], sched[n] - 1)

    def test_time_stop(self):
        """The run's end while slices are in progress: committed likewise."""
        with self.assertLogs('rankbound', 'INFO'):
            self.assertEqual(jobs.sieve_run(self.d, jobs=2, budget=runtime.Budget(seconds=0.3))[0], None)
        self.assertEqual(len(State(self.d).finished()), 2)


class TestSiever(unittest.TestCase):
    CMD = [jobs.BIN, 'sieve', '-poly', '-1,-1,0,1', '-lim', '65536', '-lpb', '20', '-mfb', '40', '-N', '16', '-server']

    def test_stop_a_slice(self):
        """The siever's server mode: SIGUSR1 ends the slice in progress before its next special-q, with its relations
        so far, '# stopped before special-q Q', the totals and '# done'; Slice.end reads Q."""
        sv = jobs.LSieve()
        srv = jobs.Server(self.CMD, 'F', sv.wake)
        self.addCleanup(srv.close)
        h = srv.sieve(200000, 50000000)
        t = time.monotonic()
        while not any(l[:1] != b'#' for l in h.out) and time.monotonic() - t < 60:     # its first relation
            time.sleep(0.02)
        h.stop()
        t = time.monotonic()
        while h.poll() is None and time.monotonic() - t < 30:
            time.sleep(0.02)
        self.assertEqual(h.poll(), 0)
        q = h.end(50000000)
        self.assertTrue(200000 < q < 50000000, q)
        lines = list(h.lines())
        # its totals and at least one relation (under a sanitizer the siever may add lines of its own, so not by
        # position)
        self.assertTrue(any(l.startswith('# Total ') for l in lines), lines[-5:])
        self.assertTrue(any(l and l[0] != '#' for l in lines), lines[-5:])

    @unittest.skipUnless(sys.platform.startswith('linux'), 'PR_SET_PDEATHSIG is Linux only')
    def test_dies_with_parent(self):
        """A siever dies with its parent, also when the parent is killed (SIGKILL: no cleanup runs)."""
        d = tmpdir(self)
        pidf = os.path.join(d, 'pid')
        code = ('import os, subprocess, sys, time\nsys.path.insert(0, %r)\nfrom rankbound import sieve\n'
                'p = sieve.spawn(%r, stdout=subprocess.PIPE, stdin=subprocess.PIPE)\n'
                'open(%r, "w").write(str(p.pid))\nos.replace(%r, %r)\ntime.sleep(60)\n'
                % (SRC, self.CMD, pidf + '.tmp', pidf + '.tmp', pidf))
        p = subprocess.Popen([sys.executable, '-c', code])
        kid = int(wait_file(pidf, 30))
        time.sleep(0.3)
        os.kill(p.pid, signal.SIGKILL)
        p.wait(10)
        self.assertTrue(gone(kid, 5))


class TestRelationsShared(unittest.TestCase):
    KEY = {'schema': 1, 'curve': ['0', '0', '1', '-1', '0'], 'hypothesis': 'GRH', 'relations': None, 'name': 'c',
           'lower': None}

    def test_rebind(self):
        """Keys that differ only in r or its points share the folder: a shard or an estimate (no rebind) leaves
        run.json as it is; a run (rebind) binds the folder to its key and drops the other key's result, which is so
        never reused for another r.  Another curve, relations or hypothesis is refused."""
        st = runtime.RunStore(tmpdir(self))
        st.begin(self.KEY)
        st.publish(self.KEY, {'name': 'c', 'rank_bound': 1})
        r31 = dict(self.KEY, lower={'r': 31, 'proved': False, 'source': 'cited', 'proof': None})
        st.begin(r31)
        self.assertEqual((st.info()['key'], st.stored(self.KEY)), (self.KEY, {'name': 'c', 'rank_bound': 1}))
        st.begin(r31, rebind=True)
        d = st.info()
        self.assertEqual((d['key'], d['result'], 'result_key' in d), (r31, None, False))
        self.assertIsNone(st.stored(r31))
        for other in (dict(r31, curve=['0', '0', '0', '-1', '0']), dict(r31, relations='abc'),
                      dict(r31, hypothesis='none')):
            with self.assertRaises(runtime.Mismatch) as e:
                st.begin(other, rebind=True)
            self.assertEqual(str(e.exception), text('other_grh' if other['hypothesis'] != r31['hypothesis'] else
                                                    'other_work'))
        self.assertEqual(st.info()['key'], r31)

    def test_shards_then_r(self):
        """Shards without --r, then `bound --r` on the same folder: it reuses their relations (it sieves nothing more,
        their check is due) and its result has its r; a run without r then gets a result of its own, on the same
        relations, never the one for r = 0."""
        w = tmpdir(self)
        E = ','.join(map(str, E571))
        with self.assertLogs('rankbound', 'INFO'), redirect_stdout(io.StringIO()):
            for k in (1, 2):
                self.assertEqual(cli.main(['bound', E, '--shard', '%d/2' % k, '--dir', w, '--compute']), 0)
        d = os.path.join(w, N571)
        S = State(d)
        names = S.finished()
        out = io.StringIO()
        with self.assertLogs('rankbound', 'INFO') as lg, redirect_stdout(out):
            self.assertEqual(cli.main(['bound', E, '--r', '0', '--dir', w, '--compute', '--json']), 0)
        res = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual((res['rank_bound'], res['rank_lower_bound']), (0, 0))
        self.assertEqual(S.finished(), names, lg.output)                # nothing sieved after the shards
        with self.assertLogs('rankbound', 'INFO'), redirect_stdout(out):
            self.assertEqual(cli.main(['bound', E, '--dir', w, '--compute', '--json']), 0)
        res = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual((res['rank_bound'], res['rank_lower_bound'], S.finished()), (0, None, names))
        with open(os.path.join(d, 'run.json')) as f:
            self.assertIsNone(json.load(f)['key']['lower'])


class TestShardEnd(FakeRun, unittest.TestCase):
    def test_done_means_the_check_can_start(self):
        """A shard is done when the folder has the relations of the next check in all, the other shard's slices
        included: the run that checks takes the folder's finished slices first (Solver.in_order), so it checks at
        once, without sieving its own slices in between."""
        S = State(self.d)
        sched = list(jobs.schedule(PRM))
        for j, poly, q0, q1 in sched[1:10:2]:
            jobs.finish_slice(S.sdir, jobs.slice_name(poly, q0), {'poly': poly, 'q0': q0, 'q1': q1},
                              iter(['%d,%d:2,3:\n' % (a, j) for a in range(R)]), S.fp)
        self.nxt = 5 * R
        fd = S.lock('shard1_2.lock')                 # shard 2 is running
        self.addCleanup(os.close, fd)
        self.assertTrue(jobs.shard_done(S, 0, 2, self.nxt, jobs.Count(S)))
        names = S.finished()
        with self.assertLogs('rankbound', 'INFO'):
            jobs.sieve_run(self.d, jobs=1, shard=(0, 2), budget=runtime.Budget(seconds=60))
        self.assertEqual(S.finished(), names)          # nothing sieved
        self.assertFalse(jobs.shard_done(S, 0, 2, self.nxt + 1, jobs.Count(S)))

    def test_idle_other(self):
        """With the other shard not running, a shard ends once the folder has the check's relations in all."""
        self.nxt = 3 * R
        with self.assertLogs('rankbound', 'INFO') as out:
            jobs.sieve_run(self.d, jobs=1, shard=(0, 2), budget=runtime.Budget(seconds=60))
        self.assertGreaterEqual(State(self.d).slice_stats()[1], self.nxt)
        self.assertEqual(out.output[-1:], ['INFO:rankbound:Shard 1 of 2 is done.'])


class TestPendingCount(FakeRun, unittest.TestCase):
    def test_count_checked(self):
        """The pending check is announced with the relations it eliminates; one that only purges (too few) is not
        announced, and the next check names its own count (before: 'on the same 40 relations', then a solve of
        60)."""
        S = State(self.d)
        for k in range(3):
            jobs.finish_slice(S.sdir, jobs.slice_name('F', 1024 + 100 * k), {'poly': 'F'},
                              iter(['%d,%d:2,3:\n' % (a, k) for a in range(R)]), S.fp)
        S.begin_check(S.finished()[:2], 2 * R)
        self.nxt = 3 * R
        real = jobs.Solver.check
        def check(V, work, dry=False):
            if V.nrel > 2 * R:
                return real(V, work, dry)
            self.checks.append((V.nrel, dry))         # a purge alone: no elimination announced
            return None
        jobs.Solver.check = check
        self.answer = lambda V, dry: 'plateau'
        with self.assertLogs('rankbound', 'INFO') as out:
            jobs.sieve_run(self.d, jobs=1, budget=runtime.Budget(seconds=60))
        self.assertEqual(self.checks, [(2 * R, False), (3 * R, False)])
        said = [l.split(':', 2)[2] for l in out.output]
        self.assertNotIn(text('pending_check', n=2 * R), said)
        self.assertIn('Found 60 relations so far in 0.00 core-hours.', said)
        S.begin_check(S.finished()[:2], 2 * R)
        self.checks.clear()
        jobs.Solver.check = real
        S.clear_stop()
        with self.assertLogs('rankbound', 'INFO') as out:
            jobs.sieve_run(self.d, jobs=1, budget=runtime.Budget(seconds=60))
        self.assertIn('INFO:rankbound:' + text('pending_check', n=2 * R), out.output)


if __name__ == '__main__':
    unittest.main()
