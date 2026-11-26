"""The HPC workflow's locks and interruptions, in small (as on a cluster: a high-rank curve interrupted and resumed
under Slurm, two jobs on one work folder).

The locks are flock locks on files in the work folder (runtime.lock_fd).  On the cluster (WekaFS, home and scratch)
they held across nodes as here across processes: a second `bound` on a folder in use is refused (Locked), shards
share sieve.lock, and a killed holder's lock does not block the resume (the kernel releases a flock with its
process: the lock file stays, its lock does not)."""
import os, shutil, signal, subprocess, sys, tempfile, time, unittest

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
from rankbound import Locked, runtime  # noqa: E402
from rankbound.messages import text  # noqa: E402

HOLD = r'''
import os, sys, time
sys.path.insert(0, %r)
from rankbound import runtime
d, ready, mode = sys.argv[1], sys.argv[2], sys.argv[3]
fds = []
if mode == 'bound':
    fds.append(runtime.lock_fd(os.path.join(d, 'bound.lock')))
    fds.append(runtime.lock_fd(os.path.join(d, 'run.lock')))
else:                                   # a shard: its own lock and sieve.lock shared
    fds.append(runtime.lock_fd(os.path.join(d, 'shard%%s_2.lock' %% mode)))
    fds.append(runtime.lock_fd(os.path.join(d, 'sieve.lock'), shared=True))
open(ready, 'w').close()
time.sleep(60)
''' % SRC


class TestLocksAcrossProcesses(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='rb-hpcflow-')
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()

    def hold(self, mode):
        ready = os.path.join(self.dir, 'ready-%s-%d' % (mode, len(self.procs)))
        p = subprocess.Popen([sys.executable, '-c', HOLD, self.dir, ready, mode])
        self.procs.append(p)
        t = time.monotonic()
        while not os.path.exists(ready):
            self.assertIsNone(p.poll(), 'the holder exited')
            self.assertLess(time.monotonic() - t, 30)
            time.sleep(0.02)
        return p

    def test_second_bound_is_locked(self):
        self.hold('bound')
        S = runtime.RunStore(self.dir)
        with self.assertRaises(Locked) as e:
            with S.own('bound'):
                pass
        self.assertEqual(str(e.exception), text('busy', folder=S.dir))

    def test_shards_run_together(self):
        self.hold('0')
        self.hold('1')                  # a second shard gets sieve.lock (shared) while the first holds it
        with self.assertRaises(Locked):                 # a local run (sieve.lock exclusive) does not
            os.close(runtime.lock_fd(os.path.join(self.dir, 'sieve.lock')))
        with self.assertRaises(Locked):                 # nor the same shard twice
            os.close(runtime.lock_fd(os.path.join(self.dir, 'shard0_2.lock')))

    def test_stale_lock_after_kill(self):
        p = self.hold('bound')
        os.kill(p.pid, signal.SIGKILL)
        p.wait()
        self.assertTrue(os.path.exists(os.path.join(self.dir, 'bound.lock')))     # the file stays ...
        t = time.monotonic()
        with runtime.RunStore(self.dir).own('bound'):                            # ... its lock does not
            os.close(runtime.lock_fd(os.path.join(self.dir, 'run.lock')))
        self.assertLess(time.monotonic() - t, 1.0)


RUNCHILD = r'''
import signal, sys
signal.signal(signal.SIGINT, signal.default_int_handler)   # as at a terminal, also if started with SIGINT ignored
sys.path.insert(0, %r)
from rankbound import runtime
runtime.run(['sh', '-c', 'echo $$ > ' + sys.argv[1] + '.tmp; mv ' + sys.argv[1] + '.tmp ' + sys.argv[1] + '; exec sleep 60'],
            budget=runtime.Budget())
''' % SRC


class TestSignalsReachChildren(unittest.TestCase):
    """A signal to the rankbound process while a program of runtime.run runs (f2elim, gp, the checker: each in its own
    session, so a job's or a terminal's signal to the rankbound process does not reach it): the program must not
    outlive rankbound, and the run's spent must be written (every finally runs) on SIGTERM as on SIGINT."""

    def child_after(self, sig):
        d = tempfile.mkdtemp(prefix='rb-hpcflow-')
        self.addCleanup(shutil.rmtree, d, True)
        pidf = os.path.join(d, 'pid')
        p = subprocess.Popen([sys.executable, '-c', RUNCHILD, pidf], stderr=subprocess.DEVNULL)
        t = time.monotonic()
        while not os.path.exists(pidf):
            self.assertLess(time.monotonic() - t, 30)
            time.sleep(0.02)
        with open(pidf) as f:
            kid = int(f.read())
        os.kill(p.pid, sig)
        p.wait(30)
        t = time.monotonic()
        while time.monotonic() - t < 10:            # runtime.terminate's grace is 5 s
            try:
                os.kill(kid, 0)
            except ProcessLookupError:
                return False
            time.sleep(0.1)
        os.kill(kid, signal.SIGKILL)
        return True

    def test_sigint_stops_the_program(self):
        self.assertFalse(self.child_after(signal.SIGINT))

    def test_sigterm_stops_the_program(self):
        self.assertFalse(self.child_after(signal.SIGTERM))


if __name__ == '__main__':
    unittest.main()
