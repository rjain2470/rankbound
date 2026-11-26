"""The lifecycle of a request: core.py's records, runtime.py's budget, locks, run identity, artifact
transactions and subprocesses, and runner.py's execute/verify/publish on a stand-in route and checker (no PARI/GP
needed): concurrent requests in one folder, a crash between data and manifest, nested budgets, resuming a
finished-but-unchecked run by checking."""
import contextlib, dataclasses, json, multiprocessing as mp, os, shutil, signal, sys, tempfile, textwrap, time, unittest
from unittest import mock

from rankbound import core, runner, runtime
from rankbound.core import (GRH, UNCONDITIONAL, LowerBound, Locked, ProofData, Request, Resources, Stopped, TimeLimit,
                            Verdict)
from helpers import ended

FORK = mp.get_context('fork')


def tmpdir(test):
    d = tempfile.mkdtemp(prefix='rb_lifecycle')
    test.addCleanup(shutil.rmtree, d, True)
    return d


# ---------- the budget

class TestBudget(unittest.TestCase):
    def test_child_never_outlives_parent(self):
        p = runtime.Budget(seconds=10, stop=lambda: Stopped('w'))
        c = p.child(seconds=1000)
        self.assertLessEqual(c.end, p.end)
        self.assertAlmostEqual(c.left(), p.left(), delta=0.05)
        c2 = p.child(seconds=1)
        self.assertLess(c2.end, p.end)
        self.assertIsNone(runtime.Budget().child().end)
        self.assertEqual(runtime.Budget().child(seconds=5).end is not None, True)

    def test_which_end_raises(self):
        run = runtime.Budget.for_run(seconds=30, workdir='W')
        stage = run.child(seconds=0)
        with self.assertRaises(TimeLimit):          # the stage's own limit: the stage gives up
            stage.check()
        run.check()                                 # the run goes on
        run2 = runtime.Budget.for_run(seconds=0, workdir='W')
        stage2 = run2.child(seconds=0)
        with self.assertRaises(Stopped) as e:       # the run's end beats the stage's
            stage2.check()
        self.assertEqual(e.exception.workdir, 'W')

    def test_scope_restores_and_check(self):
        self.assertIs(runtime.current(), runtime.UNLIMITED)
        b = runtime.Budget(seconds=0)
        with self.assertRaises(TimeLimit):
            with runtime.scope(b):
                self.assertIs(runtime.current(), b)
                runtime.check()
        self.assertIs(runtime.current(), runtime.UNLIMITED)
        runtime.check()                             # no limit leaks out of the scope

    def test_limit_and_wall(self):
        b = runtime.Budget(seconds=100)
        self.assertEqual(b.limit(10), 10)
        self.assertLessEqual(b.limit(1000), 100)
        self.assertAlmostEqual(b.wall(), time.time() + 100, delta=1)
        self.assertEqual(runtime.Budget().limit(7), 7)
        with self.assertRaises(TimeLimit):
            runtime.Budget(seconds=0).limit(5)

    def test_stopped(self):
        self.assertTrue(runtime.stopped(runtime.Budget.for_run(seconds=0).child(seconds=100)))
        self.assertFalse(runtime.stopped(runtime.Budget.for_run(seconds=60).child(seconds=0)))


class TestAlarm(unittest.TestCase):
    def test_interrupts_and_restores(self):
        outer = runtime.Budget.for_run(seconds=0.6, workdir='W')
        inner = outer.child(seconds=0.15)
        t = time.monotonic()
        with self.assertRaises(Stopped):
            with runtime.alarm(outer):
                with self.assertRaises(TimeLimit):
                    with runtime.alarm(inner):
                        while True:
                            pass
                self.assertLess(time.monotonic() - t, 0.5)
                while True:                          # the outer alarm was restored: it still fires
                    pass
        self.assertLess(time.monotonic() - t, 2)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)

    def test_no_limit_no_timer(self):
        with runtime.alarm(runtime.Budget()):
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0], 0)


# ---------- locks and the run's identity

def _hold(path, shared, ready, release):
    with runtime.lock(path, shared=shared):
        ready.set()
        release.wait(10)


def _begin(folder, key, barrier, q):
    barrier.wait()
    try:
        runtime.RunStore(folder).begin(key, name=key['curve'][0])
        q.put(('ok', key['curve'][0]))
    except runtime.Mismatch:
        q.put(('mismatch', key['curve'][0]))


class TestLocks(unittest.TestCase):
    def _other(self, path, shared):
        ready, release = FORK.Event(), FORK.Event()
        p = FORK.Process(target=_hold, args=(path, shared, ready, release))
        p.start()
        self.assertTrue(ready.wait(10))
        self.addCleanup(p.join, 10)
        self.addCleanup(release.set)
        return release

    def test_exclusive_shared(self):
        d = tmpdir(self)
        path = os.path.join(d, 'stage.lock')
        rel = self._other(path, shared=True)
        with runtime.lock(path, shared=True):       # shards: shared with shared
            pass
        with self.assertRaises(Locked):
            with runtime.lock(path):
                pass
        rel.set()
        with runtime.lock(path, wait=10):           # waits for the other to let go
            pass

    def test_own_stage(self):
        d = tmpdir(self)
        S = runtime.RunStore(d)
        self._other(S.path('run.lock'), shared=False)
        with self.assertRaises(Locked) as e:
            with S.own('run'):
                pass
        self.assertIn(d, str(e.exception))
        with S.own('sieve', shared=True):           # another stage is free
            pass


class TestRunStore(unittest.TestCase):
    KA = {'curve': ['0', '0', '0', '-1', '0'], 'r': 0, 'points': None, 'relations': None, 'GRH': False}
    KB = {'curve': ['0', '0', '0', '-25', '0'], 'r': 1, 'points': None, 'relations': None, 'GRH': False}

    def test_concurrent_identity(self):
        """Two processes start different curves in one empty folder at once: exactly one binds it."""
        for _ in range(8):
            d = tmpdir(self)
            barrier, q = FORK.Barrier(2), FORK.Queue()
            ps = [FORK.Process(target=_begin, args=(d, k, barrier, q)) for k in (self.KA, self.KB)]
            [p.start() for p in ps]
            [p.join(20) for p in ps]
            out = sorted(q.get(timeout=5) for _ in ps)
            self.assertEqual([o[0] for o in out], ['mismatch', 'ok'])
            won = next(o[1] for o in out if o[0] == 'ok')
            self.assertEqual(runtime.RunStore(d).info()['key']['curve'][0], won)

    def test_result_bound_to_key(self):
        d = tmpdir(self)
        S = runtime.RunStore(d)
        S.begin(self.KA)
        S.publish(self.KA, {'rank_bound': 0, 'name': '0_0_0_-1_0'})
        self.assertEqual(S.stored(self.KA)['rank_bound'], 0)
        self.assertEqual(S.info()['result_key'], self.KA)
        with self.assertRaises(core.RankboundError):        # B's request, A's result: refused, never returned
            S.stored(self.KB)
        with self.assertRaises(core.RankboundError):
            S.publish(self.KB, {'rank_bound': 1})
        d2 = dict(S.info(), key=self.KB)            # run.json's key changed under the result: not B's result
        runtime.write_json(S.path('run.json'), d2)
        with self.assertRaises(core.RankboundError):
            S.stored(self.KB)

    def test_hypothesis_message(self):
        d = tmpdir(self)
        S = runtime.RunStore(d)
        S.begin(dict(self.KA, GRH=True))
        with self.assertRaises(runtime.Mismatch) as e:
            S.begin(self.KA)
        self.assertIn('assuming the GRH', str(e.exception))
        with self.assertRaises(runtime.Mismatch) as e:
            S.begin(self.KB)
        self.assertIn('another curve', str(e.exception))

    def test_damaged_run_json(self):
        d = tmpdir(self)
        with open(os.path.join(d, 'run.json'), 'w') as f:
            f.write('{"key": ')
        with self.assertRaises(core.Damaged):
            runtime.RunStore(d).info()


# ---------- artifacts

def _crash_before_manifest(folder):
    real = runtime.write_json

    def wj(path, obj, indent=1):
        if path.endswith('.manifest.json'):
            os._exit(3)                             # killed after the data, before its manifest
        real(path, obj, indent)
    runtime.write_json = wj
    with runtime.transaction(folder, 'rels', 's1', 'fp') as tx:
        tx.write('a.bin', b'new data')
        tx.write('b.bin', b'more new')


class TestArtifacts(unittest.TestCase):
    def test_commit_and_load(self):
        d = tmpdir(self)
        with runtime.transaction(d, 'rels', 's1', 'fp', counts={'relations': 2}) as tx:
            tx.write('a.bin', b'12')
            with open(tx.path('b.txt'), 'w') as f:
                f.write('x')
            self.assertFalse(os.path.exists(os.path.join(d, 'a.bin')))      # nothing in place before the commit
        m = runtime.load(d, 'rels', 's1', 'fp')
        self.assertEqual(m['counts'], {'relations': 2})
        self.assertEqual(sorted(m['files']), ['a.bin', 'b.txt'])
        with open(os.path.join(d, 'a.bin'), 'rb') as f:
            self.assertEqual(f.read(), b'12')
        self.assertEqual([f for f in os.listdir(d) if f.endswith('.part')], [])

    def test_crash_between_data_and_manifest(self):
        d = tmpdir(self)
        with runtime.transaction(d, 'rels', 's1', 'fp') as tx:
            tx.write('a.bin', b'old data')
            tx.write('b.bin', b'old more')
        self.assertIsNotNone(runtime.load(d, 'rels', 's1', 'fp'))
        p = FORK.Process(target=_crash_before_manifest, args=(d,))
        p.start()
        p.join(20)
        self.assertEqual(p.exitcode, 3)
        with open(os.path.join(d, 'a.bin'), 'rb') as f:
            self.assertEqual(f.read(), b'new data')                     # new data, no manifest:
        self.assertEqual(runtime.inspect(d, 'rels')[0], 'missing')                     # neither old nor mixed
        self.assertIsNone(runtime.load(d, 'rels', 's1', 'fp'))

    def test_damaged_and_stale(self):
        d = tmpdir(self)
        with runtime.transaction(d, 'x', 's1', 'fp') as tx:
            tx.write('a.bin', b'abc')
        self.assertIsNone(runtime.load(d, 'x', 's1', 'other'))                  # stale: another input
        self.assertTrue(os.path.exists(runtime.manifest_path(d, 'x')))         # but not damaged
        self.assertIsNone(runtime.load(d, 'x', 's2', 'fp'))
        with open(os.path.join(d, 'a.bin'), 'wb') as f:
            f.write(b'abd')                                                    # same size, other bytes
        self.assertEqual(runtime.inspect(d, 'x', digests=False)[0], 'valid')
        self.assertEqual(runtime.inspect(d, 'x')[0], 'damaged')
        self.assertIsNone(runtime.load(d, 'x', 's1', 'fp'))
        self.assertTrue(os.path.exists(runtime.manifest_path(d, 'x') + '.damaged'))
        with open(runtime.manifest_path(d, 'x'), 'w') as f:
            f.write('{"format": "rankbound-manifest-1", "na')                  # a torn manifest
        self.assertEqual(runtime.inspect(d, 'x')[0], 'damaged')
        self.assertIsNone(runtime.load(d, 'x'))

    def test_abort(self):
        d = tmpdir(self)
        with self.assertRaises(KeyError):
            with runtime.transaction(d, 'x', 's', 'f') as tx:
                tx.write('a.bin', b'1')
                raise KeyError('boom')
        self.assertEqual(os.listdir(d), [])
        with self.assertRaises(ValueError):
            runtime.Transaction(d, 'x', 's', 'f').path('../a')

    def test_adopt(self):
        d = tmpdir(self)
        runtime.write_bytes(os.path.join(d, 'w.json'), b'{}')
        with runtime.transaction(d, 'proof', 's', 'f') as tx:
            tx.adopt('w.json')
        self.assertIsNotNone(runtime.load(d, 'proof', 's', 'f'))
        runtime.write_bytes(os.path.join(d, 'w.json'), b'{"x": 1}')
        self.assertIsNone(runtime.load(d, 'proof', 's', 'f'))


# ---------- subprocesses

class TestProcesses(unittest.TestCase):
    def test_budget_kills_group(self):
        d = tmpdir(self)
        pidf = os.path.join(d, 'pid')
        script = 'sleep 30 & echo $! > %s; wait' % pidf
        b = runtime.Budget.for_run(seconds=0.5, workdir=d)
        t = time.monotonic()
        with self.assertRaises(Stopped):
            runtime.run(['sh', '-c', script], budget=b)
        self.assertLess(time.monotonic() - t, 5)
        with open(pidf) as f:
            gc = int(f.read())
        time.sleep(0.1)
        self.assertTrue(ended(gc))                      # the grandchild went with its group
        self.assertEqual(runtime._CHILDREN, set())

    def test_run_ok_and_expired(self):
        rc, out, _ = runtime.run([sys.executable, '-c', 'print(6*7)'])
        self.assertEqual((rc, out.strip()), (0, '42'))
        with self.assertRaises(TimeLimit):
            runtime.run(['true'], budget=runtime.Budget(seconds=0))


# ---------- core

class TestCore(unittest.TestCase):
    def req(self, **kw):
        base = dict(source=('0', '0', '1', '-1', '0'), minimal_model=('0', '0', '1', '-1', '0'),
                    model_map=('1', '0', '0', '0'), hypothesis=GRH, name='37a', output='/w')
        base.update(kw)
        return Request(**base)

    def test_identity_excludes_resources(self):
        a, b = self.req(options={'cores': 1}), self.req(options={'cores': 64, 'max_hours': 2})
        self.assertEqual(a.key(), b.key())
        self.assertNotEqual(a.key(), self.req(hypothesis=UNCONDITIONAL).key())
        self.assertNotEqual(a.key(), self.req(lower=LowerBound(1, True, 'points', 'abc')).key())
        self.assertNotEqual(a.key(), self.req(relations='d1').key())
        self.assertNotIn('output', a.identity())

    def test_one_identity(self):
        """One run identity: run.json's key is the request's identity, the proof artifact's fingerprint its digest."""
        from rankbound import curve as cv
        c = cv.with_points(cv.from_ainvs([0, 0, 1, -1, 0], bad_primes=['37']), [(0, 0)])
        req = runner.request_of(cv.classified(c), runner.SIEVE, tmpdir(self))
        self.assertEqual(req.key(), core.digest(req.identity()))
        self.assertEqual(req.identity()['lower'], {'r': 1, 'proved': True, 'source': 'points',
                                                   'proof': runner._points_digest(c.proof)})
        self.assertEqual(runner.model([0, 0, 1, -1, 0])[2], ('1', '0', '0', '0'))
        self.assertEqual(runner.model([0, 0, 0, -81, 0])[1:], (('0', '0', '0', '-1', '0'), ('3', '0', '0', '0')))
        self.assertTrue(req.lower.proved)

    def test_result_fields_compatible(self):
        import rankbound
        old = [f.name for f in dataclasses.fields(rankbound.Result)]
        new = [f.name for f in dataclasses.fields(core.Result)]
        self.assertEqual(new[:len(old)], old)

    def test_verdict_of(self):
        self.assertEqual(Verdict.of(None).state, 'unavailable')
        v = Verdict.of({'verdict': 'rejected', 'checks': [{'check': 'format', 'ok': True}, {'check': 'rank', 'ok': False}]})
        self.assertEqual((v.state, v.check), ('rejected', 'rank'))
        self.assertEqual(Verdict.of({'verdict': 'odd'}).state, 'error')
        self.assertIsNone(core.checked_of('limit'))

    def test_resume_command(self):
        """Exact: the CLI's command as typed; the API's call with its work folder, cores, hours and budget."""
        cmd = "rankbound bound 0,0,1,-1,0 --r 1 --uncond --dir work --cores 4 --budget 2"
        self.assertEqual(self.req(options={'command': cmd}).resume_command(), cmd)
        r = self.req(options={'r': 1, 'uncond': True, 'cores': 4, 'max_hours': 0.5, 'budget': 2.0, 'compute': False},
                     hypothesis=UNCONDITIONAL, lower=LowerBound(1, False), output='/runs/w')
        self.assertEqual(r.resume_command(), "rankbound([0, 0, 1, -1, 0], r=1, uncond=True, cores=4, max_hours=0.5, "
                                             "budget=2.0, workdir='/runs/w')")

    def test_errors_reexported(self):
        import rankbound
        for n in ('RankboundError', 'NotApplicable', 'FactorisationNeeded', 'TimeLimit', 'Rejected', 'Locked', 'Stopped'):
            self.assertTrue(issubclass(getattr(core, n), Exception))
            self.assertTrue(hasattr(rankbound, n))


# ---------- the runner, on a stand-in route and checker

CHECKER = textwrap.dedent('''
    import hashlib, json, os, sys
    a = sys.argv[1:]
    path = a[-1]
    curve = a[a.index('--curve') + 1] if '--curve' in a else None
    mode = open(os.environ['FAKE_MODE']).read().strip() if os.environ.get('FAKE_MODE') else 'accept'
    w = json.load(open(path))
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    if os.environ.get('FAKE_TAMPER'):                 # the original changes while it is checked
        open(os.environ['FAKE_TAMPER'], 'w').write('{"S": 9, "GRH": "yes"}')
    if mode == 'crash':
        sys.exit(5)
    ok = mode in ('accept', 'other') and curve == w['curve']
    print(json.dumps({'verdict': 'accepted' if ok else mode if mode in ('limit', 'error') else 'rejected',
                      'sha256': sha, 'rank_upper_bound': w['S'] + (mode == 'other'), 'GRH': w['GRH'],
                      'statement': 'checked: rank <= %d' % w['S'], 'checker': 'fake', 'reason': mode,
                      'checks': [{'check': 'curve', 'ok': ok}]}))
''')


class FakeRoute:
    """A route that writes a witness 'rank <= S' at once (or sleeps first)."""
    exclusive = True

    def __init__(self, S=0, grh=False, sleep=0.0):
        self.S, self.grh, self.sleep, self.calls = S, grh, sleep, 0

    def precheck(self, curve):
        pass

    def logging(self):
        return contextlib.nullcontext()

    def prove(self, curve, req, resources, budget, **kw):
        self.calls += 1
        t = time.monotonic()
        while time.monotonic() - t < self.sleep:
            runtime.check()
            time.sleep(0.01)
        path = os.path.join(req.output, req.name + '.witness.json')
        runtime.write_json(path, {'curve': ','.join(req.source), 'S': self.S, 'GRH': 'yes' if self.grh else 'no'})
        return ProofData(witness=path, rank_bound=self.S, hypothesis=GRH if self.grh else UNCONDITIONAL,
                         statement='rank <= %d' % self.S, method='descent', record={'minimal_model': list(req.source)},
                         cost={'descent': 0.0})


def _exec(d, src, ck, barrier, q):
    barrier.wait()
    try:
        req = Request(source=src, name='E' + src[3], output=d, hypothesis=UNCONDITIONAL)
        res = runner.execute(None, req, Resources(), runner.DESCENT, FakeRoute(), checker=ck)
        q.put(('ok', src, res.details['minimal_model']))
    except (runtime.Mismatch, Locked) as e:
        q.put((type(e).__name__, src, None))


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.d = tmpdir(self)
        self.ck = os.path.join(self.d, 'checker.py')
        with open(self.ck, 'w') as f:
            f.write(CHECKER)
        self.mode = os.path.join(self.d, 'mode')
        self.set_mode('accept')
        patcher = mock.patch.dict(os.environ, {'FAKE_MODE': self.mode})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.work = os.path.join(self.d, 'work')

    def set_mode(self, m):
        with open(self.mode, 'w') as f:
            f.write(m)

    def req(self, **kw):
        base = dict(source=('0', '0', '0', '-1', '0'), name='E1', output=self.work, hypothesis=UNCONDITIONAL)
        base.update(kw)
        return Request(**base)

    def run_(self, route, req=None, **kw):
        return runner.execute(None, req or self.req(), Resources(), runner.DESCENT, route,
                              checker=[sys.executable, self.ck], **kw)

    def test_fresh_then_resumed(self):
        R = FakeRoute()
        res = self.run_(R)
        self.assertEqual((res.rank_bound, res.checked, res.verification, res.origin), (0, True, 'accepted', 'computed'))
        self.assertEqual(res.statement, 'checked: rank <= 0')
        self.assertEqual(res.witness_sha256, runtime.sha256_file(res.witness)[0])
        self.assertEqual(res.resume, 'rankbound([0, 0, 0, -1, 0], workdir=%r)' % self.work)
        rec = runtime.RunStore(self.work).info()
        self.assertEqual(rec['result_key'], self.req().identity())
        self.assertEqual(runtime.inspect(self.work, runner.PROOF)[1]['fingerprint'], self.req().key())
        self.assertEqual(rec['stage'], 'finished')
        again = self.run_(R)
        self.assertEqual((R.calls, again.origin, again.checked), (1, 'resumed', True))

    def test_resume_unchecked_by_checking(self):
        """A result published without PARI/GP is checked when the run is resumed with it."""
        R = FakeRoute()
        with mock.patch.object(runner.shutil, 'which', return_value=None):
            res = runner.execute(None, self.req(), Resources(), runner.DESCENT, R)
        self.assertEqual((res.checked, res.verification), (None, 'unavailable'))
        res = self.run_(R)
        self.assertEqual((R.calls, res.checked, res.verification, res.origin), (1, True, 'accepted', 'resumed'))

    def test_killed_after_witness_checks_it(self):
        """The checker fails (or the process dies) after the witness: a run again checks it, proving nothing."""
        R = FakeRoute()
        self.set_mode('crash')
        with self.assertRaises(RuntimeError):
            self.run_(R)
        self.assertIsNone(runtime.RunStore(self.work).info()['result'])
        self.set_mode('accept')
        res = self.run_(R)
        self.assertEqual((R.calls, res.checked, res.origin), (1, True, 'resumed'))

    def test_changed_witness_is_proved_again(self):
        R = FakeRoute()
        self.set_mode('crash')
        with self.assertRaises(RuntimeError):
            self.run_(R)
        with open(os.path.join(self.work, 'E1.witness.json'), 'a') as f:
            f.write(' ')
        self.set_mode('accept')
        res = self.run_(R)
        self.assertEqual((R.calls, res.checked), (2, True))

    def test_rejected_and_other_claim(self):
        self.set_mode('reject')
        res = self.run_(FakeRoute())
        self.assertEqual((res.checked, res.verification, res.details['check']), (False, 'rejected', 'curve'))
        d2 = os.path.join(self.d, 'w2')
        self.set_mode('other')                       # the checker's bound is not the route's claim
        with self.assertRaises(RuntimeError):
            res = self.run_(FakeRoute(), self.req(output=d2))
        self.set_mode('accept')
        self.assertEqual(self.run_(FakeRoute(), self.req(output=d2)).checked, True)

    def test_wrong_curve_rejected(self):
        """The checker is given the request's curve: a witness for another curve is rejected."""
        class Other(FakeRoute):
            def prove(self, curve, req, resources, budget, **kw):
                return super().prove(curve, dataclasses.replace(req, source=('0', '0', '0', '-25', '0')), resources, budget)
        res = self.run_(Other())
        self.assertEqual(res.verification, 'rejected')

    def test_captured_bytes(self):
        """The witness is replaced while the checker runs: the verdict is on the bytes it read."""
        R = FakeRoute()
        path = os.path.join(self.work, 'E1.witness.json')
        with mock.patch.dict(os.environ, {'FAKE_TAMPER': path}):
            res = self.run_(R)
        self.assertEqual(res.checked, True)
        self.assertNotEqual(res.witness_sha256, runtime.sha256_file(path)[0])
        m = runtime.inspect(self.work, runner.PROOF)[1]           # the bytes the route committed
        self.assertEqual(res.witness_sha256, m['files']['E1.witness.json']['sha256'])

    def test_stopped_keeps_proof(self):
        R = FakeRoute(sleep=5)
        b = runtime.Budget.for_run(seconds=0.3, workdir=self.work)
        with self.assertRaises(Stopped) as e:
            runner.execute(None, self.req(), Resources(), runner.DESCENT, R, budget=b, checker=[sys.executable, self.ck])
        self.assertIn('workdir=%r' % self.work, e.exception.resume)
        self.assertIsNone(runtime.load(self.work, runner.PROOF))
        R.sleep = 0
        self.assertEqual(self.run_(R).checked, True)

    def test_stopped_before_check(self):
        R = FakeRoute()
        self.set_mode('crash')
        with self.assertRaises(RuntimeError):
            self.run_(R)
        b = runtime.Budget.for_run(seconds=0, workdir=self.work)
        with self.assertRaises(Stopped):
            runner.execute(None, self.req(), Resources(), runner.DESCENT, R, budget=b, checker=[sys.executable, self.ck])
        self.assertIsNotNone(runtime.load(self.work, runner.PROOF))      # the witness stays committed

    def test_concurrent_requests_one_folder(self):
        """Through the runner: two curves at once in one folder; the published result is the winner's."""
        A, B = ('0', '0', '0', '-1', '0'), ('0', '0', '0', '-25', '0')
        for _ in range(4):
            d = tempfile.mkdtemp(dir=self.d)
            barrier, q = FORK.Barrier(2), FORK.Queue()
            ps = [FORK.Process(target=_exec, args=(d, s, [sys.executable, self.ck], barrier, q)) for s in (A, B)]
            [p.start() for p in ps]
            [p.join(60) for p in ps]
            out = [q.get(timeout=5) for _ in ps]
            ok = [o for o in out if o[0] == 'ok']
            self.assertEqual(len(ok), 1, out)
            self.assertEqual(tuple(ok[0][2]), ok[0][1])
            info = runtime.RunStore(d).info()
            self.assertEqual(tuple(info['key']['curve']), ok[0][1])
            self.assertEqual(tuple(info['result']['minimal_model']), ok[0][1])

    def test_lookup_table(self):
        from rankbound import curve as cv
        with open(cv.WITNESSES) as f:
            head = f.readline().rstrip('\n').split('\t')
            row = dict(zip(head, f.readline().rstrip('\n').split('\t')))
        mm = row[cv.COL_MM].split(',')
        res = runner.lookup(mm)
        self.assertEqual((res.known, res.origin, res.outcome, res.checked), (True, 'table', 'table', None))


# ---------- the real routes: the API and the CLI on one path (needs PARI/GP for the checker)

E25 = [0, 0, 0, -25, 0]         # rational 2-torsion: the 2-isogeny descent, rank 1 with the point (-4, 6)


@unittest.skipUnless(shutil.which('gp'), 'needs PARI/GP')
class TestEntryPoints(unittest.TestCase):
    def setUp(self):
        self.d = tmpdir(self)

    def cli(self, *argv):
        import contextlib, io
        from rankbound import cli
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cli.main(['bound'] + list(argv))
        return rc, [json.loads(l) for l in out.getvalue().splitlines() if l.startswith('{')]

    def test_api_cli_equivalent(self):
        """Rankbound() and `rankbound bound` take one path: the same route, record and verdict."""
        import rankbound
        api = rankbound.rankbound(E25, compute=True, workdir=os.path.join(self.d, 'api', '0_0_0_-1_0'))
        rc, lines = self.cli(','.join(map(str, E25)), '--compute', '--json', '--dir', os.path.join(self.d, 'cli'))
        self.assertEqual(rc, 0)
        j = lines[-1]
        self.assertEqual((api.rank_bound, api.grh, api.checked, api.verification, api.origin),
                         (j['rank_bound'], j['GRH'], j['checked'], 'accepted', 'computed'))
        ra = runtime.read_json(os.path.join(api.workdir, 'run.json'))
        rc_ = runtime.read_json(os.path.join(self.d, 'cli', api.name, 'run.json'))
        self.assertEqual(ra['key'], rc_['key'])
        skip = {'witness', 'cost', 'core_hours', 'witness_sha256', 'request', 'statement', 'provenance'}
        self.assertEqual({k: v for k, v in ra['result'].items() if k not in skip},
                         {k: v for k, v in rc_['result'].items() if k not in skip})
        pa, pc = ra['result']['provenance'], rc_['result']['provenance']       # the same but for the words given
        self.assertEqual((pa['versions'], pa['seeds'], dict(pa['parameters'], options=None)),
                         (pc['versions'], pc['seeds'], dict(pc['parameters'], options=None)))
        self.assertEqual(ra['result']['verification'], 'accepted')
        self.assertEqual(api.witness_sha256, runtime.sha256_file(api.witness)[0])

    def test_resume_unchecked_checks(self):
        """A run finished without PARI/GP is checked when resumed with it; nothing is proved again."""
        import rankbound
        from rankbound import descent
        w = os.path.join(self.d, 'w')
        with mock.patch.object(runner.shutil, 'which', return_value=None):
            first = rankbound.rankbound(E25, compute=True, workdir=w)
        self.assertEqual((first.checked, first.verification), (None, 'unavailable'))
        with mock.patch.object(descent.Descent, 'prove', side_effect=AssertionError('proved again')):
            again = rankbound.rankbound(E25, compute=True, workdir=w)
        self.assertEqual((again.checked, again.verification, again.origin), (True, 'accepted', 'resumed'))
        self.assertEqual(again.rank_bound, first.rank_bound)

    def test_killed_before_verdict(self):
        """Interrupted between the witness and the verdict (Ctrl-C in the checker): the resumed run checks that
        witness; the checker's process is gone."""
        import rankbound
        from rankbound import descent
        w = os.path.join(self.d, 'w')
        with mock.patch.object(runner, 'verify', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                rankbound.rankbound(E25, compute=True, workdir=w)
        self.assertIsNotNone(runtime.load(w, runner.PROOF))
        self.assertIsNone(runtime.read_json(os.path.join(w, 'run.json'))['result'])
        with mock.patch.object(descent.Descent, 'prove', side_effect=AssertionError('proved again')):
            res = rankbound.rankbound(E25, compute=True, workdir=w)
        self.assertEqual((res.checked, res.origin), (True, 'resumed'))
        self.assertEqual(runtime._CHILDREN, set())

    def test_stopped_resume_command(self):
        import rankbound
        w = os.path.join(self.d, 'w')
        with self.assertRaises(Stopped) as e:
            runner.bound(E25, compute=True, workdir=w, cores=2, max_hours=1.5,
                         clock=runtime.Budget.for_run(seconds=0))
        self.assertEqual(e.exception.resume, 'rankbound([0, 0, 0, -25, 0], compute=True, cores=2, max_hours=1.5, '
                                             'workdir=%r)' % os.path.abspath(w))
        self.assertTrue(rankbound.rankbound(E25, compute=True, workdir=w).checked)

    def test_cli_stopped_and_mismatch(self):
        w = os.path.join(self.d, 'cli')
        rc, _ = self.cli('0,0,0,-25,0', '--compute', '--dir', w, '--max-hours', '0')
        self.assertEqual(rc, 3)                     # stopped at the time limit: exit 3, resumable
        self.assertIsNone(runtime.read_json(os.path.join(w, '0_0_0_-25_0', 'run.json'))['result'])
        import rankbound
        with self.assertRaises(core.RankboundError):        # the folder holds another request
            rankbound.rankbound([0, 0, 0, -1, 0], compute=True, workdir=os.path.join(w, '0_0_0_-25_0'))


if __name__ == '__main__':
    unittest.main()
