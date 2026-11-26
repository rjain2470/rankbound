"""The entry points: the CLI only parses and renders, the API and the CLI enter one runner at the request's start;
--curves streams; --shard and status through the runner; one run identity; the request budget and the final stops
(too_long, budget); the runner writes the witness once; a stop shows the best checked bound and the attempted one; a
work folder of an older version is refused with its own message; a published witness is hashed again on reuse and
checked again if its bytes changed; the default request budget is max(3 x the estimate, 1 core-hour).  The routes
and the checker are stand-ins here (the real ones are tested in their own modules)."""
import contextlib, io, json, logging, os, shutil, tempfile, unittest
from unittest import mock

import rankbound
from rankbound import cli, core, runner, runtime
from rankbound.core import FORMAT2
from rankbound import curve as cv
from rankbound.core import UNCONDITIONAL, ProofData, Stopped, Verdict
from rankbound.messages import text

E1 = [0, 0, 0, -1, 0]           # rational 2-torsion: route DESCENT (here a stand-in)
E37 = [0, 0, 1, -1, 0]          # no 2-torsion: route SIEVE (here a stand-in)
NOSLURM = {k: '' for k in ('SLURM_JOB_ID', 'SLURM_JOB_END_TIME')}


class Route(runtime.Folder):
    """A stand-in route: prove() does what Route.plan says next and records its arguments (Route.log).
    plan items: ('ok', S) a witness rank <= S as data; ('final', S, outcome) the same as a final stop's best bound;
    ('stop', outcome) Stopped (a final stop's when outcome is in core.FINAL); ('wrote', S) a witness it wrote itself."""
    route, method = 'descent', 'descent'
    plan, log, shards = [], [], []

    def precheck(self, curve):
        pass

    def logging(self):
        return contextlib.nullcontext()

    def prove(self, curve, req, resources, budget, **kw):
        Route.log.append({'name': req.name, 'resources': resources, 'budget': budget, 'request': req,
                          'current': runtime.current()})
        step = Route.plan.pop(0)
        if step[0] == 'stop':
            raise Stopped(req.output, outcome=step[1], reason='It needs 30 hours on 64 cores.' if step[1] != 'stopped'
                          else None)
        path = os.path.join(req.output, req.name + '.witness.json')
        data = {'format': FORMAT2, 'S': step[1]}
        if step[0] == 'wrote':
            runtime.write_json(path, data)
            data = None
        return ProofData(witness=path, rank_bound=step[1], hypothesis=UNCONDITIONAL, statement='rank <= %d' % step[1],
                         method='descent', record={'minimal_model': list(req.source)}, cost={}, data=data,
                         final=step[2] if step[0] == 'final' else None,
                         reason='Spent 1.0 core-hours.' if step[0] == 'final' else None)

    def sieve_shard(self, curve, cores=1, shard=(1, 1), budget=None, budget_core_hours=None):
        Route.shards.append({'shard': shard, 'cores': cores, 'budget': budget, 'current': runtime.current()})
        return 0


class Sieve(Route):
    route, method = 'sieve', 'selmer'


class Checker:
    """A stand-in for runner.verify: accepts the witness's claim (verdict on the file's bytes), or raises Stopped
    (stop), or reads other bytes (other); records each call (Checker.log)."""
    mode, log = 'accept', []

    @staticmethod
    def verify(path, curve=None, budget=None, cores=1, logfile=None, checker=None):
        Checker.log.append(path)
        if Checker.mode == 'stop':
            raise Stopped('')
        with open(path) as f:
            S = json.load(f)['S']
        sha = 'f' * 64 if Checker.mode == 'other' else runtime.sha256_file(path)[0]
        return Verdict('accepted', sha, statement='checked: rank <= %d' % S, checker='fake',
                       raw={'verdict': 'accepted', 'rank_upper_bound': S, 'GRH': 'no'})


class Base(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='rb_entry')
        self.addCleanup(shutil.rmtree, self.d, ignore_errors=True)
        Route.plan, Route.log, Route.shards, Checker.mode, Checker.log = [], [], [], 'accept', []
        for p in (mock.patch.object(runner, 'folder', lambda meth, path: (Route if meth == runner.DESCENT else Sieve)(path)),
                  mock.patch.object(runner, 'verify', Checker.verify), mock.patch.dict(os.environ, NOSLURM)):
            p.start()
            self.addCleanup(p.stop)

    def cli(self, *argv):
        """(exit status, the log's lines, the --json objects) of `rankbound bound ARGV`."""
        out, err, h = io.StringIO(), io.StringIO(), Lines()
        runtime.LOG.addHandler(h)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cli.main(['bound'] + [str(x) for x in argv])
        finally:
            runtime.LOG.removeHandler(h)
        return rc, h.lines, [json.loads(l) for l in out.getvalue().splitlines() if l.startswith('{')]


class Lines(logging.Handler):
    """The user's lines (the 'rankbound' logger's messages)."""
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class TestOnePath(Base):
    def test_cli_only_parses_and_renders(self):
        """cli.py has no routing, preparation, table or run.json code of its own: it hands an Ask to runner.each."""
        import inspect
        src = inspect.getsource(cli)
        for word in ('curve as cv', 'route(', 'folder(', '.estimate(', 'sieve_shard', 'lookup(', 'preflight(',
                     'run_folder', 'run.json', 'json.load(', 'RunStore', 'deadline', 'factored'):
            self.assertNotIn(word, src, word)
        seen = []

        def each(ask, **kw):
            seen.append((ask, kw))
            yield runner.Item('x', ['0', '0', '0', '-1', '0'], estimate=1.5)
        with mock.patch.object(runner, 'each', each):
            rc, lines, _ = self.cli('0,0,0,-1,0', '--estimate', '--cores', 3, '--dir', self.d)
        ask, kw = seen[0]
        self.assertEqual((rc, ask.cores, ask.root, kw['mode'], kw['curve']), (0, 3, self.d, 'estimate',
                                                                             ['0', '0', '0', '-1', '0']))

    def test_budget_starts_at_the_request(self):
        """The API's and the CLI's run budgets start at the request's start: before the table and the factoring."""
        for call in (lambda: rankbound.rankbound(E1, workdir=os.path.join(self.d, 'api'), max_hours=1),
                     lambda: self.cli('0,0,0,-1,0', '--dir', self.d, '--max-hours', 1)):
            order, seconds, factor = [], runtime.run_seconds, cv.from_ainvs
            Route.plan = [('ok', 0)]
            with mock.patch.object(runtime, 'run_seconds', lambda *a, **k: order.append('clock') or seconds(*a, **k)), \
                    mock.patch.object(runner, 'lookup', lambda *a, **k: order.append('table') or None), \
                    mock.patch.object(cv, 'from_ainvs', lambda *a, **k: order.append('factor') or factor(*a, **k)):
                call()
            self.assertEqual(order, ['clock', 'table', 'factor'])
            self.assertAlmostEqual(Route.log[-1]['budget'].left(), 3600, delta=60)

    def test_curves_stream(self):
        """--curves: each curve is prepared, computed and reported before the next is prepared."""
        lst = os.path.join(self.d, 'list.json')
        with open(lst, 'w') as f:
            json.dump({'curves': [{'id': i, 'ainvs': [0, 0, 0, -n * n, 0]} for i, n in ((1, 1), (2, 2), (3, 5))]}, f)
        Route.plan = [('ok', 0)] * 3
        events, route, report = [], runner.route, cli.report
        with mock.patch.object(runner, 'route', lambda c, u=False: events.append(('prepared', c.name)) or route(c, u)), \
                mock.patch.object(cli, 'report', lambda res, a, c=None: events.append(('reported', res.name)) or
                                  report(res, a, c)):
            rc, _, js = self.cli('--curves', lst, '--dir', self.d, '--json')
        self.assertEqual(rc, 0)
        self.assertEqual(events, [(k, n) for n in ('curve1', 'curve2', 'curve3') for k in ('prepared', 'reported')])
        self.assertEqual([j['name'] for j in js], ['curve1', 'curve2', 'curve3'])

    def test_curves_stop_ends_the_list(self):
        lst = os.path.join(self.d, 'list.json')
        with open(lst, 'w') as f:
            json.dump({'curves': [{'id': i, 'ainvs': [0, 0, 0, -n * n, 0]} for i, n in ((1, 1), (2, 2))]}, f)
        Route.plan = [('stop', 'stopped')]
        rc, lines, js = self.cli('--curves', lst, '--dir', self.d, '--json')
        self.assertEqual((rc, [j['outcome'] for j in js], len(Route.log)), (3, ['stopped'], 1))

    def test_shard_under_the_budget(self):
        """--shard runs through the runner, in the request's budget (runtime.current() is it)."""
        rc, _, _ = self.cli('0,0,1,-1,0', '--compute', '--dir', self.d, '--shard', '2/3', '--cores', 2,
                            '--max-hours', 1)
        s = Route.shards[0]
        self.assertEqual((rc, s['shard'], s['cores'], s['budget'].label), (0, (2, 3), 2, 'run'))
        self.assertIs(s['current'], s['budget'])
        self.assertAlmostEqual(s['budget'].left(), 3600, delta=60)
        with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()):
            cli.main(['bound', '0,0,0,-1,0', '--compute', '--dir', self.d, '--shard', '1/2'])
        self.assertEqual(e.exception.code, 2)          # 2-torsion: nothing to sieve (a usage error, found by the runner)

    def test_status_through_runner(self):
        """status: from RunStore and the proof artifact: a witness committed but not checked is shown."""
        Route.plan, Checker.mode = [('ok', 0)], 'stop'
        self.cli('0,0,0,-1,0', '--compute', '--dir', self.d)
        Route.plan, Checker.mode = [('stop', 'too_long')], 'accept'
        self.cli('0,0,0,-4,0', '--compute', '--dir', self.d)
        self.assertEqual(sorted(r['state'] for r in runner.status(self.d)), ['pending', 'too_long'])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cli.main(['status', self.d]), 0)
        self.assertIn('witness not yet checked', out.getvalue())
        self.assertIn('needs a longer run', out.getvalue())


class TestRunner(Base):
    def test_witness_written_once_before_the_check(self):
        """The runner alone writes ProofData.data, once, before the checker reads it."""
        from rankbound import witness
        order, real = [], witness.write
        Route.plan = [('ok', 1)]
        with mock.patch.object(witness, 'write', lambda d, p: order.append('write') or real(d, p)), \
                mock.patch.object(runner, 'verify', lambda *a, **k: order.append('check') or Checker.verify(*a, **k)):
            res = rankbound.rankbound(E1, compute=True, workdir=os.path.join(self.d, 'w'))
        self.assertEqual((order, res.rank_bound, res.checked), (['write', 'check'], 1, True))
        with open(res.witness) as f:
            self.assertEqual(json.load(f), {'format': FORMAT2, 'S': 1})
        Route.plan = [('wrote', 0)]                     # a route that wrote it (data None): not written again
        with mock.patch.object(witness, 'write', side_effect=AssertionError('written twice')):
            self.assertEqual(rankbound.rankbound(E1, compute=True, workdir=os.path.join(self.d, 'w2')).rank_bound, 0)

    def test_verdict_on_committed_bytes(self):
        """The verified sha256 must be the one committed with the proof: else no result is published."""
        Route.plan, Checker.mode = [('ok', 0)], 'other'
        w = os.path.join(self.d, 'w')
        with self.assertRaisesRegex(RuntimeError, 'other bytes'):
            rankbound.rankbound(E1, compute=True, workdir=w)
        self.assertIsNone(runtime.RunStore(w).info()['result'])
        Checker.mode = 'accept'                         # resumed: that witness is checked, not proved again
        res = rankbound.rankbound(E1, compute=True, workdir=w)
        self.assertEqual((res.checked, res.origin, len(Route.log)), (True, 'resumed', 1))

    def test_budget_passed(self):
        """Request.budget, the API's budget= and the CLI's --budget reach the route as
        resources.budget_core_hours (not part of the request's identity)."""
        Route.plan = [('ok', 0), ('ok', 0)]
        rankbound.rankbound(E37, compute=True, workdir=os.path.join(self.d, 'a'), budget=2.5)
        self.cli('0,0,1,-1,0', '--compute', '--dir', os.path.join(self.d, 'c'), '--budget', 4)
        self.assertEqual([(g['resources'].budget_core_hours, g['request'].budget) for g in Route.log],
                         [(2.5, 2.5), (4.0, 4.0)])
        self.assertEqual(Route.log[0]['request'].key(), Route.log[1]['request'].key())
        with self.assertRaises(ValueError):
            rankbound.rankbound(E37, budget=-1)

    def test_budget_refused_without_a_sieve(self):
        """--budget is the sieve's: a curve with 2-torsion (the descent) and --uncond refuse it, a usage error."""
        for E, kw in ((E1, {}), (E37, {'uncond': True})):
            with self.assertRaises(core.Usage) as e:
                rankbound.rankbound(E, compute=True, workdir=os.path.join(self.d, 'u'), budget=2, **kw)
            self.assertEqual(str(e.exception), text('budget_no_sieve'))
        self.assertEqual(Route.log, [])

    def test_budget_final_and_resumed(self):
        """'budget': the route's best bound, checked and published as final; a resume reports it again without proving;
        a larger budget goes on, the final bound the best checked so far."""
        w = os.path.join(self.d, 'w')
        Route.plan = [('final', 2, 'budget')]
        res = rankbound.rankbound(E37, compute=True, workdir=w, budget=1)
        self.assertEqual((res.outcome, res.rank_bound, res.checked), ('budget', 2, True))
        self.assertEqual(res.to_json()['reason'], 'Spent 1.0 core-hours.')
        again = rankbound.rankbound(E37, compute=True, workdir=w, budget=1)
        self.assertEqual((again.outcome, again.origin, len(Route.log)), ('budget', 'resumed', 1))
        Route.plan = [('stop', 'stopped')]
        with self.assertRaises(Stopped) as e:
            rankbound.rankbound(E37, compute=True, workdir=w, budget=3)
        self.assertEqual((len(Route.log), e.exception.outcome, e.exception.result.rank_bound,
                          e.exception.result.checked), (2, 'stopped', 2, True))
        Route.plan = [('ok', 0)]
        res = rankbound.rankbound(E37, compute=True, workdir=w, budget=3)
        self.assertEqual((res.outcome, res.rank_bound, res.origin), ('completed', 0, 'computed'))
        self.assertIsNone(runtime.RunStore(w).info()['final'])

    def test_too_long_final_in_cli(self):
        """'too_long': final, its words and how to go on rendered, in --json too; the same command says it again
        without proving; a larger --max-hours goes on."""
        Route.plan = [('stop', 'too_long')]
        args = ('0,0,1,-1,0', '--compute', '--dir', self.d, '--json', '--max-hours', 1)
        for k in range(2):
            rc, lines, js = self.cli(*args)
            self.assertEqual((rc, len(Route.log)), (0, 1))
            self.assertEqual(lines[-2:], ['It needs 30 hours on 64 cores.', cli.text('final_too_long')])
            self.assertEqual({x: js[0][x] for x in ('outcome', 'reason', 'rank_bound', 'best', 'attempted')},
                             {'outcome': 'too_long', 'reason': 'It needs 30 hours on 64 cores.', 'rank_bound': None,
                              'best': None, 'attempted': None})
            self.assertEqual(js[0]['resume'], 'rankbound bound ' + ' '.join(map(str, args)))
            self.assertEqual((js[0]['name'], js[0]['curve']), ('0_0_1_-1_0', ['0', '0', '1', '-1', '0']))
        Route.plan = [('ok', 1)]
        rc, lines, js = self.cli('0,0,1,-1,0', '--compute', '--dir', self.d, '--json', '--max-hours', 2)
        self.assertEqual((rc, len(Route.log), js[0]['outcome'], js[0]['rank_bound']), (0, 2, 'completed', 1))

    def test_stop_shows_best_and_attempted(self):
        """A stop shows the best checked bound so far and the attempted (unchecked) one apart, in text and --json."""
        Route.plan = [('final', 3, 'budget'), ('ok', 1)]
        self.cli('0,0,1,-1,0', '--compute', '--dir', self.d, '--budget', 1)
        Checker.mode = 'stop'                           # the new witness is written, its check stopped
        rc, lines, js = self.cli('0,0,1,-1,0', '--compute', '--dir', self.d, '--budget', 2, '--json')
        w = os.path.join(self.d, '0_0_1_-1_0', '0_0_1_-1_0.witness.json')
        kept = os.path.join(self.d, '0_0_1_-1_0', '0_0_1_-1_0.best.witness.json')     # its bytes kept apart
        self.assertEqual(rc, core.EXIT_STOPPED)
        self.assertIn('The best checked bound so far: Rank at most 3, unconditional. Witness: %s' % cli.shown(kept),
                      lines)
        with open(kept) as f:
            self.assertEqual(json.load(f)['S'], 3)
        self.assertIn(cli.text('attempted', rank='Rank at most 1', hyp='unconditional', path=cli.shown(w)), lines)
        self.assertEqual(lines[-2:], [cli.text('resume', command=js[0]['resume']), cli.text('stopped')])
        j = js[0]
        self.assertEqual((j['outcome'], j['best']['rank_bound'], j['best']['checked'], j['attempted']['rank_bound'],
                          j['attempted']['verification']), ('stopped', 3, True, 1, 'pending'))
        Checker.mode = 'accept'                         # resumed: the attempted witness is checked, not proved again
        rc, lines, js = self.cli('0,0,1,-1,0', '--compute', '--dir', self.d, '--budget', 2, '--json')
        self.assertEqual((js[0]['rank_bound'], js[0]['checked'], js[0]['outcome'], len(Route.log)), (1, True,
                                                                                                    'completed', 2))


class TestPublicNames(unittest.TestCase):
    def test_no_job(self):
        """No public rankbound.Job, and no wrappers over the old lifecycle; the documented API stays."""
        self.assertNotIn('Job', rankbound.__all__)
        for name in ('Job', 'deadline', 'job_end'):
            self.assertFalse(hasattr(rankbound, name), name)
        with self.assertRaises(ImportError):
            exec('from rankbound import Job')
        for name in ('rankbound', 'Result', 'Curve', 'Stopped', 'RankboundError'):
            self.assertTrue(hasattr(rankbound, name), name)

    def test_outcomes(self):
        self.assertEqual(core.FINAL, ('too_long', 'budget'))
        self.assertTrue(set(core.FINAL) <= set(core.OUTCOMES))
        self.assertEqual(core.RESULT_SCHEMA, 'rankbound-result-1')     # fields added compatibly


class TestOldFolder(unittest.TestCase):
    def test_old_key(self):
        """A run.json key without 'schema' (rankbound 3.0's and earlier development versions' legacy_key) is refused as an older version's
        folder; a folder of another request in this version stays 'another run'."""
        new = {'schema': 1, 'curve': ['0', '0', '0', '-1', '0'], 'hypothesis': 'GRH', 'name': 'x'}
        legacy = {'curve': ['0', '0', '0', '-1', '0'], 'r': None, 'points': None, 'relations': None, 'GRH': True}
        self.assertEqual(str(runtime.mismatch(legacy, new)), text('old_folder'))
        self.assertEqual(str(runtime.mismatch(dict(new, name='y'), new)), text('other_work'))
        with tempfile.TemporaryDirectory() as d:
            st = runtime.RunStore(d)
            st.begin(legacy)
            with self.assertRaisesRegex(runtime.Mismatch, 'older version'):
                st.begin(new)


class TestChanged(Base):
    def test_rechecked(self):
        """A published, checked witness whose bytes changed is checked again (not reported as before); unchanged, it
        is reported as before without a check."""
        Route.plan = [('ok', 1)]
        w = os.path.join(self.d, 'w')
        res = rankbound.rankbound(E1, compute=True, workdir=w)
        self.assertTrue(res.checked)
        n = len(Checker.log)
        again = rankbound.rankbound(E1, compute=True, workdir=w)
        self.assertEqual((len(Checker.log), again.details.get('earlier')), (n, True))
        with open(res.witness, 'a') as f:
            f.write(' ')                                # other bytes, the same JSON
        self.assertTrue(runner.changed(runtime.RunStore(w).info()['result']))
        third = rankbound.rankbound(E1, compute=True, workdir=w)
        self.assertEqual((len(Checker.log), third.checked, third.rank_bound), (n + 1, True, 1))
        self.assertFalse(runner.changed(runtime.RunStore(w).info()['result']))     # its new bytes, recorded
        self.assertEqual(len(Route.log), 1)             # never proved again


class TestBudgetFloor(unittest.TestCase):
    def test_floor(self):
        with tempfile.TemporaryDirectory() as d:
            st = runtime.RunStore(d)
            st.begin({'schema': 1})
            self.assertEqual(st.set_budget(0.01), 1.0)
        with tempfile.TemporaryDirectory() as d:
            st = runtime.RunStore(d)
            st.begin({'schema': 1})
            self.assertEqual(st.set_budget(2.0), 6.0)


if __name__ == '__main__':
    unittest.main()
