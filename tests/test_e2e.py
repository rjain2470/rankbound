"""Whole runs: 571a1 (rank 0, Sha[2] = (Z/2)^2) with the line siever, 37a1 with points on two models, and ICARM #1
with CADO-NFS las."""
import glob, gzip, json, os, shutil, signal, subprocess, sys, tempfile, time, unittest
from rankbound import verify as vf
from rankbound import Job, NotApplicable, RankboundError, Stopped, field, linalg, rankbound, sieve
from rankbound.rules import JOB_MARGIN
from rankbound.state import State

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = [sys.executable, '-m', 'rankbound']
# outside any Slurm job, so that no time limit comes from a job's end
ENV = dict({k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}, PYTHONPATH=ROOT)
N571 = '0_-1_1_-929_-10595'                         # 571a1's default name: its minimal model
E571 = [0, -1, 1, -929, -10595]


# ICARM #1 (r = 12): GB 3917, las at lpb 14; G = 10, rank <= 12
ICARM1 = dict(ainvs=[0, 0, 1, -6349808647, 193146346911036], bad_primes=[60259, 550469, 11241887, 722983930261], r=12)


class TestEndToEnd(unittest.TestCase):
    """One run of 571a1 through rankbound() (setUpClass), shared by the tests below."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(); cwd = os.getcwd()
        try:
            os.chdir(cls.tmp)
            cls.res = rankbound(E571, cores=1)
        finally:
            os.chdir(cwd)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_api(self):
        """rankbound() on any model: rank_bound 2 (rank 0, Sha[2] = (Z/2)^2), verified; a second call resumes from
        result.json; NotApplicable when E(Q)[2] != 0; the default workdir is ./rankbound_work/NAME, NAME the minimal
        model's a-invariants."""
        res = self.res
        self.assertEqual((res.rank_bound, res.G, res.u, res.n, res.selmer_bound), (2, 1, 1, 0, 2))
        self.assertTrue(res.verified); self.assertEqual(res.outcome, 'plateau')
        self.assertEqual((res.rank_lower_bound, res.r_proved, res.proved_equal), (None, False, False))
        self.assertEqual(os.path.dirname(res.workdir), os.path.realpath(os.path.join(self.tmp, 'rankbound_work')))
        self.assertEqual((os.path.basename(res.workdir), res.name), (N571, N571))
        self.assertIn('verify', res.cost)
        t = time.time()
        self.assertEqual(rankbound(E571, workdir=res.workdir), res)
        self.assertLess(time.time() - t, 15)
        with self.assertRaises(NotApplicable):
            rankbound([0, 0, 0, -1, 0])                # y^2 = x^3 - x: full 2-torsion

    def test_certificate(self):
        """A verified certificate NAME.cert.json, with its keys in order; another format, a changed relation, pivot
        list or claim fails."""
        cert = self.res.certificate
        self.assertEqual(cert, os.path.join(self.res.workdir, N571 + '.cert.json'))
        with open(os.path.join(self.res.workdir, 'verify.json')) as f:      # the verifier's verdict (verify --json)
            v = json.loads(f.read())
        self.assertEqual((v['verdict'], v['certificate'], v['rank_upper_bound']), ('accepted', cert, 2))
        self.assertTrue(os.path.getsize(os.path.join(self.res.workdir, 'verify.log')))
        c0 = vf.load(cert)
        self.assertEqual(list(c0), ['name', 'curve', 'statement', 'field', 'local', 'claims', 'created', 'relations',
                                    'ops', 'pivots', 'format'])
        for what in ('format', 'rel', 'ops', 'claim'):
            c = json.loads(json.dumps(c0))
            if what == 'format':
                c['format'] = 'x'
            elif what == 'rel':
                i = next(k for k, l in enumerate(c['relations']) if not l.startswith('f:'))
                a, rest = c['relations'][i].split(',', 1); c['relations'][i] = '%d,%s' % (int(a) + 1, rest)
            elif what == 'ops':
                c['pivots'].append(c['pivots'][0]); c['claims']['dimV'] += 1; c['claims']['G'] -= 1
            else:
                c['claims']['S'] -= 2
            t = os.path.join(self.tmp, what + '.cert.json')
            with open(t, 'w') as f:
                json.dump(c, f)
            self.assertFalse(vf.verify(t, log=lambda *a: None)[0], what)     # a FAIL, not an exception

    def fresh(self, name):
        """A copy of the run's state after prepare and tune (no slice, check or result)."""
        st = os.path.join(self.tmp, name)
        shutil.copytree(self.res.workdir, st, ignore=lambda d, fs: [f for f in fs if f in (
            'slices', 'la', 'history.json', 'STOP', 'result.json', 'verify.log', 'verify.json') or
            f.endswith(('.cert.json', '.lock'))])
        return st

    def test_shards(self):
        """Two concurrent shards, one in a Slurm job, reach the local run's bound G = 1, by a plateau or, when they
        stop by time, through `rankbound solve`.  sec_per_q is raised so that a slice is one line."""
        B = self.fresh('shards')
        p = os.path.join(B, 'params.json')
        with open(p) as f:
            prm = json.load(f)
        prm['polys']['F']['sec_per_q'] = 1e3
        with open(p, 'w') as f:
            json.dump(prm, f)
        # shard 1 stops at a fake Slurm job's end, shard 0 at --max-hours
        env1 = dict(ENV, SLURM_JOB_ID='1', SLURM_JOB_END_TIME='%d' % (time.time() + JOB_MARGIN + 8))
        ps = [subprocess.Popen(CLI + ['sieve', B, '--shard', '%d/2' % k, '--cores', str(2 - k)] +
                               (['--max-hours', '0.002'] if k == 0 else ['--max-hours', '5']),
                               cwd=self.tmp, env=ENV if k == 0 else env1, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True) for k in range(2)]
        for p in ps:
            self.end_on_failure(p)
        outs = [p.communicate(timeout=600)[0] for p in ps]
        self.assertEqual([p.returncode for p in ps], [0, 0], outs)
        self.assertIn('5 minutes before the Slurm job ends.', outs[1])
        if not os.path.exists(os.path.join(B, 'STOP')):
            r = subprocess.run(CLI + ['solve', B], cwd=self.tmp, env=ENV, capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(State(B).history()[-1]['bound'], self.res.G)
        self.assertGreaterEqual(len(State(B).finished()), 2)     # how many depends on timing

    def test_time_limit(self):
        """A check that did not finish in an earlier run with as much time and as many threads (check.json) stops the
        run with a request for more cores or time.  Without check.json, a check longer than the time left starts at
        once in a run that has just started."""
        B = self.fresh('limit')
        S = State(B)
        name = State(self.res.workdir).finished()[0]           # one slice of the run: a check is due at once
        for ext in ('.rels.gz', '.json'):
            shutil.copy(os.path.join(self.res.workdir, 'slices', name + ext), S.sdir)
        S.save_history([{'relations': 1, 'rows': 1, 'excess': 1, 'bound': 5, 'solve_s': 1e6}])
        with open(os.path.join(B, 'check.json'), 'w') as f:
            json.dump({'relations': 1, 'start': time.time() - 3600, 'until': time.time(), 'threads': 1}, f)
        with self.assertRaisesRegex(RankboundError, r'^The last check did not finish in the time it had \(1.0 h on 1 core\). '
                                    'Give more cores or more time.$'):
            Job(B).sieve(1, max_hours=0.0005)
        self.assertEqual(len(S.history()), 1)
        os.remove(os.path.join(B, 'check.json'))
        with self.assertLogs('rankbound', 'INFO') as cm:
            Job(B).sieve(1, max_hours=0.0005)
        self.assertTrue(any('relations: rank <= ' in l for l in cm.output), cm.output)       # the check ran
        self.assertGreaterEqual(len(S.history()), 2)
        self.assertIsNone(S.unfinished_check())

    def test_due(self):
        """A check that an earlier run left to the next one (due.json, e.g. due because the sieve dried up) is the
        next run's first act, although the relation count has not reached the next check (here 1.15 x 10^9)."""
        B = self.fresh('due')
        S = State(B)
        for p in glob.glob(os.path.join(self.res.workdir, 'slices', '*')):
            shutil.copy(p, S.sdir)
        S.save_history([{'relations': 10 ** 9, 'rows': 1, 'excess': 1, 'bound': 5, 'solve_s': 1}])
        S.mark_due(1)
        self.assertEqual(Job(B).sieve(1, max_hours=0.002), 'plateau')
        h = S.history()
        self.assertGreater(h[1]['relations'], 0)
        self.assertEqual(h[1]['bound'], self.res.G)
        self.assertFalse(S.due())

    def test_stop(self):
        """A shard without the solver abandons its slices within seconds of STOP, leaves no partial output, and does
        not certify."""
        B = self.fresh('stop')
        p = subprocess.Popen(CLI + ['sieve', B, '--shard', '1/2', '--cores', '2'], cwd=self.tmp, env=ENV,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.end_on_failure(p)
        self.wait_slices(B, p)
        State(B).write_stop('goal'); t = time.time()
        out = p.communicate(timeout=60)[0]
        self.assertEqual(p.returncode, 0, out); self.assertLess(time.time() - t, 10)
        self.assertEqual(glob.glob(os.path.join(B, 'slices', '*.las*')), [])
        self.assertFalse(os.path.exists(os.path.join(B, 'result.json')))

    def end_on_failure(self, p):
        """Send SIGTERM to the `rankbound sieve` P if it still runs when the test ends, so that it stops its sieve
        processes."""
        def end():
            if p.poll() is None:
                p.terminate()
                try:
                    p.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill(); p.communicate()
        self.addCleanup(end)

    def wait_slices(self, B, p):
        """Wait until the `rankbound sieve` P on the state B has a slice's partial output."""
        t = time.time()
        while not glob.glob(os.path.join(B, 'slices', '*.las*')) and p.poll() is None and time.time() - t < 60:
            time.sleep(0.05)
        if p.poll() is not None:
            self.fail('it exited (%s) before it sieved: %s' % (p.returncode, p.communicate()[0]))
        self.assertTrue(glob.glob(os.path.join(B, 'slices', '*.las*')))

    def test_sigterm(self):
        """SIGTERM to `rankbound sieve` (scancel, or the Slurm job's end when it could not be read): exit 143
        within seconds, its sieve processes (in their own sessions) stopped and their partial slices deleted."""
        B = self.fresh('term')
        p = subprocess.Popen(CLI + ['sieve', B, '--shard', '0/1', '--cores', '2'], cwd=self.tmp, env=ENV,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.end_on_failure(p)
        self.wait_slices(B, p)
        p.send_signal(signal.SIGTERM); t = time.time()
        out = p.communicate(timeout=60)[0]
        self.assertEqual(p.returncode, 128 + signal.SIGTERM, out); self.assertLess(time.time() - t, 10)
        self.assertEqual(glob.glob(os.path.join(B, 'slices', '*.las*')), [])
        time.sleep(1)
        with self.subTest('no sieve process left'):
            try:
                ps = subprocess.run(['ps', '-A', '-ww', '-o', 'command'], capture_output=True, text=True, timeout=60)
            except OSError as e:
                ps = subprocess.CompletedProcess('ps', 1, '', str(e))
            if ps.returncode:                   # e.g. in a sandbox that does not let a process list others
                self.skipTest('ps cannot list the processes: %s' % ps.stderr.strip())
            self.assertEqual([l for l in ps.stdout.splitlines() if B in l], [])

    def test_solve_stopped(self):
        """`rankbound solve` on a stopped state without a certificate rebuilds the last full solve's matrix in
        la/order.txt's order, reuses its elimination, and writes a certificate with the same claims."""
        B = os.path.join(self.tmp, 'stopped')
        shutil.copytree(self.res.workdir, B, ignore=lambda d, fs: [f for f in fs if f in ('result.json', 'verify.log',
                                                                                          'verify.json') or
                                                                  f.endswith(('.cert.json', '.lock'))])
        rec = os.path.join(B, 'la', 'elim.json')
        t = os.stat(rec).st_mtime_ns
        r = subprocess.run(CLI + ['solve', B], cwd=self.tmp, env=ENV, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('RESULT %s: %s' % (N571, self.res.statement), r.stdout)
        self.assertEqual(os.stat(rec).st_mtime_ns, t)                   # no second elimination
        c = vf.load(os.path.join(B, N571 + '.cert.json'))
        self.assertEqual(c['claims'], vf.load(self.res.certificate)['claims'])
        with open(os.path.join(B, 'verify.json')) as f:
            self.assertEqual(json.loads(f.read())['verdict'], 'accepted')

    def test_solve_once(self):
        """`rankbound solve` on a state with slices but no STOP runs one check on every finished slice; here a
        plateau, so STOP is written and the certificate follows."""
        B = os.path.join(self.tmp, 'once')
        shutil.copytree(self.res.workdir, B, ignore=lambda d, fs: [f for f in fs if f in (
            'la', 'history.json', 'STOP', 'result.json', 'verify.log', 'verify.json') or
            f.endswith(('.cert.json', '.lock'))])
        State(B).save_history([{'relations': 1, 'rows': 1, 'excess': 1, 'bound': self.res.G, 'solve_s': 1}])
        r = subprocess.run(CLI + ['solve', B], cwd=self.tmp, env=ENV, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(State(B).stopped(), r.stdout)
        self.assertEqual(State(B).stop_outcome(), 'plateau')
        self.assertEqual(State(B).history()[-1]['bound'], self.res.G)
        self.assertIn('RESULT %s: %s' % (N571, self.res.statement), r.stdout)

    def test_curve_list(self):
        """`rankbound bound --curves` in a Slurm job reports a finished state's result again; a state past its time
        limit stops without a result (Stopped); `rankbound status` prints one row per state folder, tuned or not."""
        L = os.path.join(self.tmp, 'list.json')
        with open(L, 'w') as f:
            json.dump({'curves': [{'id': 1, 'ainvs': E571, 'bad_primes': ['571']}]}, f)
        D = os.path.join(self.tmp, 'list')
        shutil.copytree(self.res.workdir, os.path.join(D, 'curve1'), ignore=lambda d, fs: [f for f in fs if f.endswith('.lock')])
        env = dict(ENV, SLURM_JOB_ID='1', SLURM_JOB_END_TIME='%d' % (time.time() + 3600))
        r = subprocess.run(CLI + ['bound', '--dir', D, '--curves', L, '--ids', '1', '--cores', '1', '--max-hours', '1',
                                  '--mem', '1.5'], cwd=self.tmp, env=env, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('5 minutes before the Slurm job ends.', r.stdout)
        self.assertIn('RESULT %s: rank <= 2 under GRH, certificate verified (0.0' % N571, r.stdout)
        late = os.path.join(self.tmp, 'late')
        os.makedirs(late)
        os.rename(self.fresh('late_curve1'), os.path.join(late, 'curve1'))
        r = subprocess.run(CLI + ['bound', '--dir', late, '--curves', L, '--max-hours', '0'], cwd=self.tmp, env=ENV,
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn('The time limit has already passed; nothing to do.', r.stdout)
        self.assertIn('Stopped at the time limit. Run the same command again to resume.', r.stdout)
        self.assertNotIn('RESULT', r.stdout)
        with self.assertRaises(Stopped):
            rankbound(E571, workdir=os.path.join(late, 'curve1'), max_hours=0)
        P = os.path.join(D, 'prepared')
        os.makedirs(P)
        shutil.copy(os.path.join(self.res.workdir, 'field.json'), P)
        r = subprocess.run(CLI + ['status', D], cwd=self.tmp, env=ENV, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rows = [l.split('\t') for l in r.stdout.splitlines()]
        self.assertEqual(rows[0], ['name', 'relations', 'core-hours', 'estimated', 'rank bound', 'status'])
        self.assertEqual(len(rows), 3)
        self.assertEqual((rows[1][0], rows[1][4:]), (N571, ['2', 'finished, verified']))       # curve1
        self.assertEqual(rows[2], [N571, '0', '0.0', '', '', 'not started'])                     # prepared

    def test_relations(self):
        """The certificate's relations fed back, gzip-compressed or as bare pairs (a, b) on the command line, prove the
        same bound without a sieve, and a second call resumes.  Other relations in the same workdir, relations for a
        finished state, a wrong relation, and sieve=False without relations are refused."""
        cert = self.res.certificate
        gz = os.path.join(self.tmp, 'given.cert.json.gz')
        with open(cert, 'rb') as f, gzip.open(gz, 'wb') as g:
            g.write(f.read())
        with self.assertRaisesRegex(RankboundError, 'was started for another curve'):
            rankbound(E571, relations=gz, workdir=self.res.workdir)
        W = os.path.join(self.tmp, 'given')
        r = rankbound(E571, relations=gz, workdir=W)
        self.assertEqual((r.G, r.rank_bound, r.verified, r.outcome), (self.res.G, 2, True, 'relations'))
        with open(os.path.join(W, 'params.json')) as f:
            self.assertEqual(json.load(f)['siever'], 'given')
        self.assertEqual(State(W).finished(), ['given'])
        self.assertEqual(rankbound(E571, relations=gz, workdir=W), r)
        with self.assertRaisesRegex(RankboundError, 'was started for another curve'):      # its result is not a sieve's
            rankbound(E571, relations=gz, workdir=W, sieve=True)
        pairs = [l.split(':')[0] for l in vf.load(cert)['relations'] if not l.startswith('f:')]
        with self.assertRaisesRegex(RankboundError, 'was started for another curve'):
            rankbound(E571, relations=pairs[1:], workdir=W)
        txt = os.path.join(self.tmp, 'pairs.txt')
        with open(txt, 'w') as f:
            f.write('# 571a1: (a, b) of a - b alpha\n' + '\n'.join(pairs) + '\n')
        p = subprocess.run(CLI + ['bound', ','.join(map(str, E571)), '--relations', txt], cwd=self.tmp, env=ENV,
                           capture_output=True, text=True, timeout=600)     # default --dir: next to the sieve's folder
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertRegex(p.stdout, r'[0-9,]+ relations: rank <= 2 \(aim: rank <= 0\)\.\n')
        self.assertIn('RESULT %s: rank <= 2 under GRH, certificate verified (0.00 core-h)' % N571, p.stdout)
        rel = [d for d in os.listdir(os.path.join(self.tmp, 'rankbound_work')) if d.startswith(N571 + '_rel')]
        self.assertEqual(len(rel), 1)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, 'rankbound_work', rel[0], N571 + '.cert.json')))
        a, b = map(int, pairs[0].split(','))
        with self.assertRaisesRegex(ValueError, '^The primes given for relation 1 '):
            rankbound(E571, relations=[(a, b, [2])], workdir=os.path.join(self.tmp, 'wrong'))
        with self.assertRaisesRegex(ValueError, '^sieve=False needs relations=....$'):
            rankbound(E571, sieve=False)

    def test_batch(self):
        """`rankbound estimate` over a curve list: a curve that cannot be prepared (a composite in its discriminant) is
        skipped with a reason, the others are prepared; the exit status is then 1."""
        path = os.path.join(self.tmp, 'two.json')
        with open(path, 'w') as f:
            json.dump({'curves': [{'id': 1, 'ainvs': E571, 'bad_primes': ['571'], 'r': 0},
                                  {'id': 2, 'ainvs': E571, 'bad_primes': [], 'composites': ['3997'], 'r': 0}]}, f)
        r = subprocess.run(CLI + ['estimate', '--dir', os.path.join(self.tmp, 'batch'), '--curves', path], cwd=self.tmp,
                           env=ENV, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn('Skipped curve2. The discriminant is not fully factored. Give all its prime factors in "bad_primes".',
                      r.stdout)
        self.assertRegex(r.stdout, r'1 curves; estimated cost [0-9.]+ core-hours in all\. Skipped: curve2\.')
        self.assertTrue(os.path.exists(os.path.join(self.tmp, 'batch', 'curve1', 'params.json')))


E37 = ['0', '0', '1', '-1', '0']                   # 37a1 (minimal), rank 1: P = (0, 0)
E37_SHORT = ['0', '0', '0', '-1296', '11664']        # a short Weierstrass model of it (u = 1/6): P = (0, +-108)


class TestModels(unittest.TestCase):
    """Certificates on the reduced minimal model, whatever model the curve and its points are given on; 37a1's
    certificate (rankbound(points=...), setUpClass) shared by the tests below."""
    MM_POINTS = ([['0', '0']], [['0', '-1']])        # (0, +-108) of the short model on the minimal one

    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.r37 = rankbound(E37, points=[(0, 0)], workdir=os.path.join(cls.d, 'e37'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def test_points_of_another_model(self):
        """A state prepared from the minimal model, run from a list entry on the short model with its point (0, 108):
        the certificate is on the minimal model, with the point mapped onto it and primes chosen there, and proves
        rank = 1.  The relations are the first certificate's, without a sieve."""
        st = os.path.join(self.d, 'list37')
        Job(st).prepare(field.from_ainvs(E37, r=1, name='e37'))
        path = os.path.join(self.d, 'list.json')
        primes = [p for p in range(2, 100) if all(p % q for q in range(2, p))]
        with open(path, 'w') as f:
            json.dump({'curves': [{'id': 37, 'name': 'e37', 'ainvs': E37_SHORT, 'bad_primes': ['2', '3', '37'], 'r': 1,
                                   'proof': {'points': [['0', '108']], 'primes': primes}}]}, f)
        cur = field.from_list(path)[0]
        self.assertEqual(cur.proof['ainvs'], E37_SHORT)           # the model the list's points are on
        r = Job(st).run(cur, relations=self.r37.certificate, sieve=False)
        self.assertEqual((r['verified'], r['rank_bound']), (True, 1))
        c = vf.load(r['certificate'])
        self.assertEqual(c['curve']['ainvs'], E37)
        rl = c['rank_lower_bound']
        self.assertEqual(rl['r'], 1)
        self.assertIn(rl['points'], self.MM_POINTS)
        with open(os.path.join(st, 'verify.json')) as f:
            self.assertIn({'check': 'rank_lower_bound', 'ok': True}, [{k: c[k] for k in ('check', 'ok')}
                                                                     for c in json.loads(f.read())['checks']])
        self.assertIn('rank E(Q) = 1', r['statement'])

    def test_points(self):
        """rankbound(points=...) proves rank = 1 for 37a1 with a prime it finds.  `rankbound bound --points
        --relations --keep-sieving` on the short model gives a certificate on the minimal model.  A point off the
        curve is refused before anything is sieved."""
        r, d = self.r37, self.d
        self.assertEqual((r.rank_bound, r.rank_lower_bound, r.proved_equal, r.verified, r.outcome),
                         (1, 1, True, True, 'goal'))
        rl = vf.load(r.certificate)['rank_lower_bound']
        self.assertEqual((rl['points'], len(rl['primes'])), ([['0', '0']], 1))
        pts = os.path.join(d, 'points.txt')
        with open(pts, 'w') as f:
            f.write('0 108\n')
        p = subprocess.run(CLI + ['bound', '--dir', d, ','.join(E37_SHORT), '--name', 'short', '--points', pts,
                                  '--relations', r.certificate, '--keep-sieving'],
                           cwd=d, env=ENV, capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn('RESULT short: rank = 1 under GRH, certificate verified', p.stdout)
        c = vf.load(os.path.join(d, 'short', 'short.cert.json'))
        self.assertEqual(c['curve']['ainvs'], E37)
        self.assertIn(c['rank_lower_bound']['points'], self.MM_POINTS)
        self.assertIn('rank E(Q) = 1', c['statement'])
        W = os.path.join(d, 'off')
        with self.assertRaisesRegex(RankboundError, '^Point 1 is not on the curve.$'):
            rankbound(E37, points=[(0, 1)], workdir=W)
        self.assertFalse(os.path.exists(os.path.join(W, 'params.json')))      # no trial sieve, no sieve

    def test_cited_r(self):
        """r given without points is cited: S = r = 1 proves rank <= 1 only, so r_proved and proved_equal are False and
        the statement says r is cited (with points both are True).  The default workdir keeps r apart from a run
        without it (here on the first certificate's relations, no sieve)."""
        self.assertEqual((self.r37.r_proved, self.r37.proved_equal), (True, True))
        cwd = os.getcwd()
        try:
            os.chdir(self.d)
            r = rankbound(E37, r=1, relations=self.r37.certificate)
        finally:
            os.chdir(cwd)
        self.assertEqual((r.rank_bound, r.rank_lower_bound, r.r_proved, r.proved_equal, r.verified),
                         (1, 1, False, False, True))
        self.assertIn('(cited; not proved in this certificate)', r.statement)
        self.assertRegex(os.path.basename(r.workdir), r'^0_0_1_-1_0_r1_rel[0-9a-f]{8}$')

    def test_refused_proof(self):
        """A proof that Job.prepare refuses leaves the state's curve.json as it was (`rankbound sieve` and `solve` read
        it).  When K is a cyclic cubic field (x^3 - 3x + 1: d_K = 81), points cannot prove r, as no prime has
        #E(F_p) = 2 mod 4: refused with that reason."""
        p = os.path.join(self.r37.workdir, 'curve.json')
        with open(p) as f:
            before = f.read()
        with self.assertRaisesRegex(RankboundError, '^Point 1 is not on the curve.$'):
            Job(self.r37.workdir).prepare(field.with_points(field.from_ainvs(E37), [(0, 1)]))
        with open(p) as f:
            self.assertEqual(f.read(), before)
        with self.assertRaisesRegex(RankboundError, r'^Points cannot prove the rank of this curve \(its cubic field is '
                                    r'cyclic\)\. Give r with --r instead; it will be cited, not proved\.$'):
            rankbound([0, 0, 0, -3, 1], points=[(0, 1)], workdir=os.path.join(self.d, 'cyclic'))

    def test_recert(self):
        """linalg.recert rewrites a certificate on a short model onto its minimal model: the point is mapped, the rest
        is unchanged, and it verifies.  A cited r is kept; a certificate already on its minimal model is left alone;
        a point off the curve, or an existing OUT, is refused."""
        src = self.r37.certificate
        head, off = linalg.read_head(src)
        with open(src, 'rb') as f:
            tail = f.read()[off:]

        def write(name, h):
            path = os.path.join(self.d, name + '.cert.json')
            with open(path, 'wb') as f:
                f.write(linalg.head_text(h).encode() + tail)
            return path
        rl = linalg.rank_proof(E37_SHORT, ['37'], 1, dict(head['rank_lower_bound'], ainvs=E37))
        short = dict(head, curve=dict(head['curve'], ainvs=E37_SHORT), rank_lower_bound=rl)
        old = write('short', short)
        quiet = lambda *a: None
        self.assertTrue(vf.verify(old, log=quiet)[0])
        out = os.path.join(self.d, 'recert.cert.json')
        self.assertEqual(linalg.recert(old, out)['curve']['ainvs'], E37)
        new, k = linalg.read_head(out)
        self.assertEqual(list(new), list(short))
        self.assertEqual(new['curve'], head['curve'])                 # ainvs = minimal_model
        self.assertIn(new['rank_lower_bound']['points'], self.MM_POINTS)
        self.assertEqual(new['rank_lower_bound']['primes'], rl['primes'])  # the old primes still prove it
        self.assertEqual({x: v for x, v in new.items() if x not in ('curve', 'rank_lower_bound')},
                         {x: v for x, v in short.items() if x not in ('curve', 'rank_lower_bound')})
        with open(out, 'rb') as f:
            self.assertEqual(f.read()[k:], tail)
        self.assertTrue(vf.verify(out, log=quiet)[0])
        with self.assertRaisesRegex(ValueError, 'exists'):
            linalg.recert(old, out)
        x = os.path.join(self.d, 'x.cert.json')
        self.assertIsNone(linalg.recert(src, x))                        # already on its minimal model: nothing written
        self.assertFalse(os.path.exists(x))
        cited = {'r': 1, 'source': 'cited'}
        st = vf.statement(head['claims']['G'], head['local'], 1, 'cited')[1]
        c = write('cited', dict(short, statement=st, rank_lower_bound=cited))
        self.assertTrue(vf.verify(c, log=quiet)[0])
        self.assertEqual(linalg.recert(c, x)['rank_lower_bound'], cited)
        self.assertTrue(vf.verify(x, log=quiet)[0])
        bad = write('bad', dict(short, rank_lower_bound=dict(rl, points=[['0', '107']])))
        y = os.path.join(self.d, 'y.cert.json')
        with self.assertRaisesRegex(RuntimeError, 'Point 1 is not on the curve'):
            linalg.recert(bad, y)
        self.assertFalse(os.path.exists(y))


class TestLas(unittest.TestCase):
    def test_icarm1(self):
        """A whole run with CADO-NFS las, when it is built: ICARM #1, G = 10, rank <= 12 = r, verified.  Covers the
        trial sieve, makefb, the dummy side and two sieve processes; `make test-fast` skips it."""
        if os.environ.get('RANKBOUND_TEST_FAST'):
            self.skipTest('RANKBOUND_TEST_FAST (make test-fast)')
        try:
            sieve.binary('las'); sieve.binary('makefb')
        except RuntimeError:
            self.skipTest('CADO-NFS las not built (make cado)')
        d = tempfile.mkdtemp()
        try:
            J = Job(os.path.join(d, 'icarm1'))
            r = J.run(field.from_ainvs(ICARM1['ainvs'], ICARM1['bad_primes'], ICARM1['r'], 'icarm1'), cores=2, mem_gb=1.0)
            self.assertEqual((r['outcome'], r['G'], r['rank_bound'], r['verified']), ('goal', 10, 12, True))
            with open(os.path.join(J.dir, 'params.json')) as f:
                prm = json.load(f)
            self.assertNotEqual(prm.get('siever'), 'line'); self.assertEqual(prm['lpb0'], 14)
        finally:
            shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
