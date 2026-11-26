"""The command line (cli.py): every message renders, the arguments, the routing, run.json and `status`, --json,
--shard, the table of witnesses; what the prover is given: curve lists, values pasted into gp, given relations
and points, the default name; the Python API (rankbound()); determinism of the witnesses."""
import contextlib, dataclasses, gzip, io, json, os, re, subprocess, sys, tempfile, time, unittest
from rankbound.check import selmer as vsel
import rankbound as rb
from rankbound import curve as cv, field, sieve as relations
from rankbound import runtime, sieve
jobs = sieve
from helpers import PKG, SRC, TESTS, wit0

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestArguments(unittest.TestCase):
    def test_positive(self):
        """Bad --cores and --max-hours, and the same keyword values, are refused before anything is written."""
        from rankbound import __main__ as cli, rankbound
        with tempfile.TemporaryDirectory() as d:        # nothing may be written here
            for args in (['bound', '--dir', d, '0,0,1,-1,0', '--cores', '0'], ['bound', '--dir', d, '1,1', '--max-hours', 'nan'],
                         ['bound', '--dir', d, '0,0,1,-1,0', '--max-hours', '-1'], ['bound', '--dir', d, '--cores', 'two']):
                with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()) as err:
                    cli.main(args)
                self.assertEqual(e.exception.code, 2, args)
                self.assertRegex(err.getvalue(), "^rankbound bound: error: argument --(cores|max-hours): '[^']*' (must be "
                                 "(positive|>= 0)|is not a whole number|is not a number like 2 or 0.5). See rankbound "
                                 "bound --help.\n$", args)
            for kw in ({'cores': 0}, {'cores': 1.5}):
                self.assertRaisesRegex(ValueError, 'must be a positive', rankbound, [0, 0, 1, -1, 0],
                                       workdir=os.path.join(d, 'w'), **kw)
            self.assertEqual(os.listdir(d), [])

    def test_shard(self):
        """--shard K/N needs integers 1 <= K <= N."""
        from rankbound import cli
        with tempfile.TemporaryDirectory() as d:
            for k in ('0/1', '2/1', '0/0', 'x', '-1/2', '1/-2', '0/1/2', '0.5/2'):
                with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()) as err:
                    cli.main(['bound', '0,0,1,-1,0', '--dir', d, '--shard=' + k])
                self.assertEqual(e.exception.code, 2, k)
                self.assertIn('must be K/N with 1 <= K <= N', err.getvalue(), k)
            self.assertEqual(os.listdir(d), [])

    def test_names(self):
        """A curve list's name must be one the checker accepts, and not . or ..; a bad one is refused before any work."""
        bad = ('../x', 'a/b', 'a b', '.', '..', '', 'x' * 65, 'icarm\u0660', '/abs')
        for name in bad:
            self.assertRaisesRegex(ValueError, 'name', cv.check_name, name)
        for name in ('icarm728', 'E28', 'a.b-c_1', '...'):
            self.assertEqual(cv.check_name(name), name)
        with tempfile.TemporaryDirectory() as d:
            for name in bad[:5]:
                path = os.path.join(d, 'list.json')
                with open(path, 'w') as f:
                    json.dump({'curves': [dict(TestGpInput.GOOD, name=name)]}, f)
                self.assertRaisesRegex(ValueError, '^list.json, curve 1: The name', cv.from_list, path)
                os.remove(path)
            self.assertEqual(os.listdir(d), [])

    def test_spawn_stderr(self):
        """A slice's stderr goes to a file, not a pipe that nobody reads until the process exits: a siever that writes
        more than a pipe holds (64 KB) does not block, and slice_failed reads all of it."""
        p = sieve.spawn([sys.executable, '-c', 'import sys; sys.stderr.write("x" * 300000 + "end"); sys.exit(1)'])
        h = sieve.Slice(type('Server', (), {'p': p})())
        try:
            self.assertEqual(p.wait(timeout=60), 1)
            h.returncode = p.returncode
            with self.assertRaisesRegex(RuntimeError, '^The siever failed. x+end$'):
                jobs.slice_failed(h)
        finally:
            p.stderr.close()
            runtime.reap(p)             # sieve.spawn registers it (runtime's children)



class TestInput(unittest.TestCase):
    """What a laptop user types: one curve per command (also when it starts with a minus sign, and after --), a curve
    copied from LMFDB, Sage or PARI, numbers in ASCII digits only, the contents of --points, --relations and
    --curves read before any work; each mistake one line on stderr, exit 2, nothing written."""

    def main(self, args):
        """(exit status, stderr) of cli.main(args)."""
        from rankbound import cli
        with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()):
            try:
                code = cli.main(args)
            except SystemExit as e:
                code = e.code
        return code, err.getvalue()

    def test_one_curve(self):
        """A second CURVE is a usage error, never dropped (with a minus sign, or after --); the rewrite of a curve
        with a minus sign leaves the values of options alone."""
        from rankbound import cli
        with tempfile.TemporaryDirectory() as d:
            for args in (['1,1', '-1,1'], ['-1,1', '-2,2'], ['1,1', '2,2'], ['--', '-1,1', '-2,2'], ['1,1', '--', '-1,1']):
                self.assertEqual(self.main(['bound', '--estimate', '--dir', d] + args),
                                 (2, 'Give one curve per command, or several with --curves FILE.\n'), args)
            self.assertEqual(os.listdir(d), [])
        takes = {'--ids', '--dir', '--r'}
        self.assertEqual(cli.curve_args(['--ids', '-1,2', '-3,4', '--dir=-5,6', '--', '-7,8'], takes),
                         ['--ids', '-1,2', '--curve-arg=-3,4', '--dir=-5,6', '--', '-7,8'])
        self.assertEqual(cli.curve_args(['--di', '-1,2'], takes), ['--di', '-1,2'])      # a prefix of --dir
        code, err = self.main(['bound', '1,1', '--frobnicate'])
        self.assertEqual((code, err), (2, 'rankbound bound: error: unrecognized arguments: --frobnicate. See rankbound '
                                          'bound --help.\n'))

    def test_curve_forms(self):
        """[a1,...] and spaces around the entries are read; anything else is refused naming the entry."""
        from rankbound import cli
        for s, want in (('[0,1,1,-2,0]', ['0', '1', '1', '-2', '0']), ('0, 1, 1, -2, 0', ['0', '1', '1', '-2', '0']),
                        (' [-25, 0] ', ['-25', '0'])):
            self.assertEqual(cli.curve_text(s), want)
            cv.ainvs_given(cli.curve_text(s))
        for s, bad in (('[[0,1]]', '[0'), ('(0,1)', '(0'), ('1,\uff11', '\uff11'), ('1_0,1', '1_0'), ('+1,1', '+1'),
                       ('1,\u22121', '\u22121'), ('1,\u00a01', '\u00a01')):
            self.assertRaisesRegex(ValueError, "^The a-invariant %s is not an integer.$" % re.escape(repr(bad)),
                                   cv.ainvs_given, cli.curve_text(s))

    def test_numbers(self):
        """Numeric options take ASCII digits only, finite; the message says what was given."""
        with tempfile.TemporaryDirectory() as d:
            for o, v, why in (('--cores', '\uff12', 'is not a whole number'), ('--cores', ' 2', 'is not a whole number'),
                              ('--r', '1_0', 'is not a whole number'), ('--r', '+1', 'is not a whole number'),
                              ('--r', '-1', 'must be >= 0'), ('--max-hours', '1e400', 'is too large'),
                              ('--budget', 'inf', 'is not a number like 2 or 0.5'),
                              ('--ecm-hours', '\u0661', 'is not a number like 2 or 0.5'),
                              ('--cores', '9' * 50, 'is too large')):
                code, err = self.main(['bound', '1,1', '--dir', d, o, v])
                shown = repr(v[:20]) + '...' if len(v) > 40 else repr(v)
                self.assertEqual((code, err), (2, 'rankbound bound: error: argument %s: %s %s. See rankbound bound '
                                                  '--help.\n' % (o, shown, why)), (o, v))
            self.assertEqual(os.listdir(d), [])

    def test_ignored_options(self):
        """--ids without --curves, an empty --bad-primes or --points file: refused, not ignored."""
        with tempfile.TemporaryDirectory() as d:
            empty = os.path.join(d, 'empty.txt')
            open(empty, 'w').close()
            run = os.path.join(d, 'run')
            for args, line in ((['1,1', '--ids', '3'], '--ids works with --curves only.'),
                               (['1,1', '--bad-primes', ''], '--bad-primes is empty. Give the primes, like --bad-primes '
                                                             '2,3,37, or leave it out.'),
                               (['0,0,1,-1,0', '--points', empty], 'The file %s given with --points has no points. Give '
                                                                   'one "x y" per line, or leave it out.' % empty)):
                self.assertEqual(self.main(['bound', '--dir', run] + args), (2, line + '\n'), args)
            self.assertFalse(os.path.exists(run))

    def test_contents(self):
        """Malformed --points, --relations and --curves contents are usage errors (exit 2), found before any work."""
        with tempfile.TemporaryDirectory() as d:
            def put(name, text):
                with open(os.path.join(d, name), 'w') as f:
                    f.write(text)
                return os.path.join(d, name)
            run = os.path.join(d, 'run')
            for args, line in (
                    (['0,0,1,-1,0', '--points', put('p.txt', '0 0\n1 2 3\n')], '%s, line 2: not a point "x y".'),
                    (['0,0,1,-1,0', '--points', put('q.txt', '0 \uff10\n')],
                     "%s, line 1: The point coordinate '\uff10' is not written like -3/4."),
                    (['0,0,1,-1,0', '--points', put('d.txt', '0 0\n0 -1\n')],
                     'Points 1 and 2 have the same x, so one is plus or minus the other: they are dependent.'),
                    (['0,0,1,-1,0', '--relations', put('r.txt', '1,x\n')],
                     'In %s, relation \'1,x\' is not of the form "a,b" or "a,b:hexprimes".'),
                    (['0,0,1,-1,0', '--relations', put('e.txt', '# none\n')], '%s has no relations.'),
                    (['--curves', put('l.json', '{"curves": [{"id": 1, "ainvs": [0, 1], "ainvs": [1, 1]}]}')],
                     'l.json has the key "ainvs" twice in one object.'),
                    (['--curves', put('n.json', '{"curves": [{"id": NaN, "ainvs": [0, 1]}]}')],
                     'n.json has NaN, which is not a JSON number.'),
                    (['--curves', put('i.json', '{"curves": [{"id": 1e400, "ainvs": [0, 1]}]}')],
                     'Entry 1 of i.json has the id Infinity: give a number or a string.'),
                    (['--curves', put('b.json', '{"curves": [{"id": 1, "ainvs": [0, %s]}]}' % ('9' * 5000))],
                     "b.json has a number of more than 4000 digits, the verifier's limit."),
                    (['--curves', put('x.json', '{not json')], '%s is not valid JSON at line 1, column 2.'),
                    (['--curves', put('y.json', '{"curves": [{"ainvs": [0, 1]}]}')], 'Entry 1 of y.json has no "id".')):
                f = next(a for a in args if a.startswith(d))
                self.assertEqual(self.main(['bound', '--dir', run, '--estimate' if '--curves' in args else '--compute']
                                           + args), (2, (line % f if '%s' in line else line) + '\n'), args)
            self.assertFalse(os.path.exists(run))

    def test_status(self):
        """status without a work folder says what to do; a hand-edited run.json is shown, not a traceback."""
        with tempfile.TemporaryDirectory() as d:
            old = os.getcwd()
            os.chdir(d)
            try:
                self.assertEqual(self.main(['status']), (2, 'There is no work folder ./rankbound_work here. Give one: '
                                                            'rankbound status DIR.\n'))
                os.makedirs(os.path.join('w', 'c'))
                with open(os.path.join('w', 'c', 'run.json'), 'w') as f:
                    json.dump({'name': 'c', 'relations': 'many', 'core_hours': None, 'estimate': 'x'}, f)
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    from rankbound import cli
                    self.assertEqual(cli.main(['status', 'w']), 0)
                self.assertRegex(out.getvalue().splitlines()[1], r'^c +\? +0\.00 +\? ')
            finally:
                os.chdir(old)

    def test_words(self):
        """Long a-invariants shortened next to a hash name; an estimate below 0.01 core-hours is 'under a minute'; the
        API's messages name workdir, the CLI's --dir; an OSError without Python's [Errno]."""
        from rankbound import messages
        self.assertEqual(cv.short_ainvs([1, 1, 1, -128067973760200094197595276, 551283524310015496401062930694209419949]),
                         '1,1,1,-1280...276,5512...949')
        mm = ['1', '1', '1', '-128067973760200094197595276', '551283524310015496401062930694209419949']
        self.assertTrue(cv.hashed(cv.default_name(mm), mm))
        self.assertFalse(cv.hashed('0_0_1_-1_0', ['0', '0', '1', '-1', '0']))
        self.assertEqual((messages.about(0.004), messages.about(1.6)), ('under a minute', 'about 1.6 core-hours'))
        tok = messages.FOLDER_WORD.set('workdir')
        try:
            self.assertEqual(messages.text('other_work'), 'This work folder was started for another curve or other '
                             'relations. Use another workdir.')
            self.assertEqual(messages.text('no_write', folder='f', reason='r'), 'The work folder f cannot be written '
                             '(r). Choose another with workdir=.')
        finally:
            messages.FOLDER_WORD.reset(tok)
        self.assertTrue(messages.text('other_work').endswith('Use another --dir.'))
        mm2 = {'schema': 1, 'curve': ['0', '0', '1', '-1', '0'], 'minimal_model': ['0', '0', '1', '-1', '0'],
               'model_map': None, 'hypothesis': 'GRH', 'relations': None, 'lower': None, 'name': 'n'}
        self.assertEqual(str(runtime.mismatch(mm2, dict(mm2, hypothesis='none'))), messages.text('other_grh'))
        self.assertEqual(str(runtime.mismatch(dict(mm2, hypothesis='none'), mm2)), messages.text('other_unc'))
        self.assertEqual(str(runtime.mismatch(mm2, dict(mm2, curve=['0', '0', '0', '-16', '16'], model_map=['2', '0', '0', '0']))),
                         'This work folder was started for this curve on another model, 0,0,1,-1,0. Give the curve as '
                         '0,0,1,-1,0 to resume it, or use another --dir.')
        self.assertEqual(str(runtime.mismatch(mm2, dict(mm2, minimal_model=['0', '0', '1', '-7', '6']))),
                         messages.text('other_work'))

    def test_preflight(self):
        """A --dir that is a link to nothing, or has a name too long for the file system: refused before any work."""
        from rankbound import runner
        with tempfile.TemporaryDirectory() as d:
            os.symlink(os.path.join(d, 'nothing'), os.path.join(d, 'link'))
            for root, why in ((os.path.join(d, 'link'), 'a link to nothing'),
                              (os.path.join(d, 'x' * 300), 'a name in it is longer than')):
                with self.assertRaisesRegex(rb.RankboundError, re.escape(why)):
                    runner.preflight(root, checker=['true'])

    def test_api(self):
        """The API: points as Fractions, ints or strings are kept as canonical strings; booleans, budget > 0 and E
        are checked before any work."""
        from fractions import Fraction
        from rankbound import rankbound
        self.assertEqual(cv.points_of([(Fraction(-3, 4), 2), ('0', Fraction(1))]), [['-3/4', '2'], ['0', '1']])
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'w')
            for kw, want in (({'uncond': 'no'}, "^uncond must be True or False, not 'no'.$"),
                             ({'compute': 'False'}, "^compute must be True or False, not 'False'.$"),
                             ({'budget': 0}, '^budget must be a number > 0.$')):
                self.assertRaisesRegex(ValueError, want, rankbound, [0, 0, 1, -1, 0], workdir=w, **kw)
            self.assertRaisesRegex(ValueError, '^ainvs must be a list.$', rankbound, None, workdir=w)
            self.assertEqual(os.listdir(d), [])

    def test_environment(self):
        """The native programs' internal variables are dropped at a request's start, but the overrides."""
        env = {'F2ELIM_POOL': '1', 'F2ELIM_DEADLINE': '1', 'F2ELIM_DENSE_S': '0', 'RANKBOUND_PARENT': '1',
               'F2KERNEL_DENSE_N': '7', 'F2ELIM_TMPDIR': '/t', 'F2ELIM_DENSE_MB': '9', 'RANKBOUND_CORES': '2',
               'PATH': '/bin'}
        self.assertEqual(runtime.clean_environ(env), ['F2ELIM_DEADLINE', 'F2ELIM_DENSE_S', 'F2ELIM_POOL',
                                                      'RANKBOUND_PARENT'])
        self.assertEqual(env, {'F2KERNEL_DENSE_N': '7', 'F2ELIM_TMPDIR': '/t', 'F2ELIM_DENSE_MB': '9',
                               'RANKBOUND_CORES': '2', 'PATH': '/bin'})


@unittest.skipUnless(__import__('shutil').which('gp'), 'needs PARI/GP')
class TestInputRuns(unittest.TestCase):
    """Short runs of what a laptop user gives: a list entry a4,a6 (checked), an older gp (the witness written but not
    checked, exit 0), given relations too few, of another index form or hard to factor, points as Fractions to the
    API, an inherited internal variable of the native programs."""
    E571 = '0,-1,1,-929,-10595'

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(__import__('shutil').rmtree, self.d)

    def main(self, args, env=None):
        from rankbound import cli
        from unittest import mock
        with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()) as out, \
                mock.patch.dict(os.environ, env or {}):
            try:
                code = cli.main(args)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_list_short(self):
        """A --curves entry [a4, a6]: its witness checked (the checker gets the five a-invariants)."""
        lst = os.path.join(self.d, 'l.json')
        with open(lst, 'w') as f:
            json.dump({'curves': [{'id': 3, 'ainvs': [-25, 0]}]}, f)
        code, out, err = self.main(['bound', '--curves', lst, '--cores', '1', '--dir', os.path.join(self.d, 'w')])
        self.assertEqual(code, 0, err)
        self.assertRegex(out, r'Rank at most 1, unconditionally. Witness: .*curve3.witness.json \(checked\)')

    def test_old_gp(self):
        """PARI/GP 2.15 on the PATH: said first, the witness written unchecked, exit 0; the same command with 2.17
        checks it."""
        from rankbound import runner
        fake = os.path.join(self.d, 'bin')
        os.makedirs(fake)
        with open(os.path.join(fake, 'gp'), 'w') as f:
            f.write('#!/bin/sh\necho 2.15.4\n')
        os.chmod(os.path.join(fake, 'gp'), 0o755)
        args = ['bound', '0,0,0,-25,0', '--compute', '--cores', '1', '--dir', os.path.join(self.d, 'w')]
        code, out, err = self.main(args, {'PATH': fake + os.pathsep + os.environ['PATH']})
        self.assertEqual(code, 0, err)
        old = ('PARI/GP 2.15.4 (%s) is older than 2.17, so the witness will be written but not checked. Install '
               'PARI/GP 2.17 or later and run the same command again to check it.' % os.path.join(fake, 'gp'))
        lines = err.splitlines()                # before the work: a Slurm job's end line may come first
        first = [i for i, l in enumerate(lines) if old in l]
        work = [i for i, l in enumerate(lines) if 'Work folder ' in l]
        self.assertTrue(first and work and first[0] < work[0], err)
        self.assertTrue(out.strip().endswith('.witness.json'), out)
        code, out, err = self.main(args)
        self.assertEqual(code, 0, err)
        self.assertTrue(out.strip().endswith('(checked)'), out)
        self.assertTrue(runner.gp_usable())

    def test_relations(self):
        """One relation: no bound, said as such (not memory); a witness of another index form, or a relation without
        primes too large to factor quickly: a usage error."""
        one = os.path.join(self.d, 'one.txt')
        with open(one, 'w') as f:
            f.write('1,1\n')
        code, out, err = self.main(['bound', self.E571, '--relations', one, '--cores', '1', '--dir', os.path.join(self.d, 'a')])
        self.assertEqual(code, 'The relations given (1 relation) give no bound: too few are left once those with a prime '
                               'in no other relation are removed. Give more relations, or run without --relations to sieve.')
        old = os.path.join(TESTS, 'data', '571a1.witness.json')       # rankbound 3.0's index form of 571a1
        code, out, err = self.main(['bound', self.E571, '--relations', old, '--cores', '1', '--dir', os.path.join(self.d, 'b')])
        self.assertEqual(code, 2, err)
        self.assertIn('The relations of the witness %s are written for the index form -5,6,-2,2, and this run uses ' % old,
                      err)
        big = os.path.join(self.d, 'big.txt')
        with open(big, 'w') as f:
            f.write('%d,%d\n' % (10 ** 300 + 7, 10 ** 300 + 9))
        t = time.time()
        code, out, err = self.main(['bound', self.E571, '--relations', big, '--cores', '1', '--dir', os.path.join(self.d, 'c')])
        self.assertEqual(code, 2, err)
        self.assertRegex(err.splitlines()[-1], r'^Relation 1 has no primes, and \|F\(a, b\)\| \(\d+ digits\) does not factor '
                                               r'quickly. Give its primes, as "a,b:hexprimes", or leave it out.$')
        self.assertLess(time.time() - t, 30)
        from unittest import mock
        with mock.patch.object(relations.arith, 'factor', side_effect=TimeoutError):
            self.assertRaisesRegex(rb.RankboundError, r'^Relation 1 \(2,3\) has no primes', relations.given_lines,
                                   {'poly': [1, 0, 0, 1]}, [(2, 3, None)])
        other = relations.relation_input(old)
        self.assertEqual((other.poly, other.curve[3]), (['-5', '6', '-2', '2'], '-929'))
        self.assertRaisesRegex(rb.RankboundError, '^The witness .* is for another curve', relations.check_given,
                               {'poly': [-5, 6, -2, 2], 'minimal_model': ['0', '0', '1', '-1', '0']}, other)
        relations.check_given({'poly': [-5, 6, -2, 2], 'minimal_model': ['0', '-1', '1', '-929', '-10595']}, other)

    def test_api_fraction_points(self):
        """Points given as Fractions: the run publishes its result (the points kept as canonical strings)."""
        from fractions import Fraction
        res = rb.rankbound([0, 0, 1, -1, 0], points=[(Fraction(0), 0)], workdir=os.path.join(self.d, 'w'), cores=1)
        self.assertEqual((res.rank_bound, res.checked), (1, True))
        self.assertEqual(res.to_json()['provenance']['parameters']['options']['points'], [['0', '0']])

    def test_inherited_pool(self):
        """F2ELIM_POOL=1 (an internal variable of f2elim) in the environment changes nothing."""
        code, out, err = self.main(['bound', self.E571, '--compute', '--cores', '1', '--dir', os.path.join(self.d, 'w')],
                                   {'F2ELIM_POOL': '1'})
        self.assertEqual(code, 0, err)
        self.assertIn('Rank 0, assuming the GRH.', out)


class TestGpInput(unittest.TestCase):
    """Every value of a curve list that is pasted into gp must be a JSON integer or a canonical decimal string (the
    verifier's form): gp evaluates what it is given."""
    GOOD = {'id': 1, 'ainvs': ['0', '-1', '1', '-929', '-10595'], 'bad_primes': ['571'], 'r': 0}
    EVIL = '0*#Str(system("touch MARK"))'

    def load(self, **kw):
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, 'list.json')
            with open(path, 'w') as f:
                json.dump({'curves': [dict(self.GOOD, **kw)]}, f)
            return cv.from_list(path)

    def test_curve_list(self):
        self.assertEqual(self.load()[0].ainvs, self.GOOD['ainvs'])
        self.assertEqual(self.load(ainvs=[0, -1, 1, -929, -10595])[0].ainvs, self.GOOD['ainvs'])    # JSON integers
        self.assertEqual(self.load(ainvs=[-929, -10595])[0].ainvs, ['-929', '-10595'])
        proof = {'points': [['-3/4', '7']], 'primes': [5]}
        self.assertEqual(self.load(r=1, proof=proof)[0].proof, dict(proof, ainvs=self.GOOD['ainvs']))   # its model
        self.assertEqual(self.load(r=1, proof=dict(proof, ainvs=[0, 0, 1, -1, 0]))[0].proof['ainvs'], [0, 0, 1, -1, 0])
        for bad in ({'ainvs': ['0', '-1', '1', '-929', '-10595+' + self.EVIL]}, {'ainvs': [0, 1, 2]},
                    {'ainvs': ['0', '-1', '1', '-929', '-10595.0']}, {'ainvs': [0, -1, 1, -929, 1.5]},
                    {'bad_primes': ['571)']}, {'bad_primes': [' 571']}, {'bad_primes': ['0571']},
                    {'ainvs': '01234'}, {'bad_primes': '571'}, {'minimal_model': '01234'}, {'composites': '3997'},
                    {'ainvs': [0, 0, 0, 0, 0]}, {'ainvs': [0, 1, 0, 0, 0]},
                    {'minimal_model': ['0', '-1', '1', '-929', '1e3']}, {'r': '0'}, {'r': True}, {'r': -1},
                    {'proof': {'points': [['1', '2+' + self.EVIL]], 'primes': [5]}},
                    {'proof': {'points': [['1', '2']], 'primes': ['5']}}, {'proof': {'points': [['1']], 'primes': [5]}},
                    {'proof': {'points': [['1', '2']], 'primes': [5], 'ainvs': ['0', '0', '1', '-1', '0;' + self.EVIL]}}):
            with self.assertRaisesRegex(ValueError, '^list.json, curve 1: ', msg=bad):
                self.load(**bad)
        for key in ('id', 'ainvs'):                       # the required keys: a missing one is named
            with tempfile.TemporaryDirectory() as t:
                path = os.path.join(t, 'list.json')
                with open(path, 'w') as f:
                    json.dump({'curves': [{k: v for k, v in self.GOOD.items() if k != key}]}, f)
                with self.assertRaisesRegex(ValueError, '^Entry 1 of list.json has no "id".$' if key == 'id' else
                                            '^Curve 1 of list.json has no "%s".$' % key):
                    cv.from_list(path)
        # without bad_primes: factored later (curve.factored); composites alone are refused
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, 'list.json')
            for extra, err in (({}, None), ({'composites': ['3997']}, '"composites" needs "bad_primes"')):
                with open(path, 'w') as f:
                    json.dump({'curves': [dict({k: v for k, v in self.GOOD.items() if k != 'bad_primes'}, **extra)]}, f)
                if err:
                    self.assertRaisesRegex(ValueError, '^list.json, curve 1: %s' % err, cv.from_list, path)
                else:
                    self.assertIsNone(cv.from_list(path)[0].bad_primes)

    def test_prover_gp(self):
        """from_ainvs, compute and rank_proof refuse a non-canonical value before gp runs."""
        self.assertRaises(ValueError, cv.from_ainvs, [0, -1, 1, -929, -10595], bad_primes=['571+' + self.EVIL])
        self.assertRaises(ValueError, cv.from_ainvs, ['0', '-1', '1', '-929', ' -10595'])
        # a string is not read as a list of digits; a singular curve gets a clear message
        self.assertRaisesRegex(ValueError, 'ainvs must be a list', cv.from_ainvs, '01234')
        self.assertRaisesRegex(ValueError, 'bad_primes must be a list', cv.from_ainvs, [0, -1, 1, -929, -10595], '571')
        for ai in ([0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0]):
            self.assertRaisesRegex(ValueError, 'singular curve', cv.from_ainvs, ai)
        self.assertRaisesRegex(ValueError, 'singular curve', field.compute, ['0'] * 5, [])    # a Curve made by hand
        self.assertEqual(cv.disc([0, -1, 1, -929, -10595]), -571)
        self.assertEqual(cv.disc([0, 0, 1, -1, 0]), 37)
        self.assertRaises(ValueError, field.compute, ['0', '-1', '1', '-929', '-10595'], ['571;quit'])
        good = {'points': [['1', '2']], 'primes': [3, 5]}
        for ai, bad, proof in ((self.GOOD['ainvs'], ['571'], {'points': [['1', '2;' + self.EVIL]], 'primes': [3, 5]}),
                               (['0', '-1', '1', '-929', 'x'], ['571'], good),
                               (self.GOOD['ainvs'], ['571)'], good),
                               (self.GOOD['ainvs'], ['571'], dict(good, ainvs=['0', '-1', '1', '-929', '1e3']))):
            self.assertRaises(ValueError, cv.rank_proof, ai, bad, 1, proof)



class TestGiven(unittest.TestCase):
    """Relations and points given to rankbound(): their forms, the checks before anything is written, and the
    parameters of a run without a siever."""
    POLY = [-5, 0, 1, 1]          # an irreducible cubic: F(a, b) = a^3 + a^2 b - 5 b^3

    def test_parse(self):
        P = relations.parse_relation
        self.assertEqual(P('3,-2:2,b'), (3, -2, [2, 11]))
        self.assertEqual(P('3,-2'), (3, -2, None))
        self.assertEqual(P('1,0:'), (1, 0, []))
        self.assertIsNone(P('f:7'))
        self.assertEqual(P((3, -2)), (3, -2, None))
        self.assertEqual(P([3, -2, [2, 11]]), (3, -2, [2, 11]))
        for bad in ('3;-2', '03,1', '3,1:0b', '3,1:B', '3,1:2:', (3,), (3, '2'), (True, 1), (3, 2, [2.0]), 3, 'f:x',
                    'f:07'):
            self.assertRaises(ValueError, P, bad)
        # one digest per set of relations, whatever the form or order, repeats or not; free relations are left out
        a = relations.relation_input(['3,-2', '1,1:2,b', 'f:7'])
        self.assertEqual(a.rels, [(3, -2, None), (1, 1, [2, 11])])
        self.assertEqual(relations.relation_input([(3, -2), (1, 1, [2, 11])]).digest, a.digest)
        self.assertEqual(relations.relation_input([(1, 1, [11, 2]), (3, -2), '3,-2']).digest, a.digest)
        self.assertNotEqual(relations.relation_input([(3, -2)]).digest, a.digest)
        self.assertRaisesRegex(ValueError, '^No relations given.$', relations.relation_input, ['f:7'])
        with tempfile.TemporaryDirectory() as d:
            for name, op in (('r.txt', open), ('r.txt.gz', gzip.open)):
                p = os.path.join(d, name)
                with op(p, 'wt') as f:
                    f.write('# a comment\n3,-2\n\n1,1:2,b\n')
                self.assertEqual(relations.relation_input(p), a)
            p = os.path.join(d, 'x.cert.json')         # a witness: its relations, without the free ones
            with open(p, 'w') as f:
                json.dump({'format': vsel.FORMAT3, 'relations': ['f:7', '3,-2', '1,1:2,b']}, f)
            self.assertEqual(relations.relation_input(p)[:2], a[:2])
            self.assertEqual(relations.relation_input(p).source, p)       # a witness's: compared with the run's
            for c in ({'format': vsel.FORMAT3}, {'format': vsel.FORMAT3, 'relations': {}}, [1], {'format': 'x'}):
                with open(p, 'w') as f:          # not a witness with relations
                    json.dump(c, f)
                self.assertRaisesRegex(ValueError, 'x.cert.json is not a witness with relations.$',
                                       relations.read_relations, p)
            with open(os.path.join(d, 'r.txt'), 'w') as f:
                f.write('3,-2\nx\n')
            self.assertRaisesRegex(ValueError, '^In .*r.txt, relation \'x\' is not of the form "a,b" or "a,b:hexprimes".$',
                                   relations.read_relations, os.path.join(d, 'r.txt'))

    def test_given_lines(self):
        """Missing primes are factored by PARI; wrong primes (not multiplying to |F(a, b)|, or composite) and (a, b) not
        coprime are refused, naming the relation."""
        F = {'poly': self.POLY}
        v = abs(field.norm(self.POLY, 3, -2))       # 27 - 18 + 40 = 49
        self.assertEqual(v, 49)
        lines, nf = relations.given_lines(F, [(3, -2, None), (3, -2, [7, 7]), (1, 0, None), (2, 1, None)])
        self.assertEqual((lines[:3], nf), (['3,-2:7,7', '3,-2:7,7', '1,0:'], 3))
        from rankbound import linalg
        self.assertEqual(linalg.parse_line(lines[3])[2], [7])                          # 8 + 4 - 5 = 7
        for rel, err in (((3, -2, [49]), '^The primes given for relation 1 \\(3,-2\\) are not the prime factors'),
                         ((3, -2, [7]), 'are not the prime factors'), ((3, -2, [1, 7, 7]), 'are not the prime factors'),
                         ((2, 4, None), '^In relation 1 \\(2,4\\), a and b are not coprime.$'), ((0, 0, None), 'not coprime')):
            with self.assertRaisesRegex(ValueError, err):
                relations.given_lines(F, [rel])

    def test_points(self):
        from fractions import Fraction
        c = cv.Curve(name='e37', ainvs=['0', '0', '1', '-1', '0'], bad_primes=['37'])
        p = cv.with_points(c, [(0, 0), ('-3/4', Fraction(6, 4))])
        self.assertEqual((p.r, p.proof), (2, {'points': [['0', '0'], ['-3/4', '3/2']], 'ainvs': c.ainvs}))
        self.assertEqual(cv.with_points(c, [], None).r, 0)
        self.assertRaisesRegex(ValueError, '^r = 1 but 2 points were given.$', cv.with_points, c, [(0, 0), (1, 0)], 1)
        for bad in ([(0,)], [(0, 1.5)], [('0', '06/4')], [('0', '1;x')]):
            self.assertRaises(ValueError, cv.with_points, c, bad)
        self.assertIsNone(cv.check_proof(None))
        self.assertEqual(cv.check_proof(p.proof), p.proof)         # no primes: the prover searches them
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'p.txt')
            with open(path, 'w') as f:
                f.write('# P1, P2\n0 0\n-3/4,3/2\n')
            self.assertEqual(cv.read_points(path), [['0', '0'], ['-3/4', '3/2']])
            with open(path, 'w') as f:
                f.write('0 0 1\n')
            self.assertRaisesRegex(ValueError, 'p.txt, line 1: not a point "x y".$', cv.read_points, path)

    def test_params(self):
        """A run without a siever frees up to 2^lpb0 >= GB (every small column) and above the relations' primes (up to
        2^GIVEN_LPB_MAX); the runner's other readers of params.json find what they need."""
        for GB, top, lpb in ((10, 7, 4), (10, 251, 8), (16, 3, 4), (17, 3, 5), (1000, 2 ** 40, sieve.GIVEN_LPB_MAX)):
            prm = sieve.given_params({'GB': GB}, top)
            self.assertEqual(prm['lpb0'], lpb, (GB, top))
            self.assertGreaterEqual(prm['lim0'], GB)
            self.assertIsInstance(sieve.siever(prm), sieve.NoSiever)
        self.assertEqual(jobs.slice_cpu(prm), jobs.SLICE_MIN)



class TestCurveList(unittest.TestCase):
    def test_icarm_list(self):
        """An ICARM curve list with one curve, ICARM #728 (from its witness: tests/data/reference.json)."""
        with open(os.path.join(TESTS, 'data', 'reference.json')) as f:
            c = json.load(f)['icarm728']
        cur, lb = c['curve'], c['rank_lower_bound']
        with tempfile.TemporaryDirectory() as t:
            path = os.path.join(t, 'icarm_curves_728.json')
            with open(path, 'w') as f:
                json.dump({'curves': [dict(cur, icarm_id=728, name=c['name'], r=lb['r'], proof=lb)]}, f)
            a = cv.from_list(path)
            self.assertEqual(len(a), 1)
            c = a[0]
            self.assertEqual((c.name, c.r, len(c.proof['points']), c.source),
                             ('icarm728', 29, 29, 'ICARM #728'))
            self.assertEqual(len(c.minimal_model), 5)
            self.assertEqual(cv.Curve.from_dict(c.to_dict()), c)
            self.assertEqual(cv.from_list(path, {728}), a)
            with self.assertRaises(ValueError):
                cv.from_list(path, {692})

    def test_list_options(self):
        """--bad-primes, --r, --points, --relations and --shard go with a single curve: with --curves they are refused,
        not ignored."""
        from rankbound.__main__ import main
        for extra in (['--bad-primes', '2,3'], ['--r', '1'], ['--shard', '1/2']):
            with tempfile.TemporaryDirectory() as t, self.assertRaises(SystemExit) as e, \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                main(['bound', '--estimate', '--dir', t, '--curves', os.path.join(t, 'none.json')] + extra)
            self.assertEqual(e.exception.code, 2)
            self.assertEqual(err.getvalue(), '%s works with one curve only.\n' % extra[0])


class TestDefaultName(unittest.TestCase):
    """A curve given without a name is named by its reduced minimal model (0_-1_1_0_0; a hash when that is longer
    than 64 characters), on the command line and in rankbound(), whose default workdir is rankbound_work/NAME;
    Curve.key() of a Curve without its minimal model (as from_list gives when the list has none) computes it."""

    def test_default_name(self):
        import rankbound as rb
        import rankbound.selmer  # noqa: F401  (rb.selmer)
        self.assertEqual(cv.default_name(['0', '-1', '1', '0', '0']), '0_-1_1_0_0')
        big = ['1', '-1', '1', '-20067762415575526585033208209338542750', '30422586260593757248460328202047405873200']
        self.assertRegex(cv.default_name(big), '^[0-9a-f]{12}$')
        self.assertEqual(cv.check_name(cv.default_name(big)), cv.default_name(big))
        with open(wit0()) as f:
            cur = json.load(f)['curve']
        self.assertEqual(cv.from_ainvs([0, 0, 0, -432, 8208]).name, '0_-1_1_0_0')      # a short model of #0
        self.assertEqual(cv.from_ainvs([0, 0, 0, -432, 8208], name='e').name, 'e')
        self.assertEqual(cv.Curve(name='x', ainvs=cur['ainvs'], bad_primes=cur['bad_primes']).key(), '0_-1_1_0_0')
        seen = []

        class Stop(Exception):
            pass

        class J:
            def __init__(self, d):
                seen.append(d)
                raise Stop

        old, rb.selmer.Job = rb.selmer.Job, J          # rankbound() imports Job on use
        try:
            self.assertRaises(Stop, rb.rankbound, [0, 0, 0, -432, 8208], compute=True)     # compute: #0 is in the index
            self.assertRaises(Stop, rb.rankbound, [0, 0, 0, -432, 8208], compute=True, r=0)
        finally:
            rb.selmer.Job = old
        self.assertEqual(seen, [os.path.join('rankbound_work', '0_-1_1_0_0')] * 2)

class TestOutsider(unittest.TestCase):
    """Bad inputs to the command line and the API: each is refused with a clear reason before anything is written."""
    E37 = [0, 0, 1, -1, 0]

    def write(self, d, name, data, mode='w'):
        path = os.path.join(d, name)
        with open(path, mode) as f:
            f.write(json.dumps(data) if isinstance(data, dict) else data)
        return path

    def main(self, args):
        """Return cli.main(ARGS)'s SystemExit message (a usage error's, from stderr) or return value."""
        from rankbound import __main__ as cli
        with contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                return cli.main(args)
            except SystemExit as e:
                return err.getvalue().strip() if e.code == 2 else str(e.code)

    def test_curve_list(self):
        """A string proof, a repeated id or name, no curves, or a file that is not JSON, UTF-8 or gzip: refused,
        naming the file."""
        good = dict(TestGpInput.GOOD)
        with tempfile.TemporaryDirectory() as d:
            for data, want in (({'curves': [dict(good, proof='x')]}, 'list.json, curve 1: "proof" must be {'),
                               ({'curves': [good, dict(good, ainvs=self.E37)]}, '^list.json has two curves with the id 1.$'),
                               ({'curves': [dict(good, name='a'), dict(good, id=2, name='a')]},
                                '^list.json has two curves named a.$'),
                               ({'curves': []}, '^list.json has no curves.$'),
                               ([1], r'^list.json is not a curve list: \{"curves": \[\{"id": \.\.\., "ainvs": \[\.\.\.\]\}, \.\.\.\]\}\.$')):
                self.assertRaisesRegex(ValueError, want, cv.from_list, self.write(d, 'list.json', json.dumps(data)))
            for name, data, want in (('a.json', '{not json', 'a.json is not valid JSON at line 1, column 2.$'),
                                     ('b.json', '{"curves": []}'.encode('utf-16'), 'b.json is not UTF-8 text.$'),
                                     ('c.json.gz', 'hello', 'c.json.gz is not a valid gzip file.$')):
                path = self.write(d, name, data, 'wb' if isinstance(data, bytes) else 'w')
                self.assertRaisesRegex(ValueError, want, cv.from_list, path)

    def test_options(self):
        """A curve with --curves, no curve, bad --ids, --estimate or --shard with options they do not take, and a
        missing folder: refused with a message."""
        with tempfile.TemporaryDirectory() as d:
            lst = self.write(d, 'list.json', {'curves': [TestGpInput.GOOD]})
            run = os.path.join(d, 'run')
            self.assertEqual(self.main(['bound', '--dir', run, '--curves', lst, '0,0,1,-1,0']),
                             'Give a curve or --curves, not both.')
            self.assertEqual(self.main(['bound', '--dir', run]), 'Give a curve, like 0,1,1,-2,0, or a list of curves with '
                             '--curves.')
            self.assertEqual(self.main(['bound', '--dir', run, '--curves', lst, '--ids', '1,abc']),
                             'list.json has no curve with the id abc.')
            self.assertEqual(self.main(['bound', '--dir', run, '--curves', lst, '--ids', '1,']),
                             "--ids '1,' contains an empty id.")
            self.assertEqual(self.main(['bound', '--dir', run, '--curves', lst, '--ids', ' ']),
                             '--ids is empty. Leave it out to run the whole list.')
            self.assertEqual(self.main(['bound', '--dir', run, '0,0,1,-1,0', '--estimate', '--points', lst]),
                             '--estimate only estimates the cost, so it takes no --points.')
            self.assertEqual(self.main(['bound', '--dir', run, '0,0,1,-1,0', '--shard', '1/2', '--uncond']),
                             '--shard only sieves, so it takes no --uncond.')
            self.assertEqual(self.main(['bound', '--dir', run, '0,0,1,-1,0', '--uncond', '--relations', lst]),
                             'Given relations give bounds assuming the GRH only. Drop --uncond or --relations.')
            self.assertEqual(self.main(['status', os.path.join(d, 'none')]),
                             '%s is not a rankbound work folder.' % os.path.join(d, 'none'))
            self.assertFalse(os.path.exists(run))

    def test_numbers(self):
        """Numbers of more than 4,000 digits, the verifier's limit, are refused by the prover too."""
        for x in ('9' * 4001, 10 ** 4001, -10 ** 5000):
            self.assertRaisesRegex(ValueError, 'more than 4000 digits', cv.canon, x, 'a4')
        self.assertEqual(cv.canon('9' * 4000, 'a4'), '9' * 4000)
        self.assertRaisesRegex(ValueError, 'more than 4000 digits', cv.from_ainvs, [0, 0, 0, '9' * 5000, 1])
        self.assertRaisesRegex(ValueError, "^An a-invariant has more than 4000 digits, the verifier's limit.$",
                               cv.from_ainvs, [0, 0, 0, '9' * 5000, 1])
        self.assertRaisesRegex(ValueError, "^Relation '9{20}...' is too long for the verifier.$", relations.parse_relation,
                               '9' * 4001 + ',1')
        self.assertRaisesRegex(ValueError, 'is too long for the verifier', relations.parse_relation, (10 ** 5000, 1))
        self.assertRaisesRegex(ValueError, '^Relation 1 is too long for the verifier.$', relations.given_lines,
                               {'poly': [1, 0, 0, 1]}, [(10 ** 1500, 1, None)])

    def test_bad_primes(self):
        """A bad prime below 2 is refused as not prime; so is an empty name."""
        for p in ('1', '-37', '0'):
            self.assertRaisesRegex(ValueError, '^The bad prime %s is not prime.$' % p, cv.from_ainvs, self.E37, [p])
        self.assertRaisesRegex(ValueError, '^The bad prime -37 is not prime.$', field.compute, ['0', '0', '1', '-1', '0'],
                               ['-37'])
        self.assertRaisesRegex(ValueError, 'name', cv.from_ainvs, self.E37, None, None, '')

    def test_files(self):
        """--points and --relations files that are binary, not JSON, or not valid gzip or xz: refused, naming the file
        and the reason."""
        with tempfile.TemporaryDirectory() as d:
            b = self.write(d, 'x.bin', b'\x00\xff\xfe\x80\n', 'wb')
            self.assertRaisesRegex(ValueError, 'x.bin is not UTF-8 text.$', cv.read_points, b)
            self.assertRaisesRegex(ValueError, 'x.bin is not UTF-8 text.$', relations.read_relations, b)
            for name, want in (('c.cert.json', 'is not valid JSON at line 1, column 1.'), ('r.gz', 'is not a valid gzip file.'),
                               ('r.xz', 'is not a valid xz file.'), ('c.cert.json.xz', 'is not a valid xz file.')):
                self.assertRaisesRegex(ValueError, name + ' ' + want + '$', relations.read_relations,
                                       self.write(d, name, 'hello'))

    def test_wrong_r(self):
        """r above ns + u + n is refused before the state folder has a field; a bound S below r (ICARM #0 with r = 2)
        stops with 'Check r' and no witness."""
        from rankbound import RankboundError, rankbound
        from rankbound.selmer import Job
        from helpers import ICARM0, ICARM0_RELS
        with tempfile.TemporaryDirectory() as d:
            for r in (5, 10 ** 60):
                w = os.path.join(d, 'w%d' % len(str(r)))
                self.assertRaisesRegex(ValueError, '^r = %d is impossible for this curve: under GRH its rank is at most 4. '
                                       'Check r.$' % r,
                                       Job(w).prepare, cv.from_ainvs(self.E37, r=r))
                self.assertFalse(os.path.exists(os.path.join(w, 'field.json')))
            w = os.path.join(d, 'i0')
            with self.assertRaisesRegex(RankboundError, '^The bound is rank at most 0, but r = 2 was given. No witness was '
                                        'written. Check r.$'):
                rankbound(ICARM0, relations=ICARM0_RELS, r=2, workdir=w)
            self.assertEqual([f for f in os.listdir(w) if '.witness.json' in f], [])
            self.assertIsNone(json.load(open(os.path.join(w, 'run.json')))['result'])

    def test_minimal_model(self):
        """A curve list's minimal_model that is not the curve's is refused before the state folder exists."""
        from rankbound.selmer import Job
        c = cv.Curve(name='c', ainvs=['0', '0', '1', '-1', '0'], bad_primes=['37'], minimal_model=['0', '0', '0', '1', '1'])
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'c')
            self.assertRaisesRegex(ValueError, r'^The minimal_model given for c is wrong; it should be \[0,0,1,-1,0\]. '
                                   'Correct it or leave it out.$', Job(w).prepare, c)
            self.assertFalse(os.path.exists(os.path.join(w, 'field.json')))

    def test_api(self):
        """rankbound() given bad max_hours, points or relations, or a hand-made Curve with bad fields: a ValueError
        before anything is written."""
        from rankbound import rankbound
        from rankbound.selmer import Job
        good = cv.Curve(name='c', ainvs=['0', '0', '1', '-1', '0'], bad_primes=['37'])
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'w')
            for kw, want in (({'max_hours': float('nan')}, '^max_hours must be a number >= 0.$'),
                             ({'max_hours': -1}, 'max_hours'), ({'max_hours': 'x'}, 'max_hours'),
                             ({'points': 5}, '^points must be a list.$'), ({'points': [None]}, '^Each point must be a pair'),
                             ({'relations': 5}, '^relations must be a file name or a list.$'),
                             ({'relations': b'1,1'}, 'relations must be a file name or a list')):
                self.assertRaisesRegex(ValueError, want, rankbound, self.E37, **dict({'workdir': w}, **kw))
            for c, want in ((dataclasses.replace(good, proof='x'), '^"proof" must be {'),
                            (dataclasses.replace(good, r='1'), '^r must be an integer >= 0.$'),
                            (dataclasses.replace(good, composites='3997'), '^composites must be a list.$'),
                            (dataclasses.replace(good, source=1), '^source must be a string.$')):
                self.assertRaisesRegex(ValueError, want, Job(w).prepare, c)
            self.assertRaisesRegex(ValueError, '^Expected a Curve.$', Job(w).prepare, 5)
            self.assertRaisesRegex(ValueError, 'ainvs must be a list', rankbound, 5, workdir=w)
            self.assertEqual(os.listdir(d), [])

    def test_exit_status(self):
        """`rankbound bound` and `bound --estimate` exit 1 when a curve of a list is skipped or fails, or a witness is
        rejected; the last line of each outcome."""
        from argparse import Namespace
        from rankbound import cli
        with tempfile.TemporaryDirectory() as d:
            lst = self.write(d, 'list.json', {'curves': [TestGpInput.GOOD, dict(TestGpInput.GOOD, id=2, bad_primes=['3'])]})
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(self.main(['bound', '--estimate', '--dir', os.path.join(d, 'run'), '--curves', lst]), 1)
            self.assertIn('INFO:rankbound:Curve curve2 failed. The bad prime 3 does not divide the minimal discriminant.',
                          out.output)
            self.assertTrue(any(l.startswith('INFO:rankbound:1 curves, estimated ') and l.endswith(' Skipped curve2.')
                                for l in out.output), out.output)
        res = {'name': 'x', 'rank_bound': 2, 'core_hours': 0.014, 'rank_lower_bound': 2, 'GRH': True, 'witness': 'x.json'}
        a = Namespace(json=False, resume='rankbound bound x')
        from rankbound.runner import result_of
        res = dict(res, statement='', verification=None)
        with self.assertLogs('rankbound', 'INFO') as out:
            self.assertEqual([cli.report(result_of(dict(res, checked=v, r_proved=p, check='rank', reason='Why.'), ''), a)
                              for v, p in ((True, True), (False, False), (None, False))], [0, 1, 0])
            self.assertEqual(cli.report(result_of(dict(res, rank_lower_bound=1, checked=None, r_proved=True), ''), a), 0)
        self.assertEqual(out.output, ['INFO:rankbound:Rank 2, assuming the GRH. Witness: x.json (checked)',
                                      'INFO:rankbound:The checker rejected this witness at the check rank. Please report '
                                      'this. Why.',
                                      'INFO:rankbound:Rank at most 2, assuming the GRH. Witness: x.json',
                                      'INFO:rankbound:Rank at most 2, assuming the GRH. Witness: x.json'])

    def test_usage(self):
        """A CURVE or --bad-primes that does not parse, or a FILE (--curves, --points, --relations) that cannot be
        opened or does not parse: one line on stderr, exit 2, nothing on stdout and nothing written, with --json too; a
        singular curve is a failure (exit 1)."""
        from rankbound import __main__ as cli
        def run(args):
            with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
                try:
                    code = cli.main(args)
                except SystemExit as e:
                    code = e.code
            return code, out.getvalue(), err.getvalue()
        with tempfile.TemporaryDirectory() as d:
            run_dir, none = os.path.join(d, 'run'), os.path.join(d, 'none.txt')
            bad = self.write(d, 'bad.json', '{not json')
            for args, line in (
                    (['1,2,3'], 'Give the curve as a1,a2,a3,a4,a6 or as a4,a6.'),
                    (['-1,2,3'], 'Give the curve as a1,a2,a3,a4,a6 or as a4,a6.'),
                    (['1,2,3,4,5,6', '--estimate'], 'Give the curve as a1,a2,a3,a4,a6 or as a4,a6.'),
                    (['x,1'], "The a-invariant 'x' is not an integer."),
                    (['1.5,2', '--json'], "The a-invariant '1.5' is not an integer."),
                    (['0,0,1,-1,0', '--bad-primes', '37,x'], "The bad prime 'x' is not an integer."),
                    (['--curves', none], 'The file %s given with --curves cannot be read (No such file or directory).' % none),
                    (['--curves', d, '--estimate'], 'The file %s given with --curves cannot be read (Is a directory).' % d),
                    (['0,0,1,-1,0', '--points', none, '--json'],
                     'The file %s given with --points cannot be read (No such file or directory).' % none),
                    (['0,0,1,-1,0', '--relations', none],
                     'The file %s given with --relations cannot be read (No such file or directory).' % none),
                    (['--curves', bad, '--ids', '1,'], "--ids '1,' contains an empty id.")):
                self.assertEqual(run(['bound', '--dir', run_dir] + args), (2, '', line + '\n'), args)
            self.assertFalse(os.path.exists(run_dir))
            code, out, err = run(['bound', '--dir', run_dir, '0,0,0,0,0', '--estimate'])
            self.assertEqual((code, out), ('The a-invariants [0,0,0,0,0] give a singular curve (discriminant 0).', ''))
            code, out, err = run(['bound', '--dir', run_dir, '--curves', bad, '--estimate'])     # a usage error too
            self.assertEqual((code, out, err), (2, '', '%s is not valid JSON at line 1, column 2.\n' % bad))

    def test_commands(self):
        """`rankbound --version` prints the version (pyproject.toml's) on stdout; an unknown command is refused naming
        all three commands."""
        r = subprocess.run([sys.executable, '-m', 'rankbound', '--version'], cwd=ROOT, capture_output=True, text=True,
                           timeout=60)
        with open(os.path.join(ROOT, 'pyproject.toml')) as f:
            version = re.search(r'(?m)^version = "([^"]+)"$', f.read())[1]
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, 'rankbound %s\n' % version, ''))
        self.assertEqual(rb.__version__, version)
        r = subprocess.run([sys.executable, '-m', 'rankbound', 'nosuch'], cwd=ROOT, capture_output=True, text=True,
                           timeout=60)
        self.assertEqual((r.returncode, r.stdout), (2, ''))
        self.assertRegex(r.stderr, r"invalid choice: '?nosuch'? \(choose from '?bound'?, '?status'?, '?check'?, '?unpack'?\)")


class TestKnown(unittest.TestCase):
    """A curve that witnesses.tsv lists with a witness is not computed again: its row is given (outcome
    'known'), unless --compute (compute=True); points, relations or a curve list always compute."""
    I0 = '0,-1,1,0,0'                                 # ICARM #0 (11a3), its reduced minimal model
    SHORT = [0, 0, 0, -432, 8208]                     # a short model of it: y^2 = x^3 - 27 c4 x - 54 c6
    HEAD = ['name', 'icarm_id', 'file', 'rank lower bound', 'rank upper bound', 'rank = r?', 'sha256',
            'verified', 'minimal model']

    def index(self, d, rows):
        path = os.path.join(d, 'witnesses.tsv')
        with open(path, 'w') as f:
            f.write('\t'.join(self.HEAD) + '\n' + ''.join('\t'.join(r) + '\n' for r in rows))
        return path

    def row(self, name, S, eq, mm=None, file=True, r='0'):
        return [name, '', name + '.cert.json' if file else '', r, S, eq, 'ab' * 32, 'checked', mm or self.I0]

    def use(self, path):
        from rankbound import curve as cv
        old, cv.WITNESSES = cv.WITNESSES, path
        self.addCleanup(setattr, cv, 'WITNESSES', old)

    def test_lookup(self):
        """Exact match on the minimal model; rows without a witness or not matching the curve list are not used;
        of two rows of one curve the least S, then r proved; no index (pip install) or an index
        without the column: None."""
        from rankbound import curve as cv
        with tempfile.TemporaryDirectory() as d:
            p = self.index(d, [self.row('a', '2', 'r cited'), self.row('b', '2', 'yes'), self.row('c', '0', 'yes', file=False),
                               self.row('x', '1', 'not matching the curve list: r', mm='1,1,1,1,1'),
                               self.row('y', '3', 'yes', mm='0,-1,1,-10,-20')])
            got = cv.table_row(['0', '-1', '1', '0', '0'], p)
            self.assertEqual(got['name'], 'b')
            self.assertIsNone(cv.table_row(['1', '1', '1', '1', '1'], p))
            self.assertIsNone(cv.table_row(['0', '-1', '1', '-10', '-21'], p))
            self.assertEqual(cv.table_row([0, -1, 1, -10, -20], p)['name'], 'y')
            self.assertIsNone(cv.table_row(['0', '-1', '1', '0', '0'], os.path.join(d, 'none.tsv')))
            with open(p, 'w') as f:
                f.write('\t'.join(self.HEAD[:-1]) + '\n' + '\t'.join(self.row('b', '2', 'yes')[:-1]) + '\n')
            self.assertIsNone(cv.table_row(['0', '-1', '1', '0', '0'], p))
        st = lambda *a, **k: cv.table_statement(dict(zip(self.HEAD, self.row(*a, **k))))
        self.assertEqual(st('b', '3', 'yes', r='3'), 'rank E(Q) = 3 under GRH')
        self.assertEqual(st('b', '0', 'r cited', r='0'), 'rank E(Q) = 0 under GRH')
        self.assertEqual(st('b', '3', 'r cited', r='3'), 'rank E(Q) <= 3 under GRH (r = 3 is cited, not proved in the '
                                                         'witness)')
        self.assertEqual(st('b', '4', 'no', r='2'), '2 <= rank E(Q) <= 4 under GRH')

    def test_cli(self):
        """`rankbound bound CURVE` (any model) says the row and exits 0 with no work folder; --compute, --points,
        --relations and a curve list compute (here: prepared, then stopped by --max-hours 0, or certified)."""
        from rankbound import __main__ as cli
        from helpers import ICARM0_RELS
        job = os.environ.pop('SLURM_JOB_ID', None)      # no time limit from a Slurm job
        if job is not None:
            self.addCleanup(os.environ.__setitem__, 'SLURM_JOB_ID', job)
        with tempfile.TemporaryDirectory() as d:
            self.use(self.index(d, [self.row('icarm0', '0', 'yes')]))
            run = os.path.join(d, 'run')
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(cli.main(['bound', '--dir', run, ','.join(map(str, self.SHORT))]), 0)
            self.assertEqual(out.output, ['INFO:rankbound:This curve is already in the table of witnesses. Rank 0, '
                                          'assuming the GRH. Add --compute to compute it again.'])
            self.assertFalse(os.path.exists(run))
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(cli.main(['bound', '--dir', run, self.I0, '--compute', '--max-hours', '0']), 3)
            self.assertEqual(out.output[-1], 'INFO:rankbound:Stopped at the time limit. Run the same command again to resume.')
            rels = os.path.join(d, 'rels.txt')
            with open(rels, 'w') as f:
                f.write('\n'.join(ICARM0_RELS) + '\n')
            with self.assertLogs('rankbound', 'INFO') as out:
                self.assertEqual(cli.main(['bound', '--dir', os.path.join(d, 'g'), self.I0, '--relations', rels]), 0)
            self.assertRegex(out.output[-1], r'^INFO:rankbound:Rank 0, assuming the GRH. Witness: \S+0_-1_1_0_0.witness.json.gz \(checked\)$')
            w = os.path.join(d, 'g', '0_-1_1_0_0')
            for f in ('check.log', 'check.json', 'run.log', 'run.json'):     # the checker's lines: not in the run's log
                self.assertTrue(os.path.exists(os.path.join(w, f)), f)
            with open(os.path.join(w, 'check.json')) as f:
                self.assertEqual(json.loads(f.read())['verdict'], 'accepted')
            self.assertFalse(any('passes' in l for l in out.output), out.output)
            lst = os.path.join(d, 'list.json')
            with open(lst, 'w') as f:
                json.dump({'curves': [{'id': 0, 'ainvs': self.SHORT, 'bad_primes': ['11']}]}, f)
            with self.assertLogs('rankbound', 'INFO'):
                self.assertEqual(cli.main(['bound', '--dir', run, '--curves', lst, '--max-hours', '0']), 3)
            self.assertTrue(os.path.exists(os.path.join(run, 'curve0', 'field.json')))

    def test_api(self):
        """rankbound(): known (checked None, the witness's name), from any model; compute=True, relations or points
        compute."""
        from rankbound import Stopped, rankbound
        from helpers import ICARM0_RELS
        with tempfile.TemporaryDirectory() as d:
            self.use(self.index(d, [self.row('icarm0', '0', 'yes')]))
            res = rankbound(self.SHORT, workdir=os.path.join(d, 'w'))
            self.assertEqual((res.known, res.name, res.rank_bound, res.G, res.checked, res.witness,
                              res.proved_equal, res.r_proved, res.minimal_model),
                             (True, 'icarm0', 0, None, None, 'icarm0.cert.json', True, True, self.I0.split(',')))
            self.assertFalse(os.path.exists(os.path.join(d, 'w')))
            with self.assertLogs('rankbound', 'INFO'):
                self.assertRaises(Stopped, rankbound, self.SHORT, compute=True, workdir=os.path.join(d, 'w3'), max_hours=0)
                res = rankbound(self.SHORT, relations=ICARM0_RELS, workdir=os.path.join(d, 'w4'))
            self.assertEqual((res.known, res.rank_bound, res.G, res.details['outcome']), (False, 0, 0, 'goal'))

    def test_index(self):
        """The shipped index: every row with a witness has the 5 a-invariants of a reduced minimal model, and
        ICARM #0's model finds its row."""
        from rankbound import curve as cv
        with open(cv.WITNESSES) as f:
            head = f.readline().rstrip('\n').split('\t')
            rows = [dict(zip(head, l.rstrip('\n').split('\t'))) for l in f]
        self.assertEqual(head[-1], 'minimal model')
        for r in rows:
            if r['file']:
                self.assertEqual(len(r['minimal model'].split(',')), 5, r['name'])
                self.assertTrue(all(cv.NUM.fullmatch(x) for x in r['minimal model'].split(',')), r['name'])
        self.assertEqual(cv.table_row(self.I0.split(','))['name'], 'icarm0')
        # one row per curve; ICARM #11 (rank 28) is found by its minimal model alone, without factoring
        from rankbound import __main__ as cli
        job = os.environ.pop('SLURM_JOB_ID', None)      # no time limit from a Slurm job
        if job is not None:
            self.addCleanup(os.environ.__setitem__, 'SLURM_JOB_ID', job)
        self.assertEqual(len({r['minimal model'] for r in rows if r['file']}), len([r for r in rows if r['file']]))
        mm = next(r for r in rows if r['name'] == 'icarm11')['minimal model']
        with tempfile.TemporaryDirectory() as d, self.assertLogs('rankbound', 'INFO') as out:
            t = time.time()
            self.assertEqual(cli.main(['bound', '--dir', os.path.join(d, 'run'), mm]), 0)
            self.assertLess(time.time() - t, 30)
            self.assertEqual(os.listdir(d), [])
        self.assertEqual(out.output, ['INFO:rankbound:This curve is already in the table of witnesses. Rank 28, '
                                      'assuming the GRH. Add --compute to compute it again.'])


class TestMessages(unittest.TestCase):
    def test_render(self):
        """Every message of messages.MESSAGES renders with its placeholders, as one line of plain text; the helpers."""
        import string
        from rankbound import messages as cli
        for key, tmpl in cli.MESSAGES.items():
            fields = [f[1] for f in string.Formatter().parse(tmpl) if f[1] is not None]
            kw = {n: 1 if n in ('n', 'S', 'r', 'w', 'k', 'gb', 'digits', 'seconds', 'limit') else 'X' for n in fields}
            line = cli.text(key, **kw) if all(fields) else tmpl.format(*['x'] * len(fields))
            self.assertNotIn('\n', line, key)
            self.assertNotIn('{', line, key)
        self.assertEqual((cli.count(1), cli.count(2), cli.count(10)), ('one', 'two', '10'))
        self.assertEqual((cli.core_hours(0.004), cli.core_hours(1.64), cli.core_hours(240)),
                         ('0.00 core-hours', '1.6 core-hours', '240 core-hours'))
        self.assertEqual([cli.duration(s) for s in (40, 720, 7200, 4e5, 1.3e9)],
                         ['40 seconds', '12 minutes', '2.0 hours', '5 days', '41 years'])
        self.assertEqual((cli.conductor_text(571), cli.conductor_text(10 ** 147 * 3)), ('571', '≈ 1e147'))
        self.assertEqual((cli.big(2048), cli.big(1234567890)), ('2,048', '1.2e9'))
        self.assertEqual((cli.torsion_text([]), cli.torsion_text([3]), cli.torsion_text([2, 4])),
                         ('no rational 2-torsion', 'no rational 2-torsion', 'torsion Z/2 x Z/4'))
        self.assertEqual((cli.rank_text(0, None, False), cli.rank_text(2, 2, True), cli.rank_text(2, 2, False),
                          cli.rank_text(2, None, False), cli.rank_text(3, 2, True)),
                         ('Rank 0', 'Rank 2', 'Rank at most 2', 'Rank at most 2', 'Rank at most 3'))
        self.assertEqual(cli.text('curve', conductor='571', bad=cli.count(1), s='', w=1, torsion=cli.torsion_text([])),
                         'Conductor 571, one bad prime, root number +1, no rational 2-torsion.')


class TestRun(unittest.TestCase):
    """A run end to end through the command line: the lines of the README curve 571a1, its --json line, run.json and
    `status`; a run again says the result; another run in the same folder is refused; --shard sieves and exits, and
    a plain run then solves."""

    def test_readme_curve(self):
        from rankbound import cli
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'w')
            out = io.StringIO()
            with self.assertLogs('rankbound', 'INFO') as lg, contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(['bound', '0,-1,1,-929,-10595', '--dir', w, '--compute', '--json']), 0)
            lines = [l.split(':', 2)[2] for l in lg.output if 'Slurm job' not in l and 'This job ends' not in l]
            self.assertRegex(lines.pop(0), r'^Work folder .*, 1 core(, at most .*)?\.$')      # where and with what, first
            path = os.path.join(w, '0_-1_1_-929_-10595', '0_-1_1_-929_-10595.witness.json.gz')
            self.assertEqual(lines[0], 'Conductor 571, one bad prime, root number +1, no rational 2-torsion.')
            self.assertRegex(lines[1], '^Sieving for relations. This should take (about [0-9.]+ core-hours|under a '
                                       'minute).$')
            self.assertEqual(lines[-3:], ['Rank at most 2. Verification method: 2-Selmer group (assuming the GRH).',
                                          'Rank at most 0. Verification method: Cassels-Tate pairing (assuming the GRH).',
                                          'Rank 0, assuming the GRH. Witness: %s (checked)' % cli.shown(path)])
            j = json.loads(out.getvalue())
            self.assertEqual({k: j[k] for k in ('name', 'rank_bound', 'rank_lower_bound', 'GRH', 'checked', 'witness')},
                             {'name': '0_-1_1_-929_-10595', 'rank_bound': 0, 'rank_lower_bound': None, 'GRH': True,
                              'checked': True, 'witness': path})
            with open(os.path.join(w, '0_-1_1_-929_-10595', 'run.log')) as f:      # the 2-descent's progress, in run.log only
                log = f.read()
            for step in ('Selmer: k = ', 'Selmer: the local data', 'Selmer: kernel vectors of', 'Selmer: 4 of U = 4 elements'):
                self.assertIn(step, log)
                self.assertFalse(any(step in l for l in lines))
            run = json.load(open(os.path.join(w, '0_-1_1_-929_-10595', 'run.json')))
            self.assertEqual((run['stage'], run['result']['rank_bound'], run['GRH'], run['method']),
                             ('finished', 0, True, 'selmer'))
            with self.assertLogs('rankbound', 'INFO') as lg:
                self.assertEqual(cli.main(['bound', '0,-1,1,-929,-10595', '--dir', w, '--compute']), 0)
            self.assertEqual([l.split(':', 2)[2] for l in lg.output if 'Slurm job' not in l and 'This job ends' not in l
                              and 'Work folder ' not in l],
                             ['Already done earlier. Rank 0, assuming the GRH. Witness: %s (checked)' % cli.shown(path)])
            for args, key in ((['--uncond'], 'other_grh'),):
                with self.assertRaises(SystemExit) as e:
                    cli.main(['bound', '0,-1,1,-929,-10595', '--dir', w, '--compute'] + args)
                self.assertEqual(str(e.exception), cli.text(key))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(['status', w]), 0)
            self.assertEqual([l.split() for l in out.getvalue().splitlines()][1][0::5],
                             ['0_-1_1_-929_-10595', 'finished'])

    def test_streams(self):
        """Without --json, stdout has the result lines only and stderr the progress; -v adds the methods' inner steps
        (run.log's) to stderr."""
        import subprocess
        env = dict({k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}, PYTHONPATH=SRC)
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, '-m', 'rankbound', 'bound', '0,-1,1,-929,-10595', '--dir', d, '--compute',
                                '-v'], env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertEqual([l[9:].split('. Witness: ')[0] for l in r.stdout.splitlines()], ['Rank 0, assuming the GRH'])
            self.assertIn('Conductor 571', r.stderr)
            self.assertIn('Selmer: k = ', r.stderr)

    def test_shards(self):
        """Two shards of a curve sieve their shares to the first check's relations and exit; then a plain run solves
        on their slices (and sieves no more if they suffice)."""
        from rankbound import cli
        with tempfile.TemporaryDirectory() as d:
            w = os.path.join(d, 'w')
            for k in (1, 2):
                with self.assertLogs('rankbound', 'INFO') as lg:
                    self.assertEqual(cli.main(['bound', '0,0,1,-7,6', '--dir', w, '--compute', '--shard', '%d/2' % k]), 0)
                self.assertIn('INFO:rankbound:Shard %d of 2 is done.' % k, lg.output)
            f = os.path.join(w, '0_0_1_-7_6')
            self.assertTrue(json.load(open(os.path.join(f, 'run.json')))['result'] is None)
            n = len(os.listdir(os.path.join(f, 'slices')))
            with self.assertLogs('rankbound', 'INFO') as lg:
                self.assertEqual(cli.main(['bound', '0,0,1,-7,6', '--dir', w, '--compute']), 0)
            self.assertRegex(lg.output[-1], r'Rank (at most )?\d+, assuming the GRH. Witness: ')
            self.assertEqual(json.load(open(os.path.join(f, 'run.json')))['stage'], 'finished')
            self.assertGreaterEqual(len(os.listdir(os.path.join(f, 'slices'))), n)


class TestRequests(unittest.TestCase):
    """What a run is asked for is what it does: --uncond is never answered with a bound under GRH (a
    GRH work folder, a GRH row of witnesses.tsv, given relations); a composite cofactor is factored first."""
    COMPOSITE = [12, -81, 270, -5020785, -6677562960738]     # log10 |d_K| 18; trial division to 10^6 leaves a C18

    def test_composite(self):
        with tempfile.TemporaryDirectory() as d:
            res = rb.rankbound(self.COMPOSITE, compute=True, workdir=os.path.join(d, 'w'))
            self.assertEqual((res.rank_bound, res.checked, res.grh), (0, True, True))
            self.assertTrue(os.path.exists(os.path.join(d, 'w', 'factor.json')))

    def test_uncond(self):
        """ICARM #2's row in witnesses.tsv is under GRH: with uncond it is computed, not looked up; a GRH folder
        refuses an unconditional run."""
        with tempfile.TemporaryDirectory() as d:
            known = rb.rankbound([0, 1, 1, -2, 0])
            self.assertEqual((known.known, known.grh), (True, True))
            u = rb.rankbound([0, 1, 1, -2, 0], uncond=True, workdir=os.path.join(d, 'u'))
            self.assertEqual((u.grh, u.checked, u.known), (False, True, False))
            w = os.path.join(d, 'w')
            self.assertTrue(rb.rankbound([0, -1, 1, -10, -20], compute=True, workdir=w).grh)
            with self.assertRaisesRegex(rb.RankboundError, '^This work folder holds a bound assuming the GRH, and this '
                                        'run asks for an unconditional one. Use another workdir.$'):
                rb.rankbound([0, -1, 1, -10, -20], compute=True, workdir=w, uncond=True)

    def test_r_checked(self):
        """A lookup in witnesses.tsv checks the given r against the row, and names the method."""
        with self.assertRaisesRegex(ValueError, r'r = 3 is impossible for this curve: its witness icarm2.witness.json.gz '
                                                r'proves rank E\(Q\) <= 2 under GRH'):
            rb.rankbound([0, 1, 1, -2, 0], r=3)
        res = rb.rankbound([0, 1, 1, -2, 0], r=1)
        self.assertEqual((res.known, res.rank_bound, res.rank_lower_bound, res.method), (True, 2, 2, 'class group'))
        self.assertEqual(rb.rankbound([0, 0, 0, -82, 0]).method, '2-isogeny descent')
        self.assertEqual(cv.table_method({cv.COL_TOR: '[6, 2]'}), '2-isogeny descent')
        self.assertEqual(cv.table_method({cv.COL_TOR: '[]', 'method': '2-Selmer group'}), '2-Selmer group')


class TestDeterminism(unittest.TestCase):
    """The same input gives the same witness, byte for byte, but 'created' (the date): rankbound() in fresh folders,
    with 1 and with 2 cores, on a curve of each route (37a1 with a point: the sieve; 37a1 --uncond: SPLIT; 571a1: the 2-descent
    and the Cassels-Tate pairing; y^2 = x^3 - 25x with a point: 2-isogeny descent; n = 113 with its pairings)."""
    CASES = [([0, 0, 1, -1, 0], dict(points=[(0, 0)])), ([0, 0, 1, -1, 0], dict(uncond=True)),
             ([0, -1, 1, -929, -10595], {}), ([0, 0, 0, -25, 0], dict(points=[(-4, 6)])), ([0, 0, 0, -12769, 0], {})]

    def test_same_bytes(self):
        import re
        with tempfile.TemporaryDirectory() as d:
            for k, (ai, kw) in enumerate(self.CASES):
                out = []
                for cores in (1, 2):
                    res = rb.rankbound(ai, compute=True, cores=cores, workdir=os.path.join(d, '%d_%d' % (k, cores)), **kw)
                    with open(res.witness, 'rb') as f:
                        out.append(re.sub(rb'"created": "[0-9-]+"', b'"created": ""', f.read()))
                self.assertEqual(out[0], out[1], (ai, kw))


def pyproject():
    """Return (packages, {package: [files]}) from pyproject.toml (read by regular expressions: no tomllib before 3.11)."""
    text = open(os.path.join(ROOT, 'pyproject.toml')).read()
    packages = re.findall(r'"([\w.]+)"', re.search(r'^packages = \[(.*?)\]', text, re.M | re.S).group(1))
    data = re.search(r'^\[tool\.setuptools\.package-data\]\n(.*?)(?:^\[|\Z)', text, re.M | re.S).group(1)
    return packages, {k: re.findall(r'"([^"]+)"', v) for k, v in re.findall(r'^"([\w.]+)" = \[(.*?)\]', data, re.M)}


class TestPackage(unittest.TestCase):
    def test_installed_files(self):
        packages, data = pyproject()
        for d, ds, fs in os.walk(PKG):
            ds[:] = [x for x in ds if x not in ('__pycache__', 'cado')]
            if '__init__.py' not in fs:
                continue
            pkg = os.path.relpath(d, SRC).replace(os.sep, '.')
            self.assertIn(pkg, packages)
            for f in fs:
                if f.endswith(('.gp', '.c')):
                    self.assertIn(f, data.get(pkg, []), '%s/%s is not in package-data' % (pkg, f))


if __name__ == '__main__':
    unittest.main()
