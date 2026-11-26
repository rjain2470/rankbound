"""Records, not text: the prover's witness file owner (witness.py), the 2-descent's data, the descent's and the Cassels-Tate step's results as
records (no in-process text protocol), the local setup once across the rounds of the a,b search, triangular
checkpoints as checked records, and reuse of the descent's artifacts within one request.  Every witness is the same
as before, byte for byte (tests/test_cli.py's determinism test)."""
import json, os, shutil, tempfile, unittest
from unittest import mock
from rankbound import arith as A, check, ctp, curve as cv, descent, field as fld, selmer, witness
from helpers import TESTS

DATA = os.path.join(TESTS, 'data')
QUIET = lambda *a: None


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix='rb_records')
        self.addCleanup(shutil.rmtree, self.d, True)


class TestWitness(Tmp):
    """witness.py owns the format dispatch and the atomic write; selmer does not re-export them."""

    def test_names(self):
        for n in ('write', 'read', 'encode', 'decode', 'deltas', 'undeltas', 'packable', 'Unpackable', 'FORMATS4'):
            self.assertFalse(hasattr(selmer, n), n)
        self.assertIs(selmer.FORMAT6, witness.FORMAT6)

    def test_descent_bytes(self):
        """A descent witness is written as runtime.write_json writes JSON (indent 1, a newline), atomically."""
        with open(os.path.join(DATA, 'n113.witness.json')) as f:
            c = json.load(f)
        p = os.path.join(self.d, 'n113.witness.json')
        witness.write(c, p)
        with open(p) as f:
            self.assertEqual(f.read(), json.dumps(c, indent=1) + '\n')
        self.assertEqual(os.listdir(self.d), ['n113.witness.json'])      # no temporary file left
        self.assertEqual(check.check(p)['verdict'], 'accepted')

    def test_selmer_round_trip(self):
        """A format-5 witness read and written again: the same lists, accepted."""
        c = witness.read(os.path.join(DATA, '571a1.witness.json'))
        p = os.path.join(self.d, 'a.witness.json.gz')
        witness.write(c, p)
        d = witness.read(p)
        self.assertEqual({k: d[k] for k in ('ops', 'pivots', 'elements', 'relations')},
                         {k: c[k] for k in ('ops', 'pivots', 'elements', 'relations')})
        self.assertEqual(check.check(p)['verdict'], 'accepted')


def field_of(ai):
    c = cv.from_ainvs(ai, name='t')
    F, small = fld.compute(c.ainvs, c.bad_primes)
    return F, small, selmer.s_primes(F)[0]


class TestRecords(Tmp):
    """The 2-descent's data are records: no text line is made or parsed in process."""

    def test_setup_record(self):
        """The setup is sel_local's record (computed once) and fb_shells' a,b relations; numbers are numbers, the
        witness's strings (theta, W, the local points) strings."""
        F, _, S = field_of([0, 0, 1, -7, 6])
        mm, f, P = [int(a) for a in F['minimal_model']], F['poly'], [int(p) for p in F['bad_primes']]
        loc, k = selmer.sel_local(mm, f, P, 30, 3, selmer.SMAX, 20, 0, S)
        rels, n, top = selmer.fb_shells(k, S, 30, 200, 1, 40, 0)
        d = dict(loc, rels=loc['rels'] + rels)
        self.assertTrue(all(r[0] == 0 for r, _, _ in rels) and all(r[0] != 0 for r, _, _ in loc['rels']))
        self.assertTrue(all(isinstance(x, str) for x in d['theta'] + d['W']))
        self.assertTrue(all(isinstance(z, int) for pl in d['pld'] for z in [pl['p'], pl['cp'], pl['ct']] + pl['g']))
        self.assertEqual(len(d['pld']), len(d['cls']))
        for key, (xs, bits) in d['pts'].items():
            self.assertTrue(all(isinstance(x, str) for x in xs) and all(isinstance(b, int) for b in bits), key)
        out = selmer.batch(F, S, [r for r, _, _ in d['rels'][:5]], [[j, j, -1] for j in range(5)], d['chars'])
        self.assertEqual([j for j, _, _ in out], list(range(5)))
        self.assertTrue(all(s == -1 and isinstance(x, int) for _, s, x in out))

    def test_rounds_share_local_setup(self):
        """Prove builds the local setup once, however many rounds (each round once built it again)."""
        calls = []
        real = selmer.sel_local

        def counting(*a):
            calls.append(1)
            return real(*a)
        logs = []
        with mock.patch.object(selmer, 'sel_local', counting), mock.patch.object(selmer, 'SHELLS', 2):
            c = selmer.prove(cv.from_ainvs([0, 0, 1, -7, 6], name='5077a1'), hypothesis='GRH', log=logs.append)
        rounds = sum('2-descent round' in l for l in logs)
        self.assertGreater(rounds, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(c['claims']['S'], 3)
        p = os.path.join(self.d, 'w.witness.json.gz')
        witness.write(c, p)
        self.assertEqual(check.check(p)['verdict'], 'accepted')

    def test_triangular_checkpoints(self):
        """A range's checkpoint is a runtime artifact (its record and manifest, with the
        schema and the fingerprint of the plan and the range); a damaged or foreign one is computed again, an intact
        one is used, and the result is the same as without checkpoints."""
        F, _, S = field_of([0, 0, 1, -1, 0])
        P = [str(p) for p in F['bad_primes']]
        B0, B = 30, 3000
        fresh = selmer.triangular(F, P, S, B0, B)
        st = os.path.join(self.d, 'st')
        self.assertEqual(selmer.triangular(F, P, S, B0, B, state=st), fresh)
        tdir = os.path.join(st, 'tri')
        names = [n for n in os.listdir(tdir) if not n.startswith('plan.')]
        self.assertEqual(sorted(names), ['%d_%d.json' % (B0, B), 'range_%d_%d.manifest.json' % (B0, B)])
        path = os.path.join(tdir, '%d_%d.json' % (B0, B))
        with open(path) as f:
            c = json.load(f)
        m = selmer.runtime.inspect(tdir, 'range_%d_%d' % (B0, B))[1]
        self.assertEqual(m['schema'], selmer.TRI_SCHEMA)
        calls = []
        real = selmer.tri

        def counting(*a):
            calls.append(1)
            return real(*a)
        with mock.patch.object(selmer, 'tri', counting):
            self.assertEqual(selmer.triangular(F, P, S, B0, B, state=st), fresh)
            self.assertEqual(len(calls), 0)                             # intact: used
            c['ns'] += 1                                                # damaged: the digest no longer matches
            with open(path, 'w') as f:
                json.dump(c, f)
            self.assertEqual(selmer.triangular(F, P, S, B0, B, state=st), fresh)
            self.assertEqual(len(calls), 1)
            with open(path, 'w') as f:                                  # not JSON
                f.write('NS\t30\t3000\t5\n')
            self.assertEqual(selmer.triangular(F, P, S, B0, B, state=st), fresh)
            self.assertEqual(len(calls), 2)

    def test_par_map_work(self):
        """Par_map's worker state is one record (Work), set for the map and restored after; order kept."""
        try:
            got = selmer.par_map(lambda xs: [3 * x for x in xs], list(range(101)), 2, least=10)
        except OSError:                          # no process pool here (a sandbox)
            self.skipTest('no fork pool')
        self.assertEqual(got, [3 * x for x in range(101)])
        self.assertIsNone(selmer._WORK)


class TestDescentRecords(Tmp):
    """descent.prove and ctp.ctprove return records (no tdprove or ctprove lines); the descent's local images,
    pairings and conic points are computed once within a request, the Cassels-Tate construction reuses them."""

    def test_reuse(self):
        ai = [0, 0, 0, -12769, 0]
        mm = cv.minimal(ai)[0]
        P = [str(p) for p in A.prime_divisors(cv.cinv(mm)[2])]
        D = descent.descent(ai, P, ctp_seconds=600)
        self.assertEqual(D['S'], 0)
        n = descent.REUSE_COUNTS
        for kind in ('images', 'pairing', 'conic'):
            self.assertGreater(n[kind][0], n[kind][1], kind)             # requests > computations
        self.assertIsNone(descent._REUSE)                               # nothing outlives the request
        p = os.path.join(self.d, 'n113.witness.json')
        witness.write(descent.witness('n113', P, D, None), p)
        self.assertEqual(check.check(p)['verdict'], 'accepted')
        with descent.reuse():                                           # callers get copies
            W = descent.images(-2, 1, [2, 113])
            W[0][1].append('x')
            self.assertEqual(descent.images(-2, 1, [2, 113]), descent._images(-2, 1, [2, 113]))

    def test_prove_record(self):
        mm = cv.minimal([0, 0, 0, -25, 0])[0]
        out = descent.prove(mm, [2, 5], [], [], 1000, 0)
        self.assertIsInstance(out, dict)
        self.assertEqual((out['B'], out['S']), descent.bound([g['B'] for g in out['isogenies']], out['w']))
        self.assertTrue(all(g['dual'] and g['in_selmer'] for g in out['isogenies']))

    def test_ctprove_record(self):
        c = selmer.prove(cv.from_ainvs([0, -1, 1, -929, -10595], name='571a1'), log=QUIET)
        basis = ctp.selmer_basis(c)
        info = ctp.py_body(c, basis)
        self.assertEqual(set(info), {'basis', 'located', 'points_rank', 'complement', 'quartics_ms', 'paired',
                                     'pairs_ms', 'rank', 'section'})
        rank, Q, loc, pairs = info['section']
        self.assertEqual((rank, info['rank']), (2, 2))
        ct = ctp.ct_block(rank, Q, loc, pairs)
        self.assertTrue(all(isinstance(x, str) for g in ct['quartics'] for x in g))
        c = ctp.add(c, seconds=600, log=QUIET)
        self.assertEqual(c['cassels_tate'], ct)
        p = os.path.join(self.d, 'w.witness.json.gz')
        witness.write(c, p)
        self.assertEqual(check.check(p)['verdict'], 'accepted')


if __name__ == '__main__':
    unittest.main()
