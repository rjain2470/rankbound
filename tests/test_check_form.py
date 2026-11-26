"""The checker's check of form (rankbound/check: form() and each format's clean), done before any mathematics, on
honest witnesses of every format (tests/data/corpus.json, the two format-3 files and the new-name witnesses): every
value replaced by one of another type, every key dropped, an unknown key added, every integer pushed out of its range.
Each change the format does not allow is refused with a reason that names the place; through check(), such a witness
fails at the check 'format'."""
import copy, glob, json, os, unittest
from rankbound import check as checker
from rankbound.check import selmer
from helpers import TESTS, corpus, drop, get, paths, put

OPTIONAL = {'rank_lower_bound', 's_dropped', 'cassels_tate', 'u', 'n'}       # keys that may be absent


def clean(c, fmt):
    """The checker's check of form of (a copy of) c, as for the format fmt."""
    f = checker.clean if fmt in checker.FORMATS else selmer.clean if fmt in selmer.FORMATS else selmer.clean3
    return f(copy.deepcopy(c))


def dotted(path):
    return ''.join('[%d]' % k if isinstance(k, int) else ('.' if i else '') + k for i, k in enumerate(path))


def witnesses():
    """corpus() and the new-name witnesses of tests/data."""
    out = corpus()
    for p in sorted(glob.glob(os.path.join(TESTS, 'data', '*.witness.json'))):
        with open(p) as f:
            out[os.path.basename(p)] = (None, json.load(f))
    return out


class TestForm(unittest.TestCase):
    C = witnesses()

    def refused(self, c, fmt, where, what):
        with self.assertRaises(ValueError, msg=what) as e:
            clean(c, fmt)
        self.assertIn(where, str(e.exception), what)

    def test_honest(self):
        for name, (_, c) in self.C.items():
            clean(c, c['format'])

    def test_types(self):
        """Every value, replaced by a value of another type, is refused, and the reason names its place."""
        n = 0
        for name, (_, c) in self.C.items():
            for path in paths(c):
                v = get(c, path)
                for bad in (None, True, 1.5, 'x', [], {}):
                    if type(bad) is type(v) or (bad is None and path[-1] == 'cassels_tate'):
                        continue
                    n += 1
                    self.refused(put(c, path, bad), c['format'], dotted(path), (name, path, bad))
        self.assertGreater(n, 5000)

    def test_keys(self):
        """Every key dropped (but the optional ones) and an unknown key in every object are refused, naming the
        object."""
        for name, (_, c) in self.C.items():
            for path in [()] + list(paths(c)):
                v = get(c, path)
                if not isinstance(v, dict):
                    continue
                self.refused(put(c, path + ('zz',), 1), c['format'], dotted(path) or 'the witness',
                             (name, path, 'zz'))
                if path[-1:] == ('p',):          # local.p: keyed by the primes of S
                    continue
                for key in v:
                    if key in OPTIONAL:
                        continue
                    with self.assertRaises(ValueError, msg=(name, path, key)):
                        clean(drop(c, path + (key,)), c['format'])

    def test_ranges(self):
        """Integers and indices out of their ranges, and lengths that do not match, are refused."""
        c4, c2 = self.C['571a1'][1], self.C['n113'][1]
        cases = [
            (c4, ('claims', 'w'), 0), (c4, ('claims', 'S'), -1), (c4, ('claims', 'S_ctp'), -1),
            (c4, ('field', 'bound', 'B'), 0), (c4, ('torsion',), [2]), (c4, ('torsion',), [3, 3]),
            (c4, ('characters', 'real'), [1, 0]), (c4, ('characters', 'real'), [3]),
            (c4, ('characters', 'primes'), [[7, 7]]), (c4, ('s_columns',), [[2, 3]]), (c4, ('elements',), [[]]),
            (c4, ('elements',), [[0, 0]]), (c4, ('ops',), [[-1, 0]]), (c4, ('local', 'p', '2', 'prec'), 0),
            (c4, ('created',), '16 Nov 2026'), (c4, ('curve', 'bad_primes'), ['0']),
            (c4, ('rank_lower_bound',), {'r': 2, 'points': [['0', '0']], 'primes': [17]}),
            (c4, ('rank_lower_bound',), {'r': 1, 'source': 'line\nbreak'}),
            (c2, ('local', 'w'), -2), (c2, ('claims', 'B'), -1), (c2, ('torsion',), [2, 2, 2]),
            (c2, ('descent', 0, 'pairing', 'rank'), -1), (c2, ('descent', 0, 's'), -1),
            (c2, ('descent',), []), (c2, ('cassels_tate', 'pairs', 0, 'j'), 1),
            (c2, ('cassels_tate', 'pairs', 0, 'primes'), ['5', '3']),
            (c2, ('descent', 0, 'pairing', 'conic'), []), (c2, ('descent', 0, 'pairing', 'places'), ['R']),
        ]
        for c, path, bad in cases:
            with self.subTest(path=path, bad=bad):
                with self.assertRaises(ValueError):
                    clean(put(c, path, bad), c['format'])

    def test_verdict(self):
        """Through check(): rejected at the check 'format', the reason naming the place; never 'error'."""
        import shutil, tempfile
        d = tempfile.mkdtemp()
        try:
            for name, path, bad, where in (('37a1', ('claims', 'k'), '0', 'claims.k: '),
                                           ('n5', ('descent', 1, 'pairing', 'local', 0), [], 'descent[1].pairing: '),
                                           ('icarm0.f3', ('field', 'GB'), 1.5, 'field.GB: ')):
                p = os.path.join(d, 'x.cert.json')
                with open(p, 'w') as f:
                    json.dump(put(self.C[name][1], path, bad), f)
                v = checker.check(p)
                self.assertEqual((v['verdict'], v['checks'][0]['check']), ('rejected', 'format'), v['reason'])
                self.assertTrue(v['reason'].startswith('not a valid witness: ' + where), v['reason'])
        finally:
            shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
