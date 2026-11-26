"""Bundles (rankbound/bundle.py, `rankbound unpack`): a witness put in a bundle comes back byte for byte, its sha256
checked; a changed witness or a missing id is refused."""
import gzip, hashlib, json, os, subprocess, sys, tempfile, unittest
from rankbound import bundle, rankbound
from helpers import SRC

ENV = dict(os.environ, PYTHONPATH=SRC)


def raw(path):
    with gzip.open(path, 'rb') if path.endswith('.gz') else open(path, 'rb') as f:
        return f.read()


class TestBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.res = [rankbound(a, points=pts, workdir=os.path.join(cls.d, 'w%d' % k), compute=True)
                   for k, (a, pts) in enumerate((([0, 0, 0, -25, 0], [['-4', '6']]),        # 2-isogeny, .json
                                                 ([0, -1, 1, -929, -10595], [])))]           # 2-Selmer, .json.gz
        cls.entries = []
        for k, res in enumerate(cls.res):
            b = raw(res.witness)
            cls.entries.append({'icarm_id': k, 'name': 'c%d' % k, 'S': res.rank_bound,
                                'sha256': hashlib.sha256(b).hexdigest(), 'witness': json.loads(b)})
        cls.path = os.path.join(cls.d, 'b.json.gz')
        bundle.write(cls.path, {'source': 'test'}, cls.entries)

    def test_round_trip(self):
        self.assertEqual(bundle.header(self.path), {'source': 'test'})
        self.assertEqual([e['name'] for e in bundle.entries(self.path)], ['c0', 'c1'])
        out = os.path.join(self.d, 'out')
        r = subprocess.run([sys.executable, '-m', 'rankbound', 'unpack', self.path, '--dir', out], env=ENV,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        paths = r.stdout.split()
        self.assertEqual([os.path.basename(p) for p in paths], ['c0.witness.json', 'c1.witness.json.gz'])
        for p, res in zip(paths, self.res):
            self.assertEqual(raw(p), raw(res.witness))
        self.assertEqual(bundle.unpack(self.path, os.path.join(self.d, 'one'), ['1']),
                         [os.path.join(self.d, 'one', 'c1.witness.json.gz')])

    def test_refused(self):
        bad = os.path.join(self.d, 'bad.json.gz')
        e = json.loads(json.dumps(self.entries[0]))
        e['witness']['claims'] = dict(e['witness'].get('claims') or {}, S=99)
        bundle.write(bad, {}, [e])
        with self.assertRaisesRegex(ValueError, 'does not match its sha256'):
            bundle.unpack(bad, os.path.join(self.d, 'bad'))
        with self.assertRaisesRegex(ValueError, 'no curve with the id 7'):
            bundle.unpack(self.path, os.path.join(self.d, 'none'), ['7'])


if __name__ == '__main__':
    unittest.main()
