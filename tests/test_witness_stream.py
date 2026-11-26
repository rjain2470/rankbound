"""Writing the Selmer witness: selmer_part's element classes computed once; selmer.theta reuses field.field_of's
exact theta; the streamed witness writer (the same bytes, bounded temporary memory, atomic)."""
import gzip, json, os, tempfile, tracemalloc, unittest
from fractions import Fraction
from rankbound import witness, selmer, field as fld, curve as cv, runtime

HERE = os.path.dirname(os.path.abspath(__file__))


def joined(c):
    """The writer's text before streaming (one string, as write_selmer built it), the reference for byte identity."""
    parts = []
    for k, v in c.items():
        if k in witness.LINES or k in witness.FLAT:
            items = [v[t:t + 64] for t in range(0, len(v), 64)] if k in witness.FLAT else v
            js = [json.dumps(x, separators=(',', ':')) for x in items]
            body = ',\n'.join('  ' + (x[1:-1] if k in witness.FLAT else x) for x in js)
            parts.append(' %s: [%s]' % (json.dumps(k), '\n' + body + '\n ' if v else ''))
        else:
            parts.append(' %s: %s' % (json.dumps(k), json.dumps(v, indent=1).replace('\n', '\n ')))
    return '{\n' + ',\n'.join(parts) + '\n}\n'


def old_bytes(text, gz):
    import io
    data = text.encode()
    if gz:
        buf = io.BytesIO()
        with gzip.GzipFile(filename='', mode='wb', fileobj=buf, mtime=0, compresslevel=6) as g:
            g.write(data)
        data = buf.getvalue()
    return data


def big(n):
    """571a1's witness with its relations repeated n times (a serialization stress control)."""
    c = witness.read(os.path.join(HERE, 'data', '571a1.witness.json'))
    c['relations'] = c['relations'] * n
    return c


class TestStream(unittest.TestCase):
    def test_bytes(self):
        """The streamed file has the bytes of the joined text (plain and gzipped; small, and past CHUNK)."""
        for c in (witness.read(os.path.join(HERE, 'data', '571a1.witness.json')), big(4000)):
            enc = witness.encode({k: v for k, v in c.items() if not k.startswith('_')})
            ref = joined(enc)
            self.assertEqual(''.join(witness.selmer_text(enc)), ref)
            for gz in (False, True):
                with tempfile.TemporaryDirectory() as d:
                    p = os.path.join(d, 'w.witness.json' + ('.gz' if gz else ''))
                    witness.write(c, p)
                    with open(p, 'rb') as f:
                        self.assertEqual(f.read(), old_bytes(ref, gz))
                    self.assertEqual(witness.read(p)['relations'], c['relations'])
        self.assertGreater(len(ref), 2 * witness.CHUNK)

    def test_memory(self):
        """Writing a large witness allocates about CHUNK, not the whole text (traced peak)."""
        enc = witness.encode(big(4000))
        n = len(joined(enc))
        peaks = []
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'w.witness.json.gz')
            for write in (lambda: runtime.write_bytes(p, old_bytes(joined(enc), True)),
                          lambda: witness.atomic(p, witness.selmer_text(enc))):
                tracemalloc.start()
                write()
                peaks.append(tracemalloc.get_traced_memory()[1])
                tracemalloc.stop()
        self.assertGreater(n, 2 * witness.CHUNK)
        self.assertLess(peaks[1], n // 2)   # the joined text alone is n bytes (ASCII)
        self.assertLess(4 * peaks[1], peaks[0])     # measured: 1.3 MB against 26.8 MB

    def test_atomic(self):
        """A writer that fails mid-stream leaves the old file and no temporary file."""
        with tempfile.TemporaryDirectory() as d:
            for gz in ('', '.gz'):
                p = os.path.join(d, 'w.json' + gz)
                witness.atomic(p, 'old\n')

                def pieces():
                    yield 'x' * (2 * witness.CHUNK)
                    raise OSError('disk full')
                with self.assertRaises(OSError):
                    witness.atomic(p, pieces())
                with (gzip.open(p, 'rt') if gz else open(p)) as f:
                    self.assertEqual(f.read(), 'old\n')
            self.assertEqual(sorted(os.listdir(d)), ['w.json', 'w.json.gz'])


class TestTheta(unittest.TestCase):
    def test_reuse(self):
        """Field.field_of's theta (matching form) equals the reconstructed one and is a root of W; a theta that
        is not a root, or another form, is not reused; a cyclic field (three roots of W in K) is reconstructed."""
        for a, P in [([0, 1, 1, -2, 0], [389]), ([0, 0, 1, -1, 0], [37]), ([0, 0, 1, -7, 6], [5077]),
                     ([0, -1, 1, -10, -20], [11])]:
            mm, _ = fld.check_bad_primes(a, P)
            f, th = fld.field_of(mm, [2, 3] + P)
            b2, b4, b6, _ = cv.binv(mm)
            W = [16 * b6, 8 * b4, b2, 1]
            k = selmer.kfield(f, P)
            self.assertEqual(selmer.theta(k, W, mm, P), selmer.theta(k, W))
            self.assertTrue(selmer.is_root(k, W, [Fraction(x) for x in th]))
            self.assertFalse(selmer.is_root(k, W, [Fraction(x) + 1 for x in th]))


class TestElementClasses(unittest.TestCase):
    def test_selmer_part(self):
        """An element's classes, XORed once and then cut to a place group, are the group's XOR (the identity
        selmer_part relies on); the witnesses themselves are test_selmer's (equal before and after: REPORT.md)."""
        import random
        rnd = random.Random(5)
        cls = {j: rnd.getrandbits(40) for j in range(60)}
        els = [rnd.sample(range(60), 9) for _ in range(12)]
        ecls = [selmer.xor(cls, e) for e in els]
        for o, w in ((0, 7), (7, 13), (20, 20)):
            for e, x in zip(els, ecls):
                self.assertEqual((x >> o) & ((1 << w) - 1), (selmer.xor(cls, e) >> o) & ((1 << w) - 1))


if __name__ == '__main__':
    unittest.main()
