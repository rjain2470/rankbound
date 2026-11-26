"""A resumed run reports only what the witness in its folder proves: a replaced witness with another lower bound
is an error, a deleted witness is proved again, a copied folder uses its own witness, proved_equal needs the
checker's acceptance, and the given bad primes are checked before a published result is returned."""
import glob, os, shutil, tempfile, unittest
from unittest import mock
import rankbound
from rankbound import runner

HAVE_GP = shutil.which('gp') is not None
E25 = [0, 0, 0, -25, 0]                 # 2-torsion: rank 1, by the descent and the point (-4, 6)
E1 = [0, 0, 0, -1, 0]                   # 2-torsion: rank 0


def tmpdir(t):
    d = tempfile.mkdtemp(prefix='rb_resume')
    t.addCleanup(shutil.rmtree, d, True)
    return d


def witness_of(w):
    return glob.glob(os.path.join(w, '**', '*.witness.json'), recursive=True)[0]


@unittest.skipUnless(HAVE_GP, 'PARI/GP not found')
class TestResume(unittest.TestCase):
    def proved(self, w, **kw):
        return rankbound.rankbound(E25, points=[(-4, 6)], compute=True, workdir=w, cores=1, **kw)

    def test_replaced_witness_with_less(self):
        d = tmpdir(self)
        a, b = os.path.join(d, 'a'), os.path.join(d, 'b')
        self.assertTrue(self.proved(a).proved_equal)
        rankbound.rankbound(E25, compute=True, workdir=b, cores=1)        # rank <= 1, no lower bound
        shutil.copyfile(witness_of(b), witness_of(a))
        with self.assertRaisesRegex(Exception, 'another lower bound'):
            self.proved(a)

    def test_deleted_witness(self):
        w = os.path.join(tmpdir(self), 'w')
        self.proved(w)
        os.remove(witness_of(w))
        res = self.proved(w)
        self.assertEqual(res.verification, 'accepted')
        self.assertTrue(os.path.exists(res.witness))

    def test_copied_folder(self):
        d = tmpdir(self)
        a, c = os.path.join(d, 'a'), os.path.join(d, 'c')
        self.proved(a)
        shutil.copytree(a, c)
        res = self.proved(c)
        self.assertTrue(os.path.abspath(res.witness).startswith(os.path.abspath(c) + os.sep), res.witness)

    def test_bad_primes_checked_on_resume(self):
        w = os.path.join(tmpdir(self), 'w')
        self.proved(w, bad_primes=[2, 5])
        with self.assertRaisesRegex(Exception, 'does not divide'):
            self.proved(w, bad_primes=[2, 5, 7])


class TestUnchecked(unittest.TestCase):
    def test_proved_equal_needs_the_checker(self):
        with mock.patch.object(runner, 'gp_usable', return_value=False):
            res = rankbound.rankbound(E1, compute=True, workdir=os.path.join(tmpdir(self), 'w'), cores=1)
        self.assertEqual((res.rank_bound, res.verification, res.proved_equal), (0, 'unavailable', False))


if __name__ == '__main__':
    unittest.main()
