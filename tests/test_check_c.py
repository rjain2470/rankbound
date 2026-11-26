"""The checker's program rankbound/check/check (check.c).  check rel, the relation check (through selmer.gp_rows): its
rows against PARI's idealval (a test-only GP reference, refrows) on index forms with lc(F) != 1 (primes of lc(F) in H,
at (1:0) and elsewhere), ramified, degree-2 and inert free relations, and on a prover witness; one corrupted relation
per check, each rejected (or over a limit); fuzzed input.  check rank: its modes and its input checks (its ranks are
tested against Python's in tests/test_f2la.py and tests/test_reference.py)."""
import array, json, math, os, random, re, subprocess, unittest
from rankbound import check as checker
from rankbound.check import selmer as vs
from helpers import REFROWS, TESTS, wit0

# primitive cubic forms [c0, c1, c2, c3] that are index forms (disc F = d_K): lc(F) = 1, 6, 21, 20, 18, 24, 245; x^3 - 5
FORMS = ([-1, -1, 0, 1], [1, 3, -2, 6], [-3, 1, 4, 21], [7, -2, 5, 20], [1, 3, -2, 18], [7, -2, 5, 24], [2, 9, -5, 245],
         [-5, 0, 0, 1])


GRAMMAR = re.compile(r'F( (0|-?[1-9][0-9]*)){4}|L [1-9][0-9]* [1-9][0-9]*|\[[1-9][0-9]*\]|'
                     r'\[(0|-?[1-9][0-9]*),(0|-?[1-9][0-9]*),\[([1-9][0-9]*(,[1-9][0-9]*)*)?\]\]')   # check rel's input
REL, RANK = [vs.CHECK, 'rel'], [vs.CHECK, 'rank']


def factor(n):
    n, out, p = abs(n), [], 2
    while p * p <= n:
        while n % p == 0:
            out.append(p); n //= p
        p += 1
    return out + ([n] if n > 1 else [])


def F(poly, a, b):
    return sum(c * a ** i * b ** (3 - i) for i, c in enumerate(poly))


def ref_rows(poly, rels):
    f = vs.gp_poly(poly)
    body = REFROWS + 'K = ckinit(%s, factor(abs(poldisc(%s)))[, 1]~); refrows(K, [%s]);' % (f, f, ','.join(rels))
    out = checker.run_gp(body, seconds=600)
    return [sorted(l.split()[2:]) for l in out.splitlines() if l.startswith('R ')]


def c_rows(poly, rels):
    f = vs.gp_poly(poly)
    P = checker.run_gp('print(strjoin(apply(z -> Str(z), factor(abs(poldisc(%s)))[, 1]~), ","));' % f, seconds=60).split()[0]
    rows, names = vs.gp_rows([str(c) for c in poly], P.split(','), rels, 1)
    return [sorted(names[i] for i in r) for r in rows]


class TestRelcheck(unittest.TestCase):
    def test_forms_are_index_forms(self):
        out = checker.run_gp('foreach([%s], v, my(f = Pol(Vecrev(v))); print(poldisc(f) == nfdisc(f) && content(f) == 1));'
                              % ','.join(str(v) for v in FORMS), seconds=60)
        self.assertEqual(out.split(), ['1'] * len(FORMS))

    def test_against_pari(self):
        """Random coprime (a, b), b of either sign and with primes of lc(F), and every free relation p < 200."""
        rnd = random.Random(4)
        for poly in FORMS:
            rels = []
            while len(rels) < 150:
                a, b = rnd.randint(-300, 300), rnd.choice([rnd.randint(-60, 60), poly[3] * rnd.randint(-5, 5)])
                if math.gcd(a, b) != 1 or F(poly, a, b) == 0:
                    continue
                rels.append('[%d,%d,[%s]]' % (a, b, ','.join(map(str, factor(F(poly, a, b))))))
            rels += ['[%d]' % p for p in range(2, 200) if all(p % q for q in range(2, p))]
            self.assertEqual(c_rows(poly, rels), ref_rows(poly, rels), poly)

    def test_general(self):
        """General relations (c0 + c1 alpha + c2 alpha^2)/d, by ckgen (its v_p(N) = 1 shortcut at the roots of F mod p,
        idealval otherwise), against PARI's idealval on every prime of H, lc(F) and d."""
        rnd = random.Random(8)
        for poly in FORMS:
            f = vs.gp_poly(poly)
            gens = []
            for _ in range(120):
                c = [rnd.randint(-400, 400) for _ in range(3)] + [rnd.choice([1, 1, 1, 2, 3, 6, 35])]
                if any(c[:3]):
                    gens.append(c)
            out = checker.run_gp('K = ckinit(%s, factor(abs(poldisc(%s)))[, 1]~); foreach([%s], c, my(n = nfeltnorm(K[1], '
                                 '(c[1] + c[2] * K[2] + c[3] * K[2]^2) / c[4]), m = numerator(n) * denominator(n)); '
                                 'print("H ", strjoin(apply(z -> Str(z), factor(abs(m))[, 1]~), ",")));' % (
                                     f, f, ','.join('[%d,%d,%d,%d]' % tuple(c) for c in gens)), seconds=120)
            Hs = [l[2:].strip() for l in out.splitlines() if l.startswith('H ')]
            rels = ['[%d,%d,%d,%d,[%s]]' % (tuple(c) + (h,)) for c, h in zip(gens, Hs)]
            self.assertEqual(c_rows(poly, rels), ref_rows(poly, rels), poly)

    def test_witness(self):
        """The rows of ICARM #0's witness (as the checker reads it) against PARI's."""
        with open(wit0()) as f:
            c = json.load(f)
        rels = [vs._gpvec(l, True) for l in c['relations']]
        self.assertEqual(c_rows(c['field']['poly'], rels), ref_rows([int(x) for x in c['field']['poly']], rels))

    def test_icarm728(self):
        """1,000 relations of ICARM #728's witness (tests/data/reference.json; lc(F) with primes 2, 3, 11, 13, 23,
        1021, 1223 and 2756959289691073 at odd powers) against PARI's."""
        with open(os.path.join(TESTS, 'data', 'reference.json')) as f:
            c = json.load(f)['icarm728']
        poly = c['poly']                            # PARI's index form, which the relations are for
        rels = [vs._gpvec(l) for l in c['relations']]
        self.assertEqual(c_rows(poly, rels), ref_rows(poly, rels))

    def test_corrupted(self):
        """Each check of a relation, failed once: the reason, and the relation's index."""
        poly = FORMS[3]
        good = ['[%d,%d,[%s]]' % (a, b, ','.join(map(str, factor(F(poly, a, b))))) for a, b in ((3, 1), (-7, 2), (1, 0))]
        H = factor(F(poly, 5, 3))
        comp = sorted(H[:-2] + [H[-2] * H[-1]]) if len(H) >= 2 else None
        cases = {
            'F(a,b) = 0': ('[0,0,[]]', r'has F\(a, b\) = 0'),
            'prod H': ('[5,3,[%s]]' % ','.join(map(str, H[:-1])), 'does not factor as claimed'),
            'prod H, one more': ('[5,3,[%s]]' % ','.join(map(str, H + [2])), 'does not factor as claimed'),
            'not prime': ('[5,3,[%s]]' % ','.join(map(str, comp)), 'has %d, which is not prime' % (H[-2] * H[-1])),
            'not coprime': ('[%d,%d,[%s]]' % (10, 6, ','.join(map(str, factor(F(poly, 10, 6))))), 'has a and b not coprime'),
            'free not prime': ('[15]', r'is \(15\), and 15 is not prime'),
        }
        for what, (rel, why) in cases.items():
            with self.subTest(what):
                rels = good + [rel]
                with self.assertRaisesRegex(checker.Rejected, r'^relation 3 ' + why + '$'):
                    vs.gp_rows([str(c) for c in poly], [], rels, 1)
        # beyond check rel (a prime of 2^64 or more): checked in gp as a general element; 2^64 + 1 = 274177 x ...
        with self.assertRaisesRegex(checker.Rejected, '^relation 3 has 18446744073709551617, which is not prime$'):
            vs.gp_rows([str(c) for c in poly], [], good + ['[18446744073709551617]'], 1)

    def test_beyond_check_rel(self):
        """Relations with a prime of 2^64 or more (a free one, and a - b alpha with F(a, b) factored in gp) are checked
        in gp as general elements; their rows against PARI's, with the others' from check rel."""
        poly = FORMS[3]
        f = vs.gp_poly(poly)
        out = checker.run_gp('F2 = substvec(%s, [x], [x / y]) * y^3; for(a = 2^40, 2^40 + 60, my(n = abs(subst(subst(F2, x, a), y, 7))); '
                              'my(fa = factor(n)); if(gcd(a, 7) == 1 && fa[#fa~, 1] >= 2^64, print(a, " ", strjoin(apply(z -> Str(z), '
                              'concat(vector(#fa~, i, vector(fa[i, 2], j, fa[i, 1])))), ","))));' % f, seconds=120)
        ab = ['[%s,7,[%s]]' % tuple(l.split()) for l in out.splitlines()][:3]
        self.assertTrue(ab)
        rels = ['[3,1,[%s]]' % ','.join(map(str, factor(F(poly, 3, 1)))), '[18446744073709551629]', '[101]'] + ab
        self.assertEqual(c_rows(poly, rels), ref_rows(poly, rels))

    def test_lc_factorization(self):
        """check rel refuses L lines that are not the factorization of |c3|, and needs the F line first."""
        for inp in ('F 7 -2 5 20\nL 2 2\n[3,1,[2]]\n', 'F 7 -2 5 20\nL 2 2\nL 5 1\nL 3 1\n[3,1,[2]]\n', '[3]\n'):
            r = subprocess.run(REL, input=inp.encode(), capture_output=True)
            self.assertEqual(r.returncode, 2, inp)
            self.assertIn(b'check rel: line ', r.stderr)

    def test_fuzz(self):
        """check rel on mutated input (a fixed seed; bytes changed, inserted or deleted, lines dropped, repeated or
        swapped, from a valid input with lc(F) = 20): it never crashes (exit 0 to 3, no sanitizer report: make
        test-asan), and accepts only its grammar (GRAMMAR): exit 0 only if every line is well formed, with one R line per
        relation; a failed check (1) or a limit (3) names its relation; anything else malformed is exit 2."""
        poly = FORMS[3]
        rels = ['[%d,%d,[%s]]' % (a, b, ','.join(map(str, factor(F(poly, a, b)))))
                for a, b in ((3, 1), (-7, 2), (1, 0), (5, 3), (-11, 4), (0, 1)) if F(poly, a, b)]
        base = ['F %s' % ' '.join(map(str, poly)), 'L 2 2', 'L 5 1'] + rels + ['[2]', '[13]', '[101]']
        rnd, seen, n = random.Random(20), {}, 3000 if os.environ.get('RANKBOUND_TEST_FULL') else 600
        for k in range(n):
            lines = list(base)
            for _ in range(rnd.randint(1, 3)):
                i = rnd.randrange(len(lines))
                op, l = rnd.randrange(6), lines[i]
                j = rnd.randrange(len(l) + 1)
                if op == 0:
                    lines[i] = l[:j] + rnd.choice('0123456789-,[] \r\x00xF') + l[j + 1:]
                elif op == 1:
                    lines[i] = l[:j] + rnd.choice('0123456789-,[] 9999999999999999999') + l[j:]
                elif op == 2:
                    lines[i] = l[:j] + l[j + 1:]
                elif op == 3:
                    del lines[i]
                elif op == 4:
                    lines.insert(i, l)
                else:
                    lines[i:i + 2] = lines[i:i + 2][::-1]
            inp = '\n'.join(lines) + '\n'
            r = subprocess.run(REL, input=inp.encode(), capture_output=True, timeout=60)
            err = r.stderr.decode(errors='replace')
            self.assertIn(r.returncode, (0, 1, 2, 3), (inp, err))
            self.assertNotRegex(err, 'Sanitizer|runtime error', inp)
            seen[r.returncode] = seen.get(r.returncode, 0) + 1
            out, lines = r.stdout.decode().splitlines(), [l.rstrip('\r') for l in inp.split('\n') if l.rstrip('\r')]
            if r.returncode == 0:           # every line well formed, one R line per relation
                self.assertTrue(all(GRAMMAR.fullmatch(l) for l in lines), inp)
                self.assertEqual(len(out), sum(l[0] == '[' for l in lines), inp)
            elif r.returncode in (1, 3):    # a failed check or a limit, at the relation it names
                self.assertRegex(out[-1], r'^[EX] \d+ ', inp)
        self.assertTrue(all(seen.get(x, 0) > n // 50 for x in (0, 1, 2)), seen)



class TestModes(unittest.TestCase):
    """One program, two modes: check rel and check rank [THREADS [PREFIX]]; anything else is a usage error (exit 2).  check
    rank on bad input: exit 2 with a message."""

    def test_modes(self):
        for args in ([], ['rels'], ['rel', '1'], ['rank', '1', '2', '3'], ['--help']):
            r = subprocess.run([vs.CHECK] + args, input=b'', capture_output=True, timeout=60)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn(b'usage: check rel', r.stderr)
        words = [2, 3, 1, 0, 2, 2, 0, 1, 1, 2, 0, 1]           # rows {0, 1}, {2}; nl = 1; pivots 0, 1
        for th in ([], ['1'], ['4']):
            r = subprocess.run(RANK + th, input=array.array('I', words).tobytes(), capture_output=True, timeout=60)
            self.assertEqual((r.returncode, r.stdout), (0, b'rank 2 rankL 1 large 1 dimV 1\n'), th)
        for pre in ('-1', '4294967295', '4294967296', '18446744073709551617', '2x', ''):       # PREFIX in 0..2^32-2
            r = subprocess.run(RANK + ['1', pre], input=array.array('I', words).tobytes(), capture_output=True, timeout=60)
            self.assertEqual(r.returncode, 2, pre)
            self.assertIn(b'check rank: PREFIX must be', r.stderr)

    def test_rank_input(self):
        for words in ([1, 2], [1, 2, 3, 0, 0], [1, 2, 1, 0, 1, 1, 2, 0], [1, 2, 1, 0, 1, 2, 1, 1, 0],
                      [1, 2, 1, 1, 0, 0, 0, 1], [1, 2, 1, 0, 1, 0, 1], [1, 3, 1, 0, 1, 4, 0, 1, 2, 0, 0]):
            p = subprocess.run(RANK, input=array.array('I', words).tobytes(), capture_output=True, timeout=60)
            self.assertEqual(p.returncode, 2, words)
            self.assertIn(b'check rank: ', p.stderr)


if __name__ == '__main__':
    unittest.main()
