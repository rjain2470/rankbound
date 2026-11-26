"""The curve: input checks, triage, factoring the discriminant, the table of known curves, the rank lower bound.

The second half is the arithmetic of the curve without PARI."""
import contextlib, contextvars, copy, dataclasses, fcntl, fractions, gzip, hashlib, itertools, json, logging, math, os, re, shutil, subprocess, sys, tempfile, time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional
from .core import MAX_DIGITS, NAME, NUM, RAT, FactorisationNeeded, Locked, RankboundError, Rejected, Stopped
from .runtime import read_json, write_json
from . import arith, runtime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = logging.getLogger('rankbound').info


class _Detail:
    def info(self, *a):
        from .runtime import DETAIL as D
        D.info(*a)


DETAIL = _Detail()


# ---------- values that reach gp

WHAT = {'ainvs': 'a-invariant', 'bad_primes': 'bad prime', 'minimal_model': 'minimal_model value',
        'proof point': 'point coordinate', 'mapped point': 'point coordinate'}

def canon(x, what, pat=NUM):
    """Return x, a JSON integer or a canonical decimal string, as a string; raise ValueError otherwise."""
    if type(x) is int:
        s = str(x) if x.bit_length() <= 14000 else None      # str() refuses past 4,300 digits
    else:
        s = x if isinstance(x, str) and pat.fullmatch(x) else ''
    what = WHAT.get(what, what)
    if s is None or len(s) > MAX_DIGITS:
        raise ValueError("%s %s has more than %d digits, the verifier's limit." % (
            'An' if what[0] in 'aeiou' else 'A', what, MAX_DIGITS))
    if s:
        return s
    if pat is NUM:
        raise ValueError('The %s %.80r is not an integer.' % (what, x))
    raise ValueError('The %s %.80r is not written like -3/4.' % (what, x))


def check_name(name):
    """Return name if the checker accepts it, else raise ValueError."""
    if isinstance(name, str) and NAME.fullmatch(name) and name not in ('.', '..'):
        return name
    raise ValueError('The name %.80r is not allowed: use at most 64 letters, digits, _ . and -.' % (name,))


def r_value(r):
    """Return r if it is None or an integer >= 0, else raise ValueError."""
    if r is None or (type(r) is int and r >= 0):
        return r
    raise ValueError('r must be an integer >= 0.')


PROOF = '"proof" must be {"points": [[x, y], ...], "primes": [p, ...]}.'


def check_proof(proof):
    """Return the proof of a rank lower bound (points, optional primes and ainvs), or None; raise ValueError."""
    if proof is None:
        return None
    if not isinstance(proof, dict):
        raise ValueError(PROOF)
    pts, prs = proof.get('points'), proof.get('primes', [])
    if not (isinstance(pts, list) and (prs is None or isinstance(prs, list))):
        raise ValueError(PROOF)
    prs = prs or []
    if proof.get('ainvs') is not None:
        _ainvs5(proof['ainvs'])
    for pt in pts:
        if not (isinstance(pt, list) and len(pt) == 2):
            raise ValueError(PROOF)
        for x in pt:
            canon(x, 'proof point', RAT)
    for p in prs:
        if type(p) is not int:
            raise ValueError(PROOF)
    return proof


def point_value(x):
    """Return a point coordinate (int, Fraction or a string like '-3/4') as a canonical string."""
    if isinstance(x, fractions.Fraction):
        x = str(x)
    return canon(x, 'point coordinate', RAT)


def read_points(path):
    """Return the points of a --points file (one 'x y' or 'x,y' per line, '#' for comments) as strings."""
    pts = []
    with open(path) as f:
        try:
            for k, l in enumerate(f, 1):
                t = l.replace(',', ' ').split()
                if not t or t[0].startswith('#'):
                    continue
                if len(t) != 2:
                    raise ValueError('%s, line %d: not a point "x y".' % (path, k))
                try:
                    pts.append([point_value(x) for x in t])
                except ValueError as e:
                    raise ValueError('%s, line %d: %s' % (path, k, e))
        except UnicodeDecodeError:
            raise ValueError('%s is not UTF-8 text.' % path)
    return pts


def points_of(points):
    """Return points as pairs of canonical strings; raise ValueError for a non-pair or two with the same x."""
    if not all(isinstance(p, (list, tuple)) for p in seq(points, 'points')):
        raise ValueError('Each point must be a pair (x, y).')
    pts = [[point_value(x) for x in p] for p in points]
    if any(len(p) != 2 for p in pts):
        raise ValueError('Each point must be a pair (x, y).')
    xs = [p[0] for p in pts]
    for j, x in enumerate(xs):
        if x in xs[:j]:
            raise ValueError('Points %d and %d have the same x, so one is plus or minus the other: they are dependent.'
                             % (xs.index(x) + 1, j + 1))
    return pts


def with_points(curve, points, r=None):
    """Return curve with r = len(points), proved by the points (on the given model)."""
    pts = points_of(points)
    if r is not None and r_value(r) != len(pts):
        raise ValueError('r = %s but %d points were given.' % (r, len(pts)))
    return dataclasses.replace(curve, r=len(pts), proof={'points': pts, 'ainvs': list(curve.ainvs)}, source='cited')


# ---------- the curve

@dataclass(frozen=True)
class Curve:
    name: str
    ainvs: List[str]                          # the input model
    bad_primes: Optional[List[str]]           # None: to be found by factored()
    r: Optional[int] = None
    proof: Optional[dict] = None              # points (and primes) that prove r
    source: str = 'cited'                     # where r is from, without a proof
    minimal_model: Optional[List[str]] = None
    composites: List[str] = field(default_factory=list)
    torsion: Optional[List[int]] = None       # as elltors, e.g. [2, 2]

    def to_dict(self):
        d = {'name': self.name, 'ainvs': self.ainvs, 'bad_primes': self.bad_primes, 'composites': self.composites,
             'r': self.r, 'proof': self.proof, 'source': self.source}
        if self.minimal_model is not None:
            d['minimal_model'] = self.minimal_model
        if self.torsion is not None:
            d['torsion'] = self.torsion
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(name=d['name'], ainvs=d['ainvs'], bad_primes=d['bad_primes'], r=d.get('r'), proof=d.get('proof'),
                   source=d.get('source', 'cited'), minimal_model=d.get('minimal_model'), composites=d.get('composites', []),
                   torsion=d.get('torsion'))

    def two_torsion(self):
        """Return whether E(Q)[2] != 0, or None if the torsion is not known."""
        return None if self.torsion is None else any(n % 2 == 0 for n in self.torsion)

    def key(self):
        """Return the default name of the curve."""
        return default_name(self.minimal_model or from_ainvs(self.ainvs, bad_primes=self.bad_primes).minimal_model)


CURVE_SCHEMA = 'rankbound-curve-1'      # curve.json


def curve_fingerprint(c):
    """Return the fingerprint of the artifact 'curve'."""
    return runtime.fingerprint(schema=CURVE_SCHEMA, name=c.name, ainvs=[str(a) for a in c.ainvs])


def save(folder, c):
    """Commit the Curve c as the work folder's artifact 'curve'."""
    with runtime.transaction(folder, 'curve', CURVE_SCHEMA, curve_fingerprint(c)) as tx:
        tx.write('curve.json', json.dumps(c.to_dict(), indent=1) + '\n')


def saved(folder, c=None):
    """Return the Curve of the work folder's artifact 'curve', or None if missing, damaged or not for c."""
    m = runtime.load(folder, 'curve', CURVE_SCHEMA, None if c is None else curve_fingerprint(c))
    return None if m is None else Curve.from_dict(read_json(os.path.join(folder, 'curve.json')))


def short_ainvs(ainvs, digits=12):
    """Return the a-invariants joined by ',', long ones shortened as '-1280...276'."""
    out = []
    for x in map(str, ainvs):
        d = x.lstrip('-')
        out.append(x if len(d) <= digits else x[:len(x) - len(d)] + d[:4] + '...' + d[-3:])
    return ','.join(out)


def hashed(name, mm):
    """Return whether name is the hash that default_name gives the minimal model mm."""
    return mm is not None and name == default_name(mm) and len('_'.join(str(x) for x in mm)) > 64


def default_name(mm):
    """Return the minimal model joined by '_', or a 12-digit hash beyond 64 characters."""
    s = '_'.join(str(x) for x in mm)
    return s if len(s) <= 64 else hashlib.sha256(','.join(str(x) for x in mm).encode()).hexdigest()[:12]


def checked(c):
    """Return the hand-made Curve c after checking every field as from_ainvs would."""
    if not isinstance(c, Curve):
        raise ValueError('Expected a Curve.')
    check_name(c.name)
    _ainvs5(c.ainvs)
    _bad_primes(c.bad_primes)
    for x in seq(c.composites, 'composites'):
        canon(x, 'composite')
    if c.minimal_model is not None:
        [canon(x, 'minimal_model') for x in seq(c.minimal_model, 'minimal_model')]
    r_value(c.r)
    check_proof(c.proof)
    if not isinstance(c.source, str):
        raise ValueError('source must be a string.')
    if c.torsion is not None and not (isinstance(c.torsion, list) and len(c.torsion) <= 2
                                      and all(type(n) is int and n > 0 for n in c.torsion)):
        raise ValueError('torsion must be a list of at most two positive integers, as from elltors.')
    return c


def triage(a, bad_primes, seconds):
    """Return ['E2', torsion, minimal model, primes, composites] as strings.

    Factoring: trial division to 10^6, perfect powers, the gcd with c4 (divisible by the additive primes), then
    arith.factor for seconds per cofactor (0: leave the composites to ECM)."""
    mm, _ = minimal(a, [int(p) for p in bad_primes or []])
    P, C = set(), []
    if bad_primes is None:
        c4, D = cinv(mm)[0], abs(cinv(mm)[2])
        todo = []
        for p, e in arith.factor(D, 10 ** 6):
            (P.add(p) if p < 10 ** 6 or arith.is_prime(p) else todo.append(p))
        while todo:
            q = todo.pop()
            if q == 1:
                continue
            if arith.is_prime(q):
                P.add(q)
                continue
            b, k = power_base(q)
            g = math.gcd(q, c4)
            if k > 1:
                todo.append(b)
            elif 1 < g < q:
                todo += [g, q // g]
            else:
                try:
                    fa = arith.factor(q, seconds=seconds) if seconds else None
                except (ValueError, TimeoutError):
                    fa = None
                if fa is None:
                    C.append(q)
                else:
                    P |= {p for p, _ in fa}
    J = lambda v: ','.join(map(str, v))
    return ['E2', J(torsion_group(mm)[0]), J(mm), J(sorted(P)), J(sorted(set(C)))]


def seq(v, what):
    """Return v if it is a list or tuple (not a string), else raise ValueError."""
    if isinstance(v, (list, tuple)):
        return v
    raise ValueError('%s must be a list.' % what)


def disc(a):

    a1, a2, a3, a4, a6 = a
    b2, b4, b6 = a1 * a1 + 4 * a2, 2 * a4 + a1 * a3, a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    return -b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 * b6 + 9 * b2 * b4 * b6


def ainvs_given(ainvs):
    """Return the 5 integers of ainvs (5 a-invariants or [a4, a6]); raise ValueError if malformed."""
    a = [int(canon(x, 'a-invariant')) for x in seq(ainvs, 'ainvs')]
    if len(a) == 2:
        a = [0, 0, 0] + a
    if len(a) != 5:
        raise ValueError('Give the curve as a1,a2,a3,a4,a6 or as a4,a6.')
    return a


def _ainvs5(ainvs):
    """Return ainvs_given(ainvs); raise ValueError also if singular."""
    a = ainvs_given(ainvs)
    if disc(a) == 0:
        raise ValueError('The a-invariants [%s] give a singular curve (discriminant 0).' % ','.join(map(str, a)))
    return a


def _bad_primes(bad_primes):
    """Return bad_primes as canonical strings >= 2; raise ValueError otherwise."""
    bad_primes = [canon(p, 'bad prime') for p in seq(bad_primes, 'bad_primes')]
    for p in bad_primes:
        if int(p) < 2:
            raise ValueError('The bad prime %s is not prime.' % p)
    return bad_primes


def minimal_model(ainvs, bad_primes=None):
    """Return the reduced minimal model of ainvs as 5 strings."""
    a = _ainvs5(ainvs)
    return [str(x) for x in minimal(a, [int(p) for p in _bad_primes(bad_primes or [])])[0]]


NOT_APPLICABLE = ('This curve has a rational point of order 2, so the class group method does not apply (rankbound() '
                  'and `rankbound bound` use a 2-isogeny descent for it).')


def torsion(ainvs):
    """Return the structure of E(Q)_tors of ainvs (any model), as elltors: e.g. [2, 2]."""
    return torsion_group(minimal(_ainvs5(ainvs))[0])[0]


def classified(curve):
    """Return curve with its torsion, computed if it is not known."""
    return curve if curve.torsion is not None else dataclasses.replace(curve, torsion=torsion(curve.ainvs))


def from_ainvs(ainvs, bad_primes=None, r=None, name=None, factor_seconds=600, composites=False):
    """Return the Curve of ainvs, on any model, with its torsion.

    Without bad_primes the discriminant is factored for factor_seconds per cofactor; a composite left raises
    FactorisationNeeded or, with composites, goes into Curve.composites."""
    a = _ainvs5(ainvs)
    r = r_value(r)
    if name is not None:
        check_name(name)
    if bad_primes is not None:
        bad_primes = _bad_primes(bad_primes)
    t = triage(a, bad_primes, factor_seconds)
    left = []
    if bad_primes is None:
        left = [x for x in t[4].strip().split(',') if x]
        if left and not composites:
            raise FactorisationNeeded('The discriminant could not be factored. Give its prime factors with '
                                      '--bad-primes P1,P2,...', [{'n': c, 'digits': len(c)} for c in left])
        bad_primes = [x for x in t[3].strip().split(',') if x]
    mm = t[2].strip().split(',')
    return Curve(name=default_name(mm) if name is None else name, ainvs=[str(x) for x in a], bad_primes=[str(p) for p in bad_primes], r=r,
                 minimal_model=mm, composites=left, torsion=[int(x) for x in t[1].strip().split(',') if x])


def from_list(path, ids=None):
    """Return the curves of a JSON curve list (optionally gzipped) whose id is in ids (None: all)."""
    src = os.path.basename(path)

    def obj(pairs):                     # json would keep the last of repeated keys
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('%s has the key %s twice in one object.' % (src, json.dumps(k)[:80]))
            d[k] = v
        return d

    def constant(x):
        raise ValueError('%s has %s, which is not a JSON number.' % (src, x))

    def integer(x):                     # int()'s own error past 4,300 digits is about Python
        if len(x.lstrip('-')) > MAX_DIGITS:
            raise ValueError("%s has a number of more than %d digits, the verifier's limit." % (src, MAX_DIGITS))
        return int(x)
    try:
        with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
            d = json.load(f, object_pairs_hook=obj, parse_constant=constant, parse_int=integer)
    except RecursionError:
        raise ValueError('%s is nested too deeply.' % path)
    except UnicodeDecodeError:
        raise ValueError('%s is not UTF-8 text.' % path)
    except json.JSONDecodeError as e:
        raise ValueError('%s is not valid JSON at line %d, column %d.' % (path, e.lineno, e.colno))
    except (gzip.BadGzipFile, EOFError):
        raise ValueError('%s is not a valid gzip file.' % path)
    if not (isinstance(d, dict) and isinstance(d.get('curves'), list)
            and all(isinstance(c, dict) for c in d['curves'])):
        raise ValueError('%s is not a curve list: {"curves": [{"id": ..., "ainvs": [...]}, ...]}.' % src)
    out, found, seen = [], set(), set()
    for k, c in enumerate(d['curves'], 1):
        icarm = 'icarm_id' in c
        i = c['icarm_id'] if icarm else c.get('id')
        if i is None:
            raise ValueError('Entry %d of %s has no "id".' % (k, src))
        if isinstance(i, bool) or not (isinstance(i, (int, str)) or (isinstance(i, float) and math.isfinite(i))):
            raise ValueError('Entry %d of %s has the id %.80s: give a number or a string.' % (k, src, json.dumps(i)))
        if str(i) in seen:
            raise ValueError('%s has two curves with the id %.80s.' % (src, i))
        seen.add(str(i))
        if ids and str(i) not in {str(x) for x in ids}:
            continue
        found.add(str(i))
        try:
            ai = [canon(x, 'ainvs') for x in seq(c['ainvs'], 'ainvs')]
            _ainvs5(ai)
            mm = c.get('minimal_model')
            mm = None if mm is None else [canon(x, 'minimal_model') for x in seq(mm, 'minimal_model')]
            pf = check_proof(c.get('proof'))
            # the points are on the list's model
            if pf is not None and pf.get('ainvs') is None:
                pf = dict(pf, ainvs=ai)
            r = r_value(c.get('r'))
            if 'bad_primes' in c:
                bad = [canon(p, 'bad_primes') for p in seq(c['bad_primes'], 'bad_primes')]
            elif c.get('composites'):
                raise ValueError('"composites" needs "bad_primes" (the primes found so far).')
            else:
                bad = None
            if r is None and pf is not None and pf.get('points'):
                r = len(pf['points'])
            elif r is not None and pf is not None and pf.get('points') and r != len(pf['points']):
                raise ValueError('r = %d but the proof has %d points (the points prove r = their number).'
                                 % (r, len(pf['points'])))
            out.append(Curve(name=check_name(c.get('name', ('icarm%s' if icarm else 'curve%s') % i)), ainvs=ai,
                             bad_primes=bad,
                             composites=list(seq(c.get('composites', []), 'composites')), r=r, proof=pf,
                             source='ICARM #%s' % i if icarm else 'curve %s of %s' % (i, src), minimal_model=mm))
        except (ValueError, TypeError) as e:
            raise ValueError('%s, curve %s: %s' % (src, i, e))
        except KeyError as e:
            raise ValueError('Curve %s of %s has no "%s".' % (i, src, e.args[0]))
    if ids and found != {str(x) for x in ids}:
        miss = sorted({str(x) for x in ids} - found)
        raise ValueError('%s has no curve with the id%s %s.' % (src, 's' if len(miss) > 1 else '', ', '.join(miss)))
    names = [c.name for c in out]
    if len(set(names)) < len(names):
        raise ValueError('%s has two curves named %s.' % (src, sorted(n for n in names if names.count(n) > 1)[0]))
    if not out:
        raise ValueError('%s has no curves.' % src)
    return out


def factored(curve, ecm_hours, workdir):
    """Return the listed curve with its bad primes, from workdir if prepared there, else as from_ainvs.

    With ecm_hours, what is left goes in composites; without, cofactors get 600 s of factoring."""
    if curve.bad_primes is not None:
        return curve
    c = saved(workdir, curve)
    if c is not None and c.bad_primes is not None and not c.composites:
        return dataclasses.replace(curve, bad_primes=list(c.bad_primes), composites=[])
    try:
        c = from_ainvs(curve.ainvs, factor_seconds=0 if ecm_hours else 600, composites=bool(ecm_hours))
    except FactorisationNeeded as e:
        raise FactorisationNeeded('The discriminant could not be factored. Give its prime factors in "bad_primes".',
                                  e.composites)
    return dataclasses.replace(curve, bad_primes=c.bad_primes, composites=c.composites)


FACTOR_FORMAT = 'rankbound-factor-1'
# (factor digits, B1, expected curves), from the GMP-ECM 7 README; a finished level makes no factor below D digits
# likely, not proved
LEVELS = [(20, 11e3, 74), (25, 5e4, 221), (30, 25e4, 453), (35, 1e6, 984), (40, 3e6, 2541), (45, 11e6, 4949),
          (50, 43e6, 8266), (55, 11e7, 20158), (60, 26e7, 47173), (65, 85e7, 77666)]
# seconds per curve at 140 digits, scaled by (digits/140)^2 until measured
SPC_GUESS = [0.03, 0.1, 0.5, 2.0, 6.0, 17, 60, 150, 360, 1200]
PM1_B1 = 1e9            # P-1 once per composite,
PM1_AFTER = 3           # after the levels below this one,
PM1_MIN_S = 1200        # when at least this long is left
CALL_S = 300            # aim per GMP-ECM call
# A composite of at most PARI_DIGITS digits gets PARI_S seconds of arith.factor first: if its primes are all small,
# ECM finds the whole number, which is no split.
PARI_DIGITS = 60
PARI_S = 60
ECM_FOUND = (2, 6, 10, 14)      # GMP-ECM's exit codes for a proper factor (8: the input; odd: an error)
ECM_WHOLE = 8
MAX_FAILS = 3           # failed calls of one kind on one composite before giving up
POLL_S = 0.2


def find_ecm():
    """Return $RANKBOUND_ECM, else ecm on PATH, or None (also for RANKBOUND_ECM=none)."""
    e = os.environ.get('RANKBOUND_ECM', '').strip()
    if e.lower() == 'none':
        return None
    if e:
        if not (os.path.isfile(e) and os.access(e, os.X_OK)):
            from .messages import text
            raise RankboundError(text('ecm_bad', path=e))
        return e
    return shutil.which('ecm')


# ---------- factoring the discriminant (STATE/factor.json: a record per composite, open or split into parts)

SMALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]


def root(n, k):
    """Return the integer k-th root of n if exact, else None."""
    x = 1 << -(-n.bit_length() // k)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            break
        x = y
    return x if x ** k == n else None


def power_base(n):
    """Return (g, k) with n = g^k and k largest, by prime roots (every k with n = h^k divides the largest)."""
    g, k, ps, i = n, 1, arith.primes(2, n.bit_length() + 1), 0
    while i < len(ps) and 1 << ps[i] <= g:
        r = root(g, ps[i])
        if r is None:
            i += 1
        else:
            g, k = r, k * ps[i]
    return g, k


def label(n):
    return ('p%d' if arith.is_prime(n) else 'C%d') % len(str(n))



def new_record(n, like=None):
    """Return a new record for the composite n; a part inherits the effort of like, the whole it came from."""
    r = {'digits': len(str(n)), 'status': 'open', 'pm1': False, 'done': {}, 'spc': {}, 'core_s': 0.0, 'pari_s': 0.0,
         'found': [], 'parts': []}
    if like:
        r.update(pm1=like['pm1'], done=dict(like['done']), spc=dict(like['spc']))
    return r


def load(path):
    st = read_json(path, None)
    if st is None:
        return {'format': FACTOR_FORMAT, 'numbers': {}, 'core_s': 0.0, 'wall_s': 0.0}
    if st.get('format') != FACTOR_FORMAT:
        raise RankboundError('%s is not a factor.json of this version of rankbound.' % path)
    return st


def register(st, n, like=None):
    """Record n if it is a new composite; g^k is recorded as split into g."""
    if arith.is_prime(n) or str(n) in st['numbers']:
        return
    g, k = power_base(n)
    st['numbers'][str(n)] = new_record(n, like)
    if k > 1:
        st['numbers'][str(n)].update(status='split', parts=[str(g)])
        register(st, g, like)


def resolve(st, n):
    """Return (primes, open composites) of n, following splits."""
    if arith.is_prime(n):
        return [n], []
    r = st['numbers'].get(str(n))
    if r is None or r['status'] != 'split':
        return [], [n]
    P, C = [], []
    for m in r['parts']:
        a, b = resolve(st, int(m))
        P += a; C += b
    return P, C


def split_by(st, n, d, how):
    """Split the open composite n by d; return the parts, or None if d does not split n."""
    g = math.gcd(d, n)
    if g in (1, n):
        return None
    r = st['numbers'][str(n)]
    parts = sorted([g, n // g])
    for m in parts:
        register(st, m, r)
    r.update(status='split', parts=[str(m) for m in parts])
    r['found'].append({'factor': str(g), 'by': how, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
    return parts


def level_done(r):
    """Return the index of the first level not finished (len(LEVELS) if all are)."""
    return next((L for L in range(len(LEVELS)) if r['done'].get(str(L), 0) < LEVELS[L][2]), len(LEVELS))


def duration(s):
    """Return s seconds as '40 s', '12 min' or '1.5 h'."""
    return '%.0f s' % s if s < 90 else '%.0f min' % (s / 60) if s < 3600 else '%.1f h' % (s / 3600)


def effort(r):
    """Return the effort recorded in r as a phrase."""
    L = level_done(r)
    s = [] if not (r['done'] or r['pm1']) else ['P-1 done' if r['pm1'] else 'P-1 not yet']
    if r['done']:
        if L:
            s.append('level %d finished' % LEVELS[L - 1][0])
        if L < len(LEVELS) and r['done'].get(str(L)):
            s.append('level %d %s/%s curves' % (LEVELS[L][0], format(r['done'][str(L)], ','), format(LEVELS[L][2], ',')))
    if r['pari_s']:
        s.append("built-in rho and ECM %s" % duration(r['pari_s']))
    likely = ': no factor below ~%d digits is likely' % LEVELS[L - 1][0] if L else ''
    return 'C%d (%s%s)' % (r['digits'], '; '.join(s) or 'no effort yet', likely)



class Task:
    """One engine call on the composite n: kind 'pari', 'pm1' or a level index, with k curves."""

    def __init__(self, n, kind, k=1, limit=None):
        self.n, self.kind, self.k, self.limit = n, kind, k, limit       # limit: seconds
        self.proc = self.out = None
        self.t0 = 0.0

    def how(self):
        return {'pari': 'built-in rho and ECM', 'pm1': 'GMP-ECM P-1'}.get(self.kind) or 'GMP-ECM, level %d' % LEVELS[self.kind][0]

    def start(self, ecm, maxmem):
        if self.kind == 'pari':                 # arith.factor
            cmd, stdin = [sys.executable, '-m', 'rankbound.arith', str(self.n)], ''
        else:
            cmd = [ecm, '-q', '-one', '-maxmem', str(maxmem)]
            cmd += ['-pm1', '%.0f' % PM1_B1] if self.kind == 'pm1' else ['-c', str(self.k), '%.0f' % LEVELS[self.kind][1]]
            stdin = '%d\n' % self.n
        self.out = tempfile.TemporaryFile('w+')
        self.t0 = time.monotonic()
        try:
            self.proc = runtime.start(cmd, stdin=subprocess.PIPE, stdout=self.out, stderr=subprocess.STDOUT, text=True)
        except OSError as e:
            self.out.close()
            from .messages import text
            raise RankboundError(text('no_start', program=cmd[0], reason=str(e)))
        try:
            self.proc.stdin.write(stdin)
            self.proc.stdin.close()
        except BrokenPipeError:
            pass

    def kill(self):
        runtime.terminate(self.proc, grace=0)
        self.out.close()

    def result(self):
        """Return (divisors found, outcome).

        outcome is 'none' (no factor: the only outcome credited), 'found', 'whole' (GMP-ECM found the number itself,
        after an unknown number of curves) or an error message."""
        self.out.seek(0)
        text = self.out.read()
        self.out.close()
        runtime.reap(self.proc)
        rc = self.proc.returncode
        if self.kind == 'pari':
            err = [l for l in text.splitlines() if l.lstrip().startswith('***') and 'Warning:' not in l]
            if rc or err:
                return [], 'rankbound.arith exit %d: %s' % (rc, ' '.join(err)[:300] or text[-300:].strip())
            text = '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('***'))
        elif rc == ECM_WHOLE:
            return [], 'whole'
        elif rc != 0 and rc not in ECM_FOUND:
            return [], 'GMP-ECM exit %d: %s' % (rc, text[-300:].strip())
        ds = [int(x) for x in re.findall(r'[0-9]+', text)]
        ds = [d for d in ds if 1 < d < self.n and self.n % d == 0]
        if not ds and (self.kind == 'pari' or rc in ECM_FOUND):
            return [], 'a factor was reported but none read: %s' % text[-300:].strip()
        return ds, 'found' if ds else 'none'


def spc(r, L):
    """Return the seconds per curve at level L."""
    return r['spc'].get(str(L)) or SPC_GUESS[L] * (r['digits'] / 140) ** 2


def next_task(st, open_, running, ecm, left, cores=1):
    """Return the next Task for the composite furthest behind, or None if nothing fits in the seconds left."""
    cands = []
    for n in open_:
        r = st['numbers'][str(n)]
        mine = [t for t in running if t.n == n]
        if ecm is None:
            if not mine:
                cands.append((0, 0.0, r['digits'], Task(n, 'pari')))
            continue
        if r['digits'] <= PARI_DIGITS and not r['pari_s']:
            if not mine:
                cands.append((-1, 0.0, r['digits'], Task(n, 'pari', limit=PARI_S)))
            continue
        if (not r['pm1'] and not any(t.kind == 'pm1' for t in mine) and level_done(r) >= PM1_AFTER
                and left >= max(PM1_MIN_S, 1.5 * r.get('pm1_cut_s', 0))):     # not again into as short a time
            cands.append((PM1_AFTER - 0.5, 0.0, r['digits'], Task(n, 'pm1')))
        for L in range(len(LEVELS)):
            need = LEVELS[L][2]
            have = r['done'].get(str(L), 0) + sum(t.k for t in mine if t.kind == L)
            if have < need:
                k = min(need - have, -(-need // cores), max(1, int(CALL_S / spc(r, L))), int(left / spc(r, L)))
                if k >= 1:
                    cands.append((L, have / need, r['digits'], Task(n, L, k)))
                break
    return min(cands, key=lambda c: c[:3])[3] if cands else None



def charge(st, t):
    """Add the seconds of the call t to the effort; return them."""
    secs = time.monotonic() - t.t0
    r = st['numbers'][str(t.n)]
    r['core_s'] += secs; st['core_s'] += secs
    if t.kind == 'pari':
        r['pari_s'] += secs
    return secs


def stop_task(st, t):
    """Kill the call t: its time counts, its curves do not."""
    secs = charge(st, t)
    if t.kind == 'pm1':
        r = st['numbers'][str(t.n)]
        r['pm1_cut_s'] = max(r.get('pm1_cut_s', 0), secs)
    t.kill()


FACTOR_NOTE = contextvars.ContextVar('rankbound_factor_note', default=False)    # --estimate's note


def split(path, composites, hours=1.0, cores=1, mem_gb=1.0, budget=None, ecm=False, log=print):
    """Return the primes of the composites, factoring for at most hours on cores processes, the effort kept in path.

    ecm is the GMP-ECM binary (None: arith.factor; False: find_ecm()).  Raise FactorisationNeeded at the time limit,
    Stopped if budget ends first, and Locked if another process is factoring here."""
    if ecm is False:
        ecm = find_ecm()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fd = os.open(path + '.lock', os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise Locked('Another rankbound process is factoring in %s. Wait for it, or stop it.' % os.path.dirname(path))
    try:
        return _split(path, [int(c) for c in composites], hours, cores, mem_gb, budget or runtime.current(), ecm, log)
    finally:
        os.close(fd)


def _split(path, composites, hours, cores, mem_gb, budget, ecm, log):
    st = load(path)
    for c in composites:
        register(st, c)

    def leaves():
        P, C = [], []
        for c in composites:
            a, b = resolve(st, c)
            P += a; C += b
        return sorted(set(P)), sorted(set(C))

    P, C = leaves()
    if not C:
        return P
    t0 = time.time()
    left = budget.left()
    stop = left is not None and left < hours * 3600
    own = budget.child(seconds=hours * 3600)
    from .messages import duration as words, plural, text
    from .runtime import say
    note = text('factor_est') if FACTOR_NOTE.get() else ''
    for c in C:
        if ecm:
            say('factor_ecm', digits=len(str(c)), cores=plural(cores, 'core'), time=words(hours * 3600), note=note)
        else:
            say('factor_own', digits=len(str(c)), time=words(hours * 3600), note=note)
    running, fails, maxmem = [], {}, max(100, int(800 * mem_gb))
    try:
        while C and not own.expired():
            while len(running) < cores:
                t = next_task(st, C, running, ecm, own.left(), cores)
                if t is None:
                    break
                t.start(ecm, maxmem)
                running.append(t)
            if not running:
                break
            time.sleep(POLL_S)
            for t in [t for t in running if t.limit and time.monotonic() - t.t0 > t.limit and t.proc.poll() is None]:
                stop_task(st, t); running.remove(t)
                say('factor_next', digits=len(str(t.n)), time=words(t.limit))
                write_json(path, st)
            for t in [t for t in running if t.proc.poll() is not None]:
                if t not in running:
                    continue
                running.remove(t)
                ds, outcome = t.result()
                secs = charge(st, t)
                r = st['numbers'][str(t.n)]
                if outcome == 'none':
                    if t.kind == 'pm1':
                        r['pm1'] = True
                    elif t.kind != 'pari':
                        L = str(t.kind)
                        r['done'][L] = r['done'].get(L, 0) + t.k
                        old = r['spc'].get(L)
                        r['spc'][L] = secs / t.k if old is None else 0.7 * old + 0.3 * secs / t.k
                elif outcome == 'whole':
                    log('%s found the whole of %s (no split; nothing credited).' % (t.how(), label(t.n)))
                elif outcome != 'found':
                    key = (t.n, t.kind)
                    fails[key] = fails.get(key, 0) + 1
                    log('%s failed on %s (%d time%s; nothing credited): %s' % (
                        t.how(), label(t.n), fails[key], 's' if fails[key] > 1 else '', outcome))
                    if fails[key] >= MAX_FAILS:
                        raise RankboundError('%s failed %d times on %s: %s' % (t.how(), fails[key], label(t.n), outcome))
                    write_json(path, st)
                    continue
                for d in ds:
                    for m in resolve(st, t.n)[1]:
                        parts = split_by(st, m, d, t.how())
                        if parts:
                            log('%s = %s (%s; %s of work on it)' % (label(m), ' * '.join(label(x) for x in parts),
                                                                           t.how(), duration(r['core_s'])))
                            say('factor_found', digits=len(str(m)), parts=' * '.join(
                                '%d-digit' % len(str(x)) for x in parts) + (' prime' if all(
                                    arith.is_prime(x) for x in parts) else ''), time=words(time.time() - t0))
                if r['status'] == 'split':
                    for u in [u for u in running if u.n == t.n]:
                        stop_task(st, u); running.remove(u)
                P, C = leaves()
                write_json(path, st)
    finally:
        for t in running:
            stop_task(st, t)
        st['wall_s'] += time.time() - t0
        write_json(path, st)
    P, C = leaves()
    if not C:
        return P
    wall, core = time.time() - t0, st['core_s']
    beyond = bool(ecm) and all(level_done(st['numbers'][str(c)]) == len(LEVELS) for c in C)
    if stop and not beyond:             # the run's limit came first
        raise Stopped(os.path.dirname(path))
    left = '; '.join(effort(st['numbers'][str(c)]) for c in C)
    log('%s stopped after %s (%.1f core-h in all): %d composite%s left, %s.' % (
        'GMP-ECM' if ecm else 'Factoring', duration(wall), core / 3600, len(C), 's' if len(C) > 1 else '', left))
    raise FactorisationNeeded(
        text('factor_skip' if ecm else 'factor_skip_ecm', name=os.path.basename(os.path.dirname(os.path.abspath(path))),
             time=words(wall)),
        composites=[dict(n=str(c), **{k: st['numbers'][str(c)][k] for k in ('digits', 'pm1', 'done', 'core_s')})
                    for c in C])


# ---------- the factor step of a work folder

def check_composites(curve):
    """Raise ValueError unless each composite of curve is above 1 and divides the discriminant."""
    D = abs(disc(_ainvs5(curve.ainvs)))
    for c in curve.composites:
        if int(c) < 2:
            raise ValueError('The composite %s is not an integer above 1.' % c)
        if D % int(c):
            raise ValueError('The composite %s does not divide the discriminant.' % c)


def factor_in(workdir, curve, hours, cores=1, mem_gb=1.0, budget=None):
    """Return curve with the primes of its composites added to its bad primes (wall time in cost.json)."""
    os.makedirs(workdir, exist_ok=True)
    t = time.time()
    try:
        P = split(os.path.join(workdir, 'factor.json'), curve.composites, hours=hours, cores=cores, mem_gb=mem_gb,
                  budget=budget, log=DETAIL.info)
    finally:
        cost = os.path.join(workdir, 'cost.json')
        c = read_json(cost, {})
        c.update(factor=round(read_json(os.path.join(workdir, 'factor.json'), {}).get('wall_s', time.time() - t), 1))
        write_json(cost, c)
    bad = sorted({int(p) for p in curve.bad_primes} | set(P))
    return dataclasses.replace(curve, bad_primes=[str(p) for p in bad], composites=[])


# ---------- curves already in the table

WITNESSES = os.path.join(os.path.dirname(HERE), 'witnesses.tsv')
COL_LB, COL_UB, COL_EQ, COL_MM = 'rank lower bound', 'rank upper bound', 'rank = r?', 'minimal model'
COL_TOR, COL_GRH = 'torsion', 'GRH'
ROW_ORDER = {'yes': 0, 'no': 1, 'r cited': 2, 'no (r cited)': 3}      # preference among one curve's rows


INDEX_VERSION = 1           # of the winner rule and row shape
_INDEX = {}                 # realpath -> (INDEX_VERSION, the file's identity, {minimal model: winning row})
_INDEX_MAX = 4


def _identity(path):
    st = os.stat(path)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _index(path):
    """Return the winning row per minimal model of the table at path (None without that column), cached."""
    key = os.path.realpath(path)
    ident = _identity(path)
    got = _INDEX.get(key)
    if got and got[0] == INDEX_VERSION and got[1] == ident:
        return got[2]
    with open(path) as f:
        head = f.readline().rstrip('\n').split('\t')
        rows = [dict(zip(head, l.rstrip('\n').split('\t'))) for l in f if l.strip()] if COL_MM in head else None
    idx = None
    if rows is not None:
        idx = {}
        for r in rows:
            if r.get('file') and r.get(COL_EQ) in ROW_ORDER:
                k, o = r.get(COL_MM), idx.get(r.get(COL_MM))
                if o is None or (int(r[COL_UB]), ROW_ORDER[r[COL_EQ]], r['name']) < \
                        (int(o[COL_UB]), ROW_ORDER[o[COL_EQ]], o['name']):
                    idx[k] = r
    if key not in _INDEX and len(_INDEX) >= _INDEX_MAX:
        _INDEX.pop(next(iter(_INDEX)))
    _INDEX[key] = (INDEX_VERSION, ident, idx)
    return idx


def table_row(minimal_model, index=None):
    """Return a copy of the row of witnesses.tsv for the minimal model, or None.

    Of several rows, the least S wins, then r proved over r cited, then the name."""
    path = index or WITNESSES
    if not os.path.isfile(path):
        return None
    idx = _index(path)
    hit = idx.get(','.join(str(x) for x in minimal_model)) if idx else None
    return dict(hit) if hit else None


def table_statement(row):
    """Return what the row's witness proves, in words."""
    S, r, eq = row[COL_UB], row[COL_LB], row[COL_EQ]
    h = 'unconditionally' if row.get(COL_GRH) == 'no' else 'under GRH'
    if S == '0' or eq == 'yes':
        return 'rank E(Q) = %s %s' % (S, h)
    if eq == 'no':
        return '%s <= rank E(Q) <= %s %s' % (r, S, h)
    return 'rank E(Q) <= %s %s%s' % (S, h, ' (r = %s is cited, not proved in the witness)' % r if r else '')


def check_cyclic(dK, r):
    """Raise RankboundError if points are to prove r >= 2 though the cubic field is cyclic (no 2-signature exists;
    r = 1 is proved by a point of infinite order)."""
    if r and r >= 2 and dK > 0 and math.isqrt(dK) ** 2 == dK:
        from .messages import text
        raise RankboundError(text('cyclic'))


def table_r(row, r):
    """Raise ValueError if r exceeds the row's upper bound."""
    if r is not None and r > int(row[COL_UB]):
        raise ValueError('r = %d is impossible for this curve: its witness %s proves rank E(Q) <= %s %s. Check r.'
                         % (r, row['file'], row[COL_UB], 'unconditionally' if row.get(COL_GRH) == 'no' else 'under GRH'))


def table_method(row):
    """Return the row's method: its 'method' column, else from the torsion."""
    if row.get('method'):
        return row['method']
    tor = [int(x) for x in (row.get(COL_TOR) or '').strip('[]').split(',') if x.strip()]
    return '2-isogeny descent' if any(t % 2 == 0 for t in tor) else 'class group'


# ---------- the rank lower bound from points

PRIME_SEARCH = 10 ** 6      # primes searched beyond the given ones


def rank_proof(ainvs, bad_primes, r, proof):
    """Return the rank lower bound {'r', 'points', 'primes'} on the model ainvs; raise RuntimeError if not proved.

    The primes are r primes (the proof's own first) at which the 2-signatures have rank r."""
    check_proof(proof)
    r = r_value(r)
    if len(proof['points']) != r:
        raise RuntimeError('r = %d but %d points were given.' % (r, len(proof['points'])))
    a = _ainvs5(ainvs)
    D = cinv(a)[2]
    given = [[fractions.Fraction(canon(x, 'point coordinate', RAT)) for x in p] for p in proof['points']]
    if r == 1 and D > 0 and math.isqrt(D) ** 2 == D:     # a cyclic cubic field: no 2-signatures
        return infinite_order(a, rkmap(_ainvs5(proof.get('ainvs') or ainvs), a, [int(p) for p in bad_primes], given)[0])
    pts, prs = rkproof(_ainvs5(proof.get('ainvs') or ainvs), a, [int(p) for p in bad_primes], r, given,
                       [int(p) for p in proof.get('primes') or []], PRIME_SEARCH)
    if r == 1 and len(prs) != 1:        # a saturated point can lie in 2E(F_p) at every usable p (ICARM #1791)
        return infinite_order(a, pts[0])
    if len(pts) != r or len(prs) != r:
        raise RuntimeError('The points do not prove rank >= %d. At every usable prime below 10^%d they look '
                           'dependent; check them.' % (r, round(math.log10(PRIME_SEARCH))))
    return {'r': r, 'points': [[point_value(x) for x in P] for P in pts], 'primes': prs}


def infinite_order(a, P):
    """Return the rank lower bound 1 from the point P of the model a without primes, if nP != O for n <= 12 (Mazur's
    bound on the order of a torsion point)."""
    if any(mul(a, P, n) is None for n in range(1, 13)):
        raise RuntimeError('The point does not prove rank >= 1: it has finite order.')
    return {'r': 1, 'points': [[point_value(x) for x in P]], 'primes': []}


def rkmap(a0, a, P, pts):
    """Return the points pts of the model a0 on the model a; raise Rejected unless both models are of one curve and the
    points are on it."""
    m0, v0 = minimal(a0, P)
    m, v = minimal(a, P)
    if m0 != m:
        raise Rejected('The points are given on [%s], which is not a model of this curve.' % ','.join(map(str, a0)))
    out = []
    for i, Q in enumerate(pts, 1):
        if not on_curve(a0, Q):
            raise Rejected('Point %d is not on the curve.' % i)
        Q = from_model(to_model(Q, v0), v)
        if not on_curve(a, Q):
            raise Rejected('rkmap: point %d is not on the witness\'s model' % i)
        out.append(Q)
    return out


def _sig_cubic(a):
    """Return the 2-division cubic T in X = 4x."""
    b2, b4, b6, _ = binv(a)
    return [16 * b6, 8 * b4, b2, 1]


def _sig_root(T, p):
    """Return the single root e of T mod p if T'(e) is not a square (#E(F_p) = 2 mod 4), else None."""
    rt = arith.roots_mod(T, p)
    if len(rt) != 1 or arith.kronecker(arith.peval(arith.pderiv(T), rt[0], p), p) != -1:
        return None
    return rt[0]


def _sig_value(P, p, e):
    """Return the 2-signature of P at p: 0 for O, 1 for (e, 0), else whether X(P) - e is not a square mod p."""
    x, y = P
    if x.denominator % p == 0 or y.denominator % p == 0:
        return 0
    X = 4 * x.numerator * pow(x.denominator, -1, p) % p
    return 1 if X == e else int(arith.kronecker(X - e, p) == -1)


def _fallback_primes(pmax, skip):
    """Yield the primes 3 <= q < pmax not in skip, sieved lazily."""
    lo, w = 3, 1024
    while lo < pmax:
        hi = min(lo + w, pmax)
        for q in arith.primes(lo, hi):
            if q not in skip:
                yield q
        lo, w = hi, min(2 * w, 1 << 16)


SAT_PRIMES = 64             # usable primes past a rank deficit before a combination is taken to lie in 2E(Q)


def rkproof(a0, a, P, r, pts, PR, pmax):
    """Return (the points on the model a, the first r primes, from PR then below pmax, giving rank r).

    Independent points can be dependent in E(Q)/2E(Q): a combination that stays in the kernel of the 2-signatures
    is halved over Q and replaces one of its points (2-saturation; the new points span a lattice containing the old)."""
    Q = rkmap(a0, a, P, pts)
    if r == 0:
        return Q, []
    for _ in range(4 * r + 4):                  # each halving at least doubles the covolume's 2-part: few rounds
        keep, c = _signatures(a, Q, r, PR, pmax, SAT_PRIMES)
        if c is None:
            return Q, keep
        H = _halve(a, _combination(a, Q, c))
        if H is None:                           # not a double: only the full search can tell
            return Q, _signatures(a, Q, r, PR, pmax, None)[0]
        i = (c & -c).bit_length() - 1
        Q = Q[:i] + [H] + Q[i + 1:]
    return Q, _signatures(a, Q, r, PR, pmax, None)[0]


def _signatures(a, Q, r, PR, pmax, slack):
    """Return (the primes, from PR then below pmax, whose 2-signatures raise the rank, at most r of them; a kernel
    combination as a bitmask, or None): the combination once the rank is short of r after slack more usable primes."""
    D = cinv(a)[2]
    T = _sig_cubic(a)
    rows, keep, seen, extra = [], [], set(), 0
    for p in itertools.chain(PR, _fallback_primes(pmax, set(PR))):
        if p in seen:
            continue
        seen.add(p)
        if p < 3 or not arith.is_prime(p) or D % p == 0:
            continue
        e = _sig_root(T, p)
        if e is None:
            continue
        v = sum(_sig_value(z, p, e) << i for i, z in enumerate(Q))
        if arith.insert(rows, v):
            keep.append(p)
            extra = 0
        if len(keep) == r:
            break
        extra += 1
        if slack is not None and extra >= slack:     # short of r after slack primes in a row: a kernel combination
            return keep, _kernel(rows, r)
    return keep, None


def _kernel(rows, r):
    """Return a nonzero c in F_2^r (a bitmask) with <c, v> = 0 for every row v, rows of rank < r."""
    piv = {}                                    # reduced echelon form: pivot bit -> row
    for v in rows:
        for b, w in piv.items():
            if v >> b & 1:
                v ^= w
        if v:
            b = v.bit_length() - 1
            for k in piv:
                if piv[k] >> b & 1:
                    piv[k] ^= v
            piv[b] = v
    f = next(i for i in range(r) if i not in piv)
    return (1 << f) | sum(1 << b for b, w in piv.items() if w >> f & 1)


def _combination(a, Q, c):
    R = None
    for i, z in enumerate(Q):
        if c >> i & 1:
            R = add(a, R, z)
    return R


def _halve(a, R):
    """Return a rational point H with 2H = +-R on the integral model a, or None if R is not in 2E(Q)."""
    if R is None:
        return None
    a1, a2, a3, a4, a6 = a
    b2, b4, b6, b8 = binv(a)
    n, d = R[0].numerator, R[0].denominator
    # x(2H) = (x^4 - b4 x^2 - 2 b6 x - b8) / (4 x^3 + b2 x^2 + 2 b4 x + b6), times d; squarefree as R is not 2-torsion
    f = [-(d * b8 + n * b6), -2 * (d * b6 + n * b4), -(d * b4 + n * b2), -4 * n, d]
    M = d.bit_length() + 8
    for z in arith.real_roots_floor(f, M):     # the denominator of x(H) divides d's: d x(H) is an integer
        for k in range((z * d >> M) - 1, (z * d >> M) + 3):
            x = Fraction(k, d)
            if f[4] * x ** 4 + f[3] * x ** 3 + f[2] * x ** 2 + f[1] * x + f[0] != 0:
                continue
            u = a1 * x + a3
            disc = u * u + 4 * (x ** 3 + a2 * x * x + a4 * x + a6)
            sn, sd = math.isqrt(disc.numerator), math.isqrt(disc.denominator)
            if disc < 0 or sn * sn != disc.numerator or sd * sd != disc.denominator:
                continue
            H = (x, (-u + Fraction(sn, sd)) / 2)
            D = add(a, H, H)
            if D is not None and D[0] == R[0]:
                return H
    return None


# ========== the arithmetic of the curve
# Tate's algorithm and the root number are ported from PARI/GP 2.17.4, src/basemath/elliptic.c (GPL-2.0-or-later, as
# rankbound), so they are no independent check of the checker's PARI.  A curve is [a1, a2, a3, a4, a6] (ints); a point
# is (x, y) of Fractions, or None for the point at infinity.

kross = arith.kronecker


def binv(a):
    a1, a2, a3, a4, a6 = a
    b2, b4, b6 = a1 * a1 + 4 * a2, 2 * a4 + a1 * a3, a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    return b2, b4, b6, b8


def cinv(a):
    """Return (c4, c6, disc)."""
    b2, b4, b6, b8 = binv(a)
    return (b2 * b2 - 24 * b4, -b2 ** 3 + 36 * b2 * b4 - 216 * b6,
            -b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 * b6 + 9 * b2 * b4 * b6)


def rst(a, r, s, t):
    """Return the model after x = x' + r, y = y' + s x' + t."""
    a1, a2, a3, a4, a6 = a
    return [a1 + 2 * s, a2 - s * a1 + 3 * r - s * s, a3 + r * a1 + 2 * t,
            a4 - s * a3 + 2 * r * a2 - (t + r * s) * a1 + 3 * r * r - 2 * s * t,
            a6 + r * a4 + r * r * a2 + r ** 3 - t * a3 - t * t - r * t * a1]


def _smooth_part(x, m):
    """Return the largest divisor of x whose primes all divide m."""
    x, part = abs(x), 1
    d = math.gcd(x, m)
    while d > 1:
        part *= d; x //= d
        d = math.gcd(x, d)
    return part


def _coprime_base(xs):
    """Return a coprime base of the integers > 1 in xs."""
    base = sorted({x for x in xs if x > 1})
    while True:
        for i in range(len(base)):
            for j in range(i):
                d = math.gcd(base[i], base[j])
                if d > 1:
                    rest = [b for k, b in enumerate(base) if k not in (i, j)]
                    base = sorted(set(rest) | {x for x in (d, base[i] // d, base[j] // d) if x > 1})
                    break
            else:
                continue
            break
        else:
            return base


def _large_scaling(m, g, c6, D, L):
    """Return the part at the primes of m (all >= L > 3) of the scaling u, u^12 | g.

    Over a coprime base b (smallest roots) with g's m-part = prod b^e, b^(e // 12) | u, and that is all of b's part
    unless some v_p(b) >= ceil(12 / (e % 12)), impossible for b < L^ceil(12 / (e % 12)); else b is factored."""
    parts = [_smooth_part(x, m) for x in (c6, D, g) if x]
    u = 1
    for b in _coprime_base([m] + parts):
        k = 2
        while L ** k <= b:
            r = arith.iroot(b, k) if arith.is_prime(k) else None
            b, k = (r, 2) if r else (b, k + 1)
        e = arith.val(_smooth_part(g, b), b)
        if e % 12 == 0 or b < L ** -(-12 // (e % 12)) or arith.is_prime(b):
            u *= b ** (e // 12)
            continue
        try:
            fa = arith.factor(b, seconds=60)
        except (ValueError, TimeoutError):
            raise ValueError('The minimal model needs the prime factors of %d: give the bad primes.' % b)
        for p, v in fa:
            u *= p ** (e * v // 12)
    return u


# ---------- one request's memo of minimal models and torsion, keyed by the exact model; callers get copies

_ANALYSIS = contextvars.ContextVar('rankbound_curve_analysis', default=None)
ANALYSIS_MAX = 256


@contextlib.contextmanager
def analysis():
    """Open the memo scope of one request (nested scopes share the outer one)."""
    if _ANALYSIS.get() is not None:
        yield _ANALYSIS.get()
        return
    tok = _ANALYSIS.set({})
    try:
        yield _ANALYSIS.get()
    finally:
        _ANALYSIS.reset(tok)


def _analysed(kind, key, compute):
    memo = _ANALYSIS.get()
    if memo is None:
        return compute()
    k = (kind,) + key
    if k not in memo:
        v = compute()
        if len(memo) >= ANALYSIS_MAX:
            memo.pop(next(iter(memo)))
        memo[k] = v
    return copy.deepcopy(memo[k])


def minimal(a, primes=()):
    """Return (the reduced minimal model, [u, r, s, t] from a to it), memoised within analysis()."""
    a = [int(x) for x in a]
    primes = sorted({int(p) for p in primes})
    return _analysed('minimal', (tuple(a), tuple(primes)), lambda: _minimal(a, primes))


def _minimal(a, primes=()):
    """Return (the reduced minimal model, [u, r, s, t] from a to it) (Kraus, Laska, Connell).

    The p with p^12 | gcd(c6^2, disc) come from primes, trial division to 10^4, and _large_scaling for the rest."""
    c4, c6, D = cinv(a)
    g = math.gcd(c6 * c6, D)
    ps = {p for p in primes if g % p == 0}
    h = g
    for p in ps:
        h //= p ** arith.val(h, p)
    L = 10 ** 4
    fa = arith.factor(h, L)
    m = fa.pop()[0] if fa and fa[-1][0] >= L else 1
    ps |= {p for p, _ in fa}
    ps = {q for p in ps for q, _ in (arith.factor(p) if p >= 10 ** 8 else [(p, 1)])}
    u = _large_scaling(m, g, c6, D, L) if m > 1 else 1
    for p in sorted(ps):
        d = arith.val(g, p) // 12
        if p == 2 and d:
            A, B = c4 // 2 ** (4 * d) % 16, c6 // 2 ** (6 * d) % 32
            if B % 4 != 3 and not (A == 0 and B in (0, 8)):
                d -= 1
        elif p == 3 and d and c6 != 0 and arith.val(c6, 3) == 6 * d + 2:
            d -= 1
        u *= p ** d
    c4, c6 = c4 // u ** 4, c6 // u ** 6
    b2 = -c6 % 12
    b2 -= 12 * (b2 > 5)
    b4 = (b2 * b2 - c4) // 24
    b6 = (-b2 ** 3 + 36 * b2 * b4 - c6) // 216
    a1, a3 = b2 % 2, b6 % 2
    mm = [a1, (b2 - a1) // 4, a3, (b4 - a1 * a3) // 2, (b6 - a3) // 4]
    s = (u * mm[0] - a[0]) // 2
    r = (u * u * mm[1] - a[1] + s * a[0] + s * s) // 3
    t = (u ** 3 * mm[2] - a[2] - r * a[0]) // 2
    return mm, [u, r, s, t]


def to_model(P, v):
    """Return P on the model reached by [u, r, s, t], as ellchangepoint."""
    if P is None:
        return None
    u, r, s, t = v
    x, y = Fraction(P[0]), Fraction(P[1])
    return ((x - r) / u ** 2, (y - s * (x - r) - t) / u ** 3)


def from_model(P, v):
    """Return the inverse of to_model, as ellchangepointinv."""
    if P is None:
        return None
    u, r, s, t = v
    x, y = Fraction(P[0]), Fraction(P[1])
    return (u * u * x + r, u ** 3 * y + s * u * u * x + t)


def on_curve(a, P):
    if P is None:
        return True
    x, y = Fraction(P[0]), Fraction(P[1])
    a1, a2, a3, a4, a6 = a
    return y * y + a1 * x * y + a3 * y == x ** 3 + a2 * x * x + a4 * x + a6


# ---------- Tate's algorithm (PARI's localred_p, localred_23)

def localred(a, p):
    """Return (f_p, PARI's Kodaira code, c_p) of the model a, minimal at p."""
    c4, c6, D = cinv(a)
    nuD = arith.val(D, p)
    if nuD == 0:
        return 0, 1, 1
    if p > 3:
        nuj = max(0, nuD - 3 * arith.val(c4, p)) if c4 else 0       # -v_p(j)
        if nuj > 0:
            if nuD - nuj == 0:
                k = kross(-c6, p)
                return 1, 4 + nuj, nuD if k == 1 else (1 if nuD % 2 else 2)
            d = D // p ** (6 + nuj) % p
            if nuj % 2:
                d = d * (c6 // p ** 3) % p
            return 2, -4 - nuj, 3 + kross(d, p)
        if nuD == 2:
            return 2, 2, 1
        if nuD == 3:
            return 2, 3, 2
        if nuD == 4:
            return 2, 4, 2 + kross(-6, p) * kross(c6 // p ** 2, p)
        if nuD == 6:
            return 2, -1, 1 + len(arith.roots_mod([-2 * (c6 // p ** 3), -3 * (c4 // p ** 2), 0, 1], p))
        if nuD == 8:
            return 2, -4, 2 + kross(-6, p) * kross(c6 // p ** 4, p)
        if nuD == 9:
            return 2, -3, 2
        if nuD == 10:
            return 2, -2, 1
        raise ValueError('localred: the model is not minimal at %d' % p)
    p2, p3, p4, p5 = p ** 2, p ** 3, p ** 4, p ** 5
    b2, b4, b6, b8 = binv(a)
    if b2 % p:
        c = nuD if -c6 % (8 if p == 2 else 3) == 1 else (1 if nuD % 2 else 2)
        return 1, 4 + nuD, c
    if p == 2:
        r, s, t = a[3] % 2, a[1] % 2, a[4] % 2
        if r:
            t, s = (s + t) & 1, (s + 1) & 1
    else:
        r, s, t = -(b6 % 3), a[0] % 3, a[2] % 3
        if s:
            t = (t + r * s) % 3
    e = rst(a, r, s, t)
    if e[4] % p2:
        return nuD, 2, 1
    b2, b4, b6, b8 = binv(e)
    if b8 % p3:
        return nuD - 1, 3, 2
    if b6 % p3:
        return nuD - 2, 4, 3 if b6 % (32 if p == 2 else 27) == p2 else 1
    if e[4] % p3:
        e = rst(e, 0, 0, 2 if p == 2 else e[2] % 9)
    a21, a42, a63 = e[1] % p2 // p, e[3] % p3 // p2, e[4] % p4 // p3
    n, root = _numroots3(a21, a42, a63, p)
    if n == 3:
        c = 1 if a63 else 2
        if p == 2:
            c += (a21 + a42 + a63) & 1
        else:
            c += ((1 + a21 + a42 + a63) % 3 == 0) + ((1 - a21 + a42 - a63) % 3 == 0)
        return nuD - 4, -1, c
    if n == 2:
        if root:
            e = rst(e, root * p, 0, 0)
        nu, pk, p2k = 1, p2, p4
        while True:
            al, be, ga = 1, e[2] // pk % p, -(e[4] // p2k % p)
            k, root = _numroots2(al, be, ga, p)
            if k == 2:
                break
            if root:
                e = rst(e, 0, 0, root * pk)
            pk1, pk, p2k, nu = pk, p * pk, p * p2k, nu + 1
            al, be, ga = a21, e[3] // pk % p, e[4] // p2k % p
            k, root = _numroots2(al, be, ga, p)
            if k == 2:
                break
            if root:
                e = rst(e, root * pk1, 0, 0)
            p2k, nu = p * p2k, nu + 1
        c = (2 if ga % 2 else 4) if p == 2 else 3 + kross(be * be - al * ga, 3)
        return nuD - 4 - nu, -4 - nu, c
    if root:
        e = rst(e, root * p, 0, 0)
    a32, a64 = e[2] % p3 // p2, e[4] % p5 // p4
    k, root = _numroots2(1, a32, -a64, p)
    if k == 2:
        return nuD - 6, -4, 3 - 2 * a64 if p == 2 else 2 + kross(a32 * a32 + a64, 3)
    if root:
        e = rst(e, 0, 0, root * p2)
    if e[3] % p4:
        return nuD - 7, -3, 2
    return nuD - 8, -2, 1


def _numroots3(a, b, c, p):
    if p == 2:
        if (c + a * b) % 2:
            return 3, 0
        return (2 if (a + b) % 2 else 1), b
    if not a:
        return (3 if b else 1), -c
    if b == 2:
        return (2 if a + c == 3 else 3), a * b
    return (3 if c else 2), a * b


def _numroots2(a, b, c, p):
    if p == 2:
        return (2 if b % 2 else 1), c
    return (2 if (b * b - a * c) % 3 else 1), a * b


def conductor(mm, P):
    N = 1
    for p in P:
        N *= p ** localred(mm, p)[0]
    return N


# ---------- the root number (PARI's ellrootno; Halberstadt at 2 and 3)

def rootno(mm, P):
    """Return the root number of the minimal model mm with bad primes P."""
    w = -1
    for p in P:
        w *= _rootno_p(mm, p) if p > 3 else _rootno_23(mm, p)
    return w


def _rootno_p(mm, p):
    c4, c6, D = cinv(mm)
    nuD = arith.val(D, p)
    nuj = max(0, nuD - 3 * arith.val(c4, p)) if c4 else 0
    nu = (nuD - nuj) % 12
    if nu == 0:
        if not nuj:
            return 1
        return -kross(-c6, p)
    if nuj:
        return kross(-1, p)
    ep = 12 // math.gcd(12, nu)
    z = 2 if ep == 4 else (3 if ep % 2 else 1)
    return kross(-z, p)


def _vu(x, p, pk):
    if not x:
        return 12, 0
    v = arith.val(x, p)
    return v, x // p ** v % pk


def _neron_2(v4, v6, vD, kod):
    if kod > 4:
        return 1
    if kod == 1:
        return 2 if v6 > 0 else 1
    if kod == 2:
        return 1 if vD == 4 else 3 if vD == 7 else 2 if v4 == 4 else 4
    if kod == 3:
        return {6: 3, 8: 4, 9: 5}.get(vD, 2 if v4 == 5 else 1)
    if kod == 4:
        return 2 if v4 > 4 else 1
    if kod == -1:
        return {9: 2, 10: 4}.get(vD, 3 if v4 > 4 else 1)
    if kod == -2:
        return {12: 2, 14: 3}.get(vD, 1)
    if kod == -3:
        return {12: 2, 14: 3, 15: 4}.get(vD, 1)
    if kod == -4:
        return 2 if v6 == 7 else 1
    if kod == -5:
        return 2 if v6 == 7 or v4 == 6 else 1
    if kod == -6:
        return {12: 2, 13: 3}.get(vD, 2 if v4 == 6 else 1)
    if kod == -7:
        return 2 if vD == 12 or v4 == 6 else 1
    return 2 if v4 == 6 else 1


def _neron_3(v4, v6, vD, kod):
    if abs(kod) > 4:
        return 1
    if kod in (-1, 1):
        return 2 if v4 % 2 else 1
    if kod in (-3, 3):
        return 2 if 2 * v6 > vD + 3 else 1
    if kod in (-4, 2):
        return {4: 3, 5: 4}.get(vD % 6, 2 if v6 % 3 == 1 else 1)
    return {0: 2, 1: 3}.get(vD % 6, 1)


def _rootno_23(e, p):
    c4, c6, D = cinv(e)
    if not arith.val(D, p):
        return 1
    pk = 64 if p == 2 else 81
    (v4, u), (v6, v), (vD, D1) = _vu(c4, p, pk), _vu(c6, p, pk), _vu(D, p, pk)
    kod = localred(e, p)[1]
    if p == 3:
        return _rootno_3(v4, u, v6, v, vD, D1, kod)
    n2 = _neron_2(v4, v6, vD, kod)
    if kod >= 5:
        return 1 if (e[1] + e[2]) % 2 else -1
    if kod < -9:
        return -kross(-1, v) if n2 == 2 else -1
    x1 = u + v + v
    cases = {
        1: lambda: 1,
        2: lambda: {1: lambda: {4: kross(-1, u), 5: 1}.get(v4, -1), 2: lambda: 1 if v6 == 7 else -1,
                    3: lambda: 1 if v % 8 == 5 or u * v % 8 == 5 else -1,
                    4: lambda: kross(-1, v) if v4 > 5 else (-kross(-1, u) if v4 == 5 else -1)}.get(n2, lambda: None)(),
        3: lambda: {1: lambda: -kross(2, u * v), 2: lambda: -kross(2, v),
                    3: lambda: 1 if (u - (v << (v6 - 5))) & 15 in (7, 11) else -1,
                    4: lambda: 1 if v % 8 == 3 or (2 * u + v) % 8 == 7 else -1,
                    5: lambda: kross(2, x1) if v6 == 8 else kross(-2, u)}.get(n2, lambda: None)(),
        -1: lambda: {1: lambda: -kross(2, x1), 2: lambda: 1 if v % 8 == 7 or x1 % 32 == 11 else -1,
                     3: lambda: 1 if v4 == 6 else -1,
                     4: lambda: kross(-1, v) if v4 > 6 else (-kross(-1, u * v) if v4 == 6 else -1)}.get(n2, lambda: None)(),
        -2: lambda: kross(-2, v) if n2 == 1 else kross(-1, v),
        -3: lambda: {1: lambda: 1 if (u - 2 * v) % 64 in (3, 19) else -1, 2: lambda: kross(2 * kross(-1, u), v),
                     3: lambda: -kross(-1, u) * kross(-2 * kross(-1, u), u * v),
                     4: lambda: kross(-2, x1) if v6 == 11 else -kross(-2, u)}.get(n2, lambda: None)(),
        -5: lambda: (1 if x1 % 32 == 23 else -1) if n2 == 1 else -kross(2, 2 * u + v),
        -6: lambda: {1: lambda: 1, 2: lambda: 1 if v6 == 10 else -1,
                     3: lambda: 1 if u % 16 == 11 or (u + 4 * v) % 16 == 3 else -1}.get(n2, lambda: None)(),
        -7: lambda: 1 if n2 == 1 else (1 if (u + (v << (v6 - 8))) & 15 in ((9, 13) if v6 == 10 else (9, 5)) else -1),
        -8: lambda: kross(-1, v * D1) if n2 == 2 else -1,
        -9: lambda: -kross(-1, D1) if n2 == 2 else -1}
    order = [1, 2, 3, -1, -2, -3, -5, -6, -7, -8, -9]       # C's switch falls through
    if kod not in order:
        return -1
    for k in order[order.index(kod):]:
        r = cases[k]()
        if r is not None:
            return r
    return -1


def _rootno_3(v4, u, v6, v, vD, D1, kod):
    K6 = kross(v, 3)
    if kod > 4:
        return K6
    n2 = _neron_3(v4, v6, vD, kod)
    r6, K4 = v % 9, kross(u, 3)
    cases = {
        1: lambda: 1, 3: lambda: 1, -3: lambda: 1,
        2: lambda: {1: lambda: 1 if r6 == 4 or r6 > 6 else -1, 2: lambda: -K4 * K6, 3: lambda: 1,
                    4: lambda: -K6}.get(n2, lambda: None)(),
        4: lambda: {1: lambda: K6 * kross(D1, 3), 2: lambda: -K4, 3: lambda: -K6}.get(n2, lambda: None)(),
        -2: lambda: 1 if n2 == 2 else K6,
        -4: lambda: {1: lambda: (1 if r6 in (4, 8) else -1) if v4 == 4 else (1 if r6 in (1, 2) else -1),
                     2: lambda: -K6, 3: lambda: 1 if r6 in (2, 7) else -1, 4: lambda: K6}.get(n2, lambda: None)()}
    order = [1, 3, -3, 2, 4, -2, -4]
    if kod not in order:
        return -1
    for k in order[order.index(kod):]:
        r = cases[k]()
        if r is not None:
            return r
    return -1


# ---------- points

def add(a, P, Q):
    """Return P + Q on the model a."""
    if P is None:
        return Q
    if Q is None:
        return P
    a1, a2, a3, a4, a6 = a
    (x1, y1), (x2, y2) = P, Q
    if x1 == x2:
        if y1 + y2 + a1 * x2 + a3 == 0:
            return None
        lam = (3 * x1 * x1 + 2 * a2 * x1 + a4 - a1 * y1) / (2 * y1 + a1 * x1 + a3)
    else:
        lam = (y2 - y1) / (x2 - x1)
    nu = y1 - lam * x1
    x3 = lam * lam + a1 * lam - a2 - x1 - x2
    return (x3, -(lam + a1) * x3 - nu - a3)


def mul(a, P, n):
    R = None
    while n:
        if n & 1:
            R = add(a, R, P)
        P = add(a, P, P)
        n >>= 1
    return R


def _divpoly(A, B, n):
    """Return g_n with psi_n = g_n (n odd) or y g_n (n even) on y^2 = x^3 + A x + B."""
    def m(f, g):
        h = [0] * (len(f) + len(g) - 1) if f and g else []
        for i, x in enumerate(f):
            for j, y in enumerate(g):
                h[i + j] += x * y
        return h
    def s(f, g):
        n_ = max(len(f), len(g))
        return arith.ptrim([(f[i] if i < len(f) else 0) - (g[i] if i < len(g) else 0) for i in range(n_)])
    F2 = m([B, A, 0, 1], [B, A, 0, 1])
    g = {0: [], 1: [1], 2: [2], 3: [-A * A, 12 * B, 6 * A, 0, 3],
         4: [4 * c for c in [-8 * B * B - A ** 3, -4 * A * B, -5 * A * A, 20 * B, 5 * A, 0, 1]]}
    def get(k):
        if k not in g:
            j = k // 2
            if k % 2:
                t1, t2 = m(get(j + 2), m(get(j), m(get(j), get(j)))), m(get(j - 1), m(get(j + 1), m(get(j + 1), get(j + 1))))
                g[k] = s(m(t1, F2), t2) if j % 2 == 0 else s(t1, m(t2, F2))
            else:
                h = m(get(j), s(m(get(j + 2), m(get(j - 1), get(j - 1))), m(get(j - 2), m(get(j + 1), get(j + 1)))))
                g[k] = [c // 2 for c in h]
        return g[k]
    return get(n)


def int_roots(f):
    """Return the integer roots of the squarefree integer polynomial f."""
    f = arith.ptrim(list(f))
    out = [0] if f[0] == 0 else []
    while f[0] == 0:
        f = f[1:]
    if len(f) <= 1:
        return out
    # arith.root_bound_bits: every root has |z| < 2^(e + 1)
    d, lead = len(f) - 1, abs(f[-1]).bit_length()
    e = max([0] + [-((lead - 1 - abs(c).bit_length()) // (d - i)) for i, c in enumerate(f[:-1]) if c])
    bound = 1 << (e + 1)
    p = 5
    while f[-1] % p == 0 or len(arith.pgcd(arith.pmod(f, p), arith.pmod(arith.pderiv(f), p), p)) > 1:
        p = next(q for q in range(p + 1, 10 ** 6) if all(q % d for d in range(2, int(q ** 0.5) + 1)))
    k = 1
    while p ** k <= 2 * bound:
        k += 1
    M = p ** k
    for r in arith.padic_roots(f, p, k):
        r = r - M if r > M // 2 else r
        if arith.peval(f, r) == 0:
            out.append(r)
    return sorted(out)


def count_mod(a, p):
    """Return #E(F_p) for an odd prime p of good reduction."""
    a1, a2, a3, a4, a6 = a
    return 1 + sum(1 + kross((a1 * x + a3) ** 2 + 4 * (x ** 3 + a2 * x * x + a4 * x + a6), p) for x in range(p))


def torsion_group(a):
    """Return (E(Q)_tors as elltors, generators), memoised within analysis()."""
    a = [int(x) for x in a]
    return _analysed('torsion', tuple(a), lambda: _torsion_group(a))


def _torsion_group(a):
    """Return (E(Q)_tors as elltors, generators) of the integral model a."""
    c4, c6, D = cinv(a)
    b2 = binv(a)[0]
    # #E(Q)_tors divides #E(F_p) for good odd p
    n, p, k = 0, 3, 0
    while k < 12:
        p += 2
        if arith.is_prime(p) and D % p:
            n, k = math.gcd(n, count_mod(a, p)), k + 1
        elif p % 1000 == 1:
            arith.tick()
    # orders 1-10 or 12 (Mazur) divide 2520
    n = math.gcd(n, 2520)
    pts = [None]
    if n > 1:
        A, B = -27 * c4, -54 * c6
        xs = set(int_roots(_divpoly(A, B, n))) | (set(int_roots([B, A, 0, 1])) if n % 2 == 0 else set())
        for x in xs:
            y2 = x ** 3 + A * x + B
            y = math.isqrt(max(y2, 0))
            if y * y != y2:
                continue
            for yy in {y, -y}:
                xe = Fraction(x - 3 * b2, 36)
                P = (xe, (Fraction(yy, 108) - a[0] * xe - a[2]) / 2)
                if on_curve(a, P) and mul(a, P, n) is None:
                    pts.append(P)
    N = len(pts)
    if N == 1:
        return [], []
    order = lambda P: next(k for k in range(1, N + 1) if mul(a, P, k) is None)
    two = [P for P in pts if P is not None and order(P) == 2]
    g1 = max(pts[1:], key=order)
    if len(two) < 3:
        return [N], [g1]
    h = mul(a, g1, order(g1) // 2)
    return [N // 2, 2], [g1, next(P for P in two if P != h)]
