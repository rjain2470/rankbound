"""rankbound.check: the independent checker of rankbound's witnesses.

    rankbound check FILE... [--cores N] [--json] [--curve A] [--max-mb M] [--max-seconds S]
    main(argv)            the command line; returns its exit status
    check(FILE, ...)      the verdict as a dict; never raises for anything in the file

This module has the command, the verdict, the form checks, the statements and the 2-isogeny witnesses; selmer.py has
the witnesses of curves with E(Q)[2] = 0, check.gp the PARI/GP checks, check.c the relations and the ranks.  It imports
nothing of the prover."""
import argparse, glob, gzip, hashlib, io, json, math, os, re, shutil, stat, subprocess, sys, tempfile, time, zlib
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ('__init__.py', 'selmer.py', 'check.gp', 'check.c')     # the checker's hash covers exactly these
FORMAT = 'rankbound-2descent-witness-1'                           # then rankbound 2's name
FORMATS = (FORMAT, 'rankbound-2descent-certificate-1')

# ---------- the statements

def parity_bound(G, u, n, w):
    """Return the largest S <= G + u + n with (-1)^S = w, or G + u + n if w is not +-1."""
    return parity_floor(G + u + n, w)


def parity_floor(S, w):
    """Return the largest integer <= S with (-1)^S = w (S if w is not +-1): the 2-parity step."""
    return S - 1 if w in (1, -1) and (-1) ** S != w else S


def word(c):
    """Return the witness's word for itself in its statement: 'witness', or rankbound 2's 'certificate'."""
    return 'witness' if '-witness-' in c['format'] else 'certificate'


def _with_r(S, r, source, word):
    """Return the end of the statement rank E(Q) <= S for the lower bound r; source is where an unproved r is from."""
    if r is not None and S == r and r > 0:
        return ' As E(Q) has %d independent rational point%s%s, we therefore have rank E(Q) = %d.' % (
            r, 's' if r > 1 else '', ' (%s; not proved in this %s)' % (source, word) if source else '', r)
    if r is not None and S < r:         # fails the check bound_ge_r
        return ' CONTRADICTION: rank E(Q) >= %d.' % r
    return ''


def statement(G, F, r, source=None):
    """Return (S, the statement) of formats 3 and 2, F = {u, n, w}."""
    S = parity_bound(G, F['u'], F['n'], F['w'])
    return S, 'Assuming the GRH, we have that rank E(Q) <= %d.' % S + _with_r(S, r, source, 'certificate')


def selmer_statement(S, hyp, r, source=None, word='witness'):
    """Return the statement of formats 4 to 6 under hyp ('GRH' or 'none')."""
    return ('Assuming the GRH, we have that rank E(Q) <= %d.' if hyp == 'GRH' else 'We have that rank E(Q) <= %d.') % S + \
        _with_r(S, r, source, word)


def isogeny_statement(B, w, r, source=None, word='witness'):
    """Return (S, the statement) of the 2-isogeny format for the bound B and the root number w."""
    S = parity_floor(B, w)
    return S, 'Unconditionally, rank E(Q) <= %d.' % S + _with_r(S, r, source, word)

# ---------- gp

NO_GP = 'rankbound check needs PARI/GP 2.17 or later.'
NO_READ = 'The file %s cannot be read (%s).'


NO_LIMIT = 24 * 3600      # seconds each program of a check may run without --max-seconds


class Rejected(RuntimeError):
    """A check of the witness failed."""


class LimitExceeded(RuntimeError):
    """The witness is over a limit and was not checked; the message completes 'because ...'."""


class TimeLimit(LimitExceeded):
    """A program did not finish within seconds."""

    def __init__(self, msg, seconds):
        super().__init__(msg)
        self.seconds = seconds


def source(*names):
    """Return the text of the GP files names, relative to this folder."""
    out = []
    for n in names:
        with open(os.path.join(HERE, n)) as f:
            out.append(f.read())
    return '\n'.join(out)


def run_gp(body, libs=(), seconds=3600, stack='256M', sizemax='8G', gp='gp', threads=1):
    """Run gp on check.gp, libs and body; return its output (errors as in gp_errors, TimeLimit after seconds)."""
    src = 'default(nbthreads, %d);\ndefault(parisizemax, "%s");\n' % (threads, sizemax) + source('check.gp', *libs) + \
        '\n' + body + '\nquit;\n'
    fd, path = tempfile.mkstemp(suffix='.gp', prefix='rankbound')
    with os.fdopen(fd, 'w') as f:
        f.write(src)
    try:                # without stdin=DEVNULL, gp -f waits for input
        r = subprocess.run([gp, '-q', '-s', stack, '-f', path], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=seconds)
    except subprocess.TimeoutExpired:
        raise TimeLimit('PARI/GP did not finish within %.0f s' % seconds, seconds)
    except FileNotFoundError:
        raise RuntimeError(NO_GP)
    finally:
        os.unlink(path)
    gp_errors(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError('PARI/GP failed: %s' % (r.stderr[-2000:].strip() or 'exit %d' % r.returncode))
    return r.stdout


def gp_errors(text):
    """Raise Rejected for an error() of the GP code in text, LimitExceeded if gp ran out of memory, or RuntimeError
    for any other gp error."""
    err = [l for l in text.splitlines() if l.lstrip().startswith('***') and 'Warning:' not in l]
    if any('the PARI stack overflows' in l or 'not enough memory' in l for l in err):
        raise LimitExceeded('PARI/GP ran out of memory')
    user = [l.split('user error: ', 1)[1].strip() for l in err if 'user error: ' in l]
    if user:
        raise Rejected(user[0])
    if err:
        raise RuntimeError('PARI/GP failed: ' + '\n'.join(err[:20]))


_GP: Dict[str, str] = {}


def gp_version(gp='gp'):
    """Return gp's version ('2.17.4'), or '?' if it does not run."""
    if gp not in _GP:
        try:
            _GP[gp] = subprocess.run([gp, '--version-short'], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                     timeout=60).stdout.strip() or '?'
        except (OSError, subprocess.SubprocessError):
            _GP[gp] = '?'
    return _GP[gp]


def gp_ok(gp='gp'):
    """Return whether gp runs and is PARI/GP 2.17 or later."""
    try:
        return tuple(int(x) for x in gp_version(gp).split('.')[:2]) >= (2, 17)
    except ValueError:
        return False

# ---------- the form of a witness, checked before any mathematics

MAX_DIGITS = 4000          # below Python's int/str limit, 4300
NUM = re.compile(r'0|-?[1-9][0-9]*')
RAT = re.compile(r'0|-?[1-9][0-9]*(/[1-9][0-9]*)?')
PNUM = re.compile(r'[1-9][0-9]*')
NAME = re.compile(r'[A-Za-z0-9_.-]{1,64}')
DATE = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')
_I = lambda x: type(x) is int
# gp evaluates what it reads, so every value that reaches it must be a JSON integer or match NUM or RAT.
ATOMS = {'nat': ('a JSON integer >= 0', lambda x: _I(x) and x >= 0),
         'pos': ('a JSON integer >= 1', lambda x: _I(x) and x >= 1), 'sign': ('1 or -1', lambda x: _I(x) and x in (1, -1)),
         'num': ('an integer in canonical decimal form', lambda x: isinstance(x, str) and NUM.fullmatch(x)),
         'pnum': ('a positive integer in canonical decimal form', lambda x: isinstance(x, str) and PNUM.fullmatch(x)),
         'rat': ('a rational number p/q in canonical form', lambda x: isinstance(x, str) and RAT.fullmatch(x)),
         'small': ('an integer >= 0 (JSON, or in canonical decimal form)',
                   lambda x: (_I(x) and x >= 0) or (isinstance(x, str) and (x == '0' or PNUM.fullmatch(x)))),
         'str': ('a string', lambda x: isinstance(x, str)),
         'name': ('a name of 1 to 64 letters, digits and _ . -', lambda x: isinstance(x, str) and NAME.fullmatch(x)),
         'date': ('a date YYYY-MM-DD', lambda x: isinstance(x, str) and DATE.fullmatch(x))}


def form(x, spec, path=''):
    """Raise ValueError, naming the place, unless x has the form spec.

    spec is an atom (ATOMS); [s] or [s, n], a list of values of form s (exactly n); {key: s}, an object with exactly
    these keys ('key?' optional, '*' any prime keys); or ('null', s).  A number over MAX_DIGITS raises LimitExceeded."""
    where = path or 'the witness'
    if isinstance(spec, str):
        if isinstance(x, str) and len(x) > MAX_DIGITS and spec in ('num', 'pnum', 'rat', 'small'):
            raise LimitExceeded('it has a number of more than %d digits' % MAX_DIGITS)
        what, good = ATOMS[spec]
        if not good(x):
            raise ValueError('%s: %.80s is not %s' % (where, json.dumps(x), what))
    elif isinstance(spec, tuple):
        if x is not None:
            form(x, spec[1], path)
    elif isinstance(spec, list):
        if not isinstance(x, list) or (len(spec) == 2 and len(x) != spec[1]):
            raise ValueError('%s must be a list%s' % (where, ' of %d values' % spec[1] if len(spec) == 2 else ''))
        for i, y in enumerate(x):
            form(y, spec[0], '%s[%d]' % (path, i))
    else:
        keys = {k.rstrip('?'): s for k, s in spec.items() if k != '*'}
        if not isinstance(x, dict) or not {k for k in spec if k[-1] not in '?*'} <= set(x) or (
                '*' not in spec and not set(x) <= set(keys)):
            raise ValueError('%s must be an object with the keys %s' % (where, ', '.join(
                k.replace('*', 'p (primes)') + (' (optional)' if k[-1] == '?' else '') for k in spec).replace('?', '')))
        for k, y in x.items():
            if k not in keys and not PNUM.fullmatch(k):
                raise ValueError('%s: the key %.80s is not a prime' % (where, json.dumps(k)))
            form(y, keys[k] if k in keys else spec['*'], (path + '.' if path else '') + k)


POINT = ['rat', 2]
CURVE = {'ainvs': ['num', 5], 'minimal_model': ['num', 5], 'bad_primes': ['pnum']}
COMMON = {'name': 'name', 'curve': CURVE, 'statement': 'str', 'torsion': ['pos'], 'created': 'date', 'format': 'str',
          'rank_lower_bound?': {'r': 'nat', 'points?': [POINT], 'primes?': ['pos'], 'source?': 'str'}}
CT = ('null', {'quartics': [['num', 5]], 'local': [[POINT]], 'rank': 'nat', 'pairs': [
    {'i': 'pos', 'j': 'pos', 'sum': ['rat', 5], 'm': ['rat', 3], 'sum_local': [POINT], 'primes': ['pnum'],
     'points': [POINT]}]})


def common(c, torsion):
    """Check what form() cannot of the keys every format shares; torsion is the most invariants allowed."""
    for v in (c['curve']['ainvs'], c['curve']['minimal_model']):
        c4, c6 = _invariants([int(x) for x in v])
        if c4 ** 3 == c6 ** 2:
            raise ValueError('the curve %s is singular' % ','.join(v))
    rl = c.get('rank_lower_bound')
    if rl is not None and set(rl) not in ({'r', 'points', 'primes'}, {'r', 'source'}):
        raise ValueError('rank_lower_bound must have the keys r, points, primes or r, source')
    if rl is not None and 'points' in rl and len(rl['points']) != rl['r']:
        raise ValueError('rank_lower_bound: r must be the number of points')
    src = (rl or {}).get('source', 'x')
    if not (src.strip() and len(src) <= 200 and src.isprintable()):
        raise ValueError('the source of r must be one line of at most 200 characters')
    if len(c['torsion']) > torsion:
        raise ValueError('torsion has at most %d invariant%s' % (torsion, 's' if torsion > 1 else ''))


def ct_form(ct, nv, where='cassels_tate'):
    """Check what form() cannot of a Cassels-Tate section ct with nv places: the counts and orders."""
    if ct is None:
        return
    n = len(ct['quartics'])
    if len(ct['local']) != n or any(len(x) != nv for x in ct['local']):
        raise ValueError('%s.local: a point at each of the %d places is needed per quartic' % (where, nv))
    want = ((i, j) for i in range(1, n + 1) for j in range(i + 1, n + 1))
    if len(ct['pairs']) != n * (n - 1) // 2 or any((p['i'], p['j']) != ij for p, ij in zip(ct['pairs'], want)):
        raise ValueError('%s.pairs: one pair per 1 <= i < j <= %d is needed, in order' % (where, n))
    for k, p in enumerate(ct['pairs']):
        ps = [int(q) for q in p['primes']]
        if any(a >= b for a, b in zip(ps, ps[1:])):
            raise ValueError('%s.pairs[%d].primes must be strictly increasing' % (where, k))
        if len(p['sum_local']) != nv or len(p['points']) != nv + len(ps):
            raise ValueError('%s.pairs[%d]: a point at each place (and at each of its primes) is needed' % (where, k))


def _object(pairs):
    """Build a JSON object, refusing duplicate keys (json would silently keep the last)."""
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError('the key %s appears twice' % json.dumps(k)[:80])
        d[k] = v
    return d


def _constant(x):
    raise ValueError('%s is not a JSON number' % x)


def snapshot(path, max_mb=None):
    """Return the bytes of the file at path, read once, so that check hashes and verifies the same copy.

    Raise LimitExceeded over max_mb MB, by its size and by reading at most a byte more (in case it grows)."""
    with open(path, 'rb') as f:
        if max_mb is None:
            return f.read()
        if os.fstat(f.fileno()).st_size > max_mb * 1e6:
            raise LimitExceeded('it is larger than %g MB, the limit set by --max-mb' % max_mb)
        data = f.read(int(max_mb * 1e6) + 1)
    if len(data) > max_mb * 1e6:
        raise LimitExceeded('it is larger than %g MB, the limit set by --max-mb' % max_mb)
    return data


def load(path, max_mb=None, data=None):
    """Return the witness at path (or in data, from snapshot) as a dict; raise ValueError or LimitExceeded."""
    if data is None:
        data = snapshot(path, max_mb)
    elif max_mb is not None and len(data) > max_mb * 1e6:
        raise LimitExceeded('it is larger than %g MB, the limit set by --max-mb' % max_mb)
    if data[:2] == b'\x1f\x8b':           # gzipped: max_mb bounds the JSON too
        data = gunzip(data, max_mb)
    try:
        c = json.loads(data.decode('utf-8'), object_pairs_hook=_object, parse_constant=_constant)
    except RecursionError:
        raise ValueError('the JSON is nested too deeply')
    except UnicodeDecodeError:
        raise ValueError('it is not UTF-8 text')
    except json.JSONDecodeError as e:
        raise ValueError('it is not valid JSON at line %d, column %d' % (e.lineno, e.colno))
    from .selmer import FORMATS as SEL, FORMATS3
    if not isinstance(c, dict) or not isinstance(c.get('format'), str):
        raise ValueError(NOT_WITNESS)
    if c['format'] not in FORMATS + SEL + FORMATS3:
        raise ValueError('its format %.80s is not %s' % (json.dumps(c['format']), ', '.join(FORMATS + SEL + FORMATS3)))
    return c


NOT_WITNESS = 'it is not a rankbound witness (no known "format")'
JSON_MB = 4000          # unzipped size without --max-mb (two million relations: about 340 MB)
RATIO, RATIO_MB = 50, 100   # nor beyond RATIO_MB MB at more than RATIO times the gzip size (witnesses: at most 4)


def gunzip(data, max_mb=None):
    """Return the JSON of the gzipped bytes data, within the limits above; ValueError if not valid gzip."""
    cap = (JSON_MB if max_mb is None else max_mb) * 1e6
    out, n = bytearray(), 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as g:
            while True:
                b = g.read(1 << 24)
                if not b:
                    break
                out += b
                n += len(b)
                if n > cap:
                    raise LimitExceeded('its JSON is larger than %g MB, the limit set by --max-mb' % max_mb
                                        if max_mb is not None else
                                        'its JSON is larger than %g MB, the limit without --max-mb' % JSON_MB)
                if n > RATIO_MB * 1e6 and n > RATIO * len(data):
                    raise LimitExceeded('its JSON is more than %d times the size of its gzip file, unlike a witness'
                                        % RATIO)
    except (OSError, EOFError, zlib.error):
        raise ValueError('it is not a valid gzip file')
    return out

# ---------- the curve

def _invariants(ainvs):
    """Return (c4, c6) of the five integral a-invariants ainvs."""
    a1, a2, a3, a4, a6 = ainvs
    b2, b4, b6 = a1 * a1 + 4 * a2, 2 * a4 + a1 * a3, a3 * a3 + 4 * a6
    return b2 * b2 - 24 * b4, -b2 ** 3 + 36 * b2 * b4 - 216 * b6


def _root(x, k):
    """Return the rational k-th root of the rational x > 0, or None."""
    out = []
    for m in (x.numerator, x.denominator):
        r = _iroot(m, k)
        if r ** k != m:
            return None
        out.append(r)
    return Fraction(out[0], out[1])


def _iroot(m, k):
    """Return floor(m^(1/k)) for an integer m > 0, by Newton's method."""
    x = 1 << (m.bit_length() // k + 1)
    while True:
        y = ((k - 1) * x + m // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


def same_curve(a, b):
    """Return whether a and b are elliptic curves isomorphic over Q: c4' = u^4 c4, c6' = u^6 c6, u in Q*."""
    (c4, c6), (d4, d6) = _invariants(a), _invariants(b)
    if c4 ** 3 == c6 ** 2 or d4 ** 3 == d6 ** 2:        # singular
        return False
    if (c4 == 0) != (d4 == 0) or (c6 == 0) != (d6 == 0):
        return False
    if c4 == 0:                                         # j = 0: c6'/c6 a sixth power
        return Fraction(d6, c6) > 0 and _root(Fraction(d6, c6), 6) is not None
    if c6 == 0:                                         # j = 1728: c4'/c4 a fourth power
        return Fraction(d4, c4) > 0 and _root(Fraction(d4, c4), 4) is not None
    q = Fraction(d6, c6) / Fraction(d4, c4)             # u^2
    return q > 0 and _root(q, 2) is not None and Fraction(d4, c4) == q * q and Fraction(d6, c6) == q ** 3

# ---------- the 2-isogeny format (FORMATS): curves E/Q with a rational point of order 2
# check.gp (tdverify, vrank) checks everything up to each B_e; claims() checks w, B, S and the statement.

SIDE = {'conic': [['num', 3]], 'local': [[['num', 2]]], 'rank': 'nat'}
SPEC = dict(COMMON, GRH='str', cassels_tate=CT, local={'w': 'sign'}, claims={'B': 'nat', 'S': 'nat'}, descent=[{
    'e': 'num', 'a': 'num', 'b': 'num', 'selmer': ['num'], 'selmer_isogenous': ['num'], 's': 'nat', 's_isogenous': 'nat',
    'B': 'nat', 'cassels_tate': CT,
    'pairing': dict(SIDE, places=['str'], **{k + '_isogenous': v for k, v in SIDE.items()})}])
CHECKS = ('bad_primes_prime', 'minimal_model', 'disc_factored', 'torsion', 'two_torsion')   # then per isogeny
HYPOTHESIS = ('rank E(Q) <= rank_upper_bound unconditionally (2-isogeny descent and the 2-parity theorem; no GRH, '
              'no BSD); rank E(Q) >= rank_lower_bound unconditionally when r_proved')


def clean(c):
    """Check the form of the 2-isogeny witness c; raise ValueError if malformed."""
    form(c, SPEC)
    common(c, 2)
    bad = c['curve']['bad_primes']
    nv = len(places(bad))
    if len(c['descent']) not in (1, 3):
        raise ValueError('descent has one entry per rational root of T: 1 or 3')
    for i, d in enumerate(c['descent']):
        pr, at = d['pairing'], 'descent[%d]' % i
        if pr['places'] != places(bad):
            raise ValueError('%s.pairing.places must be %s' % (at, ', '.join(places(bad))))
        for g, sel in (('', 'selmer'), ('_isogenous', 'selmer_isogenous')):
            if len(d[sel]) > len(bad) + 2:
                raise ValueError('%s.%s is longer than the dimension of Q(S, 2)' % (at, sel))
            if len(pr['conic' + g]) != len(d[sel]) or len(pr['local' + g]) != len(d[sel]) or any(
                    len(x) != nv for x in pr['local' + g]):
                raise ValueError('%s.pairing: conic%s and local%s need an entry per element of %s, local%s a point at '
                                 'each place' % (at, g, g, sel, g))
        ct_form(d['cassels_tate'], nv, at + '.cassels_tate')
    ct_form(c['cassels_tate'], nv)


def places(bad_primes):
    """Return the places of the pairings as strings: R, then {2, 3, 5, 7} and the bad primes, increasing."""
    return ['R'] + [str(p) for p in sorted({2, 3, 5, 7} | {int(p) for p in bad_primes})]


def big_primes(c):
    """Raise LimitExceeded if a prime of the points of c is 2^64 or more."""
    if any(p >= 2 ** 64 for p in (c.get('rank_lower_bound') or {}).get('primes', [])):
        raise LimitExceeded('a prime of its points is 2^64 or more')


def gp_vec(v):
    return '[%s]' % ','.join(v)


def gp_rank(c):
    """Return the GP call vrank of the points of c, or ''."""
    rl = c.get('rank_lower_bound') or {}
    if 'points' not in rl:
        return ''
    return 'vrank(%s, %d, [%s], [%s]);\n' % (gp_vec(c['curve']['ainvs']), rl['r'],
                                             ','.join('[%s,%s]' % tuple(p) for p in rl['points']),
                                             ','.join(map(str, rl['primes'])))


def gp_body(c):
    """Return the GP calls that check the 2-isogeny witness c."""
    cur = c['curve']
    pts = lambda P: '[%s]' % ','.join('[%s]' % ','.join(gp_vec(z) for z in row) for row in P)
    D = ','.join('[%s,%s,%s,%s,%s,%d,%d,%d,%s,%s,%d,%s,%s,%d,%s]' % (
        d['e'], d['a'], d['b'], gp_vec(d['selmer']), gp_vec(d['selmer_isogenous']), d['s'], d['s_isogenous'], d['B'],
        '[%s]' % ','.join(gp_vec(z) for z in d['pairing']['conic']), pts(d['pairing']['local']), d['pairing']['rank'],
        '[%s]' % ','.join(gp_vec(z) for z in d['pairing']['conic_isogenous']), pts(d['pairing']['local_isogenous']),
        d['pairing']['rank_isogenous'], gp_ct(d['cassels_tate'])) for d in c['descent'])
    return 'tdverify(%s, %s, %s, [%s], [%s], %s);\n' % (
        gp_vec(cur['ainvs']), gp_vec(cur['bad_primes']), gp_vec(cur['minimal_model']), ','.join(map(str, c['torsion'])),
        D, gp_ct(c['cassels_tate'])) + gp_rank(c)


def gp_ct(ct):
    """Return a Cassels-Tate section as tdvct's GP vector, or 0 for None."""
    if ct is None:
        return '0'
    nest = lambda v: '[%s]' % ','.join(nest(z) for z in v) if isinstance(v, list) else str(v)
    pairs = [[pr['i'], pr['j'], pr['sum'], pr['m'], pr['sum_local'], pr['primes'], pr['points']] for pr in ct['pairs']]
    return '[%s,%s,%s,%d]' % (nest(ct['quartics']), nest(ct['local']), nest(pairs), ct['rank'])


def gp_checks(out, ok):
    """Record with ok the lines "CHK name 0|1 info" of out; return the names printed."""
    names = set()
    for l in out.splitlines():
        t = l.split(' ', 3)
        if t[0] == 'CHK':
            ok(t[1], t[2] == '1', t[3] if len(t) > 3 else '')
            names.add(t[1])
    return names


def need(printed, names, ok, out=''):
    """Record a failed check for each of names that gp did not print (fail closed)."""
    why = 'not checked: gp stopped at a failed check' if 'HALT' in out.splitlines() else 'not printed by gp'
    for name in names:
        if name not in printed:
            ok(name, False, why)


def descent(c, ok, gp, left):
    """Check the 2-isogeny witness c; return (the statement rebuilt, whether the rank lower bound is proved)."""
    rl = c.get('rank_lower_bound') or {}
    out = run_gp(gp_body(c), seconds=NO_LIMIT if left() is None else left(), gp=gp)
    need(gp_checks(out, ok), CHECKS + (('cassels_tate',) if c['cassels_tate'] else ()) + tuple(
        '%s_%d' % (k, i) for i, d in enumerate(c['descent'], 1)
        for k in ('descent', 'local_duality', 'pairing') + (('cassels_tate',) if d['cassels_tate'] else ()) + ('bound',)) +
        (('rank_lower_bound',) if 'points' in rl else ()), ok, out)
    return claims(c, out, ok), 'points' in rl


def claims(c, out, ok):
    """Check the claims against gp's line "TD B S w"; return the statement rebuilt."""
    rl = c.get('rank_lower_bound') or {}
    r = rl.get('r')
    td = [l.split() for l in out.splitlines() if l.startswith('TD ')]
    if not td:
        ok('claims', False, 'no bound recomputed (a check above failed)')
        return ''
    B, S_gp, w = map(int, td[0][1:])
    ok('root_number', w == c['local']['w'], 'w = %d' % w)
    S, st = isogeny_statement(B, w, r, None if 'points' in rl else rl.get('source'), word(c))
    ok('claims', B == c['claims']['B'] and S == c['claims']['S'] == S_gp and 0 <= S <= B,
       'B = %d, rank <= %d' % (B, S))
    ok('GRH', c['GRH'] == 'no', 'the bound is unconditional')
    ok('statement', c['statement'] == st)
    if r is not None:
        ok('bound_ge_r', S >= r, 'rank <= %d, rank >= %d' % (S, r))
    return st

# ---------- the check of one witness

def verify(path: str, cores: int = 1, gp: str = 'gp', curve: Optional[Sequence[Union[int, str]]] = None,
           max_mb: Optional[float] = None, max_seconds: Optional[float] = None, data: Optional[bytes] = None,
           rank_info: Optional[List[Any]] = None
           ) -> Tuple[bool, List[Tuple[str, bool, str]], str, Optional[Dict[str, Any]]]:
    """Check the witness at path (or in data, from snapshot); return (all passed, checks, statement, witness or None).

    Raise LimitExceeded when it is not checked, and RuntimeError or OSError when the check cannot run.  rank_info, a
    list, receives the mode of each rank check."""
    if type(cores) is not int or cores < 1:
        raise ValueError('cores must be a positive integer.')
    from . import selmer as sel
    t0 = time.time()
    left = lambda: None if max_seconds is None else max_seconds - (time.time() - t0)
    res = []
    ok = lambda name, cond, info='': res.append((name, bool(cond), info))
    try:
        c = load(path, max_mb, data)
        kind = 'td' if c['format'] in FORMATS else 'sel' if c['format'] in sel.FORMATS else 'sel3'
        rels = clean(c) if kind == 'td' else sel.clean(c) if kind == 'sel' else sel.clean3(c)
    except (ValueError, KeyError, TypeError, AttributeError) as e:      # UnicodeDecodeError is a ValueError
        return False, [('format', False, str(e))], '', None
    try:                # an error computing the limits comes from the witness's numbers: verdict 'limit'
        (big_primes if kind == 'td' else sel.limits if kind == 'sel' else sel.limits3)(c)
    except (ArithmeticError, ValueError) as e:
        raise LimitExceeded('its limits cannot be computed (%s)' % type(e).__name__)
    ok('format', True)
    st, proved = '', 'points' in (c.get('rank_lower_bound') or {})
    if curve is not None:
        same = same_curve([int(x) for x in curve], [int(x) for x in c['curve']['ainvs']])
        ok('curve', same, 'the curve given' if same else 'not isomorphic to the curve given [%s]' % ','.join(map(str, curve)))
    try:
        if curve is not None and not same:
            raise Rejected('the witness is for another curve')
        if kind == 'td':
            st, proved = descent(c, ok, gp, left)
        else:
            st, proved = (sel.verify if kind == 'sel' else sel.verify3)(c, rels, ok, gp, left, cores, rank_info)
    except Rejected as e:       # any other error propagates: no verdict
        ok('rejected', False, str(e)[:2000])
    rl = c.get('rank_lower_bound') or {}
    if not proved and rl.get('r') is not None:
        ok('rank_lower_bound', True, 'not proved here: rank >= %d is from %s' % (rl['r'], rl.get('source')))
    # fail closed: a path that stops before the claims and the statement without a failure must not accept
    if all(x[1] for x in res) and not {'claims', 'statement'} <= {x[0] for x in res if x[1]}:
        ok('complete', False, 'the claims or the statement were not checked')
    return all(x[1] for x in res), res, st, c

# ---------- provenance

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def checker_hash() -> str:
    """Return the checker's hash: the first 16 hex digits of the sha256 of the lines 'FILE SHA256' of SOURCES."""
    h = hashlib.sha256()
    for f in SOURCES:
        h.update(f.encode() + b' ' + sha256(os.path.join(HERE, f)).encode() + b'\n')
    return h.hexdigest()[:16]


def deployment():
    """Return the first word of the VERSION file at the top of the checkout, else '-'."""
    vf = os.path.join(HERE, '..', '..', 'VERSION')
    if os.path.exists(vf):
        with open(vf) as f:
            return (f.read().split() or ['-'])[0]
    return '-'


MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December')


def when(t):
    """Return t as '18 October 2026 at 10:46 PM'."""
    return '%d %s %d at %d:%02d %s' % (t.tm_mday, MONTHS[t.tm_mon - 1], t.tm_year, t.tm_hour % 12 or 12, t.tm_min,
                                       'AM' if t.tm_hour < 12 else 'PM')


def seconds(s):
    n = int(round(s))
    return '%d second%s' % (n, '' if n == 1 else 's')


def name_of(path):
    return os.path.basename(str(path).rstrip('/')) or str(path)


def sentence(s):
    """Return s as a sentence: ending with a period, capitalized unless its first word is a key or a program."""
    s = s.strip().rstrip('.')
    key = re.match(r'[^ ]*[_.[]|rankbound\b', s.split(' ', 1)[0] + ' ')
    return (s if key else s[:1].upper() + s[1:]) + '.' if s else ''

# ---------- the verdict (rankbound check --json)

EXIT = {'accepted': 0, 'rejected': 1, 'limit': 3, 'error': 4}      # exit statuses (2: a usage error)
KEYS = ('verdict', 'reason', 'witness', 'sha256', 'name', 'curve', 'rank_upper_bound', 'rank_lower_bound', 'r_proved',
        'rank', 'G', 'u', 'n', 'w', 'GRH', 'method', 'torsion', 'hypothesis', 'statement', 'checks', 'checker', 'deploy',
        'gp', 'date', 'seconds', 'rank_check')


def failure(res, name, curve):
    """Return (the reason for --json, the line) for the first failed check in res."""
    n, _, info = next((x for x in res if not x[1]), ('?', False, ''))
    if n == 'format' and info == NOT_WITNESS:
        return 'not a valid witness: %s' % info, '%s is not a rankbound witness (no known "format").' % name
    if n == 'format':
        return 'not a valid witness: %s' % info.rstrip('.'), '%s fails at the check format. %s' % (name, sentence(info))
    if n == 'curve':
        r = 'the witness for %s is not for the curve %s' % (name, ','.join(map(str, curve)))
        return r, '%s fails at the check curve. It is not a witness for the curve %s.' % (name, ','.join(map(str, curve)))
    if n == 'rejected':
        return 'failed check: rejected: %s' % info.rstrip('.'), '%s fails at the check rejected. %s' % (name, sentence(info))
    return 'failed check: %s' % n, '%s fails at the check %s.' % (name, n)


def check(path: str, curve: Optional[Sequence[Union[int, str]]] = None, cores: int = 1, gp: str = 'gp',
          max_mb: Optional[float] = None, max_seconds: Optional[float] = None,
          log: Optional[Callable[[str], Any]] = None) -> Dict[str, Any]:
    """Return the verdict on the witness at path as a JSON-ready dict (KEYS); never raise for anything in the file.

    log receives the lines printed without --json (None: quiet)."""
    from .selmer import FORMATS as SEL, HYPOTHESIS as SELH, HYPOTHESIS3
    t0, name = time.time(), name_of(path)
    v = dict.fromkeys(KEYS)
    v.update(reason='', witness=str(path), checks=[], checker=checker_hash(), deploy=deployment(), gp=gp_version(gp),
             date=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t0)))
    data = err = None
    modes = v['rank_check'] = []        # [] for a 2-isogeny descent
    try:            # one read, so that the hash and the check are of the same bytes
        data = snapshot(path, max_mb)
        v['sha256'] = hashlib.sha256(data).hexdigest()
    except (OSError, LimitExceeded) as e:
        err = e
    if log:
        log('Checker %s with PARI/GP %s, %s. Witness sha256 %s.' % (v['checker'], v['gp'], when(time.localtime(t0)),
                                                                     v['sha256'] or '?'))
    try:
        if err is not None:
            raise err
        good, res, st, c = verify(path, cores, gp, curve, max_mb, max_seconds, data, modes)
        v['checks'] = [{'check': n, 'ok': g, 'info': i} for n, g, i in res]
        if good and c is not None:
            rl = c.get('rank_lower_bound') or {}
            S, r = c['claims'].get('S_ctp', c['claims']['S']), rl.get('r')
            descent_, selmer = c['format'] in FORMATS, c['format'] in SEL
            loc = c['claims'] if selmer else c['local']
            v.update(verdict='accepted', name=c['name'], curve=c['curve']['minimal_model'], rank_upper_bound=S,
                     G=c['claims'].get('G'), u=loc.get('u'), n=loc.get('n'), w=loc['w'],
                     GRH=('yes' if c['hypothesis'] == 'GRH' else 'no') if selmer else c.get('GRH', 'yes'),
                     torsion=c.get('torsion'),
                     hypothesis=HYPOTHESIS if descent_ else SELH[c['hypothesis']] if selmer else HYPOTHESIS3,
                     method='2-isogeny descent' if descent_ else '2-descent' if selmer else 'class group',
                     rank_lower_bound=r, r_proved=r is not None and ('points' in rl or r == 0), statement=st)
            v['rank'] = S if S == 0 or (v['r_proved'] and S == r) else None
            line = '%s passes in %s. Rank %s%d (%s).' % (c['name'], seconds(time.time() - t0),
                                                         '' if v['rank'] is not None else 'at most ', S,
                                                         'assuming the GRH' if v['GRH'] == 'yes' else 'unconditional')
        else:
            v['name'] = c['name'] if c else None
            v['verdict'] = 'rejected'
            v['reason'], line = failure(res, c['name'] if c else name, curve)
    except TimeLimit as e:
        line = '%s was not checked within %d seconds%s.' % (
            name, round(max_seconds if max_seconds is not None else e.seconds),
            ', the limit set by --max-seconds' if max_seconds is not None else '')
        v.update(verdict='limit', reason=line)
    except LimitExceeded as e:
        line = '%s was not checked, because %s.' % (name, str(e).rstrip('.'))
        v.update(verdict='limit', reason=line)
    except Exception as e:          # the checker could not run: no verdict
        why = str(e)[:2000].strip() if isinstance(e, RuntimeError) else '%s: %s' % (type(e).__name__, str(e)[:2000])
        line = '%s was not checked, because the checker could not run. %s' % (name, sentence(why))
        v.update(verdict='error', reason=line)
    v['seconds'] = round(time.time() - t0, 1)
    if log:
        log(line)
    return v

# ---------- the command

def _positive(kind):
    def f(s):
        try:
            x = kind(s)
        except ValueError:
            raise argparse.ArgumentTypeError('not a number')
        if not (math.isfinite(x) and x > 0):
            raise argparse.ArgumentTypeError('must be positive')
        return x
    f.__name__ = kind.__name__
    return f


class _Parser(argparse.ArgumentParser):
    """argparse with a usage error on one line of stderr (exit 2)."""

    def error(self, message):
        print('%s: error: %s. See %s --help.' % (self.prog, message.rstrip('.'), self.prog), file=sys.stderr)
        raise SystemExit(2)


def _ainvs(s):
    """Parse --curve as `rankbound bound` does: a1,a2,a3,a4,a6 or a4,a6 (any model), optionally in [], with spaces."""
    t = s.strip(' \t')
    if t[:1] == '[' and t[-1:] == ']':
        t = t[1:-1]
    t = [x.strip(' \t') for x in t.split(',')]
    if len(t) not in (2, 5) or not all(NUM.fullmatch(x) and len(x) <= MAX_DIGITS for x in t):
        raise argparse.ArgumentTypeError('must be a1,a2,a3,a4,a6 or a4,a6 (integers), not %.80r' % s)
    a = [int(x) for x in t]
    return [0, 0, 0] + a if len(a) == 2 else a


def no_gp(gp='gp'):
    """Return NO_GP with what was found: the version and path of an older gp, or that gp is not on the PATH."""
    path = shutil.which(gp)
    if path is None:
        return NO_GP[:-1] + '; %s is not on the PATH.' % gp
    v = gp_version(gp)
    return NO_GP[:-1] + ('; found %s at %s.' % (v, path) if v != '?' else '; %s does not run as PARI/GP.' % path)


def witnesses(x):
    """Return the witnesses x names: x, or a folder's witnesses, else those one level down (a work folder)."""
    if not os.path.isdir(x):
        return [x]
    pats = ('*.witness.json', '*.witness.json.gz', '*.cert.json')
    q = glob.escape(x)
    for d in (q, os.path.join(q, '*')):
        found = sorted(f for p in pats for f in glob.glob(os.path.join(d, p)))
        if found:
            return found
    return []


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run `rankbound check FILE...` and return its exit status (EXIT, the worst verdict first: 1, 4, 3)."""
    p = _Parser(prog='rankbound check', description='Check witnesses, or folders of them.')
    p.add_argument('files', nargs='+', metavar='FILE', help='witnesses or folders of them')
    p.add_argument('--cores', type=_positive(int), default=1, metavar='N', help='cores to use')
    p.add_argument('--json', action='store_true', help='one JSON verdict per witness')
    p.add_argument('--curve', type=_ainvs, metavar='A', help='the curve the witness must be for, a1,a2,a3,a4,a6 or a4,a6')
    p.add_argument('--max-mb', type=_positive(float), metavar='M', help='skip files over M MB')
    p.add_argument('--max-seconds', type=_positive(float), metavar='S', help='time limit per witness')
    try:
        a = p.parse_args(argv)
        files = [f for x in a.files for f in witnesses(x)]
        if not files:
            p.error('No witness in %s, nor in its folders.' % ', '.join(a.files))
        if a.curve and len(files) != 1:
            p.error('--curve works with one witness only.')
    except SystemExit as e:             # argparse: 2 for a usage error, 0 for --help
        return e.code if isinstance(e.code, int) else 2
    for f in files:                     # a FILE that cannot be opened is a usage error: nothing is checked
        try:
            if not stat.S_ISREG(os.stat(f).st_mode):
                print(NO_READ % (f, 'not a regular file'), file=sys.stderr)
                return 2
            open(f, 'rb').close()
        except OSError as e:
            print(NO_READ % (f, e.strerror or type(e).__name__), file=sys.stderr)
            return 2
    if not gp_ok():
        print(no_gp(), file=sys.stderr)
        return EXIT['error']
    log = None if a.json else (lambda line: print(time.strftime('%H:%M:%S'), line, flush=True))
    verdicts = []
    for f in files:
        v = check(f, curve=a.curve, cores=a.cores, max_mb=a.max_mb, max_seconds=a.max_seconds, log=log)
        verdicts.append(v['verdict'])
        if a.json:
            print(json.dumps(v), flush=True)
    if log and len(files) > 1:
        failed = [name_of(f) for f, v in zip(files, verdicts) if v == 'rejected']
        unchecked = [name_of(f) for f, v in zip(files, verdicts) if v in ('limit', 'error')]
        log(' '.join(['{:,} of {:,} witnesses pass.'.format(verdicts.count('accepted'), len(files))] +
                     (['Failed: %s.' % ', '.join(failed)] if failed else []) +
                     (['Not checked: %s.' % ', '.join(unchecked)] if unchecked else [])))
    codes = {EXIT[v] for v in verdicts}
    return next((x for x in (1, 4, 3) if x in codes), 0)
