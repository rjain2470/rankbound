"""rankbound.verify: the independent checker of rankbound certificates.

    rankbound verify CERT... [--cores N]    exit status 0 iff every certificate is accepted
    check(CERT, curve=...)                  the verdict as a dict; never raises on the file

CERT is a certificate NAME.cert.json or a folder of them.  This package uses only Python's standard library, so
checking a certificate needs only this folder, PARI/GP and a C compiler.

What a certificate proves.  Let E/Q have E(Q)[2] = 0, and let K be the cubic field of its 2-division polynomial.
  (1) Under GRH for the zeta functions of the unramified quadratic extensions of K, g = dim Cl(K)[2] <= G, where
      G = ns - dim V.  By [BDF08, Cor. 2.2] the prime ideals of norm < GB generate Cl(K)/2.  Degree-1 primes
      suffice: a degree-2 prime P above p has P p' = (p) with p' of degree 1, and inert primes are principal.
      V is the span of the stored combinations of relations that vanish on all other primes; each relation is a
      principal ideal.
  (2) Unconditionally, dim Sel_2(E) <= g + u + n (Brumer-Kramer, [KSW19, Prop. 3.1]) and (-1)^dim Sel_2(E) = w
      (2-parity, Monsky).  With (1), dim Sel_2(E) <= S, the largest integer <= G + u + n with (-1)^S = w.
  (3) Hence rank E(Q) <= S, since E(Q)/2E(Q) embeds in Sel_2(E) and E(Q)[2] = 0.  The only hypothesis is the GRH
      of (1), not GRH for L(E, s) and not BSD.  With r independent points and S = r, rank E(Q) = r under it.

The certificate is one JSON file with these keys, in this order:
  name              e.g. icarm728
  curve             ainvs [a1,a2,a3,a4,a6], minimal_model, bad_primes (every prime of the discriminant)
  statement         the theorem proved, in words; the verifier rebuilds it and compares
  rank_lower_bound  optional: r with points on ainvs and primes p with #E(F_p) = 2 mod 4 at which the points'
                    2-signatures have rank r; or r with a source, saying where r is from
  field             poly [c0,c1,c2,c3] with K = Q(alpha) and F(alpha, 1) = 0; dK; GB, the GRH bound; ns, the
                    number of degree-1 primes of norm < GB
  local             u, n (Brumer-Kramer) and w, the root number
  claims            G (dim Cl(K)[2] <= G), dimV (G = ns - dimV), S (rank E(Q) <= S)
  created           the date
  relations         one per row: "a,b:hexprimes", the element a - b alpha with the primes of F(a, b) in hex, with
                    multiplicity; or "f:p", the ideal (p)
  ops               pairs [dst, src], applied in order as row[dst] ^= row[src]; rows are indexed from 0
  pivots            row indices; dim V is the number of echelon pivots outside the large columns among these rows
  format            "rankbound-selmer-certificate-2"

What is checked.  verify.gp recomputes the number theory: the bad primes are prime (APRCL), the minimal model, the
discriminant is the product of powers of the bad primes, E(Q)[2] = 0, the maximal order and d_K (nfcertify), the
index form F (disc F = d_K and Q(alpha) = K), u, n, w, the BDF inequality at GB in exact interval arithmetic, ns,
and the rank lower bound when points are given.  For every relation it checks that the given primes are prime and
multiply to |F(a, b)|, and recomputes the odd valuations with idealval.  This module replays the row operations and
computes dim V with f2rank.c, which shares no code with the prover.

What is trusted: PARI/GP, the C compiler and Python's standard library; nothing from CADO-NFS or the prover.  Every
certificate value that reaches gp must be a JSON integer or a number in canonical decimal form, so a certificate
cannot run gp code."""
import array, hashlib, json, math, os, re, shutil, subprocess, tempfile, time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT = 'rankbound-selmer-certificate-2'
F2RANK = os.path.join(HERE, 'f2rank')     # built by `make build`
SOURCES = ('__init__.py', '__main__.py', 'verify.gp', 'f2rank.c')     # the files of verifier_hash


# ---------- the statement (also used by the prover)

def parity_bound(G, u, n, w):
    """Return the largest S <= G + u + n with (-1)^S = w, or G + u + n if w is not +-1."""
    S = G + u + n
    if w in (1, -1) and (-1) ** S != w:
        S -= 1
    return S


def statement(G, F, r, source=None):
    """Return (S, the statement) for the class group bound G, the local terms F = {u, n, w} and the rank lower bound
    r, or None.  source says where r is from when the certificate does not prove it."""
    S = parity_bound(G, F['u'], F['n'], F['w'])
    s = 'Assuming the GRH, we have that rank E(Q) <= %d.' % S
    if r is not None and S == r and r > 0:
        s += ' As E(Q) has %d independent rational point%s%s, we therefore have rank E(Q) = %d.' % (
            r, 's' if r > 1 else '', ' (%s; not proved in this certificate)' % source if source else '', r)
    elif r is not None and S < r:       # fails the check bound_ge_r
        s += ' CONTRADICTION: rank E(Q) >= %d.' % r
    return S, s


# ---------- gp

def source(*names):
    """Return the text of the GP files names; a relative name is read from this folder."""
    out = []
    for n in names:
        with open(os.path.join(HERE, n)) as f:
            out.append(f.read())
    return '\n'.join(out)


def run_gp(body, libs=(), seconds=3600, stack='256M', sizemax='8G', gp='gp', threads=1):
    """Run gp on the GP files libs followed by body, and return its output.  An error() in the GP code raises
    Rejected, any other gp error RuntimeError, and a run longer than seconds TimeLimit."""
    src = 'default(nbthreads, %d);\ndefault(parisizemax, "%s");\n' % (threads, sizemax) + source(*libs) + '\n' + body + '\nquit;\n'
    fd, path = tempfile.mkstemp(suffix='.gp', prefix='rankbound')
    with os.fdopen(fd, 'w') as f:
        f.write(src)
    # Without stdin=DEVNULL, gp -f waits for input.
    try:
        r = subprocess.run([gp, '-q', '-s', stack, '-f', path], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                           timeout=seconds)
    except subprocess.TimeoutExpired:
        raise TimeLimit('PARI/GP did not finish within %.0f s.' % seconds, seconds)
    finally:
        os.unlink(path)
    _gp_errors(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError('PARI/GP failed: %s' % (r.stderr[-2000:].strip() or 'exit %d' % r.returncode))
    return r.stdout


# ---------- the checks

MAX_DIGITS = 4000          # longer numbers are not checked (Python's limit: 4300)
NUM = re.compile(r'0|-?[1-9][0-9]*')
RAT = re.compile(r'0|-?[1-9][0-9]*(/[1-9][0-9]*)?')
REL = re.compile(r'(0|-?[1-9][0-9]*),(0|-?[1-9][0-9]*):((?:[1-9a-f][0-9a-f]*(?:,[1-9a-f][0-9a-f]*)*)?)')
FREE = re.compile(r'f:([1-9][0-9]*)')


def _str(x, pat=NUM):
    """Return x if it is a number in canonical decimal form, else raise ValueError."""
    # gp evaluates what it reads, so every certificate value passed to gp is checked here or is a JSON integer.
    if isinstance(x, str) and len(x) > MAX_DIGITS:
        raise LimitExceeded('{cert} has a number of more than %d digits' % MAX_DIGITS)
    if isinstance(x, str) and pat.fullmatch(x):
        return x
    raise ValueError('%.80r is not a plain %s' % (x, 'integer' if pat is NUM else 'fraction'))


def _int(x):
    """Return x if it is a JSON integer, else raise ValueError."""
    if type(x) is int:
        return x
    raise ValueError('%.80r must be a JSON integer' % (x,))


def _list(v, n=None):
    """Return v if it is a list, of length n if given, else raise ValueError."""
    if not isinstance(v, list) or (n is not None and len(v) != n):
        raise ValueError('a list%s was expected' % (' of %d values' % n if n is not None else ''))
    return v


def _gpvec(line):
    """Return the relation line 'a,b:hexprimes' or 'f:p' as a gp vector [a,b,[primes]] or [p]."""
    m = FREE.fullmatch(line) if isinstance(line, str) else None
    if m:
        return '[%s]' % m[1]
    m = REL.fullmatch(line) if isinstance(line, str) else None
    if not m:
        raise ValueError('the relation line %.80r is malformed' % (line,))
    if len(line) > MAX_DIGITS:
        raise LimitExceeded('{cert} has a relation line of more than %d characters' % MAX_DIGITS)
    return '[%s,%s,[%s]]' % (m[1], m[2], ','.join(str(int(h, 16)) for h in m[3].split(',') if h))


KEYS = ('name', 'curve', 'statement', 'rank_lower_bound', 'field', 'local', 'claims', 'created', 'relations', 'ops',
        'pivots', 'format')          # rank_lower_bound may be absent
PARTS = {'curve': ('ainvs', 'minimal_model', 'bad_primes'), 'field': ('poly', 'dK', 'GB', 'ns'), 'local': ('u', 'n', 'w'),
         'claims': ('G', 'dimV', 'S')}
NAME = re.compile(r'[A-Za-z0-9_.-]{1,64}')


def _keys(d, want, what):
    """Return d if it is an object with exactly the keys want, else raise ValueError."""
    if not isinstance(d, dict) or set(d) != set(want):
        raise ValueError('%s must have the keys %s' % ('the certificate' if what == 'certificate' else '"%s"' % what,
                                                       ', '.join(sorted(want))))
    return d


def _clean(c):
    """Check the form of the certificate c and return its relations as gp vectors; raise ValueError if it is
    malformed."""
    _keys(c, [k for k in KEYS if k != 'rank_lower_bound' or k in c], 'certificate')
    for k, ks in PARTS.items():
        _keys(c[k], ks, k)
    for k in ('name', 'statement', 'created'):
        if not isinstance(c[k], str):
            raise ValueError('"%s" must be a string' % k)
    if not NAME.fullmatch(c['name']):
        raise ValueError('the name %.80r is not allowed' % c['name'])
    cur, fd, loc, cl = c['curve'], c['field'], c['local'], c['claims']
    for v, n in ((cur['ainvs'], 5), (cur['minimal_model'], 5), (cur['bad_primes'], None), (fd['poly'], 4)):
        for x in _list(v, n):
            _str(x)
    _str(fd['dK'])
    for d, ks in ((fd, ('GB', 'ns')), (loc, ('u', 'n', 'w')), (cl, ('G', 'dimV', 'S'))):
        for k in ks:
            _int(d[k])
    if 'rank_lower_bound' in c:
        rl = c['rank_lower_bound']
        if isinstance(rl, dict) and 'points' in rl:
            _keys(rl, ('r', 'points', 'primes'), 'rank_lower_bound')
            for pt in _list(rl['points']):
                for x in _list(pt, 2):
                    _str(x, RAT)
            for p in _list(rl['primes']):
                _int(p)
        else:
            _keys(rl, ('r', 'source'), 'rank_lower_bound')
            src = rl['source']
            if not (isinstance(src, str) and src.strip() and len(src) <= 200 and src.isprintable()):
                raise ValueError('the source of r must be one line of at most 200 characters')
        if _int(rl['r']) < 0:
            raise ValueError('r is negative')
    for o in _list(c['ops']):
        for x in _list(o, 2):
            _int(x)
    for x in _list(c['pivots']):
        _int(x)
    return [_gpvec(l) for l in _list(c['relations'])]


class Rejected(RuntimeError):
    """A check of the certificate failed."""


class LimitExceeded(RuntimeError):
    """The certificate is over a limit and was not checked."""


class TimeLimit(LimitExceeded):
    """A program did not finish within seconds."""

    def __init__(self, msg, seconds):
        super().__init__(msg)
        self.seconds = seconds


def _gp_errors(text):
    """Raise Rejected for an error() of the GP code in text, or RuntimeError for any other gp error."""
    err = [l for l in text.splitlines() if l.lstrip().startswith('***') and 'Warning:' not in l]
    if any('user error' in l for l in err):
        raise Rejected('gp: ' + '\n'.join(err[:20]))
    if err:
        raise RuntimeError('PARI/GP failed: ' + '\n'.join(err[:20]))


def _gp_reason(e):
    """Return the message of the GP error() in e, without gp's trace."""
    return str(e).split('user error: ')[-1].strip().split('\n')[0].strip()


def gp_poly(poly):
    """Return F(x, 1) as GP source, from the coefficients c0..c3."""
    return ' + '.join('(%s)*x^%d' % (x, i) for i, x in enumerate(poly))


def gp_rows(f, P, rels, jobs, gp='gp', seconds=None):
    """Recompute in gp the odd-valuation columns of the relations rels, in jobs parallel processes.  f is F(x, 1)
    as GP source and P its bad primes.  Return (the rows as sets of column ids, the column names 'p:k' or 'p:k:f');
    raise TimeLimit after seconds."""
    end = None if seconds is None else time.time() + seconds
    n = len(rels)
    P = ','.join(P)
    tmpd = tempfile.mkdtemp(prefix='rankbound')
    procs = []
    try:
        per = (n + jobs - 1) // max(jobs, 1)
        for k in range(jobs):
            lo, hi = k * per, min(n, (k + 1) * per)
            if lo >= hi:
                continue
            rf = os.path.join(tmpd, 'r%d.gpv' % k)
            with open(rf, 'w') as fo:
                for v in rels[lo:hi]:
                    fo.write(v + '\n')
            src = ('default(nbthreads, 1);\ndefault(parisizemax, "4G");\n' + source('verify.gp') +
                   '\nf = %s; K = ckinit(f, [%s]);\nckrels(K, f, readvec("%s"), %d);\nquit;\n' % (f, P, rf, lo))
            sf = os.path.join(tmpd, 's%d.gp' % k)
            with open(sf, 'w') as fo:
                fo.write(src)
            with open(sf + '.out', 'w') as fo:
                procs.append((subprocess.Popen([gp, '-q', '-s', '256M', '-f', sf], stdin=subprocess.DEVNULL, stdout=fo,
                                               stderr=subprocess.STDOUT), sf + '.out'))
        rows = [None] * n
        cols = {}
        for p, out in procs:
            try:
                p.wait(None if end is None else max(0.0, end - time.time()))
            except subprocess.TimeoutExpired:
                raise TimeLimit('PARI/GP did not finish the relations within %.0f s.' % seconds, seconds)
            with open(out) as fi:
                text = fi.read()
            _gp_errors(text)
            for l in text.splitlines():
                if l.startswith('R '):
                    t = l.split()
                    rows[int(t[1])] = {cols.setdefault(z, len(cols)) for z in t[2:]}
    finally:
        for p, _ in procs:          # no gp process outlives an error
            if p.poll() is None:
                p.kill()
            p.wait()
        shutil.rmtree(tmpd, ignore_errors=True)
    missing = sum(r is None for r in rows)
    if missing:
        raise RuntimeError('PARI/GP stopped early (%d of %d relations checked).' % (n - missing, n))
    names = [None] * len(cols)
    for z, i in cols.items():
        names[i] = z
    return rows, names


def rank_V(rows, names, ops, piv, GB, jobs=1, seconds=None):
    """Replay the ops on rows, in place, and return (dim V, the number of large columns, the rank of the pivot rows),
    computed by f2rank.  A column is small iff it is a degree-1 prime p:k with p < GB.  Raises TimeLimit after
    seconds."""
    if not os.path.exists(F2RANK):
        raise RuntimeError('f2rank is not built. Run `make build` in the rankbound folder.')
    for d, s in ops:
        rows[d] ^= rows[s]
    smallset = {i for i, z in enumerate(names) if z.count(':') == 1 and int(z.split(':')[0]) < GB}
    large = sorted({col for i in piv for col in rows[i] if col not in smallset})
    sm = sorted({col for i in piv for col in rows[i] if col in smallset})
    nl = len(large)
    pos = {col: k for k, col in enumerate(large)}
    pos.update({col: nl + k for k, col in enumerate(sm)})
    ncols = len(pos)
    data = array.array('I', [len(piv), ncols, nl])
    for i in piv:
        data.append(len(rows[i])); data.extend(sorted(pos[col] for col in rows[i]))
    try:
        r = subprocess.run([F2RANK, str(max(1, jobs))], input=data.tobytes(), capture_output=True, timeout=seconds)
    except subprocess.TimeoutExpired:
        raise TimeLimit('f2rank did not finish within %.0f s.' % seconds, seconds)
    if r.returncode:
        raise RuntimeError('f2rank failed: %s' % (r.stderr.decode()[-500:].strip() or 'exit %d' % r.returncode))
    t = r.stdout.split()
    return int(t[3]), nl, int(t[1])


# the checks verify.gp must print, and pass
FIELD_CHECKS = ('bad_primes_prime', 'minimal_model', 'disc_factored', 'E2_trivial', 'maximal_order', 'index_form',
                'same_field', 'bk_u', 'bk_n', 'root_number', 'bdf_GB', 'ns')


# ---------- the certificate file

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


def load(path, max_bytes=None):
    """Return the certificate at path as a dict.  Raises ValueError if it is not UTF-8 JSON in FORMAT, or has a
    duplicate key, NaN or Infinity; LimitExceeded if it has more than max_bytes bytes."""
    if max_bytes is not None and os.path.getsize(path) > max_bytes:
        raise LimitExceeded('{cert} is larger than --max-mb (%g MB)' % (max_bytes / 1e6))
    with open(path, encoding='utf-8') as f:
        try:
            c = json.load(f, object_pairs_hook=_object, parse_constant=_constant)
        except RecursionError:
            raise ValueError('the JSON is nested too deeply')
        except UnicodeDecodeError:
            raise ValueError('it is not UTF-8 text')
        except json.JSONDecodeError as e:
            raise ValueError('it is not valid JSON at line %d, column %d' % (e.lineno, e.colno))
    if not isinstance(c, dict) or c.get('format') != FORMAT:
        raise ValueError('its format is not %s' % FORMAT)
    return c


def _limits(c):
    """Raise LimitExceeded if the certificate c asks for unbounded work: GB above Bach's bound 12 log^2 |d_K|, or a
    prime of the points of 2^64 or more."""
    d = c['field']['dK'].lstrip('-')            # log |d_K| from its leading digits
    bach = 12 * ((len(d) - 15) * math.log(10) + math.log(int(d[:15])) if len(d) > 15 else math.log(max(int(d), 1))) ** 2
    if c['field']['GB'] > bach:
        raise LimitExceeded('{cert} asks for GB = %d, above Bach\'s bound %.0f' % (c['field']['GB'], bach))
    big = [p for p in (c.get('rank_lower_bound') or {}).get('primes', []) if p >= 2 ** 64]
    if big:
        raise LimitExceeded('a prime of the points in {cert} is 2^64 or more')


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
    """Return whether the a-invariants a and b give elliptic curves isomorphic over Q: c4' = u^4 c4 and
    c6' = u^6 c6 for some rational u != 0."""
    (c4, c6), (d4, d6) = _invariants(a), _invariants(b)
    if c4 ** 3 == c6 ** 2 or d4 ** 3 == d6 ** 2:        # singular: not an elliptic curve
        return False
    if (c4 == 0) != (d4 == 0) or (c6 == 0) != (d6 == 0):
        return False
    if c4 == 0:                                         # j = 0: c6'/c6 a sixth power
        return _root(Fraction(d6, c6), 6) is not None if Fraction(d6, c6) > 0 else False
    if c6 == 0:                                         # j = 1728: c4'/c4 a fourth power
        return _root(Fraction(d4, c4), 4) is not None if Fraction(d4, c4) > 0 else False
    q = Fraction(d6, c6) / Fraction(d4, c4)             # u^2
    return q > 0 and _root(q, 2) is not None and Fraction(d4, c4) == q * q and Fraction(d6, c6) == q ** 3


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def verifier_hash():
    """Return the verifier's hash: the first 16 hex digits of the sha256 of the lines 'FILE SHA256' of SOURCES."""
    h = hashlib.sha256()
    for f in SOURCES:
        h.update(f.encode() + b' ' + _sha256(os.path.join(HERE, f)).encode() + b'\n')
    return h.hexdigest()[:16]


def _provenance(gp='gp'):
    """Return the provenance: date, gp (its version, or '?'), verifier (verifier_hash), deploy (the first word of a
    VERSION file at the top of the checkout, else '-') and _date (the date for the log line)."""
    try:
        gpv = subprocess.run([gp, '--version-short'], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                             timeout=60).stdout.strip() or '?'
    except (OSError, subprocess.SubprocessError):
        gpv = '?'
    deploy = '-'
    vf = os.path.join(HERE, '..', '..', 'VERSION')
    if os.path.exists(vf):
        with open(vf) as f:
            deploy = (f.read().split() or ['-'])[0]
    now = time.localtime()
    return {'date': time.strftime('%Y-%m-%dT%H:%M:%S%z', now), 'gp': gpv, 'verifier': verifier_hash(), 'deploy': deploy,
            '_date': time.strftime('%Y-%m-%d %H:%M:%S %z', now)}


def _cert_sha(path, max_mb=None):
    """Return the sha256 of the file at path, or None if it cannot be read or is over max_mb."""
    try:
        if max_mb is not None and os.path.getsize(path) > max_mb * 1e6:
            return None
        return _sha256(path)
    except OSError:
        return None


def provenance(path, gp='gp', prov=None, sha=None):
    """Return the provenance line: the verifier's hash, the deployment if any, gp's version, the date and the
    certificate's sha256.  prov and sha: values computed already, if any."""
    p = prov or _provenance(gp)
    if sha is None and prov is None:
        sha = _cert_sha(path)
    return 'Verifier %s%s, PARI/GP %s, %s. Certificate sha256 %s.' % (
        p['verifier'], ', deployment %s' % p['deploy'] if p['deploy'] != '-' else '', p['gp'], p['_date'], sha or '?')


def _name(path):
    return os.path.basename(str(path).rstrip('/')) or str(path)


def _reason(res, name, curve, dt='0 s'):
    """Return why the certificate failed, from the first failed check in res: (the reason for --json, the log line
    after 'FAIL: ')."""
    n, _, info = next((x for x in res if not x[1]), ('?', False, ''))
    if n == 'format':
        r = 'not a valid certificate: %s' % info.rstrip('.')
        return r, '%s is %s.' % (name, r)
    if n == 'curve':
        r = 'the certificate for %s is not for the curve %s' % (name, ','.join(map(str, curve)))
        return r, r + '.'
    if n == 'rejected':
        m = _gp_reason(info).rstrip('.')
        return 'failed check: rejected: %s' % m, '%s failed a check (%s): %s.' % (name, dt, m)
    return 'failed check: %s' % n, '%s failed the check "%s" (%s).' % (name, n, dt)


def verify(cdir, cores=1, gp='gp', quiet=False, log=print, curve=None, max_mb=None, max_seconds=None):
    """Check the certificate at cdir and return (all passed, [(check, passed, info)], the statement).  With curve
    (five a-invariants), the certificate must be for a curve isomorphic to it over Q.  Raises LimitExceeded over
    max_mb or max_seconds or for unbounded work, and RuntimeError or OSError when the check cannot run."""
    return _verify(cdir, cores, gp, quiet, log, curve, max_mb, max_seconds)[:3]


def _verify(cdir, cores=1, gp='gp', quiet=False, log=print, curve=None, max_mb=None, max_seconds=None, prov=None):
    """verify(), also returning the certificate as loaded (None if malformed).  prov: _provenance() with 'sha256', if
    computed already."""
    if type(cores) is not int or cores < 1:
        raise ValueError('cores must be a positive integer.')
    t0 = time.time()
    name = _name(cdir)
    if not quiet:
        log(provenance(cdir, gp, prov, prov['sha256'] if prov else None))
    try:
        return _verify1(cdir, cores, gp, quiet, log, curve, max_mb, max_seconds, t0)
    except TimeLimit as e:
        if max_seconds is None:
            raise LimitExceeded('%s was not checked within %.0f s' % (name, e.seconds))
        raise LimitExceeded('%s was not checked within %.0f s (--max-seconds)' % (name, max_seconds))
    except LimitExceeded as e:
        raise LimitExceeded(str(e).replace('{cert}', name))


def _verify1(cdir, cores, gp, quiet, log, curve, max_mb, max_seconds, t0):
    left = lambda: None if max_seconds is None else max_seconds - (time.time() - t0)
    jobs = cores
    res = []
    ok = lambda name, cond, info='': res.append((name, bool(cond), info))
    try:
        c = load(cdir, None if max_mb is None else int(max_mb * 1e6))
        rels = _clean(c)
    except (ValueError, KeyError, TypeError, AttributeError) as e:      # also UnicodeDecodeError, a ValueError
        res = [('format', False, str(e))]
        if not quiet:
            log('FAIL: ' + _reason(res, _name(cdir), curve)[1])
        return False, res, '', None
    _limits(c)
    ok('format', True)
    cur, fd, loc, cl = c['curve'], c['field'], c['local'], c['claims']
    rl = c.get('rank_lower_bound') or {}
    r = rl.get('r')
    pr = rl if 'points' in rl else None
    st = ''
    if curve is not None:
        same = same_curve([int(x) for x in curve], [int(x) for x in cur['ainvs']])
        ok('curve', same, 'the curve given' if same else 'not isomorphic to the curve given [%s]' % ','.join(map(str, curve)))
    try:
        if curve is not None and not same:
            raise Rejected('the certificate is for another curve')
        f = gp_poly(fd['poly'])
        body = 'vfield([%s], [%s], [%s], %s, %s, %d, %d, %d, %d, %d);\n' % (
            ','.join(cur['ainvs']), ','.join(cur['bad_primes']), ','.join(cur['minimal_model']), f, fd['dK'], fd['GB'],
            fd['ns'], loc['u'], loc['n'], loc['w'])
        if pr:
            body += 'vrank([%s], %d, [%s], [%s]);\n' % (','.join(cur['ainvs']), r,
                                                       ','.join('[%s,%s]' % tuple(p) for p in pr['points']), ','.join(map(str, pr['primes'])))
        out = run_gp(body, libs=('verify.gp',), seconds=24 * 3600 if left() is None else left(), gp=gp)
        for l in out.splitlines():
            t = l.split(' ', 3)
            if t[0] == 'CHK':
                ok(t[1], t[2] == '1', t[3] if len(t) > 3 else '')
        printed = {x[0] for x in res}
        for name in FIELD_CHECKS + (('rank_lower_bound',) if pr else ()):
            if name not in printed:
                ok(name, False, 'not printed by gp')
        t1 = time.time()
        rows, names = gp_rows(f, cur['bad_primes'], rels, jobs, gp=gp, seconds=left())
        ok('relations_checked', True, '%d rows recomputed by PARI (%.0f s, %d jobs)' % (len(rows), time.time() - t1, jobs))
        ops, piv = [tuple(o) for o in c['ops']], c['pivots']
        if not all(0 <= x < len(rows) for o in ops for x in o) or not all(0 <= x < len(rows) for x in piv):
            ok('ops_in_range', False, 'a row index of ops or pivots is out of range')
        else:
            ok('ops_in_range', True)
            t2 = time.time()
            dimV, nl, rank = rank_V(rows, names, ops, piv, fd['GB'], jobs, seconds=left())
            G = fd['ns'] - dimV
            ok('dimV', dimV == cl['dimV'], 'dim V = %d (rank %d, %d large columns, %.0f s)' % (dimV, rank, nl,
                                                                                              time.time() - t2))
            S, st = statement(G, loc, r, None if pr else rl.get('source'))
            ok('claims', G == cl['G'] and S == cl['S'], 'G = %d, rank <= %d' % (G, S))
            ok('statement', c['statement'] == st)
            if r is not None:
                ok('bound_ge_r', S >= r, 'rank <= %d, rank >= %d' % (S, r))
    # A failed gp check rejects the certificate; any other error propagates, with no verdict.
    except Rejected as e:
        ok('rejected', False, str(e)[:2000])
    if pr is None and r is not None:
        res.append(('rank_lower_bound', True, 'not proved here: rank >= %d is from %s' % (r, rl.get('source'))))
    allok = all(x[1] for x in res)
    if not quiet:
        dt = '%.0f s' % (time.time() - t0)
        if allok:
            log('PASS: %s (%s). %s' % (c['name'], dt, st))
        else:
            log('FAIL: ' + _reason(res, c['name'], curve, dt)[1])
    return allok, res, st, c


# ---------- the verdict (rankbound verify --json)

HYPOTHESIS = ('rank E(Q) <= rank_upper_bound assuming GRH for the Dedekind zeta functions of the unramified quadratic '
              'extensions of K, the cubic field of the 2-division polynomial of E (not GRH for L(E, s), not BSD); '
              'rank E(Q) >= rank_lower_bound unconditionally when r_proved')
EXIT = {'accepted': 0, 'rejected': 1, 'limit': 3, 'error': 4}      # exit statuses of the command line (2: a usage error)


def check(cert, curve=None, cores=1, gp='gp', max_mb=None, max_seconds=None, log=None):
    """Return the verdict on the certificate cert as a JSON-ready dict; never raises for anything in the file.

    verdict is 'accepted', 'rejected' (a check failed, or the certificate is not for curve), 'limit' (over max_mb,
    max_seconds or unbounded work: not checked) or 'error' (the verifier could not run), and reason says why.  An
    accepted certificate gives rank_upper_bound S (rank E(Q) <= S under HYPOTHESIS), G, u, n, w, rank_lower_bound,
    r_proved and rank (S when S = 0 or S = r is proved, else None).  log receives the log lines (None: quiet)."""
    t0 = time.time()
    v = {'verdict': None, 'reason': '', 'certificate': str(cert), 'sha256': None, 'name': None, 'curve': None,
         'rank_upper_bound': None, 'rank_lower_bound': None, 'r_proved': None, 'rank': None, 'G': None, 'u': None,
         'n': None, 'w': None, 'hypothesis': HYPOTHESIS, 'statement': None, 'checks': [], 'verifier': None,
         'deploy': None, 'gp': None, 'date': None, 'seconds': None}
    try:
        prov = _provenance(gp)
        v.update((k, prov[k]) for k in ('verifier', 'deploy', 'gp', 'date'))
        if max_mb is None or os.path.getsize(cert) <= max_mb * 1e6:
            v['sha256'] = _sha256(cert)
        prov['sha256'] = v['sha256']
        good, res, st, c = _verify(cert, cores, gp, log is None, log or (lambda *a: None), curve, max_mb, max_seconds,
                                   prov)
        v['checks'] = [{'check': n, 'ok': g, 'info': i} for n, g, i in res]
        if good:
            rl = c.get('rank_lower_bound') or {}
            S, r = c['claims']['S'], rl.get('r')
            v.update(verdict='accepted', name=c['name'], curve=c['curve']['minimal_model'], rank_upper_bound=S,
                     G=c['claims']['G'], u=c['local']['u'], n=c['local']['n'], w=c['local']['w'], rank_lower_bound=r,
                     r_proved=r is not None and ('points' in rl or r == 0), statement=st)
            v['rank'] = S if S == 0 or (v['r_proved'] and S == r) else None
        else:
            v.update(verdict='rejected', reason=_reason(res, c['name'] if c else _name(cert), curve)[0],
                     name=c['name'] if c else None)
    except LimitExceeded as e:
        v.update(verdict='limit', reason=str(e))
    except Exception as e:          # the verifier could not run: no verdict
        why = str(e)[:2000].strip().rstrip('.') if isinstance(e, RuntimeError) else '%s: %s' % (type(e).__name__,
                                                                                               str(e)[:2000])
        v.update(verdict='error', reason='the verifier could not run on %s (%s)' % (_name(cert), why))
    if log is not None and v['verdict'] in ('limit', 'error'):
        log('NOT CHECKED: %s.' % v['reason'])
    v['seconds'] = round(time.time() - t0, 1)
    return v
