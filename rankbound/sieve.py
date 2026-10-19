"""Sieving for relations: a trial sieve that picks the parameters, CADO-NFS's las, and a line siever for small
fields.  Both sievers write las's output format.  Nothing a siever prints is trusted: the verifier rechecks every
relation.

For lopsided fields the smooth values of F lie in thin strips a ~ rho b, rho a real root of F(x, 1), while las
centers its region at a = 0; so F(x + k, 1) with k = round(rho) is sieved too, and its relations map back by
a -> a + k b."""
import glob, gzip, logging, math, os, re, shutil, subprocess, sys, tempfile, time
from . import field as fld, read_json, write_json

LOG = logging.getLogger('rankbound').info
HERE = os.path.dirname(os.path.abspath(__file__))

KAPPA = 0.91            # relations needed ~ KAPPA * pi(2^lpb) - free relations
LPB_SLACK = 1.25        # the lowest lpb within 25% of the cheapest sieve time
DUP = 0.9               # production rate = trial rate * DUP (las -dup)
MEM_I = {11: 0.1, 12: 0.2, 13: 0.6, 14: 1.5}    # GB per las process at sieve size I
LPB_MAX = 29            # the largest lpb the trial tries
SUPPLY_MIN = 1.0        # special-q below 2^lpb must supply the relations
LINE_BELOW = 16         # the line siever below this trial lpb
LAS_LPB_MIN = 14        # las is not run below this lpb
LINE_A = 2 ** 11        # the line siever's a-range [-A, A)
SHIFT_SKEW = 26         # log2 of the largest skew of a shifted polynomial
LIM1 = 2 ** 20          # side 1's norm (< 2^60) has <= 2 primes above this


# ---------- CADO-NFS

class LasCrash(RuntimeError):
    """las aborted with "Error, p = N does not divide at (N,x)", a CADO-NFS bug with two large primes (mfb0 > lpb0).
    default_params avoids it while lim0 = 2^lpb; otherwise the trial lowers lim0, and a slice that hits it is
    skipped."""
    def __init__(self, msg, lim0=None):
        super().__init__(msg)
        self.lim0 = lim0


LAS_BUGS = ('does not divide', 'condition fij->deg == cpoly[side]->deg failed')


def las_crashed(returncode, err):
    """Return whether las aborted on a known CADO-NFS bug (LasCrash, or the degree assertion of dummy_root)."""
    return returncode != 0 and any(b in (err or '') for b in LAS_BUGS)


def clean_tmp(out):
    """Remove what an aborted las left (it writes out.tmp.PID and renames it when done)."""
    for p in glob.glob(out + '.tmp.*'):
        os.remove(p)


def cado_failed(prog, rc, err):
    """Return the message for a failed CADO-NFS program: its exit status and the last line of err."""
    last = next((l.strip() for l in reversed((err or '').splitlines()) if l.strip()), '')
    return 'CADO-NFS failed on this curve (%s, exit %d): %s' % (prog, rc, last)


OUT_OF_REACH = 'This curve is out of reach for now.'
TOO_FEW = 'The sieve finds too few relations for this curve, even at its largest setting. ' + OUT_OF_REACH


def binary(name):
    """Return the path of a CADO-NFS program: in $RANKBOUND_CADO, else in cado/ here (`make cado`), else on the PATH."""
    for d in (os.environ.get('RANKBOUND_CADO'), os.path.join(HERE, 'cado')):
        if d and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    p = shutil.which(name)
    if not p:
        raise RuntimeError("This curve needs CADO-NFS's %s, which is not installed. Run `make cado`, or set "
                           "RANKBOUND_CADO to its folder." % ('siever (las)' if name == 'las' else name))
    return p


def polys(F):
    """Return the candidate polynomials [{'name', 'shift', 'coeffs', 'skew'}]: F, and for lopsided fields (skew >= 2^16)
    F shifted to each real root, with its declared skew capped at 2^SHIFT_SKEW (las discards special-q far below the
    declared skew)."""
    sk = 2 ** F['log2_skew']
    out = [{'name': 'F', 'shift': 0, 'coeffs': F['poly'], 'skew': max(1, round(sk))}]
    if F['log2_skew'] >= 16:
        for i, k in enumerate(F['real_roots']):
            out.append({'name': 'F_r%d' % (i + 1), 'shift': k, 'coeffs': fld.shifted(F['poly'], k),
                        'skew': round(min(sk, 2 ** SHIFT_SKEW))})
    return out


def probable_prime(n):
    """Return whether n is a strong probable prime to the first 12 prime bases (exact below 3.2e23)."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if n in small:
        return True
    if any(n % p == 0 for p in small):
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def dummy_root(f, mmax=5000):
    """Return the root m of side 1 (x - m): the least |m| with |F(m, 1)| 2^10-smooth, or 2^10-smooth times one prime
    above 2^32, so that no special-q divides it (las aborts on such a special-q).  None if no |m| <= mmax qualifies."""
    for k in range(2 * mmax + 1):
        m = (k + 1) // 2 * (1 if k % 2 else -1)
        n = abs(fld.norm(f, m, 1))
        if n < 2:
            continue
        for p in range(2, 2 ** 10):
            while n % p == 0:
                n //= p
        if n == 1 or (n > 2 ** 32 and probable_prime(n)):
            return m
    return None


def write_poly(P, path):
    """Write the CADO-NFS polynomial file of P.  Side 1 is the dummy x - m, with m from dummy_root, else the least
    m >= 0 with |F(m, 1)| >= 2."""
    f = P['coeffs']
    m = dummy_root(f)
    if m is None:
        m = 0
        while abs(fld.norm(f, m, 1)) < 2:
            m += 1
    txt = 'n: %d\nskew: %d\npoly0: %s\npoly1: %d,1\n' % (abs(fld.norm(f, m, 1)), P['skew'], ','.join(str(c) for c in f), -m)
    if os.path.exists(path):
        with open(path) as fo:
            if fo.read() == txt:
                return
    tmp = '%s.tmp%d' % (path, os.getpid())
    with open(tmp, 'w') as fo:
        fo.write(txt)
    os.replace(tmp, path)


def read_poly(path):
    with open(path) as f:
        for l in f:
            if l.startswith('poly0:'):
                return [int(x) for x in l.split(':')[1].split(',')]
    raise ValueError('no poly0 in %s' % path)


def side1_root(path):
    """Return m of the dummy side 1 (x - m) of a polynomial file."""
    with open(path) as f:
        for l in f:
            if l.startswith('poly1:'):
                return -int(l.split(':')[1].split(',')[0])
    raise ValueError('no poly1 in %s' % path)


def makefb(poly_path, lim, out, side=0):
    """Build the factor base of side up to lim (makefb compresses only if the name ends in .gz)."""
    if os.path.exists(out):
        return out
    tmp = out[:-3] + '.tmp%d.gz' % os.getpid()
    r = subprocess.run([binary('makefb'), '-poly', poly_path, '-lim', str(lim), '-maxbits', '16', '-out', tmp,
                        '-side', str(side)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(cado_failed('makefb', r.returncode, r.stderr or r.stdout))
    if os.path.getsize(tmp) < 100:
        raise RuntimeError(cado_failed('makefb', 0, 'an empty factor base'))
    os.replace(tmp, out)
    return out


def las_cmd(poly_path, fb0, fb1, prm, q0, q1, out, sample=None, seed=1, threads=1, dupqmin=None):
    """Return the las command; prm has I, lim0, lpb0, mfb0 and optionally lambda0."""
    c = [binary('las'), '-poly', poly_path, '-fb0', fb0, '-fb1', fb1, '-I', str(prm['I']), '-q0', str(q0), '-q1', str(q1),
         '-sqside', '0', '-lim0', str(prm['lim0']), '-lim1', str(LIM1), '-lpb0', str(prm['lpb0']), '-lpb1', '60',
         '-mfb0', str(prm['mfb0']), '-mfb1', '60', '-t', str(threads), '-out', out, '-allow-largesq']
    if prm.get('lambda0'):
        c += ['-lambda0', str(prm['lambda0'])]
    if sample:
        c += ['-random-sample', str(sample), '-seed', str(seed)]
    if dupqmin is not None:
        c += ['-dup', '-dup-qmin', '%d,0' % dupqmin]
    return c


def fb1_name(poly, m):
    """Return the file name of side 1's factor base, which depends on m."""
    return '%s.fb1.gz' % poly if m == 0 else '%s.fb1_m%d.gz' % (poly, m)


class Las:
    """The las siever, with factor bases from makefb."""

    def prepare(self, state, prm, poly):
        """Return (the polynomial file, the side-0 factor base, the side-1 factor base)."""
        pp = os.path.join(state, poly + '.poly')
        return (pp, makefb(pp, prm['lim0'], os.path.join(state, '%s.fb0_%d.gz' % (poly, prm['lim0']))),
                makefb(pp, LIM1, os.path.join(state, fb1_name(poly, side1_root(pp))), side=1))

    def start_slice(self, state, prm, poly, q0, q1, out):
        """Start sieving the special-q [q0, q1) of poly into out, and return the process."""
        pp, fb0, fb1 = self.prepare(state, prm, poly)
        return spawn(las_cmd(pp, fb0, fb1, prm, q0, q1, out, dupqmin=prm['q0']))


def spawn(cmd, **kw):
    """Start a slice's process in its own session.  Its stderr goes to an anonymous file, which p.stderr reads from
    the start; an unread pipe could block the process."""
    err = tempfile.TemporaryFile()
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err, start_new_session=True, **kw)
    except BaseException:
        err.close()
        raise
    p.stderr = err
    return p


# ---------- the line siever (small fields)

def primes(n):
    s = bytearray([1]) * n
    s[:2] = b'\x00\x00'
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(range(i * i, n, i)))
    return [i for i in range(n) if s[i]]


def roots(f, p):
    """Return the distinct roots of F mod p in P^1(F_p), with r = p for (1:0)."""
    c0, c1, c2, c3 = (c % p for c in f)
    rs = [r for r in range(p) if (((c3 * r + c2) * r + c1) * r + c0) % p == 0]
    if c3 == 0:
        rs.append(p)
    return rs


def write_roots(f, pmax, out):
    """Write the roots of F mod p for p < pmax in makefb's format, for the free relations."""
    tmp = '%s.tmp%d.gz' % (out, os.getpid())
    with gzip.open(tmp, 'wt') as fo:
        fo.write('# roots of %s (rankbound line siever)\n' % f)
        for p in primes(pmax):
            rs = roots(f, p)
            if rs:
                fo.write('%d: %s\n' % (p, ','.join(map(str, rs))))
    os.replace(tmp, out)


def line_sieve(f, lpb, A, b0, b1, out):
    """Sieve F(a, b) for b in [b0, b1) and a in [-A, A) by every prime below 2^lpb, trial-divide the candidates, and
    write las's format to out; return the number of relations."""
    t0 = time.process_time()
    P = primes(2 ** lpb)
    fb = [(p, roots(f, p), math.log2(p)) for p in P]
    # With gcd(a, b) = 1, only a prime with a root of F divides F(a, b).
    div = [p for p, rs, _ in fb if rs]
    c0, c1, c2, c3 = f
    n = 0
    with open(out, 'w') as fo:
        for b in range(b0, b1):
            logs = [0.0] * (2 * A)
            for p, rs, lp in fb:
                for r in rs:
                    if r == p:
                        if b % p == 0:
                            for i in range(2 * A):
                                logs[i] += lp
                        continue
                    s = (r * b + A) % p          # a = -A + i with a = r b mod p
                    for i in range(s, 2 * A, p):
                        logs[i] += lp
            for i in range(2 * A):
                a = i - A
                if math.gcd(a, b) != 1:
                    continue
                v = ((c3 * a + c2 * b) * a + c1 * b * b) * a + c0 * b ** 3
                if v == 0:
                    continue
                if logs[i] < math.log2(abs(v)) - 2 * lpb:
                    continue
                m, H = abs(v), []
                for p in div:
                    while m % p == 0:
                        m //= p; H.append(p)
                    if m == 1:
                        break
                if m == 1:
                    fo.write('%d,%d:%s:\n' % (a, b, ','.join('%x' % h for h in H))); n += 1
        cpu = time.process_time() - t0
        fo.write('# Total cpu time %.2fs\n# Total %d reports [%.3gs/r, %.1fr/sq]\n' % (cpu, n, cpu / max(n, 1), n / max(1, b1 - b0)))
    return n


LINE_CMD = 'import sys, rankbound.sieve as s; s.line_sieve(s.read_poly(sys.argv[1]), *map(int, sys.argv[2:6]), sys.argv[6])'


class Line:
    """The line siever, run as a subprocess on the lines b in [q0, q1)."""

    def prepare(self, state, prm, poly):
        pp = os.path.join(state, poly + '.poly')
        fb0 = os.path.join(state, 'F.roots_%d.gz' % prm['lim0'])
        if not os.path.exists(fb0):
            write_roots(read_poly(pp), prm['lim0'], fb0)
        return pp, fb0, None

    def start_slice(self, state, prm, poly, q0, q1, out):
        pp, _, _ = self.prepare(state, prm, poly)
        return spawn([sys.executable, '-c', LINE_CMD, pp, str(prm['lpb0']), str(prm['A']), str(q0), str(q1), out],
                     cwd=os.path.dirname(HERE))      # so the package imports when not installed


class Given:
    """No siever: the relations were given.  The free relations come from the primes where F splits completely
    (field.write_split): those below 2^lpb0, and the larger primes of the given relations."""

    def prepare(self, state, prm, poly):
        fb0 = os.path.join(state, 'F.split_%d.gz' % prm['lim0'])
        if not os.path.exists(fb0):
            with gzip.open(os.path.join(state, 'slices', 'given.rels.gz'), 'rt') as f:
                big = {p for l in f for p in parse_line(l)[2] if p >= prm['lim0']}
            fld.write_split(fld.load(state)['poly'], prm['lim0'], fb0, extra=big)
        return os.path.join(state, poly + '.poly'), fb0, None

    def start_slice(self, state, prm, poly, q0, q1, out):
        raise RuntimeError('This folder was started from given relations; to sieve, start a new folder with '
                           '--keep-sieving.')


GIVEN_LPB_MAX = 26      # the largest lpb0 of a run on given relations


def given_params(F, maxprime):
    """Return params.json for a run on given relations: lpb0 covers GB and the relations' primes, up to
    GIVEN_LPB_MAX."""
    lpb = max(math.ceil(math.log2(F['GB'])), min(max(maxprime, 2).bit_length(), GIVEN_LPB_MAX))
    return {'siever': 'given', 'lpb0': lpb, 'mfb0': lpb, 'lim0': 2 ** lpb, 'I': 0, 'q0': 1,
            'polys': {'F': {'shift': 0, 'skew': 1, 'weight': 1.0}},
            'trial': {'cpu_s': 0.0},
            'model': {'relations_needed': 0, 'rate': 0.0, 'sec_per_q': 0.0, 'rels_per_q': 0.0, 'core_hours': 0.0}}


def siever(prm):
    """Return the siever of a parameter set: 'line', 'given' (none) or, by default, las."""
    return {'line': Line, 'given': Given}.get(prm.get('siever'), Las)()


# ---------- slices: las's output format

STAT = {'cpu': re.compile(r'# Total cpu time ([\d.]+)s'), 'reports': re.compile(r'# Total (\d+) reports'),
        'nq': re.compile(r'# Total \d+ reports \[[\d.e+-]+s/r, ([\d.]+)r/sq\]'), 'mem': re.compile(r'PeakMemusage \(MB\) = (\d+)')}


def stats(path):
    """Return the CPU seconds, reports, special-q and peak MB of a las output file, and whether las discarded every
    special-q."""
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        txt = ''.join(l for l in f if l.startswith('#'))
    g = {k: (float(m.group(1)) if m else None) for k, m in ((k, r.search(txt)) for k, r in STAT.items())}
    nq = txt.count('# Now sieving side-0 q')
    g['discarded'] = nq > 0 and txt.count('# Discarding side-0 q') == nq
    if g['reports'] and g['nq']:
        g['nq'] = round(g['reports'] / g['nq'])
    return g


def relations(path, shift):
    """Yield (a, b, [primes of F(a, b)]) from a las output file, with a mapped back to F's coordinates."""
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        for l in f:
            if l[0] == '#' or ':' not in l:
                continue
            ab, s0 = l.split(':', 2)[:2]
            a, b = ab.split(',')
            b = int(b)
            yield int(a) + shift * b, b, [int(h, 16) for h in s0.split(',')] if s0 else []


def parse_line(line):
    """Parse a slice line 'a,b:hexprimes' of the prover's own files into (a, b, [primes of F(a, b)])."""
    ab, s0 = line.split(':', 1)
    a, b = ab.split(',')
    return int(a), int(b), [int(h, 16) for h in s0.split(',')] if s0.strip() else []


def finish_slice(sdir, name, meta, shift, out):
    """Turn the slice output out into sdir/name.rels.gz, in F's coordinates, and name.json, with meta and the
    statistics; return (CPU seconds, relations).  name.rels.gz appears only when complete."""
    g = stats(out)
    n = 0
    tmp = os.path.join(sdir, name + '.tmp.gz')
    with gzip.open(tmp, 'wt', compresslevel=4) as f:
        for a, b, H in relations(out, shift):
            f.write('%d,%d:%s\n' % (a, b, ','.join('%x' % h for h in H))); n += 1
    fd = os.open(tmp, os.O_RDONLY)      # synced before the rename: no partial slice
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    write_json(os.path.join(sdir, name + '.json'), dict(meta, cpu=g['cpu'], relations=n, special_q=g['nq']), None)
    os.replace(tmp, os.path.join(sdir, name + '.rels.gz'))
    os.remove(out)
    return g['cpu'] or 0.0, n


# ---------- the trial sieve and the cost model

def li(x):
    l = math.log(x)
    return x / l * (1 + 1 / l + 2 / l ** 2 + 6 / l ** 3)


def need(lpb):
    return KAPPA * li(2 ** lpb) - li(2 ** lpb) / 6


def default_params(lpb, I=13, lim_cap=None):
    """Return the default parameters at lpb.  With lim0 = 2^lpb every prime below 2^lpb is in the factor base, so
    mfb0 = lpb (a larger mfb0 only costs time, and hits LasCrash); with lim0 < 2^lpb, mfb0 = 2 lpb."""
    lim0 = min(2 ** min(lpb, 26), lim_cap or 2 ** 64)
    return {'lpb0': lpb, 'mfb0': lpb if lim0 >= 2 ** lpb else 2 * lpb, 'lim0': lim0, 'I': I, 'q0': 2 ** (lpb - 1)}


class Trial:
    def __init__(self, state, F, log=LOG):
        self.dir = os.path.join(state, 'trial')
        os.makedirs(self.dir, exist_ok=True)
        self.F, self.log, self.cpu = F, log, 0.0
        self.polys = {P['name']: P for P in polys(F)}
        for P in self.polys.values():
            write_poly(P, os.path.join(state, P['name'] + '.poly'))
        self.state = state

    def rate(self, name, prm, nq=10):
        """Return the relations per CPU second at nq random special-q in [q0, 2 q0), or at 4 nq if that takes under
        3 s or finds under 50 relations."""
        pp, fb0, fb1 = Las().prepare(self.state, prm, name)
        out = os.path.join(self.dir, '%s_lpb%d_I%d.out' % (name, prm['lpb0'], prm['I']))
        for n in (nq, 4 * nq):
            r = subprocess.run(las_cmd(pp, fb0, fb1, prm, prm['q0'], 2 * prm['q0'], out, sample=n), capture_output=True,
                               text=True)
            if r.returncode:
                clean_tmp(out)
                msg = cado_failed('las', r.returncode, r.stderr)
                # LasCrash makes tune lower lim0; anything else is an error.
                raise LasCrash(msg, prm['lim0']) if LAS_BUGS[0] in r.stderr else RuntimeError(msg)
            g = stats(out)
            self.cpu += g['cpu']
            if g['cpu'] >= 3 and (g['reports'] or 0) >= 50:
                break
        nq = g['nq'] or n                 # las prints no per-q rate without relations
        r = (g['reports'] or 0) / max(g['cpu'], 1e-3)
        return {'rate': r, 'sec_per_q': g['cpu'] / nq, 'rels_per_q': (g['reports'] or 0) / nq, 'mem_MB': g['mem'],
                'discarded': g['discarded']}


def tune(state, F, mem_gb=1.0, log=LOG, lpb_min=None):
    """Run the trial sieve and write STATE/params.json, once.  lpb_min: the least lpb, after a sieve that dried up
    (run.Job.retune)."""
    path = os.path.join(state, 'params.json')
    if os.path.exists(path):
        return read_json(path)
    t0 = time.time()
    lpbA = math.ceil(math.log2(F['GB'])) + 4
    if lpbA < LINE_BELOW and not lpb_min:
        prm = line_params(state, F, log)
    else:
        cap = None
        if lpb_min:
            lpb_min = max(lpb_min, LAS_LPB_MIN)
        while True:
            try:
                prm = las_params(state, F, lpbA, mem_gb, log, cap, lpb_min)
                break
            except LasCrash as e:     # again with lim0 halved, never below GB
                cap = e.lim0 // 2
                if cap < F['GB']:
                    raise RuntimeError(str(e))
    prm['trial']['wall_s'] = round(time.time() - t0, 1)
    write_json(path, prm)
    return prm


def las_params(state, F, lpbA, mem_gb, log, lim_cap=None, lpb_min=None):
    """Return the parameters from the trial: (A) every polynomial at lpbA, raised by 2 until one finds relations;
    (B) on the best one, lpb in {lpbA - 2, lpbA, lpbA + 2}; (C) on that, I in {12, 13, 14}.  The chosen lpb is the
    lowest whose special-q can supply the relations and whose sieve time is within LPB_SLACK of the best."""
    if lpb_min:
        lpbA = max(lpbA, lpb_min)
    T = Trial(state, F, log)
    while True:
        A = {n: T.rate(n, default_params(lpbA, lim_cap=lim_cap)) for n in T.polys}
        best = max(A, key=lambda n: A[n]['rate'])
        if A[best]['rate'] > 0:
            break
        if all(a.get('discarded') for a in A.values()):
            raise RuntimeError("CADO-NFS's siever cannot handle this curve's cubic field. " + OUT_OF_REACH)
        if lpbA + 2 > LPB_MAX:
            raise RuntimeError(TOO_FEW)
        lpbA += 2
    B = {lpbA: A[best]}
    for lpb in (lpbA - 2, lpbA + 2):
        if 2 ** lpb > F['GB'] * 2 and lpb >= LAS_LPB_MIN:
            B[lpb] = T.rate(best, default_params(lpb, lim_cap=lim_cap))
    # An lpb whose trial found no relation is no candidate.
    supply = lambda l: B[l]['rels_per_q'] * li(2 ** l) / need(l)
    ok = lambda l: B[l]['rate'] > 0 and supply(l) >= SUPPLY_MIN and l >= (lpb_min or 0)
    while not any(ok(l) for l in B):
        top = max(B) + 2
        if top > LPB_MAX:
            raise RuntimeError(TOO_FEW)
        B[top] = T.rate(best, default_params(top, lim_cap=lim_cap))
    cost = {lpb: need(lpb) / (B[lpb]['rate'] * DUP) for lpb in B if ok(lpb)}
    # The lowest lpb within LPB_SLACK of the cheapest: fewer relations make a smaller matrix.
    lpb = min(l for l in cost if cost[l] <= LPB_SLACK * min(cost.values()))
    C = {13: B[lpb]}
    for I in (12, 14):
        if MEM_I[I] <= mem_gb:
            C[I] = T.rate(best, default_params(lpb, I, lim_cap))
    I = max(C, key=lambda i: C[i]['rate'])
    # The polynomials within half of the best rate share the sieving.
    share = {n: A[n]['rate'] / A[best]['rate'] for n in A if A[n]['rate'] >= 0.5 * A[best]['rate']}
    prm = default_params(lpb, I, lim_cap)
    rate = C[I]['rate'] * sum(share.values()) / len(share)
    R = need(lpb)
    # Seconds per special-q of each polynomial, scaled from stage A.
    scale = C[I]['sec_per_q'] / A[best]['sec_per_q']
    prm.update({
        'polys': {n: {'shift': T.polys[n]['shift'], 'skew': T.polys[n]['skew'], 'weight': round(w, 3),
                      'sec_per_q': A[n]['sec_per_q'] * scale} for n, w in share.items()},
        'trial': {'lpb_A': lpbA, 'lpb_min': lpb_min, 'lim_cap': lim_cap, 'A': A, 'B': {str(k): v for k, v in B.items()}, 'C': {str(k): v for k, v in C.items()},
                  'cpu_s': round(T.cpu, 1)},
        'model': {'kappa': KAPPA, 'dup': DUP, 'relations_needed': round(R), 'rate': round(rate * DUP, 1),
                  'sec_per_q': C[I]['sec_per_q'], 'rels_per_q': C[I]['rels_per_q'], 'supply': round(supply(lpb), 2),
                  'core_hours': round(R / (rate * DUP) / 3600, 3)},
    })
    return prm


def line_params(state, F, log):
    """Return the line siever's parameters for a small field: lpb = ceil(log2 GB) + 3, at least 8."""
    lpb = max(8, math.ceil(math.log2(F['GB'])) + 3)
    P = polys(F)[0]
    write_poly(P, os.path.join(state, 'F.poly'))
    d = os.path.join(state, 'trial'); os.makedirs(d, exist_ok=True)
    t = time.process_time(); n = line_sieve(F['poly'], lpb, LINE_A, 1, 5, os.path.join(d, 'line.out'))
    cpu = max(time.process_time() - t, 1e-3)
    rate = n / cpu
    R = max(need(lpb), 1.2 * F['ns'] + 50)
    return {'siever': 'line', 'lpb0': lpb, 'mfb0': 2 * lpb, 'lim0': 2 ** lpb, 'I': 0, 'q0': 1, 'A': LINE_A,
            'polys': {'F': {'shift': 0, 'skew': P['skew'], 'weight': 1.0}},
            'trial': {'cpu_s': round(cpu, 2), 'line_rate': rate},
            'model': {'kappa': KAPPA, 'dup': 1.0, 'relations_needed': round(R), 'rate': round(rate, 1),
                      'sec_per_q': cpu / 4, 'rels_per_q': n / 4, 'core_hours': round(R / max(rate, 1e-3) / 3600, 4)}}


def estimate_lines(F, prm, cores):
    """Return the cost estimate as log lines."""
    ch = prm['model']['core_hours']
    return ['Estimated cost: %.2f core-hours (%.2f h on %d core%s).' % (ch, ch / cores, cores, 's' if cores > 1 else '')]
