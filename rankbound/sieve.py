"""Relations from the native siever for the route SIEVE: slices, their schedule, and the checks.

Nothing the siever prints is trusted; the checker rechecks every relation. Every stage is an artifact of the work
folder, so a run resumes, and shards (--shard K/N) sieve shares of one folder.
"""
from types import SimpleNamespace
import contextlib, copy, dataclasses, glob, gzip, hashlib, json, lzma, math, os, re, shutil, signal, \
    subprocess, tempfile, threading, time, zlib
from collections import namedtuple
from . import arith, field as fld, linalg
from .core import FORMAT4, FORMAT5, FORMATS3, Locked, MAX_DIGITS, RankboundError, Stopped, Usage
from .runtime import DETAIL, read_json, say, write_json
from .messages import core_hours, duration, plural, text
from .witness import FORMAT6
from . import runtime

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- the siever

KAPPA = 0.91            # relations needed ~ KAPPA * pi(2^lpb) - free relations
LPB_SLACK = 1.25        # the lowest lpb within 25% of the cheapest sieve time
DUP = 0.9               # production rate = trial rate * DUP (duplicates across special-q)
LPB_MIN = 14            # the least lpb
LPB_MAX = 29            # the largest lpb the trial tries
COEF_BITS = 447         # sieve.c's big has 8 limbs and parses below 2^448
SUPPLY_MIN = 1.0        # special-q below 2^lpb must supply the relations
N_TRIAL = (21, 23, 25)  # the log2 positions per special-q that the trial compares


def roots(f, p):
    """Return the distinct roots of F mod p in P^1(F_p), with r = p for (1:0)."""
    c0, c1, c2, c3 = (c % p for c in f)
    rs = [r for r in range(p) if (((c3 * r + c2) * r + c1) * r + c0) % p == 0]
    if c3 == 0:
        rs.append(p)
    return rs


def write_poly(F, path):
    """Write the polynomial file of the field F (c0..c3 and skew, in CADO's format), once."""
    txt = 'skew: %d\npoly0: %s\n' % (max(1, round(2 ** F['log2_skew'])), ','.join(str(c) for c in F['poly']))
    if os.path.exists(path):
        with open(path) as fo:
            if fo.read() == txt:
                return
    tmp = '%s.tmp%d' % (path, os.getpid())
    with open(tmp, 'w') as fo:
        fo.write(txt)
    os.replace(tmp, path)


def factor_base(state, poly, lim):
    """Return the gzipped factor base of poly up to lim in makefb's format (projective roots as r = p)."""
    roots = os.path.join(state, '%s.lsroots_%d' % (poly, lim))
    if not os.path.exists(roots):
        tmp = '%s.tmp%d' % (roots, os.getpid())
        rc, _, err = runtime.run([BIN, 'roots', poly_file(os.path.join(state, poly + '.poly'))[0], str(lim), tmp])
        if rc:
            raise RuntimeError('sieve roots failed: %s' % err.strip())
        os.replace(tmp, roots)
    fb0 = roots + '.gz'
    if not os.path.exists(fb0):
        tmp = '%s.tmp%d.gz' % (roots, os.getpid())
        with open(roots, 'rb') as f, gzip.open(tmp, 'wb', compresslevel=1) as g:
            shutil.copyfileobj(f, g)
        os.replace(tmp, fb0)
    return fb0


BIN = os.path.join(HERE, 'sieve')


def poly_file(pp):
    """Return (the coefficients c0..c3, the skew) of a polynomial file."""
    with open(pp) as f:
        d = dict(l.split(':', 1) for l in f if ':' in l)
    return d['poly0'].strip(), d.get('skew', '1').strip()


def cmd(pp, roots, prm):
    """Return the siever's command; N is log2 of the positions per special-q (las's I gives N = 2 I - 1)."""
    c, skew = poly_file(pp)
    return [BIN, 'sieve', '-poly', c, '-skew', skew, '-roots', roots, '-lim', str(prm['lim0']), '-lpb', str(prm['lpb0']),
            '-mfb', str(prm['mfb0']), '-N', str(prm['N'] if 'N' in prm else 2 * prm['I'] - 1)]


class LSieve:
    """The native siever: long-lived processes, one per core, each reading slices "Q0 Q1" on stdin."""

    def __init__(self):
        self.servers = []
        self.wake = threading.Event()

    def prepare(self, state, prm, poly):
        """Return (the polynomial file, the gzipped factor base for the free relations, None)."""
        return os.path.join(state, poly + '.poly'), factor_base(state, poly, prm['lim0']), None

    def start_slice(self, state, prm, poly, q0, q1):
        """Start sieving [q0, q1) of poly on an idle process, or a new one, and return the Slice."""
        pp, fb0, _ = self.prepare(state, prm, poly)
        for s in self.servers:
            if s.p.poll() is not None:
                runtime.reap(s.p)
        self.servers = [s for s in self.servers if s.p.poll() is None]
        sv = next((s for s in self.servers if s.poly == poly and s.cur is None), None)
        if sv is None:
            other = next((s for s in self.servers if s.cur is None), None)
            if other is not None:
                other.close(); self.servers.remove(other)
            sv = Server(cmd(pp, fb0[:-3], prm) + ['-server'], poly, self.wake)
            self.servers.append(sv)
        return sv.sieve(q0, q1)

    def wait(self):
        self.wake.wait(1.0)
        self.wake.clear()

    def close(self):
        for s in self.servers:
            s.close()
        self.servers = []


class Server:
    """One siever process: slices go in on stdin, one at a time."""

    def __init__(self, c, poly, wake):
        self.poly, self.wake, self.cur = poly, wake, None
        self.p = spawn(c, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def sieve(self, q0, q1):
        s = self.cur = Slice(self)
        try:
            self.p.stdin.write(b'%d %d\n' % (q0, q1)); self.p.stdin.flush()
        except (OSError, ValueError):       # the process has exited; the reader reports it
            pass
        return s

    def _read(self):
        try:
            for l in self.p.stdout:
                s = self.cur
                if s is None:
                    continue
                if l.startswith(b'# done'):
                    s.returncode, self.cur = 0, None
                    self.wake.set()
                else:
                    s.out.append(l)
                    s.found += not l.startswith(b'#')
        except (OSError, ValueError):       # closed by close(); the slice in progress is dropped
            pass
        self.p.wait()
        s = self.cur
        if s is not None:
            s.returncode, self.cur = self.p.returncode or 1, None
        self.wake.set()

    def close(self):
        """Stop and reap the process and its group; the reader drains stdout first."""
        runtime.terminate(self.p)
        self.reader.join(10)
        for f in (self.p.stdin, self.p.stdout, self.p.stderr):
            try:
                f.close()
            except OSError:
                pass


class Slice:
    """A slice on a Server, with a process-like interface."""

    def __init__(self, server):
        self.server, self.pid, self.out, self.returncode = server, server.p.pid, [], None
        self.found = 0          # relation lines so far

    def poll(self):
        return self.returncode

    def error(self):
        self.server.p.stderr.seek(0)
        return self.server.p.stderr.read().decode(errors='replace')

    def lines(self):
        return (l.decode() for l in self.out)

    def release(self):
        self.out = []

    def kill(self):
        if self.returncode is None:
            self.server.close()
        self.release()

    def stop(self):
        """Ask the siever (SIGUSR1) to end the slice before its next special-q, continuing it if paused."""
        if self.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(self.pid, signal.SIGUSR1)
                os.kill(self.pid, signal.SIGCONT)

    def end(self, q1):
        """Return the end of the special-q range sieved: q1, or where a stop ended it."""
        for l in reversed(self.out):
            if l.startswith(STOPPED):
                return int(l[len(STOPPED):])
        return q1


STOPPED = b'# stopped before special-q '        # the siever's line for a slice ended by stop()
SAMPLE = re.compile(r'# special-q sample: (\d+) pairs .*?: (\d+) (\d+) (\d+)\), (\d+) repeats drawn again, (\d+) draws')


def trial(state, prm, poly, nq=10, seed=1):
    """Return the trial's rates at nq special-q pairs (q, r) drawn without replacement from [q0, 2 q0)."""
    pp, fb0, _ = LSieve().prepare(state, prm, poly)
    rc, out, err = runtime.run(cmd(pp, fb0[:-3], prm) + ['-q0', str(prm['q0']), '-q1', str(2 * prm['q0']), '-sample',
                                                       str(nq), '-seed', str(seed)])
    if rc:
        raise RuntimeError('sieve failed: %s' % err.strip())
    g = stats_of(''.join(l + '\n' for l in out.splitlines() if l.startswith('#')))
    m = SAMPLE.search(out)
    if not m:
        raise RuntimeError('sieve printed no special-q sample line')
    pairs, r1, r2, r3, rep, _ = map(int, m.groups())
    n, cpu, k = g['reports'] or 0, max(g['cpu'] or 0, 1e-3), max(pairs, 1)
    return {'rate': n / cpu, 'sec_per_q': cpu / k, 'rels_per_q': n / k, 'pairs': pairs, 'roots': [r1, r2, r3],
            'repeats': rep, 'seed': seed}


class NoSiever:
    """The siever stand-in for given relations; free relations come from the primes where F splits."""

    def prepare(self, state, prm, poly):
        fb0 = os.path.join(state, 'F.split_%d.gz' % prm['lim0'])
        if not os.path.exists(fb0):
            with gzip.open(os.path.join(state, 'slices', 'given.rels.gz'), 'rt') as f:
                big = {p for l in f for p in linalg.parse_line(l)[2] if p >= prm['lim0']}
            fld.write_split(fld.load(state)['poly'], prm['lim0'], fb0, extra=big)
        return os.path.join(state, poly + '.poly'), fb0, None

    def start_slice(self, state, prm, poly, q0, q1):
        raise RuntimeError('This work folder holds given relations only.')

    def wait(self):
        time.sleep(1)

    def close(self):
        pass


GIVEN_LPB_MAX = 26      # the largest lpb0 of a run on given relations


def given_params(F, maxprime):
    """Return params.json for given relations: lpb0 covers GB and their primes, up to GIVEN_LPB_MAX."""
    lpb = max(math.ceil(math.log2(F['GB'])), min(max(maxprime, 2).bit_length(), GIVEN_LPB_MAX))
    return {'siever': 'given', 'lpb0': lpb, 'mfb0': lpb, 'lim0': 2 ** lpb, 'N': 0, 'q0': 1,
            'polys': {'F': {'shift': 0, 'skew': 1, 'weight': 1.0}},
            'trial': {'cpu_s': 0.0},
            'model': {'relations_needed': 0, 'rate': 0.0, 'sec_per_q': 0.0, 'rels_per_q': 0.0, 'core_hours': 0.0}}


def siever(prm):
    """Return the siever of a parameter set: NoSiever for given relations, else LSieve."""
    return NoSiever() if prm.get('siever') == 'given' else LSieve()


def signal_group(p, sig):
    """Signal a slice's process group, skipping a process that has exited (macOS gives EPERM until it is reaped)."""
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def spawn(cmd, stdout=subprocess.DEVNULL, **kw):
    """Start a siever process in its own group; stderr goes to a file, since an unread pipe could block it."""
    err = tempfile.TemporaryFile()
    kw.setdefault('env', dict(os.environ, RANKBOUND_PARENT=str(os.getpid())))    # so it dies with this process
    try:
        p = runtime.start(cmd, stdout=stdout, stderr=err, **kw)
    except BaseException:
        err.close()
        raise
    p.stderr = err
    return p


# ---------- slices: las's output format

STAT = {'cpu': re.compile(r'# Total cpu time ([\d.]+)s'), 'reports': re.compile(r'# Total (\d+) reports'),
        'nq': re.compile(r'# Total \d+ reports \[[\d.e+-]+s/r, ([\d.]+)r/sq\]')}


def stats_of(txt):
    """Return the CPU seconds, reports and special-q of a slice from the comment lines txt of its output."""
    g = {k: (float(m.group(1)) if m else None) for k, m in ((k, r.search(txt)) for k, r in STAT.items())}
    g['nq'] = round(g['reports'] / g['nq']) if g['reports'] and g['nq'] else None
    return g


def relations(lines):
    """Yield (a, b, [primes of F(a, b)]) from lines of las's output."""
    for l in lines:
        if l[0] == '#' or ':' not in l:
            continue
        ab, s0 = l.split(':', 2)[:2]
        a, b = ab.split(',')
        yield int(a), int(b), [int(h, 16) for h in s0.split(',')] if s0 else []


def finish_slice(sdir, name, meta, lines, fp=None):
    """Commit a finished slice's lines, streamed to name.rels.gz, and return (CPU seconds, relations)."""
    n, com = 0, []
    with runtime.transaction(sdir, name, SLICE_SCHEMA, fp if fp is not None else field_fingerprint(
            os.path.dirname(os.path.abspath(sdir)))) as tx:
        with gzip.open(tx.path(name + '.rels.gz'), 'wt', compresslevel=1) as f:
            for l in lines:
                if l[0] == '#':
                    com.append(l)
                    continue
                t = l.split(':', 2)
                if len(t) < 2:
                    continue
                f.write('%s:%s\n' % (t[0], t[1].rstrip('\n'))); n += 1
        g = stats_of(''.join(com))
        tx.write(name + '.json', json.dumps(dict(meta, cpu=g['cpu'], relations=n, special_q=g['nq'])))
        tx.counts['relations'] = n
    return g['cpu'] or 0.0, n


# ---------- the trial sieve and the cost model

def li(x):
    l = math.log(x)
    return x / l * (1 + 1 / l + 2 / l ** 2 + 6 / l ** 3)


def need(lpb):
    return KAPPA * li(2 ** lpb) - li(2 ** lpb) / 6


def default_params(lpb, N=23):
    """Return the default parameters at lpb (mfb0 = lpb when lim0 = 2^lpb, else 2 lpb)."""
    lim0 = 2 ** min(lpb, 26)
    return {'lpb0': lpb, 'mfb0': lpb if lim0 >= 2 ** lpb else 2 * lpb, 'lim0': lim0, 'N': N, 'q0': 2 ** (lpb - 1)}


def proc_gb(prm):
    """Return the memory (GB) of a siever process: its buckets and 16 special-q roots per factor-base prime."""
    return (1.3 * 2 ** (prm['N'] + 2) + 100 * prm['lim0'] / math.log(prm['lim0'])) / 2 ** 30


class Trial:
    def __init__(self, state, F):
        self.F, self.cpu, self.state = F, 0.0, state
        write_poly(F, os.path.join(state, 'F.poly'))

    def rate(self, prm, nq=10):
        """Return the trial's rates at nq pairs, or at 4 nq if that is under 3 s or 50 relations."""
        for n in (nq, 4 * nq):
            t = trial(self.state, prm, 'F', n)
            self.cpu += t['sec_per_q'] * max(t['pairs'], 1)
            if t['sec_per_q'] * t['pairs'] >= 3 and t['rels_per_q'] * t['pairs'] >= 50:
                break
        return t


# ---------- the parameter table

# Rows: log10 |d_K| up to; relations and CPU seconds per special-q there, for the slice size and the cost estimate
# only; N.  Measured on fields at |d_K| 1e20-1e93: 2^17 positions per special-q sieve fastest below 1e30, 2^21
# above; lpb = ceil(log2 GB) + 2 from a trial at 1e44-1e77.
TABLE = ((14, 5000, 0.004, 17), (30, 1500, 0.003, 17), (48, 2000, 0.010, 21), (56, 1150, 0.010, 21), (62, 500, 0.010, 21), (70, 180, 0.009, 21),
         (80, 70, 0.009, 21))
TABLE_MIN = 0           # the table covers log10 |d_K| from here to its last row,
TABLE_SKEW = 30         # and log2 skew up to this; elsewhere the trial sieve
TABLE_LPB = 2           # lpb = ceil(log2 GB) + TABLE_LPB


def model(lpb, rq, sq):
    """Return the cost model at rq relations and sq CPU seconds per special-q."""
    R, rate = need(lpb), rq / sq * DUP
    return {'kappa': KAPPA, 'dup': DUP, 'relations_needed': round(R), 'rate': round(rate, 1), 'sec_per_q': sq,
            'rels_per_q': rq, 'core_hours': round(R / max(rate, 1e-9) / 3600, 4)}


SIEVE_MORE = 2.0        # measured ratio of the sieve's CPU to the last check over the model's time (1.05 to 2.2),
                        # from duplicates, a first bound above the relations needed, and slices in flight


def sieve_estimate(prm):
    """Return the estimated sieve core-hours: the model's time to the relations needed, times SIEVE_MORE."""
    return round(prm['model']['core_hours'] * SIEVE_MORE, 4)


def table_params(F):
    """Return the parameters from the table, or None for a field outside it."""
    x = F.get('log2_dK', 0) * math.log10(2)
    lpb = max(math.ceil(math.log2(F['GB'])) + TABLE_LPB, LPB_MIN)
    if not TABLE_MIN <= x <= TABLE[-1][0] or F['log2_skew'] > TABLE_SKEW:
        return None
    _, rq, sq, N = next(t for t in TABLE if x <= t[0])
    prm = default_params(lpb, N)
    prm.update({'polys': {'F': {'weight': 1.0, 'sec_per_q': sq}}, 'trial': {'cpu_s': 0.0, 'table': True},
                'model': model(lpb, rq, sq)})
    return prm


def tune(state, F, mem_gb=1.0, lpb_min=None, trial=False):
    """Write STATE/params.json once, from the table or the trial sieve (always the trial with lpb_min)."""
    S = State(state)
    prm = S.params()
    if prm is not None:
        return prm
    bits = max(abs(int(c)) for c in F['poly']).bit_length()
    if bits > COEF_BITS:
        raise RuntimeError(text('poly_big', bits=bits))
    t0 = time.time()
    prm = None if trial or lpb_min else table_params(F)
    if prm is not None:
        DETAIL.info('Sieve parameters from the table: lpb %d, 2^%d positions per special-q.' % (prm['lpb0'], prm['N']))
    else:
        prm = trial_params(state, F, max(math.ceil(math.log2(F['GB'])) + 4, LPB_MIN), mem_gb, lpb_min)
        DETAIL.info('Sieve parameters from the trial: lpb %d, 2^%d positions per special-q (%.1f s).' % (
            prm['lpb0'], prm['N'], prm['trial']['cpu_s']))
    prm['trial']['wall_s'] = round(time.time() - t0, 1)
    S.save_params(prm)
    return prm


def trial_params(state, F, lpbA, mem_gb, lpb_min=None):
    """Return the parameters from a trial over lpb in {lpbA - 2, lpbA, lpbA + 2}, then N in N_TRIAL.

    The lowest lpb whose special-q can supply the relations and within LPB_SLACK of the cheapest is chosen.
    """
    lpbA = max(lpbA, lpb_min or 0)
    T = Trial(state, F)
    while True:
        A = T.rate(default_params(lpbA))
        if A['rate'] > 0:
            break
        if lpbA + 2 > LPB_MAX:
            raise RuntimeError(text('too_few'))
        lpbA += 2
    B = {lpbA: A}
    for lpb in (lpbA - 2, lpbA + 2):
        if 2 ** lpb > F['GB'] * 2 and lpb >= LPB_MIN:
            B[lpb] = T.rate(default_params(lpb))
    # An lpb whose trial found no relation is no candidate.  Only the band [2^(lpb-1), 2^lpb) counts toward supply:
    # with lim0 = 2^lpb the special-q below it give mostly duplicates, and counting them let a sieve dry up.
    supply = lambda l: B[l]['rels_per_q'] * (li(2 ** l) - li(2 ** (l - 1))) * DUP / need(l)
    ok = lambda l: B[l]['rate'] > 0 and supply(l) >= SUPPLY_MIN and l >= (lpb_min or 0)
    while not any(ok(l) for l in B):
        top = max(B) + 2
        if top > LPB_MAX:
            raise RuntimeError(text('too_few'))
        B[top] = T.rate(default_params(top))
    cost = {lpb: need(lpb) / (B[lpb]['rate'] * DUP) for lpb in B if ok(lpb)}
    # The lowest lpb within LPB_SLACK of the cheapest: fewer relations make a smaller matrix.
    lpb = min(l for l in cost if cost[l] <= LPB_SLACK * min(cost.values()))
    C = {N: B[lpb] if N == 23 else T.rate(default_params(lpb, N)) for N in N_TRIAL if proc_gb(default_params(lpb, N)) <= mem_gb}
    N = max(C, key=lambda n: C[n]['rate'])
    prm = default_params(lpb, N)
    prm.update({
        'polys': {'F': {'weight': 1.0, 'sec_per_q': C[N]['sec_per_q']}},
        'trial': {'lpb_A': lpbA, 'lpb_min': lpb_min, 'B': {str(k): v for k, v in B.items()}, 'C': {str(k): v for k, v in C.items()},
                  'cpu_s': round(T.cpu, 1)},
        'model': dict(model(lpb, C[N]['rels_per_q'], C[N]['sec_per_q']), supply=round(supply(lpb), 2)),
    })
    return prm


# ---------- given relations

FORMATS = FORMATS3 + (FORMAT4, FORMAT5, FORMAT6)     # witnesses whose relations can be given
INT = r'(0|-?[1-9][0-9]*)'
GIVEN_RE = re.compile(INT + ',' + INT + r'(?::((?:[1-9a-f][0-9a-f]*(?:,[1-9a-f][0-9a-f]*)*)?))?')
FREE = re.compile(r'f:[1-9][0-9]*')


class Given(namedtuple('Given', 'rels digest source poly curve', defaults=(None, None, None))):
    """Given relations, the sha256 of their canonical lines, and from a witness its file, index form and curve."""


def parse_relation(x):
    """Parse a given relation; return (a, b, primes or None), or None for a free relation 'f:p'."""
    if isinstance(x, str) and len(x.strip()) > MAX_DIGITS:
        raise ValueError(too_long(x))
    if isinstance(x, str):
        if FREE.fullmatch(x.strip()):
            return None
        m = GIVEN_RE.fullmatch(x.strip())
        if not m:
            raise ValueError(not_form(x))
        return int(m[1]), int(m[2]), None if m[3] is None else [int(h, 16) for h in m[3].split(',') if h]
    if isinstance(x, (tuple, list)) and len(x) in (2, 3) and all(type(v) is int for v in x[:2]):
        if max(abs(x[0]), abs(x[1])).bit_length() > 3.32 * MAX_DIGITS:
            raise ValueError(too_long(x))
        if len(x) == 2 or x[2] is None:
            return x[0], x[1], None
        if isinstance(x[2], (tuple, list)) and all(type(p) is int for p in x[2]):
            return x[0], x[1], list(x[2])
    raise ValueError(not_form(x))


def short(x):
    """Return x for a message, in about 80 characters at most, never str() of a huge int."""
    if isinstance(x, str):
        return repr(x[:77] + '...' if len(x) > 80 else x)
    if isinstance(x, int) and x.bit_length() > 256:
        return '%s...' % ('-' if x < 0 else '')
    if isinstance(x, (tuple, list)) and len(x) <= 8:
        o, c = '()' if isinstance(x, tuple) else '[]'
        return o + ', '.join(short(v) for v in x) + c
    return '%.80r' % (x,) if not isinstance(x, (tuple, list)) else type(x).__name__


def too_long(x):
    return 'Relation %s is too long for the verifier.' % (short(x[:20] + '...') if isinstance(x, str) else '(a, b)')


def not_form(x):
    return 'Relation %s is not of the form "a,b" or "a,b:hexprimes".' % short(x)


def open_file(path):
    """Open path as text, through gzip or lzma for a name ending in .gz or .xz."""
    return gzip.open(path, 'rt') if path.endswith('.gz') else lzma.open(path, 'rt') if path.endswith('.xz') else open(path)


def read_relations(path, meta=None):
    """Return the non-free relations of a witness or of a text file; meta receives 'poly' and 'curve'."""
    try:
        with open_file(path) as f:
            if path.endswith(tuple(s + z for s in ('.cert.json', '.witness.json') for z in ('', '.gz', '.xz'))):
                c = json.load(f)
                lines = c.get('relations') if isinstance(c, dict) and c.get('format') in FORMATS else None
                if not isinstance(lines, list):
                    raise ValueError('%s is not a witness with relations.' % path)
                lines = [l for l in lines if not (isinstance(l, str) and l.startswith('e:'))]     # general elements
                if meta is not None:
                    fd, cu = c.get('field'), c.get('curve')
                    meta['poly'] = fd.get('poly') if isinstance(fd, dict) else None
                    meta['curve'] = cu.get('minimal_model') if isinstance(cu, dict) else None
            else:
                lines = [l.strip() for l in f if l.strip() and not l.lstrip().startswith('#')]
    except UnicodeDecodeError:
        raise ValueError('%s is not UTF-8 text.' % path)
    except json.JSONDecodeError as e:
        raise ValueError('%s is not valid JSON at line %d, column %d.' % (path, e.lineno, e.colno))
    except (gzip.BadGzipFile, lzma.LZMAError, EOFError):
        raise ValueError('%s is not a valid %s file.' % (path, 'gzip' if path.endswith('.gz') else 'xz'))
    out = []
    for l in lines:
        try:
            r = parse_relation(l)
        except ValueError as e:
            m = str(e)
            raise ValueError('In %s, %s' % (path, m[:1].lower() + m[1:]))
        if r is not None:
            out.append(r)
    return out


def relation_input(rels):
    """Return rels, a file name or an iterable of relations, as a Given."""
    if isinstance(rels, Given):
        return rels
    if isinstance(rels, (bytes, bytearray)) or not (isinstance(rels, (str, os.PathLike)) or hasattr(rels, '__iter__')):
        raise ValueError('relations must be a file name or a list.')
    meta = {}
    rs = read_relations(os.fspath(rels), meta) if isinstance(rels, (str, os.PathLike)) else \
        [r for r in map(parse_relation, rels) if r is not None]
    if not rs:
        raise ValueError('%s has no relations.' % os.fspath(rels) if isinstance(rels, (str, os.PathLike)) else
                         'No relations given.')
    text = '\n'.join(sorted({'%d,%d' % (a, b) + ('' if ps is None else ':' + ','.join('%x' % p for p in sorted(ps)))
                             for a, b, ps in rs}))
    return Given(rs, hashlib.sha256(text.encode()).hexdigest(), os.fspath(rels) if meta else None, meta.get('poly'),
                 meta.get('curve'))


def check_given(F, given):
    """Raise core.Usage for relations of another curve or index form, which would be misread."""
    def ints(v):
        try:
            return [int(x) for x in v]
        except (TypeError, ValueError):
            return None
    if given.source is None:
        return
    if given.curve is not None and ints(given.curve) != ints(F['minimal_model']):
        raise Usage(text('rel_other_curve', file=given.source, curve=','.join(map(str, given.curve))))
    if given.poly is not None and ints(given.poly) != ints(F['poly']):
        raise Usage(text('rel_other_form', file=given.source, poly=','.join(map(str, given.poly)),
                         ours=','.join(map(str, F['poly']))))


GIVEN_DIGITS = 200          # a relation given without its primes has |F(a, b)| of at most this many digits,
GIVEN_FACTOR_S = 10         # factored in at most this many seconds


def given_lines(F, rels):
    """Return the given relations as slice lines 'a,b:hexprimes' and the number factored here."""
    vals, todo, primes = [], [], set()       # primes already proved prime; they recur across relations
    for k, (a, b, ps) in enumerate(rels):
        v = abs(fld.norm(F['poly'], a, b))
        if len(str(a)) + len(str(b)) + v.bit_length() // 4 > MAX_DIGITS:    # a, b and the hex primes
            raise ValueError('Relation %d is too long for the verifier.' % (k + 1))
        if math.gcd(a, b) != 1 or v == 0:
            raise ValueError('In relation %d (%d,%d), a and b are not coprime.' % (k + 1, a, b))
        if ps is not None:
            prod = 1
            for p in ps:
                prod *= p
            if prod != v or not all(p in primes or (p >= 2 and arith.is_prime(p) and not primes.add(p))
                                    for p in ps):
                raise ValueError('The primes given for relation %d (%d,%d) are not the prime factors of |F(a, b)|.' % (
                    k + 1, a, b))
        else:
            todo.append(k)
        vals.append(v)
    found = {}
    for k in todo:                  # relations without primes are factored here, briefly
        a, b, _ = rels[k]
        hard = Usage(text('rel_hard', k=k + 1, ab=' (%d,%d)' % (a, b) if max(abs(a), abs(b)) < 10 ** 30 else '',
                          digits=len(str(vals[k]))))
        if len(str(vals[k])) > GIVEN_DIGITS:
            raise hard
        try:
            found[k] = [p for p, e in arith.factor(vals[k], seconds=GIVEN_FACTOR_S) for _ in range(e)]
        except (TimeoutError, ValueError):
            raise hard from None
    lines = []
    for k, (a, b, ps) in enumerate(rels):
        ps = found[k] if ps is None else ps
        lines.append('%d,%d:%s' % (a, b, ','.join('%x' % p for p in ps)))
    return lines, len(todo)


# ---------- the rules

SLICE_FRACTION = 1 / 30         # CPU per slice: this share of the sieve time,
SLICE_MIN, SLICE_MAX = 0.25, 300.0     # between these (seconds),
SLICE_NEED = 0.5                # and at most this share of the relations needed, but at least one special-q
SLICE_FLIGHT = 0.25             # the slices in flight on all processes hold at most this share of the relations
                                # needed and of the sieve time, so the sieve stops near a check's count (without it,
                                # many cores overshot a check by 2-2.5x)
FIRST_CHECK = 1.0               # first check at the estimate's relations
SLOPE_DEFAULT = 0.6             # excess per relation, until measured
NEXT = 1.2                      # the next check at 20% more relations
GIVE_UP = 2.0                   # give up ('plateau') once the cost, sieving and checks, is this times the cost when
                                # G last improved
GIVE_UP_ABOVE = 3.0             # the same while G is above a known target (lb, the goal or the norm target): curves
                                # that plateaued at G = lb + 1 at 1.3x the relations reached lb at 1.5-2.7x
Q_FLOOR = 2 ** 10               # the special-q go down to this


# ---------- the slices and the checks

def slice_share(prm, share):
    """Return the share of the sieve per slice, at most SLICE_FLIGHT over all processes."""
    c = prm.get('slice_cores') or 1
    return share if c <= 1 else min(share, SLICE_FLIGHT / c)


def slice_cpu(prm):
    """Return the CPU seconds per slice."""
    return min(SLICE_MAX, max(SLICE_MIN, prm['model']['core_hours'] * 3600 * slice_share(prm, SLICE_FRACTION)))


def size_slices(state, cores):
    """Record once in params.json the cores the slices are sized for.

    This happens before the first slice, so the schedule is the same for every run and shard of the folder.
    """
    S = State(state)
    fd = S.lock('prepare.lock', wait=3600)
    try:
        prm = S.params()
        if prm is None or 'slice_cores' in prm or prm.get('siever') == 'given':
            return prm
        if not any(S.done(slice_name(poly, prm['q0'])) for poly in prm['polys']):   # each polynomial starts at q0
            prm['slice_cores'] = max(1, int(cores))
            S.save_params(prm)
        return prm
    finally:
        os.close(fd)


def q_ranges(prm, q0, width):
    """Yield the special-q ranges [lo, hi) of one polynomial: up from q0 to 2^lpb0, then down to Q_FLOOR.

    A special-q above 2^lpb0 is a large prime of every relation it gives, so most would be purged.
    """
    top, bot = max(q0, 2 ** prm['lpb0']), min(q0, Q_FLOOR)
    lo = q0
    while lo < top:
        yield lo, min(lo + width, top); lo += width
    hi = q0
    while hi > bot:
        yield max(bot, hi - width), hi; hi -= width


def schedule(prm):
    """Yield (j, poly, q_lo, q_hi), the polynomials interleaved by weight; deterministic, so shards share it."""
    polys = sorted(prm['polys'].items())
    m = prm['model']
    def width(v):          # special-q per slice
        nq = max(1, slice_cpu(prm) / max(v.get('sec_per_q', m['sec_per_q']), 1e-3))
        if m.get('rels_per_q'):     # at most SLICE_NEED of the relations needed, but at least one special-q
            nq = min(nq, max(1, slice_share(prm, SLICE_NEED) * m['relations_needed'] / m['rels_per_q']))
        return max(2, round(nq * math.log(prm['q0'])))
    rng = {n: q_ranges(prm, prm['q0'], width(v)) for n, v in polys}
    credit = {n: 0.0 for n, _ in polys}
    tot = sum(v['weight'] for _, v in polys)
    j = 0
    while True:
        for n, v in polys:
            credit[n] += v['weight']
        n = max(credit, key=lambda k: (credit[k], k))
        credit[n] -= tot
        x = next(rng[n], None)
        if x is None:                   # this polynomial's special-q are used up
            del rng[n], credit[n]
            polys = [(m, v) for m, v in polys if m != n]
            tot = sum(v['weight'] for _, v in polys)
            if not polys:
                return
            continue
        yield (j, n) + x
        j += 1


def next_check(history, prm):
    """Return the relation count of the next check."""
    h = history
    if not h:
        return FIRST_CHECK * prm['model']['relations_needed']
    last = h[-1]
    if last.get('bound') is None and (last['excess'] or 0) < 0:      # a purge-only check
        E = last['excess']
        slope = SLOPE_DEFAULT
        if len(h) >= 2 and h[-2]['relations'] < last['relations'] and h[-2]['excess'] is not None:
            s = (last['excess'] - h[-2]['excess']) / (last['relations'] - h[-2]['relations'])
            if s > 0.05:
                slope = s
        return last['relations'] + max(0.03 * last['relations'], 1.05 * (-E) / slope + 0.01 * last['relations'])
    return last['relations'] * NEXT


def goal_stop(F):
    """Return whether the goal may stop a run of the field data F.

    Below the goal the 2-descent can still beat Brumer-Kramer, unless S_bk at the goal is already r (or r + 1).
    """
    r = F.get('rank_lower_bound')
    return r is not None and fld.parity_bound(0, F['u'], F['n'], F['w']) <= (r if F['w'] == (-1) ** r else r + 1)


def stop_target(goal, use_goal, norm=None):
    """Return the G at or below which a check stops ('goal'), or None."""
    t = goal if use_goal else -1
    if norm is not None:
        t = max(t, norm)
    return t if t >= 0 else None


def cost_key(history):
    """Return the history key measuring cost: 'cost_ch' if every full solve has it, else 'relations'."""
    full = [x for x in history if x.get('bound') is not None]
    return 'cost_ch' if full and all(x.get('cost_ch') for x in full) else 'relations'


def decide(history, target, rank=None):
    """Return 'goal', 'exact', 'plateau' or None (go on) after a check; rank maps G to the rank bound.

    'exact' means the characters' lower bound leaves no better rank bound.
    """
    e = history[-1] if history else {}
    if e.get('bound') is None:
        return None
    if target is not None and e['bound'] <= target:
        return 'goal'
    if rank is not None and e.get('lb') is not None and rank(e['bound']) <= rank(e['lb']):
        return 'exact'
    full = [x for x in history if x.get('bound') is not None]
    key = cost_key(history)
    best = min(x['bound'] for x in full)
    imp = next(x for x in full if x['bound'] == best)       # when G last improved
    above = e.get('lb') is not None and best > e['lb'] or target is not None and best > target
    if full[-1] is not imp and full[-1][key] >= (GIVE_UP_ABOVE if above else GIVE_UP) * imp[key]:
        return 'plateau'
    return None


# ---------- the time a check takes, and when it runs

FIRST_FULL = 0.5        # before any full check, guess one takes this share of a run
FRESH = 0.9             # a check starting with this share of the window left has as much time as any run


def check_need(history, nrel, threads=1):
    """Return the estimated seconds of a check at nrel relations on threads, or None before any timed check.

    An untrimmed solve given up as too dense says nothing about the trimmed one after it, so it is no measure alone. The
    estimate only decides when a check runs, never a bound.
    """
    checks = {}
    for e in history:
        checks.setdefault(e['relations'], []).append(e)
    need = None
    for n, es in checks.items():
        cs = lambda e, late=False: (max(e.get('solve_s') or 0, e.get('need_s') or 0) if late else
                                    e.get('solve_s') or 0) * (e.get('threads') or threads)
        late = [cs(e, True) for e in es if e.get('late')]
        if any(e.get('bound') is not None for e in es):
            c = sum(cs(e) for e in es if not e.get('late'))
        elif late:
            c = max(late)
        else:
            continue
        c *= max(1.0, nrel / max(n, 1)) ** 2
        need = c if need is None else max(need, c)
    return None if need is None else need / max(1, threads)


def plan_check(need, left, window):
    """Return when a due check runs: 'now', 'later' (the next run's first work) or 'too_long'."""
    if left is None:
        return 'now'
    if need is not None and window is not None and need > window:
        return 'too_long'
    if window is None or left >= FRESH * window:
        return 'now'
    return 'now' if (need if need is not None else FIRST_FULL * window) <= left else 'later'


# ---------- the slice tally

MAX_EMPTY = 100         # stop after this many empty slices in a row
DRY_SLICES, DRY_RATIO = 20, 0.1         # dried up: the last 20 slices under 10% of the mean


def count_slice(tally, n):
    """Tally a finished slice of n relations; raise after MAX_EMPTY empty slices in a row."""
    tally['done'] += 1
    tally['rels'] = tally.get('rels', 0) + n
    tally.setdefault('recent', []).append(n)
    del tally['recent'][:-DRY_SLICES]
    tally['empty'] = 0 if n else tally.get('empty', 0) + 1
    if tally['empty'] >= MAX_EMPTY:
        raise RuntimeError('The sieve finds too few relations for this curve, even at its largest setting. This curve '
                           'is out of reach for now.')


def dry(tally):
    """Return whether the last DRY_SLICES slices each gave under DRY_RATIO of the mean."""
    r = tally.get('recent', [])
    return len(r) >= DRY_SLICES and sum(r) < DRY_RATIO * DRY_SLICES * tally['rels'] / tally['done']


# ---------- the state folder: its artifacts (runtime.transaction: data first, the manifest last)

SLICE_SCHEMA = 'rankbound-slice-1'          # NAME.rels.gz and NAME.json, one finished special-q slice
PARAMS_SCHEMA = 'rankbound-params-1'        # params.json
HISTORY_SCHEMA = 'rankbound-history-1'      # history.json: one entry per solve
CHECKING_SCHEMA = 'rankbound-checking-1'    # checking.json: the pending or running check, with its exact slices
BEST_SCHEMA = 'rankbound-best-1'            # la/best.ops, .piv, .vs, .json and la/order.txt: the solve to certify


def field_fingerprint(state):
    """Return the fingerprint of the state folder's field."""
    try:
        F = read_json(os.path.join(state, 'field.json'))
        return runtime.fingerprint(poly=[str(c) for c in F['poly']], dK=str(F.get('dK')))
    except (OSError, ValueError, KeyError, TypeError):
        return runtime.fingerprint(poly=None)


def read_artifact(folder, name, fname, schema, fp, default=None):
    """Return the JSON file fname of the artifact name if valid, else default."""
    state, _ = runtime.inspect(folder, name, schema, fp)
    if state != 'valid':
        if state == 'damaged':
            runtime.load(folder, name, schema, fp)          # set aside
        return default
    try:
        return read_json(os.path.join(folder, fname))
    except (OSError, ValueError):
        return default


def write_artifact(folder, name, fname, schema, fp, obj, counts=None):
    """Write obj as the JSON file fname of the artifact name, with its manifest."""
    with runtime.transaction(folder, name, schema, fp, counts) as tx:
        tx.write(fname, json.dumps(obj, indent=1) + '\n')


def drop_artifact(folder, name, *fnames):
    """Remove the artifact: its manifest, then its files."""
    runtime.invalidate(folder, name)
    for f in fnames:
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(folder, f))


GIVEN = 'given'                 # the slice of the given relations


def slice_name(poly, q0):
    return '%s_%d' % (poly, q0)


class State:
    """The sieve's state in the work folder; each file is an artifact with a manifest, except STOP and locks."""

    def __init__(self, path):
        self.path = path
        self.sdir = os.path.join(path, 'slices')
        os.makedirs(self.sdir, exist_ok=True)
        self._fp = None

    @property
    def fp(self):
        if self._fp is None:
            self._fp = field_fingerprint(self.path)
        return self._fp

    def done(self, name):
        """Return whether the slice name is committed."""
        return os.path.exists(runtime.manifest_path(self.sdir, name))

    def finished(self):
        return sorted(os.path.basename(p)[:-len('.manifest.json')]
                      for p in glob.glob(os.path.join(self.sdir, '*.manifest.json')))

    def valid(self, name):
        """Return whether the finished slice name is valid, removing it if not."""
        if runtime.load(self.sdir, name, SLICE_SCHEMA, self.fp) is not None:
            return True
        runtime.invalidate(self.sdir, name)
        DETAIL.info('slice %s is not valid: sieved again', name)
        return False

    # ---- STOP: the outcome, once the sieve is finished

    def stopped(self):
        return os.path.exists(os.path.join(self.path, 'STOP'))

    def stop_record(self):
        return read_json(os.path.join(self.path, 'STOP'), {}) or {}

    def stop_outcome(self):
        return self.stop_record().get('outcome', 'stopped')

    def write_stop(self, outcome, **kw):
        write_json(os.path.join(self.path, 'STOP'), dict(kw, outcome=outcome, time=time.strftime('%Y-%m-%d %H:%M:%S')))

    def clear_stop(self):
        """Remove STOP and the 2-descent's checkpoint, so the sieve goes on."""
        runtime.invalidate(self.path, 'selmer')
        for f in ('STOP', 'selmer.json'):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(self.path, f))

    # ---- params.json, history.json

    def params(self):
        return read_artifact(self.path, 'params', 'params.json', PARAMS_SCHEMA, self.fp)

    def save_params(self, prm):
        write_artifact(self.path, 'params', 'params.json', PARAMS_SCHEMA, self.fp, prm)

    def drop_params(self, aside=None):
        """Remove params.json, keeping its file as aside if given."""
        runtime.invalidate(self.path, 'params')
        p = os.path.join(self.path, 'params.json')
        if os.path.exists(p):
            if aside:
                os.replace(p, os.path.join(self.path, aside))
            else:
                os.remove(p)

    def history(self):
        return read_artifact(self.path, 'history', 'history.json', HISTORY_SCHEMA, self.fp, [])

    def save_history(self, h):
        write_artifact(self.path, 'history', 'history.json', HISTORY_SCHEMA, self.fp, h, {'entries': len(h)})

    # ---- checking.json: the check pending or running, on exactly its slices

    def pending(self):
        """Return the pending check from checking.json, or None."""
        d = read_artifact(self.path, 'checking', 'checking.json', CHECKING_SCHEMA, self.fp)
        return d if isinstance(d, dict) and isinstance(d.get('slices'), list) else None

    def begin_check(self, names, nrel, dry=False, **kw):
        """Write checking.json before a check on the slices names starts.

        It stays if the check is killed or late, so the next run's only work is that check on the same relations.
        """
        write_artifact(self.path, 'checking', 'checking.json', CHECKING_SCHEMA, self.fp,
                       dict(kw, slices=list(names), relations=nrel, dry=dry, start=time.time()), {'relations': nrel})

    def update_check(self, **kw):
        d = self.pending()
        if d is not None:
            d.update(kw)
            write_artifact(self.path, 'checking', 'checking.json', CHECKING_SCHEMA, self.fp, d,
                           {'relations': d['relations']})

    def end_check(self):
        drop_artifact(self.path, 'checking', 'checking.json')

    def lock(self, name, shared=False, wait=0):
        """Lock STATE/name and return the fd, as runtime.lock_fd."""
        return runtime.lock_fd(os.path.join(self.path, name), shared, wait, text('busy', folder=self.path))

    def slice_stats(self):
        """Return the total (CPU seconds, relations) of the finished slices."""
        cpu = n = 0
        for name in self.finished():
            p = os.path.join(self.sdir, name + '.json')
            try:
                d = read_json(p); cpu += d['cpu'] or 0; n += d['relations']
            except (OSError, ValueError, KeyError, TypeError):     # e.g. a node crashed while writing it
                raise RankboundError(text('damaged', file=p, file2=p[:-len('.json')] + '.rels.gz'))
        return cpu, n

    # ---- la/best: the solve to certify, with the slices of its rows in load order (order.txt)

    BEST_FILES = ('best.ops', 'best.piv', 'best.vs', 'best.json', 'order.txt')

    def commit_best(self, work):
        """Commit la/best.* and order.txt as the artifact 'best'."""
        rec = read_json(os.path.join(work, 'best.json'), {}) or {}
        with runtime.transaction(work, 'best', BEST_SCHEMA, self.fp, {'keep': rec.get('keep')}) as tx:
            for f in self.BEST_FILES:
                if os.path.exists(os.path.join(work, f)):
                    tx.adopt(f)

    def best(self, work):
        """Return (order.txt's slice names, best.json) if 'best' is valid, else (None, None)."""
        if runtime.load(work, 'best', BEST_SCHEMA, self.fp) is None:
            return None, None
        with open(os.path.join(work, 'order.txt')) as f:
            names = [l.strip() for l in f if l.strip()]
        if not all(self.done(n) for n in names):
            return None, None
        return names, read_json(os.path.join(work, 'best.json'), None)

    def finish_slice(self, poly, q0, q1, lines):
        """Commit the finished slice [q0, q1) of poly and return (CPU seconds, relations)."""
        return finish_slice(self.sdir, slice_name(poly, q0), {'poly': poly, 'q0': q0, 'q1': q1}, lines, self.fp)

    def given(self):
        """Return the digest of the given relations, or None."""
        p = os.path.join(self.sdir, GIVEN + '.json')
        return read_json(p).get('given') if self.done(GIVEN) and os.path.exists(p) else None

    def write_given(self, lines, digest):
        """Commit the given relations as the slice given."""
        runtime.invalidate(self.sdir, GIVEN)
        with runtime.transaction(self.sdir, GIVEN, SLICE_SCHEMA, self.fp, {'relations': len(lines)}) as tx:
            with gzip.open(tx.path(GIVEN + '.rels.gz'), 'wt', compresslevel=4) as f:
                f.writelines(l + '\n' for l in lines)
            tx.write(GIVEN + '.json', json.dumps({'poly': GIVEN, 'q0': None, 'q1': None, 'cpu': 0.0,
                                                  'relations': len(lines), 'special_q': None, 'given': digest}))


def open_state(S):
    """Return (field.json, params.json), rewriting missing polynomial files."""
    F = fld.load(S.path)
    prm = S.params()
    if prm is None:
        raise RankboundError(text('damaged', file=os.path.join(S.path, 'params.json'),
                                  file2=os.path.join(S.path, 'params.manifest.json')))
    write_poly(F, os.path.join(S.path, 'F.poly'))
    return F, prm


# ---------- the checks

class DriedUp(RuntimeError):
    """The sieve dried up before the excess turned positive; the job re-tunes at a larger lpb."""


UNTRIMMED_S = 60                # checks are untrimmed while the dense phase is estimated at most this many seconds
UNTRIMMED_MB = 4000             # and this many MB, then trimmed to excess ns for the rest of the run.  Untrimmed gives
                                # better bounds (S = 2 against 8 on one curve) but can cost 50x the time and memory.


def too_dense_text(r, folder):
    """Return why the solve r gave no bound for its size."""
    if r.disk:
        return text('disk', folder=folder)
    return text('memory', gb=(r.need_mb if r.need_mb is not None else r.dense_mb or 0) / 1000 / linalg.DENSE_SHARE)


def load_order(S, prm, first=frozenset()):
    """Yield the slice names in a check's load order; first are those finished when the run started."""
    if prm.get('siever') == 'given':
        yield from S.finished()
        return
    names = [slice_name(poly, q0) for _, poly, q0, _ in schedule(prm)]
    mine = set(names)
    yield from (n for n in S.finished() if n not in mine)     # the given ones, and those from before a re-tune
    yield from (n for n in names if n in first)
    yield from (n for n in names if n not in first)


class Solver:
    """The relation matrix, filled from finished slices, and its checks."""
    threads = 1                   # threads of the dense phase
    pids = ()                     # paused sieve processes during a check
    budget = None                 # the run's budget; eliminations end by it or are late
    announce = staticmethod(lambda: None)     # called when an elimination starts
    stop_at = None                # the G that stops a check ('goal'): stop_target
    sieve_ch = 0.0                # the sieve's core-hours at the last check
    late_need = None              # a late solve: f2elim's estimate of its seconds (0: unknown)
    had = None                    # the seconds the last solve had when it started (None: no limit)

    def __init__(self, S, F, small, prm, norm=None):
        self.S, self.prm, self.F = S, prm, F
        self.M = linalg.Matrix(F, small, folder=S.path)
        pp, fb0, _ = siever(prm).prepare(S.path, prm, 'F')
        top = 2 ** prm['lpb0']
        if prm.get('siever') == 'given':      # its split file has every free relation
            top = math.inf
        elif prm['lim0'] < top:     # free relations up to 2^lpb, not lim0
            fb0 = factor_base(S.path, 'F', top)
        for p, labs in linalg.free_rows(F, fb0, top):
            self.M.add_free(p, labs)
        self.nfree = len(self.M.rows)
        self.loaded = set()
        self.order = []           # the slices loaded, in order
        self.nrel = 0
        self.target = fld.goal(F)
        self.goal_stop = goal_stop(F)
        self.stop_at = stop_target(self.target, self.goal_stop, norm)
        self.top = top
        self.hist = S.history()
        self.last = None          # the best solve so far, on the first last_rows rows
        self.last_rows = 0
        self.first = frozenset(S.finished())    # the slices finished when the run started (in_order)

    @classmethod
    def of(cls, S, F, prm, threads=1, norm=None):
        """Return the Solver of the state S with no slice loaded."""
        V = cls(S, F, fld.load_small(S.path), prm, norm)
        V.threads = threads
        return V

    def in_order(self):
        """Yield the slice names in load order: outside the schedule, then finished at the start, then the rest.

        So a check sees the same relations however this run's slices finished.
        """
        return load_order(self.S, self.prm, self.first)

    def load(self, names=None, upto=None):
        """Add the relations of unloaded finished slices: names, else the schedule's prefix up to upto relations."""
        prefix = names is None
        for name in self.in_order() if prefix else names:
            if prefix and upto is not None and self.nrel >= upto:
                break
            if name in self.loaded:
                continue
            if self.budget is not None:
                self.budget.check()
            if not self.S.done(name) or not self.S.valid(name):
                if prefix:
                    break
                continue
            path = os.path.join(self.S.sdir, name + '.rels.gz')
            try:
                with gzip.open(path, 'rt') as f:
                    for l in f:
                        self.nrel += self.M.add_relation(l)
            except (EOFError, OSError, zlib.error, ValueError):    # changed since its manifest
                raise RankboundError(text('damaged', file=path, file2=os.path.join(self.S.sdir, name + '.json')))
            self.loaded.add(name)
            self.order.append(name)

    def rank(self, G):
        """Return the rank bound for the bound G on dim Cl(K)[2]."""
        return fld.parity_bound(G, self.F['u'], self.F['n'], self.F['w'])

    def chars(self):
        """Return the characters above every relation's primes (none for given relations)."""
        if math.isinf(self.top):
            return []
        return linalg.characters([int(x) for x in self.F['poly']], max(self.top, self.M.maxp))

    def solve(self, work, keep, name='elim', chars=(), **kw):
        b = self.budget if self.budget is not None else runtime.current()
        self.had = b.left()
        return linalg.solve(self.M, work, self.target, gate=True, keep=keep, threads=self.threads, others=self.pids,
                            budget=b, name=name, chars=chars, **kw)

    def decide(self):
        """Return decide() with the stop target."""
        t = self.stop_at if self.stop_at is not None else (self.target if self.goal_stop else None)
        return decide(self.hist, t, lambda g: g)

    def check(self, work, dry=False):
        """Run one check and return 'goal', 'exact', 'plateau', 'late' or None.

        Checks are untrimmed while cheap, then trimmed for the rest of the run. A trimmed plateau is retried with less
        trim, since that can still improve the bound.
        """
        t = time.time()
        self.sieve_ch = self.S.slice_stats()[0] / 3600
        keep = next((h['keep'] for h in reversed(self.hist) if 'keep' in h), -1)
        if not dry:                         # below the cliff only the purge runs
            E = linalg.purge_excess(self.M, work, self.target)
            if E < 0:
                self._record(SimpleNamespace(excess=E), t, {'keep': keep, 'gated': True})
                return None
        self.announce()             # an elimination runs now, not a purge alone
        if keep < 0:
            r = self.solve(work, -1, chars=self.chars(), dense_s=UNTRIMMED_S, dense_cap_mb=UNTRIMMED_MB)
            if r.too_dense and r.late is None:      # not cheap untrimmed: trim from now on
                self._record(r, t, {'keep': -1, 'dense_MB': r.dense_mb, 'dense_s': r.dense_s})
                keep, t = self.M.ns, time.time()
        if keep >= 0:
            r = self.solve(work, keep, chars=self.chars())
        if self._entry(r, t, keep, work) == 'late':
            return 'late'
        out = self.decide()
        said = False
        while out == 'plateau' and keep >= 0:       # a trimmed plateau: retry the same relations less trimmed
            if not said:
                say('last_check')
                said = True
            k, keep = keep, -1 if keep * 10 >= r.excess else keep * 10
            t = time.time()
            r = self.solve(work, keep, chars=self.chars())
            if r.bound is None and r.late is None:  # does not fit: the bound at excess k stands
                self._record(r, t, {'keep': k, 'keep_tried': keep, 'dense_MB': r.dense_mb})
                return 'plateau'
            if self._entry(r, t, keep, work) == 'late':     # the next run starts with this check again
                self.hist[-1]['keep'] = k
                self.S.save_history(self.hist)
                return 'late'
            out = self.decide()
        if dry and out is None:
            if r.bound is None:
                raise DriedUp(text('too_few'))
            out = 'plateau'
        return out

    def _entry(self, r, t, keep, work):
        """Record the solve r; return 'late' if it could not end in time."""
        e = {'keep': keep, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        if r.late is not None:      # no bound; check_need uses f2elim's estimate
            self.late_need = r.late
            self._record(r, t, dict(e, late=True, need_s=round(r.late)))
            return 'late'
        if r.too_dense:
            self._record(r, t, dict(e, dense_MB=r.dense_mb, killed=r.killed, disk=r.disk))
            raise RankboundError(too_dense_text(r, self.S.path))
        if r.bound is not None:
            self._keep(r, work, keep)
            e.update({'bound': r.bound, 'dimV': r.dimV, 'rank': r.rank, 'pivots': r.pivots, 'dense_s': r.dense_s})
            if r.crank is not None:
                e.update({'crank': r.crank, 'lb': r.crank - linalg.units_rank(self.F)})
                if e['lb'] > r.bound:       # impossible under GRH: a fault in the characters, so unused
                    DETAIL.info(text('chars_high'))
                    e['lb'] = None
            DETAIL.info('{:,} relations: G <= {}, rank <= {} so far (aim: rank <= {}){}.'.format(
                self.nrel, r.bound, self.rank(r.bound), self.rank(self.target),
                '' if e.get('lb') is None else ', characters G >= %d' % e['lb']))
        self._record(r, t, e)
        return None

    def _keep(self, r, work, keep):
        """Keep r as the solve to certify if its bound is the best; the first such has the smallest witness."""
        if self.last is not None and r.bound >= self.last.bound:
            return
        best = os.path.join(work, 'best')
        runtime.invalidate(work, 'best')
        if all(os.path.exists(x) for x in (r.ops, r.piv, r.vs)):
            rec = read_json(r.ops[:-len('.ops')] + '.json', None)
            for a, b in ((r.ops, best + '.ops'), (r.piv, best + '.piv'), (r.vs, best + '.vs')):
                shutil.copyfile(a, b + '.tmp')
                os.replace(b + '.tmp', b)
            r = dataclasses.replace(r, ops=best + '.ops', piv=best + '.piv', vs=best + '.vs')
            if rec is not None:
                write_json(best + '.json', dict(rec, keep=keep, shrunk=False), None)
        self.last = r
        self.last_rows = len(self.M.rows)
        runtime.write_bytes(os.path.join(work, 'order.txt'), ''.join(n + '\n' for n in self.order).encode())
        self.S.commit_best(work)

    def shrink(self, work):
        """Re-solve the solve to certify trimmed, keeping the first trim with the same bound.

        A trim only removes rows, so the bound cannot drop; the witness gets 7-74% fewer rows.
        """
        path = os.path.join(work, 'best.json')
        rec = read_json(path, {})
        if self.last is None or rec.get('shrunk') or rec.get('keep', -1) >= 0:
            return
        P = copy.copy(self.M)
        P.rows, P.written = self.M.rows[:self.last_rows], None
        b = self.budget if self.budget is not None else runtime.current()
        for keep in (self.M.ns, 2 * self.M.ns):
            r = linalg.solve(P, work, self.target, keep=keep, threads=self.threads, name='shrink', budget=b)
            if r.late is not None:
                say('late')
                raise Stopped(self.S.path)
            if r.bound == self.last.bound:
                self.last = None
                self._keep(r, work, keep)
                self.last_rows = len(P.rows)
                break
        if os.path.exists(path):
            write_json(path, dict(read_json(path), shrunk=True), None)
            self.S.commit_best(work)

    def _record(self, r, t, e):
        """Append the solve r's entry, with its cost, to history.json."""
        e = {'relations': self.nrel, 'rows': len(self.M.rows), 'excess': r.excess, **e}
        e.setdefault('solve_s', round(time.time() - t, 1))
        e['threads'] = self.threads
        e['check_ch'] = round(e['solve_s'] * self.threads / 3600, 6)
        e['cost_ch'] = round(self.sieve_ch + sum(x.get('check_ch') or (x.get('solve_s') or 0) * (x.get('threads') or 1)
                                                  / 3600 for x in self.hist) + e['check_ch'], 6)
        self.hist.append(e)
        self.S.save_history(self.hist)
        return e


# ---------- the runners: sieve_run, run_check and solve_given

POLL = 2                # seconds between a shard's checks of the folder


def slice_failed(h):
    """Raise for a slice whose siever exited nonzero."""
    err, rc = h.error(), h.poll()
    raise RuntimeError(text('failed', program='The siever', reason=next(
        (l.strip() for l in reversed(err.splitlines()) if l.strip()), 'It exited with status %d.' % rc)))


STOP_WAIT = 5.0          # seconds the slices in progress have to report at a stop


def save_slices(S, running, wait=STOP_WAIT):
    """Commit the slices in progress, and finished uncommitted ones, at a stop short of the sieve's end.

    A slice in progress is committed for the range it covered; one that does not answer within wait is dropped.
    """
    for h in running:
        h.stop()
    t = time.monotonic()
    while any(h.poll() is None for h in running) and time.monotonic() - t < wait:
        time.sleep(0.05)
    for h, (poly, q0, q1) in list(running.items()):
        if h.poll() == 0:
            try:
                S.finish_slice(poly, q0, h.end(q1), h.lines())
            finally:
                h.release()
                del running[h]


def shard_list(ks):
    """Return 'Shards 1, 3 and 4 are' or 'Shard 1 is' for the 0-based ks."""
    ks = [k + 1 for k in sorted(ks)]
    if len(ks) == 1:
        return 'Shard %d is' % ks[0]
    return 'Shards %s and %d are' % (', '.join(map(str, ks[:-1])), ks[-1])


def sieve_run(state, jobs=1, shard=(0, 1), budget=None, window=None, meter=None, redecide=False, norm=None):
    """Sieve the slices j = k mod n of the schedule; return (outcome or None, Solver or None).

    A check that would not end in this run is left pending as the next run's only work, so no run sieves past a pending
    check and a check that never fits ends within two runs. A shard (n > 1) does not check: it sieves until the folder
    has the next check's relations.
    """
    b = runtime.current() if budget is None else budget
    k, n = shard
    S = State(state)
    lk = S.lock('shard%d_%d.lock' % (k, n), wait=1.5)
    # A local run holds sieve.lock alone; the shards share it.
    try:
        lks = S.lock('sieve.lock', shared=n > 1)
    except Locked:
        os.close(lk)
        raise
    solver = n == 1
    running = {}            # Slice -> (poly, q0, q1)
    outcome = V = sv = None
    nxt = math.inf
    solved, save = False, True          # a check finished in this run; keep the slices in progress at the end
    due_now = False                     # the pending check is due on the slices the folder had, not left by a run
    with runtime.interruptible():
        try:
            F, prm = open_state(S)
            if S.stopped():
                return S.stop_outcome(), None
            if b.expired():
                say('passed')
                return None, None
            window = b.left() if window is None else window
            pend = S.pending()
            if not solver and pend is not None:         # no sieving past a pending check
                say('shard_pending')
                return None, None
            sv = siever(prm)
            nxt = next_check(S.history(), prm)
            if solver:
                V = Solver.of(S, F, prm, jobs, norm)
                V.budget = b
            sched = (x for x in schedule(prm) if x[0] % n == k)
            tally = {'done': 0}
            polled = 0.0
            count = Count(S)

            def stop(oc, **kw):
                S.write_stop(oc, window=window, threads=jobs, **kw)
                return oc

            def too_long(need):                 # the runner shows the reason
                return stop('too_long', need_s=round(need or 0), relations=V.nrel)

            def best():
                return V.last is not None or S.best(os.path.join(S.path, 'la'))[0] is not None

            def stopped_in(pending, left):
                """Handle the run's end inside a check: 'too_long' if the check had a whole run, else a plain stop."""
                if window is not None and (pending or left is not None and left >= FRESH * window):
                    too_long(None)
                    if not best():
                        raise runtime.final_stop(state, 'too_long', too_long_text(None, window, jobs))

            def check(dry, pending=False):
                """Run a check on the loaded relations; return its outcome, 'later' or None.

                With pending, the check is the run's only work: it runs unless it needs more than the window, and
                if it is late, no run can do it.
                """
                nonlocal solved
                left = b.left()
                need = check_need(V.hist, V.nrel, jobs)
                if pending:
                    plan = 'too_long' if need is not None and window is not None and need > window else 'now'
                else:
                    plan = plan_check(need, left, window)
                if plan == 'too_long':
                    S.begin_check(V.order, V.nrel, dry, threads=jobs)
                    return too_long(need)
                if plan == 'later':
                    S.begin_check(V.order, V.nrel, dry, threads=jobs, window=window)
                    say('late')
                    return 'later'
                def announce():
                    announced.append(True)
                    if pending:
                        say('due_check' if due_now else 'pending_check', n=V.nrel)
                    else:
                        cpu, _ = S.slice_stats()
                        say('found', n=V.nrel, core_hours=core_hours(cpu / 3600))
                    need = check_need(V.hist, V.nrel, jobs)
                    if need is not None:
                        say('solving', n=V.nrel, time=duration(need))
                    else:
                        say('solving_first', n=V.nrel)
                announced = []
                V.announce = announce
                try:
                    oc = run_check(S, V, running, dry=dry)
                except Stopped:
                    stopped_in(pending, left)
                    raise
                if oc != 'late' and announced:
                    solved = True
                if oc == 'late':
                    need = V.late_need or 0
                    if window is not None and (pending or need > window or V.had is not None and V.had >= FRESH * window):
                        return too_long(need if need > window else None)
                    S.update_check(late=True, need_s=round(need), window=window)
                    say('late')
                    return 'later'
                return oc

            if V is not None and pend is None and redecide and V.hist and V.hist[-1].get('bound') is not None:
                outcome = V.decide()            # the last check against this run's stop target
                if outcome:
                    S.write_stop(outcome)
            if V is not None and pend is None and not outcome:  # a check due at the start becomes pending
                names, due = due_slices(S, V, nxt)
                if names is not None:
                    S.begin_check(names, due, threads=jobs)
                    pend = S.pending()
                    due_now = True
            if V is not None and pend is not None:      # this run's only work, before sieving
                try:
                    V.load(names=pend['slices'])
                except Stopped:
                    stopped_in(True, None)
                    raise
                outcome = check(bool(pend.get('dry')), pending=True)
                if not outcome:
                    nxt = next_check(V.hist, prm)
            fresh = True
            while outcome is None:
                if S.stopped() or b.expired():
                    break
                if meter is not None and (solved or V is None) and meter.over():
                    outcome = stop('budget', budget_core_hours=meter.budget, spent_core_hours=round(
                        meter.core_hours(), 3)) if V is not None else None
                    break
                if V is not None and fresh:     # load the finished slices and check when due
                    V.load(upto=nxt)
                    fresh = False
                    if V.nrel >= nxt or dry(tally):
                        outcome = check(V.nrel < nxt)
                        if outcome:
                            break
                        nxt = next_check(V.hist, prm)
                if V is None and time.time() - polled > POLL:           # a shard: is it done?
                    polled = time.time()
                    if S.pending() is not None or shard_done(S, k, n, nxt, count, running):
                        break
                while len(running) < jobs and sched is not None:
                    x = next(sched, None)
                    if x is None:
                        sched = None
                        break
                    j, poly, q0, q1 = x
                    if not S.done(slice_name(poly, q0)):
                        running[sv.start_slice(state, prm, poly, q0, q1)] = (poly, q0, q1)
                if sched is None and not running:
                    if V is not None:           # the last check
                        V.load()
                        outcome = check(True)
                    break
                sv.wait()                       # until a slice may have finished
                for h in [h for h in running if h.poll() is not None]:
                    poly, q0, q1 = running.pop(h)
                    try:
                        if h.poll():
                            slice_failed(h)
                        else:
                            count_slice(tally, S.finish_slice(poly, q0, q1, h.lines())[1])
                    finally:
                        h.release()
                    fresh = True
        except BaseException as e:
            save = isinstance(e, (Stopped, KeyboardInterrupt))     # a stop or signal, not a failure
            raise
        finally:
            try:
                if save and running and outcome in (None, 'later', 'budget') and not S.stopped():
                    save_slices(S, running)
            finally:
                for h in running:
                    h.kill()
                if sv is not None:
                    sv.close()
                os.close(lks)
                os.close(lk)
    if outcome == 'later':
        outcome = None
    cpu, nrel = S.slice_stats()
    if n > 1:
        busy, idle = others_left(S, prm, k, n)
        got = count.total()                     # distinct, as a check counts them
        if b.expired() and got < nxt:
            say('paused', n=nrel)
        elif busy:
            say('shard_others', k=k + 1, n=n, others=shard_list(busy))
        else:
            say('shard_done', k=k + 1, n=n)
        for j in idle if got < nxt else ():
            say('shard_idle', k=j + 1)
    elif outcome in FINAL:                      # stopped short of a decision
        say('sieve_stopped', n=nrel, core_hours=core_hours(cpu / 3600))
    elif outcome:
        say('sieved', n=nrel, core_hours=core_hours(cpu / 3600))
    elif not S.stopped():
        say('paused', n=nrel)
    return outcome, V


def too_long_text(need, window, cores):
    """Return why no run can do a check."""
    if need:
        return text('check_too_long', need=duration(need), cores=plural(cores, 'core'), window=duration(window or 0))
    return text('too_long', time=duration(window or 0), cores=plural(cores, 'core'))


class Count:
    """The distinct relations of the finished slices, as a check counts them.

    rate() is the share of new relations in the slices read last; measured, it can be well below DUP.
    """
    RATE_SHARE = 0.25

    def __init__(self, S):
        self.S, self.keys, self.read, self.recent = S, set(), set(), []      # recent: (lines, new) per slice read

    def total(self):
        for name in self.S.finished():
            if name in self.read:
                continue
            n = new = 0
            try:
                with gzip.open(os.path.join(self.S.sdir, name + '.rels.gz'), 'rt') as f:
                    for l in f:
                        a, b = l.split(':', 1)[0].split(',')
                        k = linalg.seen_key(int(a), int(b))
                        n += 1
                        if k not in self.keys:
                            self.keys.add(k)
                            new += 1
            except (OSError, EOFError, ValueError, zlib.error):     # damaged: the check that loads it removes it
                continue
            self.read.add(name)
            self.recent.append((n, new))
        return len(self.keys)

    def rate(self):
        lines = sum(n for n, _ in self.recent)
        if not lines:
            return DUP
        n = new = 0
        for x, y in reversed(self.recent):
            n, new = n + x, new + y
            if n >= self.RATE_SHARE * lines:
                break
        return new / n if n else DUP


def shards_running(S, k, n):
    """Return the number of other shards running."""
    busy = 0
    for j in range(n):
        if j != k:
            try:
                os.close(S.lock('shard%d_%d.lock' % (j, n)))
            except Locked:
                busy += 1
    return busy


def shard_done(S, k, n, nxt, count, running=()):
    """Return whether the folder, with the slices in progress, has the next check's nxt relations."""
    flight = sum(h.found for h in running)
    total = count.total()
    if flight:
        total += count.rate() * flight * (1 + shards_running(S, k, n))
    return total >= nxt


def due_slices(S, V, nxt):
    """Return (slices, relations) of a check due now on the finished prefix, or (None, n) if short."""
    names, n = [], 0
    for name in V.in_order():
        if n >= nxt:
            break
        if not S.done(name):
            break
        try:
            n += read_json(os.path.join(S.sdir, name + '.json'))['relations']
        except (OSError, ValueError, KeyError, TypeError):
            break
        names.append(name)
    return (names, n) if n >= nxt else (None, n)


def others_left(S, prm, k, n):
    """Return the other shards with unfinished slices, as (running, not running)."""
    done = set(S.finished())
    left = sorted({j % n for j, poly, q0, _ in schedule(prm) if j % n != k and slice_name(poly, q0) not in done})
    busy, idle = [], []
    for j in left:
        try:
            os.close(S.lock('shard%d_%d.lock' % (j, n)))
            idle.append(j)
        except Locked:
            busy.append(j)
    return busy, idle


def run_check(S, V, running, dry=False):
    """Run one check with the sieve paused, and write STOP on an outcome.

    checking.json stays if the check is late, killed or stopped, so the next run redoes it on the same relations.
    """
    for p in running:
        signal_group(p, signal.SIGSTOP)
    V.pids = [p.pid for p in running]
    try:
        fd = S.lock('run.lock')
        try:
            if S.stopped():
                return None
            V.hist = S.history()
            S.begin_check(V.order, V.nrel, dry, threads=V.threads)
            try:
                oc = V.check(os.path.join(S.path, 'la'), dry=dry)
            except Stopped:
                raise
            except Exception:
                S.end_check()
                raise
            if oc != 'late':
                S.end_check()
            if oc and oc != 'late':
                S.write_stop(oc)
        finally:
            os.close(fd)
    finally:
        V.pids = ()
        for p in running:
            signal_group(p, signal.SIGCONT)
    return oc


def solve_given(state, budget=None, threads=1, window=None):
    """Eliminate every finished slice once for the bound the given relations prove; return (outcome, Solver)."""
    b = runtime.current() if budget is None else budget
    window = b.left() if window is None else window
    S = State(state)
    F, prm = open_state(S)
    V = Solver.of(S, F, prm, threads)
    V.budget = b
    V.load()
    fd = S.lock('run.lock')
    try:
        t = time.time()
        V.sieve_ch = 0.0
        say('solving_first', n=V.nrel)
        la, keep = os.path.join(S.path, 'la'), -1
        r = V.solve(la, -1, dense_s=UNTRIMMED_S, dense_cap_mb=UNTRIMMED_MB)
        if r.too_dense and r.late is None:          # not cheap untrimmed: trim to excess ns
            keep = V.M.ns
            r = V.solve(la, keep)
        if r.late is not None:
            if window is not None and (r.late > window or V.had is not None and V.had >= FRESH * window):
                need = r.late if r.late > window else None
                S.write_stop('too_long', need_s=round(need or 0), window=window, threads=threads)
                raise runtime.final_stop(state, 'too_long', too_long_text(need, window, threads))
            say('late')
            raise Stopped(state)
        if r.bound is None and not r.too_dense:     # no bound with memory to spare: too few relations
            raise RankboundError(text('no_bound', rels=plural(V.nrel, 'relation')))
        if r.bound is None:
            raise RankboundError(too_dense_text(r, S.path))
        V._keep(r, la, keep)
        V._record(r, t, {'keep': keep, 'bound': r.bound, 'dimV': r.dimV, 'rank': r.rank, 'pivots': r.pivots})
    finally:
        os.close(fd)
    oc = 'goal' if r.bound <= V.target else 'relations'
    S.write_stop(oc)
    return oc, V


def write_given_params(state):
    """Write params.json for given relations, unless the folder has one."""
    S = State(state)
    prm = S.params()
    if prm is not None:
        return prm
    with gzip.open(os.path.join(state, 'slices', GIVEN + '.rels.gz'), 'rt') as f:
        top = max((p for l in f for p in linalg.parse_line(l)[2]), default=2)
    prm = given_params(fld.load(state), top)
    S.save_params(prm)
    return prm


# ---------- the job

FINAL = ('too_long', 'budget')      # final stops, which certify only a solve they already have

