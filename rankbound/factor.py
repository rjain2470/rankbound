"""The factorisation of the discriminant when its primes are not given: GMP-ECM level by level on several processes,
or PARI's factor() without GMP-ECM, for a limited time.  The effort is kept in STATE/factor.json, so a run stopped at
its limit resumes where it stopped.

  factor.json   {"format": FORMAT, "numbers": {N: record}, "core_s": ..., "wall_s": ...}: one record per composite
                ever worked on, "open" with its effort, or "split" into its parts (primes or other records)"""
import fcntl, math, os, re, shutil, subprocess, tempfile, time
from . import FactorisationNeeded, Locked, RankboundError, Stopped, read_json, write_json

FORMAT = 'rankbound-factor-1'
# (factor digits, B1, expected curves with the default B2): the GMP-ECM 7 README.  A level is done when that many
# curves ran; "no factor up to D digits" is then likely, not proved.
LEVELS = [(20, 11e3, 74), (25, 5e4, 221), (30, 25e4, 453), (35, 1e6, 984), (40, 3e6, 2541), (45, 11e6, 4949),
          (50, 43e6, 8266), (55, 11e7, 20158), (60, 26e7, 47173), (65, 85e7, 77666)]
# seconds per curve at each level for 140 digits, scaled by (digits/140)^2 until measured (e4-rb2, 2026-10-21)
SPC_GUESS = [0.03, 0.1, 0.5, 2.0, 6.0, 17, 60, 150, 360, 1200]
PM1_B1 = 1e9            # P-1 once per composite,
PM1_AFTER = 3           # after the levels below this one,
PM1_MIN_S = 1200        # when at least this long is left
CALL_S = 300            # aim per GMP-ECM call
ECM_FOUND = (2, 6, 8, 10, 14)   # GMP-ECM's exit codes for a factor found (0: none; odd: an error)
MAX_FAILS = 3           # engine failures in a row that stop the step
POLL_S = 0.2


def find_ecm():
    """Return the GMP-ECM binary: $RANKBOUND_ECM, else ecm on PATH; None without one or with RANKBOUND_ECM=none."""
    e = os.environ.get('RANKBOUND_ECM', '').strip()
    if e.lower() == 'none':
        return None
    if e:
        if not (os.path.isfile(e) and os.access(e, os.X_OK)):
            raise RankboundError('RANKBOUND_ECM=%s is not an executable file.' % e)
        return e
    return shutil.which('ecm')


# ---------- numbers

SMALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89]


def probable_prime(n):
    """Miller-Rabin to the 24 primes below 90: a filter only (field.gp tests again, the verifier proves each prime)."""
    if n < 2:
        return False
    for p in SMALL:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in SMALL:
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


def root(n, k):
    """Return the integer k-th root of n if n is a k-th power, else None."""
    x = 1 << -(-n.bit_length() // k)               # >= the root
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            break
        x = y
    return x if x ** k == n else None


def power_base(n):
    """Return (g, k) with n = g^k and k >= 2 largest, or (n, 1)."""
    for k in range(n.bit_length(), 1, -1):
        g = root(n, k)
        if g is not None and g > 1:
            return g, k
    return n, 1


def label(n):
    return ('p%d' if probable_prime(n) else 'C%d') % len(str(n))


# ---------- the state

def new_record(n, like=None):
    """A composite's record; a part of a split composite inherits its effort (no factor found up to a level of the
    whole is none of the part either)."""
    r = {'digits': len(str(n)), 'status': 'open', 'pm1': False, 'done': {}, 'spc': {}, 'core_s': 0.0, 'pari_s': 0.0,
         'found': [], 'parts': []}
    if like:
        r.update(pm1=like['pm1'], done=dict(like['done']), spc=dict(like['spc']))
    return r


def load(path):
    st = read_json(path, None)
    if st is None:
        return {'format': FORMAT, 'numbers': {}, 'core_s': 0.0, 'wall_s': 0.0}
    if st.get('format') != FORMAT:
        raise RankboundError('%s is not a factor.json of this version of rankbound.' % path)
    return st


def register(st, n, like=None):
    """Record n, if it is a new composite; a perfect power g^k is recorded as split into g."""
    if probable_prime(n) or str(n) in st['numbers']:
        return
    g, k = power_base(n)
    st['numbers'][str(n)] = new_record(n, like)
    if k > 1:
        st['numbers'][str(n)].update(status='split', parts=[str(g)])
        register(st, g, like)


def resolve(st, n):
    """Return (primes, open composites) of n, following splits."""
    if probable_prime(n):
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
    """Split the open composite n by a divisor d (any found by an engine); return the parts, or None if d does not
    split n."""
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


def effort(n, r):
    """Return the effort on the composite n as a phrase."""
    L = level_done(r)
    s = [] if not (r['done'] or r['pm1']) else ['P-1 done' if r['pm1'] else 'P-1 not yet']
    if r['done']:
        if L:
            s.append('level %d finished' % LEVELS[L - 1][0])
        if L < len(LEVELS) and r['done'].get(str(L)):
            s.append('level %d %s/%s curves' % (LEVELS[L][0], format(r['done'][str(L)], ','), format(LEVELS[L][2], ',')))
    if r['pari_s']:
        s.append("PARI's factor() %s" % duration(r['pari_s']))
    likely = ': no factor below ~%d digits is likely' % LEVELS[L - 1][0] if L else ''
    return 'C%d (%s%s)' % (r['digits'], '; '.join(s) or 'no effort yet', likely)


# ---------- the engines

class Task:
    """One engine call on the composite n: kind 'pari', 'pm1' or a level index, with k curves."""

    def __init__(self, n, kind, k=1):
        self.n, self.kind, self.k = n, kind, k
        self.proc = self.out = None
        self.t0 = 0.0

    def how(self):
        return {'pari': "PARI's factor()", 'pm1': 'GMP-ECM P-1'}.get(self.kind) or 'GMP-ECM, level %d' % LEVELS[self.kind][0]

    def start(self, ecm, gp, maxmem):
        if self.kind == 'pari':
            cmd = [gp, '-q', '-f', '-s', '64M']
            stdin = 'default(parisizemax, "2G")\nprint(factor(%d)[, 1]~)\nquit\n' % self.n     # one command a line
        else:
            cmd = [ecm, '-q', '-one', '-maxmem', str(maxmem)]
            cmd += ['-pm1', '%.0f' % PM1_B1] if self.kind == 'pm1' else ['-c', str(self.k), '%.0f' % LEVELS[self.kind][1]]
            stdin = '%d\n' % self.n
        self.out = tempfile.TemporaryFile('w+')
        self.t0 = time.time()
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=self.out, stderr=subprocess.STDOUT,
                                         text=True, start_new_session=True)
        except OSError as e:
            self.out.close()
            raise RankboundError('Could not start %s: %s' % (cmd[0], e))
        try:
            self.proc.stdin.write(stdin)
            self.proc.stdin.close()
        except BrokenPipeError:
            pass

    def kill(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 9)
            except OSError:
                pass
            self.proc.wait()
        self.out.close()

    def result(self):
        """Return (divisors found, outcome): 'none' (ran to the end without a factor: the only one credited), 'found',
        or an error message."""
        self.out.seek(0)
        text = self.out.read()
        self.out.close()
        rc = self.proc.returncode
        if self.kind == 'pari':
            err = [l for l in text.splitlines() if l.lstrip().startswith('***') and 'Warning:' not in l]
            if rc or err:
                return [], 'gp exit %d: %s' % (rc, ' '.join(err)[:300] or text[-300:].strip())
            text = '\n'.join(l for l in text.splitlines() if not l.lstrip().startswith('***'))
        elif rc != 0 and rc not in ECM_FOUND:
            return [], 'GMP-ECM exit %d: %s' % (rc, text[-300:].strip())
        ds = [int(x) for x in re.findall(r'[0-9]+', text)]
        ds = [d for d in ds if 1 < d < self.n and self.n % d == 0]
        if self.kind == 'pari' and not ds:
            return [], 'factor() gave no factor: %s' % text[-300:].strip()
        return ds, 'found' if ds else 'none'


def spc(r, L):
    """Seconds per curve at level L: measured, else the guess."""
    return r['spc'].get(str(L)) or SPC_GUESS[L] * (r['digits'] / 140) ** 2


def next_task(st, open_, running, ecm, left):
    """Return the next Task: the open composite furthest behind, at its first level not covered by finished and
    running curves; P-1 once its first levels are done.  None if nothing fits in the seconds left."""
    cands = []
    for n in open_:
        r = st['numbers'][str(n)]
        mine = [t for t in running if t.n == n]
        if ecm is None:
            if not mine:
                cands.append((0, 0.0, r['digits'], Task(n, 'pari')))
            continue
        if (not r['pm1'] and not any(t.kind == 'pm1' for t in mine) and left >= PM1_MIN_S
                and level_done(r) >= PM1_AFTER):
            cands.append((PM1_AFTER - 0.5, 0.0, r['digits'], Task(n, 'pm1')))
        for L in range(len(LEVELS)):
            need = LEVELS[L][2]
            have = r['done'].get(str(L), 0) + sum(t.k for t in mine if t.kind == L)
            if have < need:
                k = min(need - have, max(1, int(CALL_S / spc(r, L))), int(left / spc(r, L)))
                if k >= 1:
                    cands.append((L, have / need, r['digits'], Task(n, L, k)))
                break
    return min(cands, key=lambda c: c[:3])[3] if cands else None


# ---------- the step

def charge(st, t):
    """Add the seconds of the call t to its composite's effort and the total; return them."""
    secs = time.time() - t.t0
    r = st['numbers'][str(t.n)]
    r['core_s'] += secs; st['core_s'] += secs
    if t.kind == 'pari':
        r['pari_s'] += secs
    return secs


def stop_task(st, t):
    """Kill the call t; its time counts, its curves do not."""
    charge(st, t)
    t.kill()


def split(path, composites, hours=1.0, cores=1, mem_gb=1.0, until=None, ecm=False, gp='gp', log=print):
    """Return the primes of the composites (sorted ints), factoring for at most hours (wall clock) on cores
    processes, with the effort kept in path (factor.json).  ecm is the GMP-ECM binary (None: PARI's factor(); False:
    find_ecm()).  Raises FactorisationNeeded at the time limit, Stopped at until (a run's --max-hours or a Slurm
    job's end) if that comes first, and Locked if another process is factoring here."""
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
        return _split(path, [int(c) for c in composites], hours, cores, mem_gb, until, ecm, gp, log)
    finally:
        os.close(fd)


def _split(path, composites, hours, cores, mem_gb, until, ecm, gp, log):
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
    end = t0 + hours * 3600
    stop = until is not None and until < end
    end = min(end, until) if until is not None else end
    log('Factoring: %s left; %s on %d core%s, for at most %s.' % (
        ', '.join(label(c) for c in C), 'GMP-ECM (%s)' % ecm if ecm else "PARI's factor() (no GMP-ECM found)", cores,
        's' if cores > 1 else '', duration(hours * 3600)))
    running, fails, maxmem = [], 0, max(100, int(800 * mem_gb))
    try:
        while C and time.time() < end:
            while len(running) < cores:
                t = next_task(st, C, running, ecm, end - time.time())
                if t is None:
                    break
                t.start(ecm, gp, maxmem)
                running.append(t)
            if not running:
                break                       # nothing fits in the time left, or every level is done
            time.sleep(POLL_S)
            for t in [t for t in running if t.proc.poll() is not None]:
                if t not in running:                # killed below, after a split
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
                elif outcome != 'found':
                    fails += 1
                    log('%s failed on %s (%d in a row; nothing credited): %s' % (t.how(), label(t.n), fails, outcome))
                    if fails >= MAX_FAILS:
                        raise RankboundError('%s failed %d times in a row: %s' % (t.how(), fails, outcome))
                    write_json(path, st)
                    continue
                fails = 0
                for d in ds:                        # each divisor splits what is still open of t.n
                    for m in resolve(st, t.n)[1]:
                        parts = split_by(st, m, d, t.how())
                        if parts:
                            log('%s = %s (%s; %s of work on it)' % (label(m), ' * '.join(label(x) for x in parts),
                                                                   t.how(), duration(r['core_s'])))
                if r['status'] == 'split':          # the other calls on it are redundant now
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
    if stop and not beyond:             # the run's own limit came first
        raise Stopped(os.path.dirname(path))
    left = '; '.join(effort(c, st['numbers'][str(c)]) for c in C)
    raise FactorisationNeeded(
        '%s stopped after %s (%.1f core-h in all): %d composite%s left, %s. %s' % (
            'GMP-ECM' if ecm else "PARI's factor()", duration(wall), core / 3600, len(C), 's' if len(C) > 1 else '', left,
            'Every ECM level is done: factor it by NFS (e.g. CADO-NFS) and give the prime factors (--bad-primes, '
            'bad_primes).' if beyond else
            'Run again to continue (--ecm-hours, ecm_hours sets the time), or give the prime factors (--bad-primes, '
            'bad_primes).' + ('' if ecm else ' With GMP-ECM (RANKBOUND_ECM or ecm on PATH) the effort is kept and '
                              'resumed level by level.')),
        composites=[dict(n=str(c), **{k: st['numbers'][str(c)][k] for k in ('digits', 'pm1', 'done', 'core_s')})
                    for c in C])
