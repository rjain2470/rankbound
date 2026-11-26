"""The checks of the witnesses of curves with E(Q)[2] = 0.

Formats 4-6 (FORMATS) bound the 2-Selmer group; rankbound 2's formats 3 and 2 (FORMATS3) prove the Brumer-Kramer bound
under GRH alone.  check.c checks the relations a - b alpha and (p) and computes dim V and k; check.gp checks the field,
the "e:" relations and the local data, and prints the classes from which this module computes v, the local images,
dim(Sel_2 cap V), U, the norm kernel, S and the statement."""
import array, concurrent.futures, math, os, re, shutil, subprocess, tempfile, time
from . import (COMMON, CT, HERE, MAX_DIGITS, NO_LIMIT, LimitExceeded, Rejected, TimeLimit, big_primes, common, ct_form,
               form, gp_checks, gp_ct, gp_errors, gp_rank, need, parity_bound, parity_floor, places, run_gp,
               selmer_statement, source, statement, word)

FORMAT = 'rankbound-selmer-witness-4'
FORMAT5 = 'rankbound-selmer-witness-5'
FORMAT6 = 'rankbound-selmer-witness-6'
FORMATS = (FORMAT6, FORMAT5, FORMAT, 'rankbound-selmer-certificate-4')
PACKED = (FORMAT5, FORMAT6)                                 # long lists packed (unpack)
FORMAT3 = 'rankbound-selmer-certificate-3'                  # read only
FORMATS3 = (FORMAT3, 'rankbound-selmer-certificate-2')
CHECK = os.path.join(HERE, 'check')                         # check.c, built by `make build`


# ---------- the relations (formats 3 and 4)

REL = re.compile(r'(0|-?[1-9][0-9]*),(0|-?[1-9][0-9]*):((?:[1-9a-f][0-9a-f]*(?:,[1-9a-f][0-9a-f]*)*)?)')
FREE = re.compile(r'f:([1-9][0-9]*)')
GEN = re.compile(r'e:(0|-?[1-9][0-9]*),(0|-?[1-9][0-9]*),(0|-?[1-9][0-9]*),([1-9][0-9]*):'
                 r'((?:[1-9a-f][0-9a-f]*(?:,[1-9a-f][0-9a-f]*)*)?)')        # (c0 + c1 a + c2 a^2)/d


def _gpvec(line, general=False):
    """Return a relation line as a gp vector: [a,b,[primes]], [p], or with general [c0,c1,c2,d,[primes]]."""
    m = FREE.fullmatch(line) if isinstance(line, str) else None
    if m:
        return '[%s]' % m[1]
    m = GEN.fullmatch(line) if general and isinstance(line, str) else None
    if m:
        if len(line) > MAX_DIGITS:
            raise LimitExceeded('it has a relation line of more than %d characters' % MAX_DIGITS)
        return '[%s,%s,%s,%s,[%s]]' % (m[1], m[2], m[3], m[4], ','.join(str(int(h, 16)) for h in m[5].split(',') if h))
    m = REL.fullmatch(line) if isinstance(line, str) else None
    if not m:
        raise ValueError('the relation line %.80r is malformed' % (line,))
    if len(line) > MAX_DIGITS:
        raise LimitExceeded('it has a relation line of more than %d characters' % MAX_DIGITS)
    return '[%s,%s,[%s]]' % (m[1], m[2], ','.join(str(int(h, 16)) for h in m[3].split(',') if h))


def gp_poly(poly):
    """Return F(x, 1) as GP source, from the coefficients c0..c3."""
    return ' + '.join('(%s)*x^%d' % (x, i) for i, x in enumerate(poly))


RELCHECK_MAX = 2 ** 64      # check rel's primes are below this


def _as_general(v):
    """Return v as a general relation [c0,c1,c2,d,[H]] if a prime of it is beyond check rel, else None."""
    t = v.strip('[]').replace('[', '').replace(']', '')
    x = t.split(',')
    if len(x) == 1:
        return '[%s,0,0,1,[%s]]' % (x[0], x[0]) if int(x[0]) >= RELCHECK_MAX else None
    H = [h for h in x[2:] if h]
    if not any(len(h) > 19 and int(h) >= RELCHECK_MAX for h in H):
        return None
    return '[%s,%d,0,1,[%s]]' % (x[0], -int(x[1]), ','.join(H))


def gp_rows(poly, P, rels, jobs, gp='gp', seconds=None):
    """Check the relations rels (gp vectors) and return (their odd-valuation rows as sets of column ids, the names).

    check rel takes a - b alpha and (p); gp takes the general ones, in jobs processes, with the same column names.
    Raise Rejected for a failed check, TimeLimit after seconds (None: NO_LIMIT)."""
    seconds = NO_LIMIT if seconds is None else seconds
    end = time.time() + seconds
    left = lambda: max(0.0, end - time.time())
    if not os.path.exists(CHECK):
        raise RuntimeError('rankbound/check/check is not built. Run make build in the rankbound folder.')
    n = len(rels)
    rels = list(rels)
    for i, v in enumerate(rels):
        g = None if v.split('[')[1].count(',') == 4 else _as_general(v)
        if g:
            rels[i] = g
    gen = [i for i, v in enumerate(rels) if v.split('[')[1].count(',') == 4]      # [c0,c1,c2,d,[H]]
    isgen = set(gen)
    f = gp_poly(poly)
    Ps = ','.join(P)
    tmpd = tempfile.mkdtemp(prefix='rankbound')
    procs = []
    rows = [None] * n
    cols = {}
    try:
        # check rel takes the primes of lc(F) as proved here (APRCL)
        lcout = run_gp('fa = factor(abs(%s)); for(i = 1, #fa~, if(!isprime(fa[i, 1]), error("the factor ", fa[i, 1], '
                       '" of lc(F) is not prime")); print("L ", fa[i, 1], " ", fa[i, 2]));' % poly[3],
                       seconds=left(), gp=gp)
        per = (len(gen) + jobs - 1) // max(jobs, 1)
        for k in range(jobs):
            part = gen[k * per:(k + 1) * per]
            if not part:
                continue
            rf = os.path.join(tmpd, 'r%d.gpv' % k)
            with open(rf, 'w') as fo:
                fo.write('[%s]\n' % ','.join(rels[i] for i in part))
            src = ('default(nbthreads, 1);\ndefault(parisizemax, "4G");\n' + source('check.gp') +
                   '\nf = %s; K = ckinit(f, [%s]);\nckrels(K, f, read("%s"), [%s]);\nquit;\n' % (
                       f, Ps, rf, ','.join(map(str, part))))
            sf = os.path.join(tmpd, 's%d.gp' % k)
            with open(sf, 'w') as fo:
                fo.write(src)
            with open(sf + '.out', 'w') as fo:
                procs.append((subprocess.Popen([gp, '-q', '-s', '256M', '-f', sf], stdin=subprocess.DEVNULL, stdout=fo,
                                               stderr=subprocess.STDOUT), sf + '.out'))
        inp = ['F %s' % ' '.join(poly)] + [l for l in lcout.splitlines() if l.startswith('L ')]
        ab = [i for i in range(n) if i not in isgen]
        inp += [rels[i] for i in ab]
        try:
            r = subprocess.run([CHECK, 'rel'], input=('\n'.join(inp) + '\n').encode(), capture_output=True,
                               timeout=left())
        except subprocess.TimeoutExpired:
            raise TimeLimit('check rel did not finish within %.0f s' % seconds, seconds)
        out = r.stdout.decode()
        last = out.splitlines()[-1].split(' ', 2) if out.strip() else []
        if r.returncode == 1 and last[:1] == ['E']:
            raise Rejected('relation %d %s' % (ab[int(last[1])], last[2]))
        if r.returncode == 3 and last[:1] == ['X']:
            raise LimitExceeded('relation %d %s' % (ab[int(last[1])], last[2]))
        if r.returncode:
            raise RuntimeError('check rel failed: %s' % (r.stderr.decode()[-500:].strip() or 'exit %d' % r.returncode))
        for l in out.splitlines():
            t = l.split()
            rows[ab[int(t[1])]] = {cols.setdefault(z, len(cols)) for z in t[2:]}
        for p, out in procs:
            try:
                p.wait(left())
            except subprocess.TimeoutExpired:
                raise TimeLimit('PARI/GP did not finish the relations within %.0f s' % seconds, seconds)
            with open(out) as fi:
                text = fi.read()
            gp_errors(text)
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
        raise RuntimeError('The relation check stopped early (%d of %d relations checked).' % (n - missing, n))
    names = [None] * len(cols)
    for z, i in cols.items():
        names[i] = z
    return rows, names


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def memory_free(root='/'):
    """Return the memory in bytes this process tree may still take, or None if unknown.

    On Linux, the least of MemAvailable and each memory cgroup's limit less its usage plus its inactive file pages
    (which it reclaims first); elsewhere half the physical memory.  root is for the tests."""
    info = _read(os.path.join(root, 'proc/meminfo'))
    if info is None:
        try:
            return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') // 2 if root == '/' else None
        except (ValueError, OSError, AttributeError):
            return None
    m = re.search(r'^MemAvailable:\s+(\d+) kB', info, re.M)
    free = [int(m.group(1)) * 1024] if m else []
    for line in (_read(os.path.join(root, 'proc/self/cgroup')) or '').splitlines():
        hid, ctl, path = (line.split(':', 2) + ['', ''])[:3]
        v2 = hid == '0' and ctl == ''
        if not v2 and 'memory' not in ctl.split(','):
            continue
        base = os.path.join(root, 'sys/fs/cgroup' if v2 else 'sys/fs/cgroup/memory')
        files = ('memory.max', 'memory.current', 'inactive_file') if v2 else \
                ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'total_inactive_file')
        parts = [x for x in path.strip().split('/') if x]
        for k in range(len(parts), -1, -1):        # the cgroup and its parents
            d = os.path.join(base, *parts[:k])
            lim, use = (_read(os.path.join(d, f)) for f in files[:2])
            if lim is None or use is None or not lim.strip().isdigit() or not use.strip().isdigit():
                continue
            st = re.search(r'^%s (\d+)$' % files[2], _read(os.path.join(d, 'memory.stat')) or '', re.M)
            free.append(max(0, int(lim) - int(use) + (int(st.group(1)) if st else 0)))
    return min(free) if free else None


MEM_SHARE, DISK_SHARE = 0.85, 0.9       # the shares of free memory and disk that check rank may use
MODE = re.compile(rb'check rank: mode (whole|blocks) by ([A-Za-z_]+), (\d+) of (\d+) words a block, dense (\d+) bytes, '
                  rb'held (\d+) bytes, memory (\d+) bytes, temporary file (\d+) bytes, clearings (\d+), threaded (\d+), '
                  rb'threads started (\d+), up to (\d+)')


def rank_env(env=None):
    """Return check rank's environment, with CHECK_RANK_MEM_MB, _TMP_MB and _TMPDIR filled in unless set."""
    e = dict(os.environ if env is None else env)
    if not e.get('CHECK_RANK_MEM_MB'):
        e.pop('CHECK_RANK_MEM_MB', None)
        free = memory_free()
        if free is not None:
            e['CHECK_RANK_MEM_MB'] = str(max(1, int(free * MEM_SHARE) >> 20))
    d = e.get('CHECK_RANK_TMPDIR') or e.get('TMPDIR') or '/tmp'
    e['CHECK_RANK_TMPDIR'] = d
    if not e.get('CHECK_RANK_TMP_MB'):
        e.pop('CHECK_RANK_TMP_MB', None)
        try:
            e['CHECK_RANK_TMP_MB'] = str(max(1, int(shutil.disk_usage(d).free * DISK_SHARE) >> 20))
        except OSError:
            pass                            # check rank reports the folder if it needs it
    return e


def rank_mode(err):
    """Return the mode line of check rank's stderr as a dict (for the verdict's rank_check), or None."""
    m = list(MODE.finditer(err))
    if not m:
        return None
    g = m[-1].groups()
    n = [int(x) for x in g[2:]]
    return {'mode': g[0].decode(), 'by': g[1].decode(), 'block_words': n[0], 'words': n[1],
            'dense_mib': round(n[2] / 2 ** 20, 1), 'held_mib': round(n[3] / 2 ** 20, 1),
            'memory_mib': round(n[4] / 2 ** 20, 1) if n[4] else None, 'temp_mib': round(n[5] / 2 ** 20, 1),
            'clearings': n[6], 'threaded': n[7], 'threads_started': n[8], 'threads_max': n[9]}


def rank_V(rows, names, ops, piv, GB, jobs=1, seconds=None, prefix=None, info=None):
    """Return (dim V, the number of large columns of M, rank M) by check rank, M the rows piv after ops.

    A column is small iff it is a degree-1 prime p:r with p < GB.  With prefix, return two such triples from one run:
    for M and for its first prefix rows.  info, a list, receives the mode that ran.  Raise TimeLimit after seconds,
    LimitExceeded if the matrix does not fit."""
    if not os.path.exists(CHECK):
        raise RuntimeError('rankbound/check/check is not built. Run make build in the rankbound folder.')
    small = [z.count(':') == 1 and int(z.split(':')[0]) < GB for z in names]
    pos, nl = [0] * len(names), len(names) - sum(small)        # the large columns first
    ks = [0, nl]
    for i, s in enumerate(small):
        pos[i] = ks[s]; ks[s] += 1
    data = array.array('I', [len(rows), len(names), nl, len(ops), len(piv)])
    for r in rows:
        data.append(len(r)); data.extend(pos[c] for c in r)
    for o in ops:
        data.extend(o)
    data.extend(piv)
    try:
        p = subprocess.run([CHECK, 'rank', str(max(1, jobs))] + ([] if prefix is None else [str(prefix)]),
                           input=data.tobytes(), capture_output=True, env=rank_env(),
                           timeout=None if seconds is None else max(0.0, seconds))
    except subprocess.TimeoutExpired:
        raise TimeLimit('check rank did not finish within %.0f s' % seconds, seconds)
    if p.returncode == 4:
        m = re.findall(rb'check rank: (.*)', p.stderr)
        raise LimitExceeded('its rank check does not fit this machine (%s)'
                            % '; '.join(x.decode(errors='replace') for x in m[-2:]))
    if p.returncode:                    # its last line says why
        why = (p.stderr.decode(errors='replace').strip().splitlines() or ['exit %d' % p.returncode])[-1][-500:]
        raise RuntimeError('check rank failed: %s' % why)
    if info is not None:
        info.append(rank_mode(p.stderr))
    t = p.stdout.split()
    if prefix is None:
        return int(t[7]), int(t[5]), int(t[1])
    if len(t) == 8:                     # the prefix is every row
        return (int(t[7]), int(t[5]), int(t[1])), (int(t[7]), int(t[5]), int(t[1]))
    if t[8] != b'prefix' or int(t[9]) != prefix:
        raise RuntimeError('check rank printed %r' % p.stdout[:200])
    return (int(t[7]), int(t[5]), int(t[1])), (int(t[17]), int(t[15]), int(t[11]))


def counting(ok):
    """Return ok wrapped, and failed(): whether a check recorded through it failed."""
    bad = []

    def rec(name, cond, info=''):
        if not cond:
            bad.append(name)
        ok(name, cond, info)
    return rec, lambda: bool(bad)


# ---------- format 3

SPEC3 = dict(COMMON, GRH='str', field={'poly': ['num', 4], 'dK': 'num', 'GB': 'pos', 'ns': 'nat'},
             local={'u': 'nat', 'n': 'nat', 'w': 'sign'}, claims={'G': 'nat', 'dimV': 'nat', 'S': 'nat'},
             relations=['str'], ops=[['nat', 2]], pivots=['nat'])


def odd_torsion(c):
    """Raise ValueError unless the torsion of c is [] or [n] with n odd."""
    common(c, 1)
    if any(x % 2 == 0 for x in c['torsion']):
        raise ValueError('torsion must be [] or [n] with n odd: E(Q)[2] = 0')


def clean3(c):
    """Check the form of the format-3 witness c and return its relations as gp vectors; raise ValueError if
    malformed."""
    old = c.get('format') == FORMATS3[1]
    form(c, {k: v for k, v in SPEC3.items() if not (old and k in ('torsion', 'GRH'))})
    cubic_dk(c)
    if old:
        common(dict(c, torsion=[]), 0)
    else:
        odd_torsion(c)
    return [_gpvec(l) for l in c['relations']]


def cubic_dk(c):
    """Raise ValueError unless |field.dK| >= 23, the least discriminant of a cubic field."""
    d = c['field']['dK'].lstrip('-')       # no int(): d_K may have more digits than int() accepts
    if len(d) < 3 and int(d) < 23:
        raise ValueError('field.dK must be the discriminant of a cubic field, so |dK| >= 23')


def logd(dK):
    """Return log |d_K| from the leading digits of the decimal dK."""
    d = dK.lstrip('-')
    return (len(d) - 15) * math.log(10) + math.log(int(d[:15])) if len(d) > 15 else math.log(max(int(d), 1))


def limits3(c):
    """Raise LimitExceeded if the format-3 witness c asks for unbounded work (GB above Bach's bound, big primes)."""
    bach = 12 * logd(c['field']['dK']) ** 2
    if c['field']['GB'] > bach:
        raise LimitExceeded('it asks for GB = %d, above Bach\'s bound %.0f' % (c['field']['GB'], bach))
    big_primes(c)


FIELD_CHECKS3 = ('bad_primes_prime', 'minimal_model', 'disc_factored', 'E2_trivial', 'maximal_order', 'index_form',
                'same_field', 'bk_u', 'bk_n', 'root_number', 'bdf_GB', 'ns')

HYPOTHESIS3 = ('rank E(Q) <= rank_upper_bound assuming GRH for the Dedekind zeta functions of the unramified quadratic '
               'extensions of K, the cubic field of the 2-division polynomial of E (not GRH for L(E, s), not BSD); '
               'rank E(Q) >= rank_lower_bound unconditionally when r_proved')


def verify3(c, rels, ok, gp, left, jobs, rank_info=None):
    """Check the format-3 or 2 witness c; return (the statement rebuilt, whether the rank lower bound is proved)."""
    ok, failed = counting(ok)
    cur, fd, loc, cl = c['curve'], c['field'], c['local'], c['claims']
    old = c['format'] == FORMATS3[1]
    rl = c.get('rank_lower_bound') or {}
    r = rl.get('r')
    pr = rl if 'points' in rl else None
    st = ''
    f = gp_poly(fd['poly'])
    body = 'vfield([%s], [%s], [%s], %s, %s, %d, %d, %d, %d, %d);\n' % (
        ','.join(cur['ainvs']), ','.join(cur['bad_primes']), ','.join(cur['minimal_model']), f, fd['dK'], fd['GB'],
        fd['ns'], loc['u'], loc['n'], loc['w'])
    if not old:
        body += 'vtors([%s], [%s]);\n' % (','.join(cur['ainvs']), ','.join(map(str, c['torsion'])))
    body += gp_rank(c)
    out = run_gp(body, seconds=NO_LIMIT if left() is None else left(), gp=gp)
    need(gp_checks(out, ok), FIELD_CHECKS3 + (() if old else ('torsion',)) + (('rank_lower_bound',) if pr else ()),
         ok, out)
    if not old:
        ok('GRH', c['GRH'] == 'yes', 'the bound assumes GRH')
    if failed():        # no relation check on a field or curve that failed a check
        return st, pr is not None
    t1 = time.time()
    rows, names = gp_rows(fd['poly'], cur['bad_primes'], rels, jobs, gp=gp, seconds=left())
    ok('relations_checked', True, '%d rows recomputed (%.0f s, %d jobs)' % (len(rows), time.time() - t1, jobs))
    ops, piv = [tuple(o) for o in c['ops']], c['pivots']
    if not all(0 <= x < len(rows) for o in ops for x in o) or not all(0 <= x < len(rows) for x in piv):
        ok('ops_in_range', False, 'a row index of ops or pivots is out of range')
    else:
        ok('ops_in_range', True)
        t2 = time.time()
        dimV, nl, rank = rank_V(rows, names, ops, piv, fd['GB'], jobs, seconds=left(), info=rank_info)
        G = fd['ns'] - dimV
        ok('dimV', dimV == cl['dimV'], 'dim V = %d (rank %d, %d large columns, %.0f s)' % (dimV, rank, nl,
                                                                                          time.time() - t2))
        S, st = statement(G, loc, r, None if pr else rl.get('source'))
        ok('claims', G == cl['G'] and S == cl['S'], 'G = %d, rank <= %d' % (G, S))
        ok('statement', c['statement'] == st)
        if r is not None:
            ok('bound_ge_r', S >= r, 'rank <= %d, rank >= %d' % (S, r))
    return st, pr is not None


# ---------- format 4

CLAIMS = ('G', 'dimV', 'k', 'U', 'v', 'dim_sel_cap_V', 'S_bk', 'S', 'w')
CLAIMS6 = ('G', 'dimV', 'k', 'U', 'v', 'dim_sel_cap_V', 'dim_V_cap_ker_N', 'S_norm', 'S_bk', 'S', 'w')
PAIR = ['small', 2]
SPEC = dict(COMMON, hypothesis='str', weierstrass=['num', 4], theta=['rat', 3], s_primes=['small'], s_columns=[PAIR],
            relations=['str'], ops=[['nat', 2]], pivots=['nat'], elements=[['nat']],
            characters={'real': ['nat'], 'primes': [PAIR]},
            field={'poly': ['num', 4], 'dK': 'num', 'bound': {'type': 'str', 'B': 'pos', 'ns': 'nat'}},
            local={'inf': ['rat'], 'p': {'*': {'points': ['rat'], 'prec': 'pos'}}, 'u?': 'nat', 'n?': 'nat'},
            claims=dict({k: 'nat' for k in CLAIMS}, w='sign', **{'u?': 'nat', 'n?': 'nat', 'S_ctp?': 'nat'}),
            **{'s_dropped?': [PAIR], 'ops_s?': [['nat', 2]], 'pivots_s?': ['nat'], 'cassels_tate?': CT})
FIELD_CHECKS = ('bad_primes_prime', 'minimal_model', 'disc_factored', 'E2_trivial', 'weierstrass', 'maximal_order',
                'index_form', 'same_field', 'theta', 'root_number', 'bound', 'ns', 'torsion', 'roots_inf', 'points_inf')
HYPOTHESIS = {'GRH': 'rank E(Q) <= rank_upper_bound assuming GRH for the Dedekind zeta functions of the unramified '
                     'quadratic extensions of K, the cubic field of the 2-division polynomial of E (not GRH for '
                     'L(E, s), not BSD); rank E(Q) >= rank_lower_bound unconditionally when r_proved',
              'none': 'rank E(Q) <= rank_upper_bound unconditionally (2-descent, class group from the Minkowski '
                      'bound; no GRH, no BSD); rank E(Q) >= rank_lower_bound unconditionally when r_proved'}


SPEC5 = dict(SPEC, ops=['nat'], pivots=['nat'], **{'ops_s?': ['nat']})
_CL6 = dict({k: 'nat' for k in CLAIMS6}, w='sign', **{'u?': 'nat', 'n?': 'nat', 'S_ctp?': 'nat'})
SPEC6 = dict(SPEC5, claims=_CL6)
SPEC6U = dict(SPEC, claims=_CL6)                            # after unpack


def unpack(c):
    """Unpack the format-5 or 6 witness c into format 4's lists, in place; raise ValueError if malformed.

    pivots, pivots_s and the elements are delta-coded; pivots_s lists only the rows beyond pivots; no ops_s means ops."""
    form(c, SPEC6 if c.get('format') == FORMAT6 else SPEC5)
    if 'ops_s' in c and 'pivots_s' not in c:
        raise ValueError('ops_s needs pivots_s')

    def run(ds, what):
        if any(d < 1 for d in ds[1:]):
            raise ValueError('%s: a delta after the first is below 1' % what)
        out, t = [], 0
        for d in ds:
            t += d
            out.append(t)
        return out

    def pairs(f, what):
        if len(f) % 2:
            raise ValueError('%s must have an even number of values' % what)
        return [f[k:k + 2] for k in range(0, len(f), 2)]
    c['ops'] = pairs(c['ops'], 'ops')
    c['pivots'] = run(c['pivots'], 'pivots')
    if 'pivots_s' in c:
        c['ops_s'] = pairs(c['ops_s'], 'ops_s') if 'ops_s' in c else c['ops']
        c['pivots_s'] = c['pivots'] + run(c['pivots_s'], 'pivots_s')
    c['elements'] = [run(e, 'elements[%d]' % i) for i, e in enumerate(c['elements'])]


def clean(c):
    """Check the form of the format-4, 5 or 6 witness c; return its relations as gp vectors."""
    if isinstance(c, dict) and c.get('format') in PACKED:
        unpack(c)
    form(c, SPEC6U if c.get('format') == FORMAT6 else SPEC)
    cubic_dk(c)
    odd_torsion(c)
    if ('ops_s' in c) != ('pivots_s' in c):
        raise ValueError('ops_s and pivots_s go together')
    if ('S_ctp' in c['claims']) != (c.get('cassels_tate') is not None):
        raise ValueError('claims.S_ctp is given exactly with a cassels_tate section')
    if c['hypothesis'] not in ('GRH', 'none') or c['field']['bound']['type'] not in ('BDF', 'minkowski'):
        raise ValueError('hypothesis must be "GRH" or "none", field.bound.type "BDF" or "minkowski"')
    if c['weierstrass'][0] == '1' and c['weierstrass'][3] != '1':      # high to low degree (16 b6 != 1)
        c['weierstrass'] = c['weierstrass'][::-1]
    if any(not J or len(set(J)) != len(J) for J in c['elements']):
        raise ValueError('elements: each lists distinct relations, at least one')
    real = c['characters']['real']
    if real != sorted(set(real)) or any(x > 2 for x in real):
        raise ValueError('characters.real must be increasing indices 0, 1, 2 of the real roots')
    if any(int(r) >= int(q) for q, r in c['characters']['primes']) or any(
            int(r) > int(p) for p, r in c['s_columns']):
        raise ValueError('characters.primes [q, r] need r < q, s_columns [p, r] r <= p')
    ct_form(c.get('cassels_tate'), len(places(c['curve']['bad_primes'])))
    return [_gpvec(l, general=True) for l in c['relations']]


def limits(c):
    """Raise LimitExceeded if the format-4 witness c asks for unbounded work.

    That is B above Bach's bound (BDF) or 2 M + 100 (M the Minkowski bound), or a prime of 2^64 or more."""
    B, ld = c['field']['bound']['B'], logd(c['field']['dK'])
    if c['field']['bound']['type'] == 'BDF':
        cap = 12 * ld ** 2
        over, what, shown = B > cap, "Bach's bound", '%.0f' % cap
    else:                                   # (4/pi)^r2 <= 4/pi for r2 = 0, 1; in logs, as |d_K| may have 4,000 digits
        lm = math.log(2 * (4 / math.pi) * 2 / 9) + ld / 2       # log of the cap less 100
        over, what = B > 100 and math.log(B - 100) > lm, "twice Minkowski's bound"
        shown = '%.0f' % (math.exp(lm) + 100) if lm < 700 else 'about 10^%d' % (lm / math.log(10))
    if over:
        raise LimitExceeded('it asks for B = %s, above %s %s' % (B if B < 10 ** 40 else 'about 10^%d' % (
            len(str(B)) - 1), what, shown))
    big_primes(c)
    if any(int(qc[0]) >= 2 ** 64 for qc in c['characters']['primes']):
        raise LimitExceeded('a prime of its characters is 2^64 or more')


# ---------- F_2 linear algebra on Python ints (bit i = coordinate i)

def _bits(s):
    return int(s[::-1], 2) if s else 0


def _echelon(vecs):
    """Return an echelon basis {leading bit: vector} of the span of vecs."""
    B = {}
    for v in vecs:
        v = _reduce(v, B)
        if v:
            B[v.bit_length() - 1] = v
    return B


def _reduce(v, B):
    """Return the element of v + span(B) with no bit at a leading bit of B (linear, zero exactly on span(B))."""
    for lead in sorted(B, reverse=True):
        if v >> lead & 1:
            v ^= B[lead]
    return v


def _join(parts):
    """Concatenate (value, width) pairs into one int."""
    out, shift = 0, 0
    for val, w in parts:
        out |= val << shift
        shift += w
    return out


# ---------- the check

def gp_body(c, rels, used=None):
    """Return (the GP code that checks the format-4 witness c, S, the relations used whose classes it prints)."""
    cur, fd, loc, ch = c['curve'], c['field'], c['local'], c['characters']
    SP = sorted({int(p) for p in c['s_primes']})        # S: verify() checks it against the bad primes
    if used is None:
        used = sorted({j for J in c['elements'] for j in J})
    vec = lambda v: '[%s]' % ','.join(v)
    body = 'KT = v4field(%s, %s, %s, %s, %s, %s, %s, %d, %d, %d, %d, %s, [%s]);\n' % (
        vec(cur['ainvs']), vec(cur['bad_primes']), vec(cur['minimal_model']), gp_poly(fd['poly']), fd['dK'],
        vec(c['weierstrass']), vec(c['theta']), 1 if fd['bound']['type'] == 'BDF' else 2, fd['bound']['B'],
        fd['bound']['ns'], c['claims']['w'], vec(map(str, SP)),
        ','.join('[%d,%d]' % tuple(map(int, z)) for z in c.get('s_dropped', [])))
    body += 'vtors(%s, [%s]);\n' % (vec(cur['ainvs']), ','.join(map(str, c['torsion']))) + gp_rank(c)
    X = '[%s]' % ','.join(vec(loc['p'].get(str(p), {'points': []})['points']) for p in SP)
    body += 'v4main(KT, %s, %s, [%s], %s, %s, [%s], %s, %s, [%s]);\n' % (
        gp_poly(fd['poly']), vec(map(str, SP)), ','.join('[%d,%d]' % tuple(map(int, z)) for z in c['s_columns']),
        X, vec(loc['inf']), ','.join(rels[j] for j in used), vec(map(str, used)), vec(map(str, ch['real'])),
        ','.join('[%d,%d]' % tuple(map(int, z)) for z in ch['primes']))
    if c.get('cassels_tate') is not None:
        body += 'v4ct(%s, %s, %s);\n' % (vec(cur['minimal_model']), vec(cur['bad_primes']), gp_ct(c['cassels_tate']))
    return body, SP, used


PAR_RELS = 2000         # with more relations in elements than this, their classes are split over jobs gp processes


def gp_rels(c, rels, used):
    """Return the GP code that prints the classes of the relations used, without checks (v4relsonly)."""
    vec = lambda v: '[%s]' % ','.join(v)
    return 'v4relsonly(%s, %s, %s, [%s], %s, %s, [%s]);\n' % (
        gp_poly(c['field']['poly']), vec(c['curve']['bad_primes']), vec(map(str, sorted({int(p) for p in c['s_primes']}))),
        ','.join(rels[j] for j in used), vec(map(str, used)), vec(map(str, c['characters']['real'])),
        ','.join('[%d,%d]' % tuple(map(int, z)) for z in c['characters']['primes']))


def verify(c, rels, ok, gp, left, jobs, rank_info=None):
    """Check the format-4 witness c; return (the statement rebuilt, whether the rank lower bound is proved)."""
    ok, failed = counting(ok)
    cur, fd, cl, loc = c['curve'], c['field'], c['claims'], c['local']
    rl = c.get('rank_lower_bound') or {}
    r = rl.get('r')
    nrel = len(rels)
    st = ''
    SPc = [int(p) for p in c['s_primes']]
    SP0 = sorted({2} | {int(p) for p in cur['bad_primes']})
    if 's_dropped' not in c:
        ok('s_primes', len(set(SPc)) == len(SPc) and sorted(SPc) == SP0, 'S = {2} and the bad primes')
    else:
        D = [int(z[0]) for z in c['s_dropped']]
        ok('s_primes', len(set(SPc)) == len(SPc) and len(set(D)) == len(D) and not set(SPc) & set(D) and
           sorted(SPc + D) == SP0 and 2 in SPc, 'S and s_dropped partition {2} and the bad primes, 2 in S')
    SPs = sorted(set(SPc))
    ok('s_columns', all(int(z[0]) in set(SPs) for z in c['s_columns']), 'every s_column lies above S')
    # with no elements no local image is needed, so local data may cover only part of S
    lp = sorted(int(p) for p in loc['p'])
    ok('local_primes', lp == SPs or (not c['elements'] and set(lp) <= set(SPs)),
       'local data for every p in S' if c['elements'] else 'no elements: local data for primes of S only')
    # S or an s_column off {2} and the bad primes (which gp proves prime) is rejected already; gp would fail at
    # places that may not be primes
    if set(SPc) - set(SP0) or {int(z[0]) for z in c['s_columns']} - set(SP0):
        return st, 'points' in rl
    if any(not 0 <= j < nrel for J in c['elements'] for j in J):
        ok('elements_in_range', False, 'a relation index of elements is out of range')
        return st, 'points' in rl
    # with many relations, the classes are split over gp processes; only the first makes the checks
    allused = sorted({j for J in c['elements'] for j in J})
    k = jobs if jobs > 1 and len(allused) >= PAR_RELS else 1
    shares = [allused[t * len(allused) // k:(t + 1) * len(allused) // k] for t in range(k)]
    body, SP, used = gp_body(c, rels, shares[0])
    secs = NO_LIMIT if left() is None else left()
    with concurrent.futures.ThreadPoolExecutor(k) as ex:
        runs = [ex.submit(run_gp, body, seconds=secs, gp=gp)] + [ex.submit(run_gp, gp_rels(c, rels, sh), seconds=secs, gp=gp)
                                                               for sh in shares[1:]]
        outs = [r.result() for r in runs]
    out = outs[0]
    used = allused
    printed = gp_checks(out, ok)
    num, blk, pi, sc, rc, ctr = {}, [], {}, {}, {}, None
    for l in out.splitlines() + [l for o in outs[1:] for l in o.splitlines() if l.startswith('RC ')]:
        t = l.split(' ')
        if t[0] == 'NUM':
            num[t[1]] = int(t[2])
        elif t[0] == 'BLK':
            blk.append((int(t[1]), int(t[2]), int(t[3])))
        elif t[0] == 'PI':
            pi.setdefault(int(t[1]), []).append(t[3] if len(t) > 3 else '')
        elif t[0] == 'SC':
            sc[int(t[1])] = t[2]
        elif t[0] == 'RC':
            rc[int(t[1])] = t[2] if len(t) > 2 else ''
        elif t[0] == 'CTR':
            ctr = int(t[1])
    ct = c.get('cassels_tate')
    need(printed, FIELD_CHECKS + tuple('%s_%d' % (k, p) for p in SP for k in ('roots', 'points')) +
         (('rank_lower_bound',) if 'points' in rl else ()) + (('cassels_tate',) if ct is not None else ()) +
         (('s_dropped',) if c.get('s_dropped') else ()), ok, out)
    if failed():        # no relation check on data that failed a check
        return st, 'points' in rl
    if len(blk) != len(SP) + 1 or len(sc) != len(c['s_columns']) or set(rc) != set(used) or len(num) != 6:
        ok('local_data', False, 'gp did not compute every class')
        return st, 'points' in rl
    rows, names = gp_rows(fd['poly'], cur['bad_primes'], rels, jobs, gp=gp, seconds=left())
    ok('relations_checked', True, '%d rows recomputed' % len(rows))
    inS = {i for i, z in enumerate(names) if int(z.split(':')[0]) in set(SP)}
    bad = [i for i, J in enumerate(c['elements']) if any(col not in inS for col in _xor(rows[j] for j in J))]
    ok('elements_in_KS2', not bad, 'the rows of every element vanish outside S' if not bad else
       'element %d has an odd valuation outside S' % bad[0])
    B = fd['bound']['B']
    ops, piv = [tuple(o) for o in c['ops']], c['pivots']
    nunit = len(c['s_columns'])
    ops_s, piv_s = [tuple(o) for o in c.get('ops_s', [])], c.get('pivots_s')
    rng = lambda xs, n: all(0 <= x < n for x in xs)
    if not (rng([x for o in ops for x in o] + piv, nrel) and
            rng([x for o in ops_s for x in o] + (piv_s or []), nrel + nunit)):
        ok('ops_in_range', False, 'a row index of ops, pivots, ops_s or pivots_s is out of range')
        return st, 'points' in rl
    ok('ops_in_range', True)
    if piv_s is None:
        dimV, nl, _ = rank_V([set(x) for x in rows], names, ops, piv, B, jobs, seconds=left(), info=rank_info)
        G = k = fd['bound']['ns'] - dimV
    else:
        names2 = list(names)
        col = {z: i for i, z in enumerate(names2)}
        units = []
        for t in range(nunit):
            if sc[t] not in col:
                col[sc[t]] = len(names2)
                names2.append(sc[t])
            units.append({col[sc[t]]})
        R2 = [set(x) for x in rows] + units
        more = sorted(set(piv_s) - set(piv))
        if ops_s == ops and set(piv) <= set(piv_s):
            # one run: the same ops, and piv_s contains piv, so dim V comes from the first rows
            (dimVs, _, _), (dimV, nl, _) = rank_V(R2, names2, ops, list(piv) + more, B, jobs, seconds=left(),
                                                prefix=len(piv), info=rank_info)
        else:
            dimV, nl, _ = rank_V([set(x) for x in rows], names, ops, piv, B, jobs, seconds=left(), info=rank_info)
            dimVs, _, _ = rank_V(R2, names2, ops_s, piv_s, B, jobs, seconds=left(), info=rank_info)
        G = fd['bound']['ns'] - dimV
        k = fd['bound']['ns'] - dimVs
    U = num['r1'] + num['r2'] + num['nSK'] + k
    sizes = [(p, n, d) for p, n, d in blk]
    width = sum(n for p, n, d in sizes if p != 0)
    r1 = num['r1']
    nN, nR, nQ = 1 + len(SP), len(c['characters']['real']), len(c['characters']['primes'])
    if any(len(rc[j]) != width + r1 + nN + nR + nQ or not _BITS.fullmatch(rc[j]) for j in used):
        ok('local_data', False, 'gp printed classes of the wrong length or not as bits')
        return st, 'points' in rl
    X = [_xor_bits(rc[j] for j in J) for J in c['elements']]
    v = len(X)

    def part(s, lo, n):
        return _bits(s[lo:lo + n])

    offs, o = {}, 0
    for p, n, d in sizes:
        if p != 0:
            offs[p] = (o, n)
            o += n
    offs[0] = (width, r1)
    chars = [part(s, width + r1 + nN, nR + nQ) for s in X]
    rk = len(_echelon(chars))
    ok('characters', rk == v, 'the %d x %d character matrix has rank %d' % (v, nR + nQ, rk))
    ok('v_le_U', v <= U, 'v = %d <= U = %d' % (v, U))
    conds = [[] for _ in X]
    good = True
    for p, n, d in sizes:
        lo, w = offs[p]
        pts = [_bits(s) for s in pi.get(p, [])]
        if any(len(s) != w for s in pi.get(p, [])):
            good = False
            ok('local_data', False, 'gp printed a point class of the wrong length at %s' % (p or 'inf'))
        Bp = _echelon(pts)
        if not v:
            ok('image_%s' % (p or 'inf'), True, 'not needed: no elements (v = 0)')
            continue
        good = good and len(Bp) == d
        ok('image_%s' % (p or 'inf'), len(Bp) == d, 'the points span a space of dimension %d = dim E(Q_%s)/2' % (
            len(Bp), p or 'R'))
        for i, s in enumerate(X):
            conds[i].append((_reduce(part(s, lo, w), Bp), w))
    for i, s in enumerate(X):
        conds[i].append((part(s, width + r1, nN), nN))           # N(x) in Q*^2
    if not good:
        return st, 'points' in rl
    dsel = v - len(_echelon([_join(cd) for cd in conds]))
    S_bk = parity_bound(G, num['u'], num['n'], num['w'])
    got = {}
    if c['format'] != FORMAT6:          # dim Sel_2 <= dim(Sel_2 cap V) + U - v, then 2-parity
        S = min(dsel + U - v if v == U else parity_floor(dsel + U - v, num['w']), S_bk)
        if v == U:
            ok('parity', (-1) ** dsel == num['w'], 'v = U: dim Sel_2 = %d, w = %d' % (dsel, num['w']))
    else:                               # Sel_2 lies in ker(N: K(S, 2) -> Q(S, 2)), of dim <= U - 1 - #S
        vkn = v - len(_echelon([part(s, width + r1, nN) for s in X]))
        slack = U - 1 - len(SP) - vkn
        ok('norm_kernel', slack >= 0, 'dim(V cap ker N) = %d <= U - 1 - #S = %d - 1 - %d' % (vkn, U, len(SP)))
        if slack < 0:
            return st, 'points' in rl
        if slack == 0:                  # ker N lies in V
            S_norm = dsel
            ok('parity', (-1) ** dsel == num['w'], 'ker N in V: dim Sel_2 = %d, w = %d' % (dsel, num['w']))
        else:
            S_norm = parity_floor(dsel + slack, num['w'])
        S = min(S_norm, S_bk)
        got = {'dim_V_cap_ker_N': vkn, 'S_norm': S_norm}
    for k_, x in (('u', num['u']), ('n', num['n'])):
        for d in (loc, cl):
            if k_ in d:
                ok('claim_' + k_, d[k_] == x, '%s = %d' % (k_, x))
    got.update({'G': G, 'dimV': dimV, 'k': k, 'U': U, 'v': v, 'dim_sel_cap_V': dsel, 'S_bk': S_bk, 'S': S,
                'w': num['w']})
    names = CLAIMS6 if c['format'] == FORMAT6 else CLAIMS
    wrong = [x for x in names if cl[x] != got[x]]
    ok('claims', not wrong, ', '.join('%s = %d' % (x, got[x]) for x in names) +
       ('; claimed otherwise: %s' % ', '.join(wrong) if wrong else ''))
    ok('hypothesis', c['hypothesis'] == ('GRH' if fd['bound']['type'] == 'BDF' else 'none'),
       'hypothesis %s for the %s bound' % (c['hypothesis'], fd['bound']['type']))
    if ct is not None:
        good = ctr is not None and ctr >= 0 and ctr == ct['rank'] and cl['S_ctp'] == S - ctr and S - ctr >= 0
        ok('S_ctp', good, 'S_ctp = S - rank CT = %d - %s' % (S, ctr))
        if not good:
            return st, 'points' in rl
        S = S - ctr
    st = selmer_statement(S, c['hypothesis'], r, None if 'points' in rl else rl.get('source'), word(c))
    ok('statement', c['statement'] == st)
    if r is not None:
        ok('bound_ge_r', S >= r, 'rank <= %d, rank >= %d' % (S, r))
    return st, 'points' in rl


def _xor(sets):
    out = set()
    for s in sets:
        out ^= s
    return out


_BITS = re.compile('[01]*')


def _xor_bits(strs):
    """Return the F_2 sum of bit strings of one validated common width, keeping leading zeros ('' for none)."""
    n, acc = None, 0
    for s in strs:
        if n is None:
            n = len(s)
        if s:
            acc ^= int(s, 2)
    return format(acc, '0%db' % n) if n else ''
