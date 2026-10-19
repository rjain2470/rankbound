"""The F_2 linear algebra and the certificate: relations to rows, the elimination by f2elim.c, and the certificate
file (format: rankbound.verify).  Given relations are read and checked here too.

A relation a - b alpha gives the row of its odd valuations: each prime p of odd multiplicity in F(a, b) gives the
label (p, r) with (a:b) = (r:1) mod p, or r = p when p | b, and the J labels are added mod 2.  This is exact for p not
dividing lc(F) and checked by field.gp for the others; the verifier recomputes every row in PARI regardless."""
import array, collections, datetime, gzip, hashlib, json, logging, lzma, math, os, re, shutil, signal, subprocess, \
    tempfile, threading
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional
from .verify import FORMAT, MAX_DIGITS, parity_bound, run_gp, statement
from . import RankboundError, field as fld, sieve, write_json

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f2elim')     # built by `make build`


# ---------- relations to rows

def relation_labels(a, b, primes, J):
    """Return the odd-valuation labels of a - b alpha."""
    odd = {}
    for p in primes:
        odd[p] = odd.get(p, 0) ^ 1
    out = set()
    for p, o in odd.items():
        if o:
            out.add((p, p) if b % p == 0 else (p, a * pow(b, -1, p) % p))
    for l in J:
        out ^= {l}
    return out


def free_rows(F, fb_path, pmax):
    """Yield the free relations (p, labels) for p < pmax: unramified p with three roots of F, and ramified p whose
    primes all have degree 1."""
    ram = {int(r['p']): r for r in F['ramified']}
    with gzip.open(fb_path, 'rt') as fb:
        for l in fb:
            if l[0] == '#' or l.count(':') != 1:
                continue
            ps, rs = l.split(':')
            p = int(ps)
            if p >= pmax:
                break
            if p in ram:
                continue
            roots = [int(x) for x in rs.split(',') if x.strip()]
            if len(set(roots)) == 3:
                yield p, {(p, r) for r in roots}
    for p, r in sorted(ram.items()):
        if p < pmax and r['all_degree_1']:
            yield p, {(p, lab[0]) for lab in r['labels'] if lab[1] % 2}


# ---------- given relations (rankbound(relations=...))

INT = r'(0|-?[1-9][0-9]*)'
GIVEN = re.compile(INT + ',' + INT + r'(?::((?:[1-9a-f][0-9a-f]*(?:,[1-9a-f][0-9a-f]*)*)?))?')
FREE = re.compile(r'f:[1-9][0-9]*')


class Given(namedtuple('Given', 'rels digest')):
    """Relations given to a run: rels [(a, b, primes or None)], and digest, the sha256 of their sorted canonical
    lines."""


def parse_relation(x):
    """Parse one given relation, 'a,b', 'a,b:hexprimes', (a, b) or (a, b, [primes]), in the coordinates of the index
    form F.  Return (a, b, primes or None), or None for a free relation 'f:p'; raise ValueError if it is malformed or
    too long for the verifier."""
    if isinstance(x, str) and len(x.strip()) > MAX_DIGITS:
        raise ValueError(too_long(x))
    if isinstance(x, str):
        if FREE.fullmatch(x.strip()):
            return None
        m = GIVEN.fullmatch(x.strip())
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


def read_relations(path):
    """Return the relations of a certificate, without the free ones, or of a text file with one relation per line
    (blank and '#' lines are skipped)."""
    try:
        with open_file(path) as f:
            if path.endswith(('.cert.json', '.cert.json.gz', '.cert.json.xz')):
                c = json.load(f)
                lines = c.get('relations') if isinstance(c, dict) and c.get('format') == FORMAT else None
                if not isinstance(lines, list):
                    raise ValueError('%s is not a certificate with relations.' % path)
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
    rs = read_relations(os.fspath(rels)) if isinstance(rels, (str, os.PathLike)) else \
        [r for r in map(parse_relation, rels) if r is not None]
    if not rs:
        raise ValueError('No relations given.')
    text = '\n'.join(sorted({'%d,%d' % (a, b) + ('' if ps is None else ':' + ','.join('%x' % p for p in sorted(ps)))
                             for a, b, ps in rs}))
    return Given(rs, hashlib.sha256(text.encode()).hexdigest())


def given_lines(F, rels, gp='gp'):
    """Return the given relations as slice lines 'a,b:hexprimes' and the number that PARI factored.  Raises
    ValueError for a relation with gcd(a, b) != 1, or whose primes are not the prime factors of |F(a, b)|."""
    vals, todo = [], []
    for k, (a, b, ps) in enumerate(rels):
        v = abs(fld.norm(F['poly'], a, b))
        if len(str(a)) + len(str(b)) + v.bit_length() // 4 > MAX_DIGITS:    # a, b and the primes of |F(a, b)| in hex
            raise ValueError('Relation %d is too long for the verifier.' % (k + 1))
        if math.gcd(a, b) != 1 or v == 0:
            raise ValueError('In relation %d (%d,%d), a and b are not coprime.' % (k + 1, a, b))
        if ps is not None:
            prod = 1
            for p in ps:
                prod *= p
            if prod != v or not all(p >= 2 and sieve.probable_prime(p) for p in ps):
                raise ValueError('The primes given for relation %d (%d,%d) are not the prime factors of |F(a, b)|.' % (
                    k + 1, a, b))
        else:
            todo.append(k)
        vals.append(v)
    found = {}
    if todo:
        d = tempfile.mkdtemp(prefix='rankbound')
        try:
            vf = os.path.join(d, 'v.gpv')
            with open(vf, 'w') as f:
                f.writelines('%d\n' % vals[k] for k in todo)
            out = run_gp('{ my(V = readvec("%s")); for(i = 1, #V, my(fa = factor(V[i]), L = List());\n'
                         '  for(j = 1, #fa~, for(t = 1, fa[j, 2], listput(L, Str(fa[j, 1]))));\n'
                         '  print("R ", i, " ", strjoin(Vec(L), ","))); }' % vf, seconds=24 * 3600, gp=gp)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        for l in out.splitlines():
            t = l.split()
            if t and t[0] == 'R':
                found[todo[int(t[1]) - 1]] = [int(x) for x in t[2].split(',')] if len(t) > 2 else []
        if len(found) != len(todo):
            raise RuntimeError('PARI/GP stopped early (%d of %d relations factored).' % (len(found), len(todo)))
    lines = []
    for k, (a, b, ps) in enumerate(rels):
        ps = found[k] if ps is None else ps
        lines.append('%d,%d:%s' % (a, b, ','.join('%x' % p for p in ps)))
    return lines, len(todo)


class Matrix:
    def __init__(self, F, small):
        self.ns = len(small)
        self.col = {lab: i for i, lab in enumerate(small)}
        self.J = [tuple(l) for l in F['J']] + [('J', z) for z in F.get('J_large', [])]
        self.src = []            # per row: the relation line, or "f:p"
        self.rows = []           # per row: array of column ids
        self.seen = set()

    def _add(self, labels, src):
        col = self.col
        ids = array.array('I', (col.setdefault(l, len(col)) for l in labels))
        self.rows.append(ids)
        self.src.append(src)

    def add_relation(self, line):
        a, b, primes = sieve.parse_line(line)
        if (a, b) in self.seen:
            return False
        self.seen.add((a, b))
        self._add(relation_labels(a, b, primes, self.J), line.strip())
        return True

    def add_free(self, p, labels):
        self._add(labels, 'f:%d' % p)

    def write(self, path):
        with open(path + '.%d.tmp' % os.getpid(), 'wb') as f:
            array.array('I', [len(self.rows), len(self.col)]).tofile(f)
            for r in self.rows:
                array.array('I', [len(r)]).tofile(f)
                r.tofile(f)
        os.replace(path + '.%d.tmp' % os.getpid(), path)


# ---------- the bound

@dataclass
class SolveResult:
    bound: Optional[int]          # ns - dim V; None if f2elim gave no bound
    dimV: Optional[int]
    rank: Optional[int]
    pivots: Optional[int]
    excess: Optional[int]         # the purge excess
    ops: str                      # the file of row operations
    piv: str                      # the file of pivot rows
    threaded: Optional[int] = None      # dense panels updated on threads
    too_dense: bool = False             # the dense phase did not fit in memory
    dense_mb: Optional[float] = None    # the dense phase's size
    killed: bool = False                # f2elim was killed (SIGKILL)


WMAX = 64   # f2elim's merge weight limit
DENSE_SHARE = 0.75      # the dense phase's share of the memory
PROGRESS = 300          # seconds between f2elim's progress lines


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def rss(pids):
    """Return the resident memory in bytes of the processes pids, from /proc (0 without /proc)."""
    tot = 0
    for pid in pids:
        try:
            with open('/proc/%d/status' % pid) as f:
                tot += next((int(l.split()[1]) * 1024 for l in f if l.startswith('VmRSS:')), 0)
        except (OSError, ValueError, IndexError):
            pass
    return tot


def dense_limit(reserve=0):
    """Return the bytes f2elim may hold in its dense phase: DENSE_SHARE of the job's memory (Slurm's, else the
    machine's) less reserve, at least 1, or 0 (no limit) if the memory cannot be read.  Above it, f2elim gives no
    bound instead of being killed."""
    try:
        mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        return 0
    e = os.environ
    try:
        node, cpu, cpus = (int(e.get(k) or 0) for k in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_CPUS_ON_NODE'))
        if node > 0:
            mem = min(mem, node << 20)
        elif cpu > 0 and cpus > 0:
            mem = min(mem, (cpu * cpus) << 20)
    except ValueError:
        pass
    return max(1, int(DENSE_SHARE * (mem - reserve)))


def _files_ok(out, res):
    """Return whether out.ops and out.piv have the sizes in f2elim's result line."""
    try:
        return (os.path.getsize(out + '.ops') == 8 * res['ops'] and os.path.getsize(out + '.piv') == 4 * res['pivots'])
    except (OSError, KeyError):
        return False


def _stamp(out):
    """Return (inode, size, mtime) of out.ops and out.piv, or None if one is missing."""
    try:
        return tuple((s.st_ino, s.st_size, s.st_mtime_ns) for s in (os.stat(out + '.ops'), os.stat(out + '.piv')))
    except OSError:
        return None


def _parse(stdout):
    """Parse f2elim's stdout into (purge statistics, result values, dense MB)."""
    purge, res, mb = {}, {}, None
    for l in stdout.splitlines():
        t = l.split()
        kv = {t[i]: t[i + 1] for i in range(1, len(t) - 1, 2)}
        if t[0] == 'purge':
            purge = {k: float(v) for k, v in kv.items()}
        elif t[0] == 'dense' and 'MB' in t:
            mb = float(t[t.index('MB') + 1])
        elif t[0] == 'result':
            res.update({k: int(v) for k, v in kv.items() if k in ('rank', 'dimV', 'bound', 'pivots', 'ops', 'threaded')})
    return purge, res, mb


def run_engine(args, log):
    """Run f2elim with args and return (exit status, stdout, the last lines of stderr); f2elim is killed if this is
    interrupted."""
    tail = collections.deque(maxlen=40)
    def relay(f):
        for l in f:
            tail.append(l)
    with subprocess.Popen([ENGINE] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:
        th = threading.Thread(target=relay, args=(p.stderr,), daemon=True)
        th.start()
        try:
            out = p.stdout.read()
            p.wait()
        except BaseException:
            p.kill()
            raise
        finally:
            th.join()
    return p.returncode, out, ''.join(tail)


def solve(M, work, target, gate=False, keep=1000, dense=1000, threads=1, dense_max=None, log=None, progress=None,
          others=()):
    """Run f2elim on M and return a SolveResult.  There is no bound if f2elim stops after the purge (gate and a
    negative excess), if the dense phase needs more than dense_max bytes, or if f2elim is killed.  others: processes
    whose memory counts against dense_max."""
    if not os.path.exists(ENGINE):
        raise RuntimeError('f2elim is not built. Run `make build` in the rankbound folder.')
    os.makedirs(work, exist_ok=True)
    inp = os.path.join(work, 'matrix.bin')
    M.write(inp)
    out = os.path.join(work, 'elim')
    t = target if target is not None and target > 0 else 0
    # An elimination of the same matrix, settings and binary, recorded in elim.json, is reused.
    key = '%s ns %d target %d dense %d wmax %d keep %d f2elim %s' % (sha256_file(inp), M.ns, t, dense, WMAX, keep,
                                                                     sha256_file(ENGINE))
    rec = out + '.json'
    stdout = None
    if os.path.exists(rec):
        with open(rec) as f:
            d = json.load(f)
        if d.get('key') == key and _files_ok(out, _parse(d['stdout'])[1]):
            stdout = d['stdout']
    too_dense = killed = False
    if stdout is None:
        # The record is removed while f2elim runs, and restored if f2elim ends without a result and leaves the old
        # files in place.
        old = None
        if os.path.exists(rec):
            with open(rec) as f:
                old = f.read()
            os.remove(rec)
        before = _stamp(out)
        if dense_max is None:
            dense_max = dense_limit(rss([os.getpid()] + list(others)))
        log = log or logging.getLogger('rankbound').info
        rc, stdout, err = run_engine([inp, str(M.ns), str(t), out, str(dense), str(WMAX), str(int(gate)), str(keep),
                                      str(max(1, threads)), str(dense_max), str(PROGRESS if progress is None else progress)],
                                     log)
        killed = rc == -signal.SIGKILL
        too_dense = rc == 3 or killed
        if killed:
            log('The elimination ran out of memory and was stopped; no bound from this check.')
        if rc and not too_dense:
            raise RuntimeError('The elimination failed: %s' % next(
                (l.strip() for l in reversed(err.splitlines()) if l.strip()), 'exit %d' % rc))
        _, res, _ = _parse(stdout)
        if res and not _files_ok(out, res):      # a last check of f2elim's files
            raise RuntimeError("The elimination's output is inconsistent. Please report it.")
        if res:
            write_json(rec, {'key': key, 'stdout': stdout}, None)
        elif old is not None and before is not None and _stamp(out) == before:
            with open(rec + '.tmp', 'w') as f:
                f.write(old)
            os.replace(rec + '.tmp', rec)
    purge, res, mb = _parse(stdout)
    os.remove(inp)
    return SolveResult(bound=res.get('bound'), dimV=res.get('dimV'), rank=res.get('rank'), pivots=res.get('pivots'),
                       excess=int(purge['excess']) if 'excess' in purge else None, ops=out + '.ops', piv=out + '.piv',
                       threaded=res.get('threaded'), too_dense=too_dense, dense_mb=mb, killed=killed)


def cone(ops_path, piv_path):
    """Return the ops and rows the pivot rows depend on: (kept ops [(dst, src)], pivots, needed rows sorted)."""
    ops = array.array('I'); piv = array.array('I')
    with open(ops_path, 'rb') as f:
        ops.frombytes(f.read())
    with open(piv_path, 'rb') as f:
        piv.frombytes(f.read())
    need = set(piv); kept = []
    for t in range(len(ops) // 2 - 1, -1, -1):
        d, s = ops[2 * t], ops[2 * t + 1]
        if d in need:
            need.add(s); kept.append((d, s))
    kept.reverse()
    return kept, list(piv), sorted(need)


# ---------- the certificate

PRIME_SEARCH = 10 ** 6      # primes searched when the given ones fall short


def rank_proof(ainvs, bad_primes, r, proof, gp='gp'):
    """Return the rank lower bound {'r', 'points', 'primes'} on the model ainvs: the proof's points mapped onto it,
    and r primes at which their 2-signatures have rank r, the proof's own first.  Raises RuntimeError if there are
    not r such points and primes."""
    fld.check_proof(proof)
    r = fld.r_value(r)
    if len(proof['points']) != r:
        raise RuntimeError('r = %d but %d points were given.' % (r, len(proof['points'])))
    a = lambda v: ','.join(fld.canon(x, 'a-invariant') for x in fld._ainvs5(v))
    body = 'rkproof([%s], [%s], [%s], %d, [%s], [%s], %d);' % (
        a(proof.get('ainvs') or ainvs), a(ainvs), ','.join(fld.canon(p, 'bad prime') for p in bad_primes), r,
        ','.join('[%s,%s]' % tuple(p) for p in proof['points']), ','.join(map(str, proof.get('primes') or [])),
        PRIME_SEARCH)
    out = run_gp(body, libs=(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field.gp'),), seconds=600, gp=gp)
    pts, prs = [], None
    for l in out.splitlines():
        t = l.split('\t')
        if t[0] == 'PT':
            pts.append([fld.canon(t[1], 'mapped point', fld.RAT), fld.canon(t[2], 'mapped point', fld.RAT)])
        elif t[0] == 'PR':
            prs = [int(x) for x in t[1].split(',') if x]
    if len(pts) != r or prs is None or len(prs) != r:
        raise RuntimeError('The points do not prove rank >= %d. At every usable prime below 10^%d they look dependent; '
                           'check them.' % (r, round(math.log10(PRIME_SEARCH))))
    return {'r': r, 'points': pts, 'primes': prs}


TAIL = ',\n "relations": ['       # where a certificate's proof data start


def head_text(c):
    """Return the fields of the certificate c before "relations", as the certificate writes them."""
    return json.dumps(c, indent=1)[:-2]


def certificate(state, M, res, path, proof, source, gp='gp'):
    """Write the certificate path for the solve res of M, on the reduced minimal model, with only the rows and ops the
    pivots depend on, and return it as a dict.  proof: the points of the rank lower bound; else source says where r is
    from.  Raises RankboundError, writing nothing, if S < r."""
    F = fld.load(state)
    ainvs = [str(x) for x in F['minimal_model']]
    rl = None
    r = F.get('rank_lower_bound')
    S = parity_bound(res.bound, F['u'], F['n'], F['w'])
    if r is not None and S < r:
        raise RankboundError('The relations prove rank <= %d under GRH, but r = %d was given. No certificate written. %s'
                             % (S, r, 'This contradicts the points: please report it.' if proof else
                                'Check r, or run without it.'))
    if r is not None:          # the rank proof first: it is cheap
        rl = rank_proof(ainvs, F['bad_primes'], r, dict(proof, ainvs=proof.get('ainvs') or F['input_ainvs']), gp) \
            if proof else {'r': r, 'source': source}
    kept, piv, need = cone(res.ops, res.piv)
    idx = {i: k for k, i in enumerate(need)}
    G = res.bound
    S, st = statement(G, F, r, None if proof else source)
    c = {'name': F['name'],
         'curve': {'ainvs': ainvs, 'minimal_model': F['minimal_model'], 'bad_primes': F['bad_primes']},
         'statement': st}
    if rl:
        c['rank_lower_bound'] = rl
    c.update(field={'poly': [str(x) for x in F['poly']], 'dK': F['dK'], 'GB': F['GB'], 'ns': F['ns']},
             local={'u': F['u'], 'n': F['n'], 'w': F['w']}, claims={'G': G, 'dimV': res.dimV, 'S': S},
             created=datetime.date.today().isoformat())
    data = {'relations': [M.src[i] for i in need], 'ops': [[idx[d], idx[s]] for d, s in kept],
            'pivots': [idx[i] for i in piv]}
    body = ',\n'.join(' "%s": [\n%s\n ]' % (k, ',\n'.join('  ' + json.dumps(x, separators=(',', ':')) for x in v))
                      for k, v in data.items())
    assert body.startswith(TAIL[2:])
    with open(path + '.%d.tmp' % os.getpid(), 'w') as f:
        f.write(head_text(c) + ',\n' + body + ',\n "format": %s\n}\n' % json.dumps(FORMAT))
    os.replace(path + '.%d.tmp' % os.getpid(), path)
    return dict(c, **data, format=FORMAT)


# ---------- a certificate rewritten on its minimal model

HEAD_MAX = 1 << 26      # bytes in which the fields before "relations" end


def read_head(path):
    """Return (the fields of the certificate before "relations", the byte offset of TAIL), reading only that far;
    raise ValueError if the file is not a certificate as rankbound writes it."""
    tail, buf = TAIL.encode(), b''
    with open(path, 'rb') as f:
        while True:
            k = buf.find(tail)
            if k >= 0:
                break
            chunk = f.read(1 << 20)
            if not chunk or len(buf) > HEAD_MAX:
                raise ValueError('%s: no %r in its first %d bytes: not a certificate as rankbound writes it' % (
                    path, TAIL, len(buf)))
            buf += chunk
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 200))
        if not f.read().rstrip().endswith(('"format": %s\n}' % json.dumps(FORMAT)).encode()):
            raise ValueError('%s: its last line is not the format %s' % (path, FORMAT))
    return json.loads(buf[:k].decode() + '\n}'), k


def recert(src, dst, gp='gp'):
    """Rewrite the certificate src on its reduced minimal model as dst, copying the proof data unchanged; the points
    are mapped onto the new model and checked with the verifier's vrank.  dst is never overwritten.  Returns None if
    src is on its minimal model already, else the new fields before "relations"."""
    if os.path.exists(dst):
        raise ValueError('%s exists: not overwritten' % dst)
    head, off = read_head(src)
    cur = head['curve']
    old, mm = [fld.canon(x, 'ainvs') for x in cur['ainvs']], [fld.canon(x, 'minimal_model') for x in cur['minimal_model']]
    if old == mm:
        return None
    if fld.from_ainvs(old, bad_primes=cur['bad_primes'], gp=gp).minimal_model != mm:
        raise ValueError('%s: curve.minimal_model is not the reduced minimal model of curve.ainvs' % src)
    new = dict(head, curve=dict(cur, ainvs=mm))
    rl = head.get('rank_lower_bound')
    if rl and 'points' in rl:
        p = rank_proof(mm, cur['bad_primes'], rl['r'], {'points': rl['points'], 'primes': rl['primes'], 'ainvs': old},
                       gp)
        new['rank_lower_bound'] = {k: p[k] for k in rl}
        pts = ','.join('[%s,%s]' % tuple(x) for x in p['points'])
        chk = run_gp('vrank([%s], %d, [%s], [%s]);' % (','.join(mm), p['r'], pts, ','.join(map(str, p['primes']))),
                     libs=('verify.gp',), gp=gp)
        if 'CHK rank_lower_bound 1' not in chk:
            raise RuntimeError('%s: the mapped points fail the verifier\'s vrank on the minimal model' % src)
    text = head_text(new).encode()
    if json.loads(text.decode() + '\n}') != new:
        raise RuntimeError('recert: the new fields do not read back')
    tmp = '%s.tmp%d' % (dst, os.getpid())
    try:
        with open(src, 'rb') as f, open(tmp, 'wb') as g:
            g.write(text)
            f.seek(off)
            shutil.copyfileobj(f, g, 1 << 24)
            g.flush()
            os.fsync(g.fileno())        # no short dst after a crash
        os.link(tmp, dst)           # never overwrites dst
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return new
