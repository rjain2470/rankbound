"""F_2 linear algebra by the native f2elim: relations to rows, and their elimination.

A row's labels are exact for primes not dividing lc(F); field.py checks the others, and the checker recomputes every
row.
"""
import array, gzip, hashlib, itertools, json, logging, os, shutil, signal, tempfile
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional
from . import machine, runtime
from .runtime import read_json, write_json
from .arith import is_prime

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f2elim')     # built by `make build`
KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'f2kernel')


# ---------- relations to rows

def parse_line(line):
    """Parse a slice line 'a,b:hexprimes' into (a, b, [primes of F(a, b)])."""
    ab, s0 = line.split(':', 1)
    a, b = ab.split(',')
    return int(a), int(b), [int(h, 16) for h in s0.split(',')] if s0.strip() else []


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
    """Yield the free relations (p, labels) for p < pmax whose primes all have degree 1."""
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


def _size_time(path):
    """Return (size, mtime) of path, or None, to detect another writer."""
    try:
        s = os.stat(path)
        return s.st_size, s.st_mtime_ns
    except OSError:
        return None


INT62 = 1 << 62         # an element (a, b) beyond this has no characters


def ncols(M):
    """Return the column count of M's header: its columns, or M.pad if larger.

    The pad lets a resume rewrite an earlier solve's matrix with the same columns, so its recorded elimination holds.
    """
    return max(len(M.col), M.pad)


class Rows:
    """The rows of a Matrix in memory, in f2elim's layout, in one flat uint32 array (an array per row costs ~80 bytes).

    rows[:n] shares the array; rows are only appended, or the last popped, never below a prefix in use.
    """
    __slots__ = ('data', 'off', 'lim')

    def __init__(self, rows=()):
        self.data, self.off, self.lim = array.array('I'), array.array('Q'), None
        for r in rows:
            self.append(r)

    def __len__(self):
        return len(self.off) if self.lim is None else self.lim

    def append(self, ids):
        if self.lim is not None:
            raise TypeError('a prefix of rows is read-only')
        self.off.append(len(self.data))
        self.data.append(len(ids))
        self.data.extend(ids)

    def pop(self):
        if self.lim is not None:
            raise TypeError('a prefix of rows is read-only')
        o = self.off.pop()
        r = self.data[o + 1:]
        del self.data[o:]
        return r

    def _end(self, j):
        """Return the offset in data after the first j rows."""
        return self.off[j] if j < len(self.off) else len(self.data)

    def __getitem__(self, i):
        n = len(self)
        if isinstance(i, slice):
            i, j, step = i.indices(n)
            out = Rows()
            if i == 0 and step == 1:
                out.data, out.off, out.lim = self.data, self.off, max(j, 0)
            elif step == 1 and i < j:
                base = self.off[i]
                out.data = self.data[base:self._end(j)]
                out.off = array.array('Q', (o - base for o in self.off[i:j]))
            else:
                for k in range(i, j, step):
                    out.append(self[k])
            return out
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError('row index out of range')
        o = self.off[i]
        return self.data[o + 1:o + 1 + self.data[o]]

    def __iter__(self):
        d = self.data
        for t in range(len(self)):
            o = self.off[t]
            yield d[o + 1:o + 1 + d[o]]

    def tofile(self, f, start=0):
        """Write rows start.. to the binary file f."""
        if start < len(self):
            with memoryview(self.data) as m:
                f.write(m[self.off[start]:self._end(len(self))])


class _Spill:
    """An append-only temporary file of records, unlinked when made so it is removed on every exit."""
    __slots__ = ('f', 'off', 'size')

    def __init__(self, folder=None):
        self.f = tempfile.TemporaryFile(dir=folder)
        self.off, self.size = array.array('Q'), 0

    def __del__(self):
        f = getattr(self, 'f', None)
        if f is not None:
            f.close()

    def add(self, data):
        self.off.append(self.size)
        self.f.write(data)
        self.size += len(data)

    def pop(self):
        """Remove the last record and return its bytes."""
        o = self.off.pop()
        data = self.read(o, self.size)
        self.f.flush()
        os.ftruncate(self.f.fileno(), o)
        self.f.seek(o)
        self.size = o
        return data

    def end(self, j):
        """Return the offset after the first j records."""
        return self.off[j] if j < len(self.off) else self.size

    def read(self, a, b):
        """Return the bytes a..b-1."""
        self.f.flush()
        out, fd = [], self.f.fileno()
        while a < b:
            x = os.pread(fd, min(b - a, 1 << 22), a)
            if not x:
                raise OSError('a temporary file of the matrix is short')
            out.append(x)
            a += len(x)
        return b''.join(out)

    def chunks(self, a, b, size=1 << 22):
        """Yield the bytes a..b-1 in pieces of about size."""
        while a < b:
            n = min(b - a, size)
            yield self.read(a, a + n)
            a += n


class FileRows:
    """The rows of a Matrix on disk, with Rows' interface and 8 bytes a row in memory."""
    __slots__ = ('sp', 'lim')

    def __init__(self, folder=None, rows=()):
        self.sp, self.lim = _Spill(folder), None
        for r in rows:
            self.append(r)

    def __len__(self):
        return len(self.sp.off) if self.lim is None else self.lim

    def append(self, ids):
        if self.lim is not None:
            raise TypeError('a prefix of rows is read-only')
        a = array.array('I', [len(ids)])
        a.extend(ids)
        self.sp.add(a.tobytes())

    def pop(self):
        if self.lim is not None:
            raise TypeError('a prefix of rows is read-only')
        a = array.array('I')
        a.frombytes(self.sp.pop())
        return a[1:]

    def __getitem__(self, i):
        n = len(self)
        if isinstance(i, slice):
            i, j, step = i.indices(n)
            if i == 0 and step == 1:
                out = FileRows.__new__(FileRows)
                out.sp, out.lim = self.sp, max(j, 0)
                return out
            return Rows(self[k] for k in range(i, j, step))
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError('row index out of range')
        a = array.array('I')
        a.frombytes(self.sp.read(self.sp.off[i], self.sp.end(i + 1)))
        return a[1:]

    def __iter__(self):
        """Yield the rows in order, read in pieces of a few MB."""
        n, t, rest = len(self), 0, b''
        for piece in self.sp.chunks(0, self.sp.end(n)):
            a = array.array('I')
            data = rest + piece
            a.frombytes(data[:len(data) - len(data) % 4])
            o = 0
            while o < len(a) and o + 1 + a[o] <= len(a):
                yield a[o + 1:o + 1 + a[o]]
                o += 1 + a[o]
                t += 1
            rest = data[4 * o:]
        if t != n:
            raise OSError('a temporary file of the matrix is short')

    def tofile(self, f, start=0):
        """Write rows start.. to the binary file f."""
        if start < len(self):
            for piece in self.sp.chunks(self.sp.off[start], self.sp.end(len(self))):
                f.write(piece)


class FileLines:
    """The relation lines of a Matrix's rows on disk, with their offsets in memory."""
    __slots__ = ('sp',)

    def __init__(self, folder=None):
        self.sp = _Spill(folder)

    def __len__(self):
        return len(self.sp.off)

    def append(self, line):
        self.sp.add(line.encode() + b'\n')

    def pop(self):
        return self.sp.pop()[:-1].decode()

    def __getitem__(self, i):
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError('line index out of range')
        return self.sp.read(self.sp.off[i], self.sp.end(i + 1))[:-1].decode()

    def __iter__(self):
        rest = b''
        for piece in self.sp.chunks(0, self.sp.size):
            data = rest + piece
            cut = data.rfind(b'\n') + 1
            for l in data[:cut].split(b'\n')[:-1]:
                yield l.decode()
            rest = data[cut:]


def _mix(k, bits):
    """Return a slot of a table of 2^bits slots for k < 2^64 (Fibonacci hashing)."""
    return ((k * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF) >> (64 - bits)


class Seen:
    """The set of relations (a, b) added, for deduplication, in 4 to 8 bytes a relation.

    A half-full table of uint32 slots holds 1 + the row whose element in elt is (a, b); elements beyond int63 are kept
    by exact key in a set.
    """
    __slots__ = ('elt', 'slots', 'bits', 'n', 'big')

    def __init__(self, elt):
        self.elt, self.bits, self.n, self.big = elt, 10, 0, set()
        self.slots = array.array('I', bytes(4 << self.bits))

    def __len__(self):
        return self.n + len(self.big)

    def _slot(self, a, b):
        return _mix((a * 0x100000001B3 + b) & 0xFFFFFFFFFFFFFFFF, self.bits)

    def add(self, a, b, row):
        """Add the relation (a, b) of row, whose element goes to elt next; return False if already added."""
        if abs(a) >= INT62 or abs(b) >= INT62:
            k = seen_key(a, b)
            if k in self.big:
                return False
            self.big.add(k)
            return True
        if 2 * (self.n + 1) > 1 << self.bits:      # grow before inserting: the new element is not in elt yet
            self._grow()
        s, elt, mask = self.slots, self.elt, (1 << self.bits) - 1
        h = self._slot(a, b)
        while s[h]:
            i = 2 * (s[h] - 1)
            if elt[i] == a and elt[i + 1] == b:
                return False
            h = (h + 1) & mask
        s[h] = row + 1
        self.n += 1
        return True

    def _grow(self):
        old = self.slots
        self.bits += 1
        self.slots = s = array.array('I', bytes(4 << self.bits))
        mask, elt = (1 << self.bits) - 1, self.elt
        for v in old:
            if v:
                h = self._slot(elt[2 * (v - 1)], elt[2 * v - 1])
                while s[h]:
                    h = (h + 1) & mask
                s[h] = v


P32 = 1 << 32


class Columns:
    """A dict-like map from labels to column ids in order of first appearance, in about 20 bytes a column.

    A label (p, r) with p < 2^32 - 1 and r < 2^32 is the key 2^32 p + r + 1 of a half-full table; others are in a dict.
    """
    __slots__ = ('keys', 'vals', 'bits', 'lab', 'other', 'inv')

    def __init__(self, labels=()):
        self.bits = 10
        self.keys, self.vals = array.array('Q', bytes(8 << self.bits)), array.array('I', bytes(4 << self.bits))
        self.lab, self.other, self.inv = array.array('Q'), {}, {}
        for l in labels:
            self.id(l)

    def __len__(self):
        return len(self.lab)

    @staticmethod
    def _key(l):
        if type(l) is tuple and len(l) == 2:
            p, r = l
            if type(p) is int and type(r) is int and 0 <= p < P32 - 1 and 0 <= r < P32:
                return (p << 32 | r) + 1
        return 0

    def _find(self, k):
        """Return the slot of the key k, or the empty slot where it goes."""
        keys, mask = self.keys, (1 << self.bits) - 1
        h = _mix(k, self.bits)
        while True:
            x = keys[h]
            if x == k or not x:
                return h
            h = (h + 1) & mask

    def id(self, l):
        """Return the id of the label l, assigning the next id if l is new."""
        k = self._key(l)
        if not k:
            c = self.other.get(l)
            if c is None:
                c = self.other[l] = len(self.lab)
                self.inv[c] = l
                self.lab.append(0)
            return c
        h = self._find(k)
        if self.keys[h]:
            return self.vals[h]
        c = len(self.lab)
        self.keys[h], self.vals[h] = k, c
        self.lab.append(k)
        if 2 * len(self.lab) > (1 << self.bits):
            self._grow()
        return c

    def _grow(self):
        keys, vals = self.keys, self.vals
        self.bits += 1
        self.keys, self.vals = array.array('Q', bytes(8 << self.bits)), array.array('I', bytes(4 << self.bits))
        for x, v in zip(keys, vals):
            if x:
                h = self._find(x)
                self.keys[h], self.vals[h] = x, v

    def get(self, l, default=None):
        k = self._key(l)
        if not k:
            return self.other.get(l, default)
        h = self._find(k)
        return self.vals[h] if self.keys[h] else default

    def __getitem__(self, l):
        c = self.get(l)
        if c is None:
            raise KeyError(l)
        return c

    def __contains__(self, l):
        return self.get(l) is not None

    def label(self, c):
        """Return the label of the column c."""
        k = self.lab[c]
        return ((k - 1) >> 32, (k - 1) & (P32 - 1)) if k else self.inv[c]

    def __iter__(self):
        return (self.label(c) for c in range(len(self.lab)))

    def items(self):
        return ((self.label(c), c) for c in range(len(self.lab)))

    def with_prime(self, ps, lo=0):
        """Return the ids c >= lo, in order, of the labels whose prime is in ps."""
        out, lab, inv = [], self.lab, self.inv
        for c in range(lo, len(lab)):
            k = lab[c]
            if (((k - 1) >> 32) if k else inv[c][0]) in ps:
                out.append(c)
        return out


def seen_key(a, b):
    """Return the deduplication key of (a, b): one exact int when |b| < 2^63, else (a, b)."""
    return (a << 64) + (b + (1 << 63)) if -(1 << 63) <= b < 1 << 63 else (a, b)


class Matrix:
    pad = 0                     # the least column count of the header (ncols)
    maxp = 0                    # the largest prime of any relation; the characters lie above it

    def __init__(self, F, small, folder=None):
        """Start a matrix with small labels small; rows and lines are kept in temporary files in folder."""
        self.ns = len(small)
        self.col = Columns(small)      # label -> column id
        self.J = [tuple(l) for l in F['J']]
        self.sp = {2} | {int(p) for p in F.get('bad_primes', ())}     # the primes S of the 2-descent
        self.src = FileLines(folder)   # per row: the relation line, or "f:p"
        self.rows = FileRows(folder)   # per row: its column ids
        self.elt = array.array('q')     # per row: a, b of its element a - b alpha (a free relation p: p, 0)
        self.big = False         # whether an element did not fit in elt (stored as 0, 0)
        self.seen = Seen(self.elt)
        self.written = None      # (path, rows, bytes) of the last write

    def _add(self, labels, src, a=None, b=0):
        """Add the row of labels for the relation line src, with element a - b alpha (a None: none)."""
        cid = self.col.id
        self.rows.append([cid(l) for l in labels])
        self.src.append(src)
        if a is None or abs(a) >= INT62 or abs(b) >= INT62:
            self.big, a, b = True, 0, 0
        self.elt.append(a); self.elt.append(b)

    def add_relation(self, line):
        a, b, primes = parse_line(line)
        if not self.seen.add(a, b, len(self.rows)):
            return False
        if primes:
            self.maxp = max(self.maxp, max(primes))
        self._add(relation_labels(a, b, primes, self.J), line.strip(), a, b)
        return True

    def add_free(self, p, labels):
        self._add(labels, 'f:%d' % p, p, 0)

    def held(self):
        """Return f2elim's held columns: the large columns above 2 and the bad primes."""
        if isinstance(self.col, Columns):
            return self.col.with_prime(self.sp, self.ns)
        return sorted(c for z, c in self.col.items() if c >= self.ns and z[0] in self.sp)

    def write(self, path):
        """Write the matrix to path for f2elim; a second write to the same path appends the new rows."""
        w = self.written
        if w and w[0] == path and len(self.rows) >= w[1] and _size_time(path) == w[2]:
            f, start, tmp = open(path, 'r+b'), w[1], None
            f.seek(0, 2)
        else:
            tmp = path + '.%d.tmp' % os.getpid()
            f, start = open(tmp, 'wb'), 0
            f.write(bytes(8))
        with f:
            if isinstance(self.rows, (Rows, FileRows)):
                self.rows.tofile(f, start)
            else:
                for r in self.rows[start:]:
                    array.array('I', [len(r)]).tofile(f)
                    array.array('I', r).tofile(f)
            f.seek(0)
            array.array('I', [len(self.rows), ncols(self)]).tofile(f)
        if tmp is not None:
            os.replace(tmp, path)
        self.written = (path, len(self.rows), _size_time(path))


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
    vs: str = ''                  # the file of the basis of V_H (f2elim's OUT.vs), its held columns in held
    held: tuple = ()
    threaded: Optional[int] = None      # dense panels updated on threads
    too_dense: bool = False             # the dense phase did not fit in memory
    dense_mb: Optional[float] = None    # the dense phase's size
    killed: bool = False                # f2elim was killed (SIGKILL)
    late: Optional[float] = None        # stopped for the deadline: seconds the dense phase needs (0: unknown)
    dense_s: Optional[float] = None     # the dense phase's estimate (seconds at the threads)
    crank: Optional[int] = None         # the characters' rank on the full dependencies (None: no characters)
    disk: bool = False                  # too_dense because the blocks' temporary files did not fit the disk
    blocks: Optional[int] = None        # the dense phase in blocks of this many words (None: the whole matrix)
    need_mb: Optional[float] = None     # too_dense by memory: MB of the dense phase in blocks of one word


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
    """Return the resident bytes of the processes pids, from /proc (0 without it)."""
    tot = 0
    for pid in pids:
        try:
            with open('/proc/%d/status' % pid) as f:
                tot += next((int(l.split()[1]) * 1024 for l in f if l.startswith('VmRSS:')), 0)
        except (OSError, ValueError, IndexError):
            pass
    return tot


def dense_limit(reserve=0):
    """Return the bytes f2elim may hold in its dense phase, or 0 (no limit) if the memory is unknown.

    Above it f2elim gives no bound instead of being killed.
    """
    mem = machine.memory()
    if mem is None:
        return 0
    return max(1, int(DENSE_SHARE * (mem - reserve)))


def _files_ok(out, res, words):
    """Return whether out.ops, out.piv and out.vs have the sizes in f2elim's result line."""
    try:
        return (os.path.getsize(out + '.ops') == 8 * res['ops'] and os.path.getsize(out + '.piv') == 4 * res['pivots']
                and os.path.getsize(out + '.vs') == 8 * words * res['vs'])
    except (OSError, KeyError):
        return False


def _digests(out, exts):
    """Return the sha256 of out + each extension (None if missing).

    The record is written last, so a reuse matching these digests reads one generation of complete files.
    """
    return {e: sha256_file(out + e) if os.path.exists(out + e) else None for e in exts}


def vs_words(nheld, ns):
    """Return the uint64 words of a row of f2elim's OUT.vs."""
    return (nheld + 63) // 64 + (ns + 63) // 64


def _stamp(out):
    """Return (inode, size, mtime) of out.ops and out.piv, or None if one is missing."""
    try:
        return tuple((s.st_ino, s.st_size, s.st_mtime_ns) for s in (os.stat(out + e) for e in ('.ops', '.piv', '.vs')))
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
            res.update({k: int(v) for k, v in kv.items()
                        if k in ('rank', 'dimV', 'bound', 'crank', 'pivots', 'ops', 'threaded', 'vs')})
    return purge, res, mb


WATCHDOG_S = 2.0       # seconds past its deadline before a program is stopped (it should exit 4 by itself first)


def run_engine(args, budget=None, deadline=False, dense_s=None, env=None, engine=None):
    """Run f2elim (or engine) with args within the budget; return (status, stdout, last lines of stderr).

    With deadline, f2elim is told the end and exits 4 if it cannot finish; it is stopped WATCHDOG_S after it.
    """
    more, env = env or {}, {k: v for k, v in os.environ.items() if k != 'F2ELIM_DEADLINE'}
    env.update(more)
    if dense_s:
        env['F2ELIM_DENSE_S'] = '%g' % dense_s
    rc, out, err = runtime.run([engine or ENGINE] + args, budget=budget, env=env,
                               deadline_env='F2ELIM_DEADLINE' if deadline else None, slack=WATCHDOG_S)
    return rc, out, ''.join(err.splitlines(True)[-40:])


def solve(M, work, target, gate=False, keep=1000, dense=1000, threads=1, dense_max=None, log=None, progress=None,
          others=(), name='elim', budget=None, chars=(), dense_s=None, dense_cap_mb=None):
    """Run f2elim on M and return a SolveResult, with the files work/NAME.ops, .piv, .vs and .json.

    The dense phase works in blocks from temporary files when the matrix does not fit in dense_max. With a target and no
    characters, f2elim first tries the rows of a structural matching, a smaller witness if it reaches the target.
    """
    if not os.path.exists(ENGINE):
        raise RuntimeError('f2elim is not built. Run `make build` in the rankbound folder.')
    os.makedirs(work, exist_ok=True)
    inp = os.path.join(work, 'matrix.bin')
    M.write(inp)
    out = os.path.join(work, name)
    t = target if target is not None and target > 0 else 0
    cf = '-'
    if chars and not M.big:
        cf = os.path.join(work, 'chars.txt')
        with open(cf, 'w') as f:
            f.writelines('%d %d\n' % c for c in chars)
        with open(inp + '.ab', 'wb') as f:
            M.elt.tofile(f)
    held = M.held()
    hf = os.path.join(work, 'held.bin')
    with open(hf, 'wb') as f:
        array.array('I', held).tofile(f)
    # Reuse a recorded elimination of the same matrix, held columns, settings, characters and binary.
    key = '%s ns %d held %s target %d dense %d wmax %d keep %d dense_s %s cap %s chars %s ab %s f2elim %s' % (
        sha256_file(inp), M.ns, sha256_file(hf), t, dense, WMAX, keep, dense_s, dense_cap_mb,
        hashlib.sha256(repr(list(chars) if cf != '-' else []).encode()).hexdigest()[:16],
        sha256_file(inp + '.ab') if cf != '-' else '-', sha256_file(ENGINE))
    rec = out + '.json'
    stdout = None
    if os.path.exists(rec):
        with open(rec) as f:
            d = json.load(f)
        if d.get('key') == key and _files_ok(out, _parse(d['stdout'])[1], vs_words(len(held), M.ns)) and \
                d.get('digests') == _digests(out, ('.ops', '.piv', '.vs')):
            stdout = d['stdout']
    too_dense = killed = False
    late = None
    if stdout is None:
        # Remove the record while f2elim runs; restore it if f2elim ends without a result and the old files remain.
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
                                      str(max(1, threads)), str(dense_max), str(PROGRESS if progress is None else progress), '0', cf, hf],
                                     budget, True, dense_s,
                                     env={'F2ELIM_DENSE_MB': '%g' % dense_cap_mb} if dense_cap_mb else None)
        if rc == 4:
            late = next((float(l.split()[2]) for l in stdout.splitlines() if l.startswith('late ')), 0.0)
            rc = 0
        killed = rc == -signal.SIGKILL
        too_dense = rc == 3 or killed
        if killed:
            log('The elimination ran out of memory and was stopped; no bound from this check.')
        if rc and not too_dense:
            raise RuntimeError('The elimination failed: %s' % next(
                (l.strip() for l in reversed(err.splitlines()) if l.strip()), 'exit %d' % rc))
        _, res, _ = _parse(stdout)
        if res and not _files_ok(out, res, vs_words(len(held), M.ns)):
            raise RuntimeError("The elimination's output is inconsistent. Please report it.")
        if res:
            # Keep what a caller added to the replaced record, and the matrix's size (for Matrix.pad on resume).
            extra = {}
            if old is not None:
                try:
                    extra = {k: v for k, v in json.loads(old).items()
                             if k not in ('key', 'stdout', 'nrows', 'ncols', 'digests')}
                except ValueError:
                    pass
            write_json(rec, dict(extra, key=key, stdout=stdout, nrows=len(M.rows), ncols=ncols(M),
                                 digests=_digests(out, ('.ops', '.piv', '.vs'))), None)
        elif old is not None and before is not None and _stamp(out) == before:
            with open(rec + '.tmp', 'w') as f:
                f.write(old)
            os.replace(rec + '.tmp', rec)
    purge, res, mb = _parse(stdout)
    est = next((float(l.split()[-1]) for l in stdout.splitlines() if l.startswith('dense ') and ' s ' in l), None)
    bw = next((int(l.split()[2]) for l in stdout.splitlines() if l.startswith('blocks words ')), None)
    need = next((float(l.split()[3]) for l in stdout.splitlines() if l.startswith('nofit memory MB ')), None)
    return SolveResult(bound=res.get('bound'), dimV=res.get('dimV'), rank=res.get('rank'), pivots=res.get('pivots'),
                       excess=int(purge['excess']) if 'excess' in purge else None, ops=out + '.ops', piv=out + '.piv',
                       vs=out + '.vs', held=tuple(held),
                       threaded=res.get('threaded'), too_dense=too_dense, dense_mb=mb, killed=killed, late=late,
                       crank=res['crank'] if res.get('crank', -1) >= 0 else None, dense_s=est,
                       disk=too_dense and 'nofit disk' in stdout, blocks=bw, need_mb=need if too_dense else None)


def purge_excess(M, work, target):
    """Return f2elim's purge excess of M; only the purge runs, in seconds.

    A negative excess means the bound cannot reach the target unless the surviving large columns are dependent; it
    decides only when to eliminate.
    """
    if not os.path.exists(ENGINE):
        raise RuntimeError('f2elim is not built. Run `make build` in the rankbound folder.')
    os.makedirs(work, exist_ok=True)
    inp = os.path.join(work, 'matrix.bin')
    M.write(inp)
    rc, stdout, err = run_engine([inp, str(M.ns), '-1', os.path.join(work, 'purge'), '1000', str(WMAX), '1', '-1', '1',
                                  '0', str(PROGRESS)])
    purge, _, _ = _parse(stdout)
    if rc or 'excess' not in purge:
        raise RuntimeError('The purge failed: %s' % next((l.strip() for l in reversed(err.splitlines()) if l.strip()),
                                                          'exit %d' % rc))
    return int(purge['excess']) + (target if target is not None and target > 0 else 0)


# ---------- the characters' lower bound

def _pmulmod(a, b, f, q):
    """Return a b mod (f, q), low to high, f monic of degree 3."""
    r = [0] * 5
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] += x * y
    for k in (4, 3):
        c = r[k] % q
        if c:
            for t in range(3):
                r[k - 3 + t] -= c * f[t]
    return [x % q for x in r[:3]]


def _ppowmod(a, e, f, q):
    out = [1, 0, 0]
    while e:
        if e & 1:
            out = _pmulmod(out, a, f, q)
        a = _pmulmod(a, a, f, q)
        e >>= 1
    return out


def _pgcd(a, b, q):
    """Return the monic gcd of a and b mod q (low to high)."""
    def trim(p):
        p = [x % q for x in p]
        while len(p) > 1 and not p[-1]:
            p.pop()
        return p
    a, b = trim(a), trim(b)
    while b != [0]:
        lb = pow(b[-1], -1, q)
        while len(a) >= len(b) and a != [0]:
            c, d = a[-1] * lb % q, len(a) - len(b)
            for i in range(len(b)):
                a[d + i] = (a[d + i] - c * b[i]) % q
            a = trim(a)
        a, b = b, a
    return [x * pow(a[-1], -1, q) % q for x in a]


def cubic_root(poly, q):
    """Return a root of F(x, 1) mod the odd prime q not dividing c3, or None; the root is checked."""
    inv = pow(poly[3] % q, -1, q)
    f = [c * inv % q for c in poly[:3]]
    xq = _ppowmod([0, 1, 0], q, f, q)
    g = _pgcd(f + [1], [xq[0], xq[1] - 1, xq[2]], q)
    t = 0
    while len(g) > 2 and t < 200:
        h = _ppowmod([t, 1, 0], (q - 1) // 2, f, q)
        d = _pgcd(g, [h[0] - 1, h[1], h[2]], q)
        if 1 < len(d) < len(g):
            g = d
        t += 1
    if len(g) != 2:
        return None
    r = -g[0] % q
    if sum(c * pow(r, i, q) for i, c in enumerate(poly)) % q:
        raise RuntimeError('cubic_root: %d is not a root mod %d' % (r, q))
    return r


CHARACTERS = 64         # quadratic characters (one word of f2elim)


def characters(poly, qmin, k=CHARACTERS):
    """Return k degree-1 primes (q, r) of K with q > qmin, for quadratic characters.

    q must exceed every prime of every relation, so no relation's element lies in (q, alpha - r).
    """
    out, q = [], qmin + 1 + (qmin % 2)
    while len(out) < k:
        if is_prime(q) and poly[3] % q:
            r = cubic_root(poly, q)
            if r is not None:
                out.append((q, r))
        q += 2
    return out


def units_rank(F):
    """Return dim O_K*/O_K*^2 = r1 + r2 for the cubic field of F."""
    return 3 if int(F['dK']) > 0 else 2


Elim = namedtuple('Elim', 'dimV rank ops pivots')


def eliminate(rows, ncols, nsmall, threads=1, dense=1000, bits=None, keep=-1):
    """Run f2elim on rows and return Elim(dim V, rank after the purge, ops, pivot rows).

    Columns below nsmall are small. Without a trim (the default) dim V is exact. Every small column enters the dense
    phase, so callers number only the small columns they use.
    """
    if not os.path.exists(ENGINE):
        raise RuntimeError('f2elim is not built. Run `make build` in the rankbound folder.')
    if not rows:
        return Elim(0, 0, [], [])
    words = _words(bits)
    work = tempfile.mkdtemp(prefix='f2elim')
    try:
        inp, out = os.path.join(work, 'matrix.bin'), os.path.join(work, 'elim')
        _matrix(inp, rows, ncols, bits)
        rc, stdout, err = run_engine([inp, str(nsmall), '0', out, str(dense), str(WMAX), '0', str(keep), str(max(1, threads)),
                                      str(dense_limit(rss([os.getpid()]))), str(PROGRESS)] + (['%d' % words] if words else []))
        if rc:
            raise RuntimeError('The elimination failed (exit %d%s): %s' % (rc, ', too dense' if rc == 3 else '', next(
                (l.strip() for l in reversed(err.splitlines()) if l.strip()), '')))
        _, res, _ = _parse(stdout)
        if not res or not _files_ok(out, res, vs_words(0, nsmall)):
            raise RuntimeError("The elimination's output is inconsistent. Please report it.")
        ops, piv, _ = cone(out + '.ops', out + '.piv')
        return Elim(res['dimV'], res['rank'], ops, piv)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _words(bits):
    """Return f2elim's WORDS for the rows' bits."""
    return (max(x.bit_length() for x in bits) + 63) // 64 if bits else 0


def _matrix(path, rows, ncols, bits):
    """Write rows to path in f2elim's input format and return WORDS."""
    words, n = _words(bits), 0
    with open(path, 'wb') as f:
        array.array('I', [len(rows), ncols]).tofile(f)
        a = array.array('I')
        for r, x in zip(rows, bits if words else itertools.repeat(0)):
            if words:
                a.frombytes(x.to_bytes(8 * words, 'little'))
            a.append(len(r))
            a.extend(r)
            n += 1
            if len(a) >= 1 << 20:           # in pieces: the rows may be streamed from disk
                a.tofile(f)
                a = array.array('I')
        a.tofile(f)
    if n != len(rows):
        raise RuntimeError('the rows of a matrix and their bits differ in number: a bug')
    return words


Kernel = namedtuple('Kernel', 'ops vectors info')
# f2kernel's products are memory-bound, so more threads are not faster (1,000,000 rows: 508 s at 16, 714 s at 126)
KERNEL_THREADS = 16


def kernel(rows, ncols, nsmall, threads=1, bits=None, keep=-1, seed=1, work=None):
    """Return Kernel(ops, vectors, info): combinations of rows whose large columns cancel.

    The vectors are pool rows meant after the ops; elements_of maps them back. The small columns only keep rows from
    vanishing in the merge. With work, the files are kept so a stopped run resumes after the merge or f2kernel.
    """
    for e in (ENGINE, KERNEL):
        if not os.path.exists(e):
            raise RuntimeError('%s is not built. Run `make build` in the rankbound folder.' % os.path.basename(e))
    tmp = work is None
    work = tempfile.mkdtemp(prefix='f2kernel') if tmp else work
    os.makedirs(work, exist_ok=True)
    try:
        inp, out = os.path.join(work, 'kernel.bin'), os.path.join(work, 'kernel')
        words = _matrix(inp, rows, ncols, bits) if rows else 0
        # WORDS says how kernel.bin's bytes are read, so it is part of the key.
        key = '%s nsmall %d words %d keep %d wmax %d f2elim %s' % (sha256_file(inp) if rows else '-', nsmall, words, keep,
                                                                  WMAX, sha256_file(ENGINE))
        rec = out + '.json'
        d = read_json(rec, {})
        if not rows:
            return Kernel(array.array('I'), [], '')
        if d.get('key') != key or d.get('digests') != _digests(out, ('.ops', '.pool')) or None in d['digests'].values():
            d = {'key': key}
            rc, stdout, err = run_engine([inp, str(nsmall), '0', out, '1000', str(WMAX), '0', str(keep),
                                          str(max(1, threads)), '0', str(PROGRESS)] + (['%d' % words] if words else []),
                                         env={'F2ELIM_POOL': '1'})
            if rc or not stdout.strip().splitlines()[-1].startswith('pool '):
                raise RuntimeError('The kernel\'s elimination failed (exit %d): %s' % (rc, next(
                    (l.strip() for l in reversed(err.splitlines()) if l.strip()), '')))
            d['pool'] = stdout
            d['digests'] = _digests(out, ('.ops', '.pool'))
            write_json(rec, d)
        vf = '%s.vec%d' % (out, seed)
        kk = 'f2kernel %d %s' % (seed, sha256_file(KERNEL))
        if kk not in d or d.get('vec%d' % seed) != _digests(vf, ('',))['']:
            rc, stdout, err = run_engine([out + '.pool', vf, str(min(KERNEL_THREADS, max(1, threads))), str(seed),
                                          str(PROGRESS)], engine=KERNEL)
            if rc:
                raise RuntimeError('f2kernel failed (exit %d): %s' % (rc, next(
                    (l.strip() for l in reversed(err.splitlines()) if l.strip()), '')))
            d[kk] = stdout.strip()
            d['vec%d' % seed] = _digests(vf, ('',))['']
            write_json(rec, d)
        ops = array.array('I')
        with open(out + '.ops', 'rb') as f:
            ops.frombytes(f.read())
        a = array.array('I')
        with open(vf, 'rb') as f:
            a.frombytes(f.read())
        vectors, t = [], 1
        for _ in range(a[0] if a else 0):
            n = a[t]
            vectors.append(a[t + 1:t + 1 + n].tolist())
            t += 1 + n
        if t != len(a):
            raise RuntimeError("f2kernel's output is inconsistent. Please report it.")
        return Kernel(ops, vectors, d[kk])
    finally:
        if tmp:
            shutil.rmtree(work, ignore_errors=True)


def cone(ops_path, piv_path):
    """Return the ops and rows the pivot rows depend on: (kept ops, pivots, needed rows sorted)."""
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
