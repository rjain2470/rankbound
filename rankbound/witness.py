"""Witness files: their formats, statements and packed lists, and the atomic write.

The prover builds a witness as a dict (selmer.witness_of, descent.witness, ctp.add); this is the only place it
becomes bytes.  The 2-isogeny descent's witness is plain JSON.  A 2-Selmer witness is written in format 5 (packed
lists; format 6 is format 5 with the norm-kernel claims), or in format 4, unpacked, when format 5 cannot keep the
lists' meaning.  Files are written atomically, gzipped when the path ends in .gz.  read() is the prover's inverse;
the checker has its own reader.
"""
import contextlib, gzip, json
from typing import Any, Dict, Optional, Tuple
from . import runtime
from .core import FORMAT2, FORMAT4, FORMAT5 as FORMAT, RankboundError, parity_bound, parity_floor
from .messages import text

FORMATS4 = (FORMAT4, 'rankbound-selmer-certificate-4')     # format 4's labels, as the checker reads them
FORMAT6 = 'rankbound-selmer-witness-6'          # format 5 with the norm-kernel claims
LINES = ('relations', 'elements')                   # one item per line
FLAT = ('ops', 'pivots', 'ops_s', 'pivots_s')       # 64 numbers per line


# ---------- the statements (the checker reads them verbatim)

def with_r(S: int, r: Optional[int], source: Optional[str]) -> str:
    """Return the end of the statement rank E(Q) <= S, given the rank lower bound r (or None)."""
    if r is not None and S == r and r > 0:
        return ' As E(Q) has %d independent rational point%s%s, we therefore have rank E(Q) = %d.' % (
            r, 's' if r > 1 else '', ' (%s; not proved in this witness)' % source if source else '', r)
    return ' CONTRADICTION: rank E(Q) >= %d.' % r if r is not None and S < r else ''


def isogeny_statement(B: int, w: int, r: Optional[int], source: Optional[str] = None) -> Tuple[int, str]:
    """Return (S, statement) of a 2-isogeny witness for the bound B, root number w and rank lower bound r."""
    S = parity_bound(B, 0, 0, w)
    return S, 'Unconditionally, rank E(Q) <= %d.' % S + with_r(S, r, source)


def statement(S, hyp, r, source=None):
    """Return the statement of a format-4 witness."""
    s = ('Assuming the GRH, we have that' if hyp == 'GRH' else 'We have that') + ' rank E(Q) <= %d.' % S
    if r is not None and S == r and r > 0:
        s += ' As E(Q) has %d independent rational point%s%s, we therefore have rank E(Q) = %d.' % (
            r, 's' if r > 1 else '', ' (%s; not proved in this witness)' % source if source else '', r)
    elif r is not None and S < r:
        s += ' CONTRADICTION: rank E(Q) >= %d.' % r
    return s


# ---------- format 5's packed lists

def deltas(xs):
    """Return the strictly increasing xs delta-coded: [xs[0], xs[1] - xs[0], ...]."""
    out = [x - y for x, y in zip(xs, [0] + xs[:-1])]
    if any(d <= 0 for d in out[1:]) or (out and out[0] < 0):
        raise RuntimeError('selmer: a list to delta-code is not strictly increasing: a bug')
    return out


def undeltas(ds):
    """Return the running sums of ds, the inverse of deltas."""
    out, s = [], 0
    for d in ds:
        s += d
        out.append(s)
    return out


class Unpackable(ValueError):
    """Raised when format 5 cannot write a witness's lists with their meaning."""


def packable(wit):
    """Return why format 5 cannot write the unpacked witness wit, or None if it can.

    pivots and pivots_s are sets of rows, so sorting and deduplicating them keeps their meaning.  Format 5 writes
    pivots_s as the rows beyond pivots, so pivots_s must contain pivots.  An element is a sum of relations, so a
    repeated relation cannot be dropped.
    """
    if any(len(o) != 2 for o in wit['ops']) or any(len(o) != 2 for o in wit.get('ops_s') or []):
        return 'an operation is not a pair'
    if wit.get('pivots_s') is not None and not set(wit['pivots']) <= set(wit['pivots_s']):
        return 'the rows of pivots_s do not contain those of pivots'
    if any(len(set(e)) != len(e) for e in wit['elements']):
        return 'an element repeats a relation'
    return None


def encode(wit):
    """Return the unpacked witness wit in format 5, or raise Unpackable (see packable).

    ops are flattened; pivots, pivots_s (the rows beyond pivots) and each element are sorted and delta-coded; ops_s
    is dropped when it equals ops.  A format-6 label is kept.
    """
    why = packable(wit)
    if why:
        raise Unpackable('format 5 cannot write this witness: ' + why)
    c = dict(wit)
    if c.get('format') in FORMATS4:                 # any other label is kept
        c['format'] = FORMAT
    pv = sorted(set(wit['pivots']))
    c['ops'] = [x for o in wit['ops'] for x in o]
    c['pivots'] = deltas(pv)
    if wit.get('pivots_s') is not None:
        if [list(o) for o in wit['ops_s']] == [list(o) for o in wit['ops']]:
            c.pop('ops_s')
        else:
            c['ops_s'] = [x for o in wit['ops_s'] for x in o]
        c['pivots_s'] = deltas(sorted(set(wit['pivots_s']) - set(pv)))
    else:
        c.pop('ops_s', None)
        c.pop('pivots_s', None)
    c['elements'] = [deltas(sorted(e)) for e in wit['elements']]
    return c


def decode(c):
    """Return the inverse of encode."""
    c = dict(c)
    pairs = lambda f: [f[k:k + 2] for k in range(0, len(f), 2)]
    pv = undeltas(c['pivots'])
    c['ops'] = pairs(c['ops'])
    c['pivots'] = pv
    if 'pivots_s' in c:
        c['ops_s'] = pairs(c['ops_s']) if 'ops_s' in c else c['ops']
        c['pivots_s'] = pv + undeltas(c['pivots_s'])
    c['elements'] = [undeltas(e) for e in c['elements']]
    return c


# ---------- reading and writing

def read(path: str) -> Dict[str, Any]:
    """Return the witness at path with its lists unpacked; formats other than 5 and 6 are returned as read."""
    with gzip.open(path, 'rt') if path.endswith('.gz') else open(path) as f:
        c = json.load(f)
    return decode(c) if c.get('format') in (FORMAT, FORMAT6) else c


CHUNK = 1 << 16             # characters buffered per write in atomic(), bounding its memory


def atomic(path, pieces):
    """Write pieces (a str or an iterable of str) to path atomically, without holding the whole text in memory.

    A path ending in .gz is gzipped deterministically (no name, mtime 0).
    """
    if isinstance(pieces, str):
        pieces = (pieces,)
    with runtime.atomic_file(path) as f, (gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0, compresslevel=6)
                                          if path.endswith('.gz') else contextlib.nullcontext(f)) as g:
        buf, n = [], 0
        for x in pieces:
            buf.append(x)
            n += len(x)
            if n >= CHUNK:
                g.write(''.join(buf).encode())
                buf, n = [], 0
        g.write(''.join(buf).encode())


def write(wit: Dict[str, Any], path: str) -> None:
    """Write the witness wit to path in its format."""
    if wit.get('format') == FORMAT2:
        atomic(path, json.dumps(wit, indent=1) + '\n')
    else:
        write_selmer(wit, path)


def write_selmer(wit, path):
    """Write the unpacked 2-Selmer witness wit to path in format 5 (or 6), else in format 4.

    Keys starting with '_' are private to the prover and left out.  A format-6 witness that cannot be packed is
    converted with as_format4.
    """
    rl, S = wit.get('rank_lower_bound') or {}, wit['claims'].get('S_ctp', wit['claims']['S'])
    if rl.get('r') is not None and S < rl['r']:
        raise RankboundError(text('points_high', S=S, n=rl['r']) if 'points' in rl else text('r_high', S=S, r=rl['r']))
    c = {k: v for k, v in wit.items() if not k.startswith('_')}
    try:
        c = encode(c)
    except Unpackable:
        if c.get('format') == FORMAT6:
            c = as_format4(c)
        elif c.get('format') not in FORMATS4:
            c['format'] = FORMAT4
    atomic(path, selmer_text(c))


def selmer_text(c):
    """Yield the JSON text of the witness c, with the LINES lists one item per line and FLAT 64 numbers per line."""
    yield '{\n'
    for t, (k, v) in enumerate(c.items()):
        if t:
            yield ',\n'
        if k in LINES or k in FLAT:
            yield ' %s: [' % json.dumps(k)
            items = (v[t:t + 64] for t in range(0, len(v), 64)) if k in FLAT else v
            for i, x in enumerate(items):
                js = json.dumps(x, separators=(',', ':'))
                yield ('\n  ' if i == 0 else ',\n  ') + (js[1:-1] if k in FLAT else js)
            yield '\n ]' if v else ']'
        else:
            yield ' %s: %s' % (json.dumps(k), json.dumps(v, indent=1).replace('\n', '\n '))
    yield '\n}\n'


def as_format4(wit):
    """Return the unpacked format-6 witness wit as format 4, whose bound may be weaker.

    The norm-kernel claims are dropped, S = min(dim(Sel_2 cap V) + U - v (parity_floor when v < U), S_bk),
    S_ctp = S minus the pairing's rank, and the statement is rewritten.
    """
    c = dict(wit)
    cl = {x: y for x, y in c['claims'].items() if x not in ('dim_V_cap_ker_N', 'S_norm')}
    dsel, U, v, w = cl['dim_sel_cap_V'], cl['U'], cl['v'], cl['w']
    S = min(dsel + U - v if v == U else parity_floor(dsel + U - v, w), cl['S_bk'])
    if 'S_ctp' in cl:
        cl['S_ctp'] = S - (c['claims']['S'] - c['claims']['S_ctp'])
    cl['S'] = S
    rl = c.get('rank_lower_bound') or {}
    c['claims'] = cl
    c['statement'] = statement(cl.get('S_ctp', S), c['hypothesis'], rl.get('r'),
                               None if 'points' in rl else rl.get('source'))
    c['format'] = FORMAT4
    return c
