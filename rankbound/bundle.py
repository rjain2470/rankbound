"""Bundles of witnesses: one gzipped JSON file holding the entries of many curves, each with its witness.

A bundle is a JSON object whose last key is "curves", a list with one entry per line, so that it can be read one
entry at a time.  Each entry holds the curve, the bounds and the witness itself ("witness"), whose file is recovered
byte for byte (its sha256 is the entry's "sha256", of the uncompressed file).
"""
import gzip, hashlib, json, os
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .core import FORMAT2
from .witness import selmer_text


def witness_text(w: Dict[str, Any]) -> str:
    """Return the text of the witness file of w, as the prover wrote it."""
    return json.dumps(w, indent=1) + '\n' if w.get('format') == FORMAT2 else ''.join(selmer_text(w))


def witness_name(e: Dict[str, Any]) -> str:
    """Return the file name of the entry's witness: gzipped unless it is a 2-isogeny witness."""
    return e['name'] + ('.witness.json' if e['witness'].get('format') == FORMAT2 else '.witness.json.gz')


def write(path: str, header: Dict[str, Any], entries: Iterable[Dict[str, Any]]) -> int:
    """Write the bundle: header's keys, then "curves" with one entry per line; return the number of entries."""
    assert 'curves' not in header
    n = 0
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        f.write(json.dumps(header)[:-1] + (', ' if header else '') + '"curves": [\n')
        for e in entries:
            f.write((',\n' if n else '') + json.dumps(e, separators=(',', ':')))
            n += 1
        f.write('\n]}\n')
    os.replace(tmp, path)
    return n


def entries(path: str) -> Iterator[Dict[str, Any]]:
    """Yield the bundle's entries one at a time."""
    with gzip.open(path, 'rt') if path.endswith('.gz') else open(path) as f:
        head = f.readline()
        if not head.rstrip().endswith('"curves": ['):
            raise ValueError('%s is not a bundle of witnesses.' % path)
        for line in f:
            line = line.rstrip('\n')
            if line == ']}':
                return
            yield json.loads(line[:-1] if line.endswith(',') else line)
    raise ValueError('%s ends early.' % path)


def header(path: str) -> Dict[str, Any]:
    """Return the bundle's keys other than "curves"."""
    with gzip.open(path, 'rt') if path.endswith('.gz') else open(path) as f:
        head = f.readline().rstrip()
    return json.loads(head[:-len('"curves": [')].rstrip(', ') + '}')


def unpack(path: str, folder: str, ids: Optional[Sequence[str]] = None) -> List[str]:
    """Write the witnesses of the bundle's entries (those of ids, if given) into folder; return their paths.

    Raise ValueError if a witness's sha256 is not its entry's, or an id is not in the bundle."""
    os.makedirs(folder, exist_ok=True)
    want = None if ids is None else {str(i) for i in ids}
    out, seen = [], set()
    for e in entries(path):
        if want is not None and str(e['icarm_id']) not in want:
            continue
        seen.add(str(e['icarm_id']))
        raw = witness_text(e['witness']).encode()
        if hashlib.sha256(raw).hexdigest() != e['sha256']:
            raise ValueError('The witness of %s in %s does not match its sha256.' % (e['name'], path))
        p = os.path.join(folder, witness_name(e))
        with (gzip.open(p, 'wb') if p.endswith('.gz') else open(p, 'wb')) as f:
            f.write(raw)
        out.append(p)
    if want is not None and want - seen:
        raise ValueError('%s has no curve with the id %s.' % (path, ', '.join(sorted(want - seen))))
    return out
