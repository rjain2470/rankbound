"""The state folder of a run, from which a resumed run or another shard starts.

  curve.json, field.json, small.txt, params.json, cost.json, result.json    the stages' outputs
  slices/NAME.rels.gz   a finished special-q slice, and slices/NAME.json its statistics; slices/given.rels.gz holds
                        the given relations
  history.json          one entry per check
  check.json            a check in progress; left behind if the check is killed
  due.json              a check postponed to the next run
  STOP                  the outcome, once the run is finished
  la/                   the last elimination, which certify reuses
  sieve.lock            shared by the shards of a run, exclusive for a local run
  NAME.cert.json, verify.log, verify.json    the certificate and its verification"""
import fcntl, glob, gzip, os, time
from . import Locked, RankboundError, field as fld, read_json, sieve, write_json

GIVEN = 'given'                 # the slice of the given relations


def slice_name(poly, q0):
    return '%s_%d' % (poly, q0)


class State:
    def __init__(self, path):
        self.path = path
        self.sdir = os.path.join(path, 'slices')
        os.makedirs(self.sdir, exist_ok=True)

    def done(self, name):
        return os.path.exists(os.path.join(self.sdir, name + '.rels.gz'))

    def finished(self):
        return sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(self.sdir, '*.rels.gz')))

    def stopped(self):
        return os.path.exists(os.path.join(self.path, 'STOP'))

    def stop_outcome(self):
        return read_json(os.path.join(self.path, 'STOP')).get('outcome', 'stopped')

    def write_stop(self, outcome):
        write_json(os.path.join(self.path, 'STOP'), {'outcome': outcome, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})

    def history(self):
        return read_json(os.path.join(self.path, 'history.json'), [])

    def save_history(self, h):
        write_json(os.path.join(self.path, 'history.json'), h)

    def begin_check(self, nrel, until, threads=1):
        """Write check.json: a check at nrel relations starts now, in a run that stops by until."""
        write_json(os.path.join(self.path, 'check.json'), {'relations': nrel, 'start': time.time(), 'until': until,
                                                           'threads': threads})

    def end_check(self):
        p = os.path.join(self.path, 'check.json')
        if os.path.exists(p):
            os.remove(p)

    def stop_check(self):
        """Mark check.json as stopped by Ctrl-C."""
        p = os.path.join(self.path, 'check.json')
        try:
            if os.path.exists(p):
                write_json(p, dict(read_json(p), stopped=time.time()))
        except (OSError, ValueError):       # then it reads as a timeout
            pass

    def unfinished_check(self):
        """Return check.json as a dict, or None."""
        return read_json(os.path.join(self.path, 'check.json'), None)

    def mark_due(self, nrel):
        """Write due.json: a check at nrel relations waits for the next run."""
        write_json(os.path.join(self.path, 'due.json'), {'relations': nrel, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})

    def due(self):
        return os.path.exists(os.path.join(self.path, 'due.json'))

    def clear_due(self):
        p = os.path.join(self.path, 'due.json')
        if os.path.exists(p):
            os.remove(p)

    def lock(self, name, shared=False):
        """Lock STATE/name, exclusive or shared, and return the fd (close it to release); raise Locked if another
        process holds it."""
        path = os.path.join(self.path, name)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise Locked('Another rankbound process is working on %s. Wait for it, or stop it.' % self.path)
        return fd

    def slice_stats(self):
        """Return (CPU seconds, relations) of every finished slice."""
        cpu = n = 0
        for p in glob.glob(os.path.join(self.sdir, '*.json')):
            try:
                d = read_json(p); cpu += d['cpu'] or 0; n += d['relations']
            except (OSError, ValueError, KeyError, TypeError):     # e.g. cut short by a node crash
                raise RankboundError('%s is damaged. Delete it and %s; that part is sieved again.'
                                     % (p, p[:-len('.json')] + '.rels.gz'))
        return cpu, n

    def save_order(self, work, names):
        """Write work/order.txt: the slices of a full solve, in load order."""
        tmp = os.path.join(work, 'order.txt.tmp%d' % os.getpid())
        with open(tmp, 'w') as f:
            f.writelines(n + '\n' for n in names)
        os.replace(tmp, os.path.join(work, 'order.txt'))

    def load_order(self, work):
        """Return the slice names in work/order.txt if all are finished, else None."""
        p = os.path.join(work, 'order.txt')
        if not os.path.exists(p):
            return None
        with open(p) as f:
            names = [l.strip() for l in f if l.strip()]
        return names if all(self.done(n) for n in names) else None

    def finish_slice(self, prm, poly, q0, q1, out):
        return sieve.finish_slice(self.sdir, slice_name(poly, q0), {'poly': poly, 'q0': q0, 'q1': q1},
                                  prm['polys'][poly]['shift'], out)

    def given(self):
        """Return the digest of the given relations, or None if there are none."""
        p = os.path.join(self.sdir, GIVEN + '.json')
        return read_json(p).get('given') if self.done(GIVEN) and os.path.exists(p) else None

    def write_given(self, lines, digest):
        """Write the given relations as the finished slice slices/given.rels.gz."""
        tmp = os.path.join(self.sdir, GIVEN + '.tmp%d.gz' % os.getpid())
        with gzip.open(tmp, 'wt', compresslevel=4) as f:
            f.writelines(l + '\n' for l in lines)
        write_json(os.path.join(self.sdir, GIVEN + '.json'), {'poly': GIVEN, 'q0': None, 'q1': None, 'cpu': 0.0,
                                                              'relations': len(lines), 'special_q': None,
                                                              'given': digest}, None)
        os.replace(tmp, os.path.join(self.sdir, GIVEN + '.rels.gz'))

    def skip_slice(self, poly, q0, q1, out):
        """Record a slice that las aborted on as finished, with no relation; fewer relations never make a bound
        wrong."""
        name = slice_name(poly, q0)
        sieve.clean_tmp(out)
        with gzip.open(os.path.join(self.sdir, name + '.tmp.gz'), 'wt'):
            pass
        write_json(os.path.join(self.sdir, name + '.json'),
                   {'poly': poly, 'q0': q0, 'q1': q1, 'cpu': None, 'relations': 0, 'special_q': None, 'las_crash': True}, None)
        os.replace(os.path.join(self.sdir, name + '.tmp.gz'), os.path.join(self.sdir, name + '.rels.gz'))


def open_state(S):
    """Return (field.json, params.json), rewriting missing polynomial files."""
    F = fld.load(S.path)
    prm = read_json(os.path.join(S.path, 'params.json'))
    for P in sieve.polys(F):
        sieve.write_poly(P, os.path.join(S.path, P['name'] + '.poly'))
    return F, prm
