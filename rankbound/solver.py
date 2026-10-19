"""The checks (Solver): the relation matrix, filled from the finished slices, and one check at a time, recorded in
STATE/history.json.  The purge gives the excess E = rows - large columns - (ns - target); the full elimination runs
once E >= 0.  A run stops at the goal, or at a plateau: two full solves PLATEAU apart with the same bound."""
import gzip, logging, math, os, time, zlib
from . import RankboundError, field as fld, linalg, sieve
from .rules import decide, next_check

LOG = logging.getLogger('rankbound').info

def too_dense(mb):
    """Return the message for a dense phase of mb MB that does not fit in memory."""
    return ('The final elimination needs %.1f GB of memory, more than is available here. Run on a machine with more '
            'memory.' % ((mb or 0) / 1000))


KEEP = 1000                     # the excess f2elim's trim keeps
                                # (x10 at each apparent plateau, then no trim)


class DriedUp(RuntimeError):
    """The sieve dried up before the excess turned positive.  run.Job.run re-tunes at a larger lpb and sieves on."""


class Solver:
    """The relation matrix, filled from finished slices, and the checks: the purge, and the full elimination when
    the excess allows."""
    threads = 1                   # threads of the dense phase
    pids = ()                     # paused sieve processes during a check

    def __init__(self, S, F, small, prm, log=LOG):
        self.S, self.prm, self.log, self.F = S, prm, log, F
        self.M = linalg.Matrix(F, small)
        pp, fb0, _ = sieve.siever(prm).prepare(S.path, prm, 'F')
        top = 2 ** prm['lpb0']
        if prm.get('siever') == 'given':      # its split file has every free relation
            top = math.inf
        elif prm.get('siever') != 'line' and prm['lim0'] < top:     # the free relations up to 2^lpb, not lim0
            fb0 = sieve.makefb(pp, top, os.path.join(S.path, 'F.free_%d.gz' % top))
        for p, labs in linalg.free_rows(F, fb0, top):
            self.M.add_free(p, labs)
        self.nfree = len(self.M.rows)
        self.loaded = set()
        self.order = []           # the slices loaded, in order
        self.nrel = 0
        self.target = fld.goal(F)
        self.hist = S.history()
        self.keep = next((h['keep'] for h in reversed(self.hist) if 'keep' in h), KEEP)
        self.last = None          # the last full solve

    @classmethod
    def of(cls, S, F, prm, log=LOG, threads=1):
        """Return the Solver of the state S, with no slice loaded yet."""
        V = cls(S, F, fld.load_small(S.path), prm, log)
        V.threads = threads
        return V

    def load(self, names=None):
        """Add the relations of the finished slices not loaded yet: names, in that order, else every finished slice."""
        for name in self.S.finished() if names is None else names:
            if name in self.loaded:
                continue
            path = os.path.join(self.S.sdir, name + '.rels.gz')
            try:
                with gzip.open(path, 'rt') as f:
                    for l in f:
                        self.nrel += self.M.add_relation(l)
            except (EOFError, OSError, zlib.error, ValueError) as e:    # cut short (a node crash) or damaged
                raise RankboundError('%s is damaged. Delete it and %s; that part is sieved again.' % (
                    path, os.path.join(self.S.sdir, name + '.json')))
            self.loaded.add(name)
            self.order.append(name)

    def rank(self, G):
        """Return the rank bound S given by the bound G on dim Cl(K)[2]."""
        return fld.parity_bound(G, self.F['u'], self.F['n'], self.F['w'])

    def progress(self, G):
        """Return the log line of a check that gave the bound G."""
        return '{:,} relations: rank <= {} so far (aim: rank <= {}).'.format(self.nrel, self.rank(G), self.rank(self.target))

    def next_check(self):
        return next_check(self.hist, self.prm)

    def check(self, work, dry=False):
        """Run one check and return 'goal', 'plateau' or None.  At an apparent plateau the elimination is repeated
        with 10x the trim's excess, then untrimmed; the plateau stands only if the bound does not improve.  dry: a
        plateau whatever the history, after the untrimmed check."""
        while True:
            t = time.time()
            r = linalg.solve(self.M, work, self.target, gate=True, keep=self.keep, threads=self.threads, log=self.log,
                             others=self.pids)
            e = {'keep': self.keep, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
            msg = None              # a purge-only check is not logged
            if r.bound is not None:
                self.last = r
                self.S.save_order(work, self.order)
                e.update({'bound': r.bound, 'dimV': r.dimV, 'rank': r.rank, 'pivots': r.pivots})
                msg = self.progress(r.bound)
            elif r.killed:          # linalg.solve logged it
                e.update({'dense_MB': r.dense_mb, 'killed': True})
            elif r.too_dense:
                e['dense_MB'] = r.dense_mb
                msg = too_dense(r.dense_mb)
            e = self._record(r, t, e)
            if msg:
                self.log(msg)
            out = decide(self.hist, self.target)
            if dry and out is None:
                if r.too_dense:
                    raise RuntimeError(too_dense(r.dense_mb))
                if r.bound is None:
                    raise DriedUp(sieve.TOO_FEW)
                out = 'plateau'
            if out != 'plateau' or self.keep < 0:
                return out
            prev = r.bound
            self.log('No progress; checking once more with every relation kept.')
            # dry: go on to the untrimmed matrix even if the bound improves.
            while self.keep >= 0 and (r.bound == prev or dry):
                k, self.keep = self.keep, -1 if self.keep * 10 >= r.excess else self.keep * 10
                r = self._again(work)
                if r.bound is None:         # does not fit: the bound at excess k stands
                    self.keep = k
                    self.log(too_dense(r.dense_mb))
                    return 'plateau'
                if r.bound <= (self.target if self.target is not None else -1):
                    return 'goal'
            if r.bound < prev and not dry:
                return None
            return 'plateau'

    def _again(self, work):
        """Run the elimination again at self.keep on the same relations.  If it does not fit, self.last stays and the
        history records the excess tried as keep_tried."""
        t = time.time()
        r = linalg.solve(self.M, work, self.target, keep=self.keep, threads=self.threads, log=self.log, others=self.pids)
        e = {'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        if r.bound is None:
            e.update({'keep_tried': self.keep, 'dense_MB': r.dense_mb})
            if r.killed:
                e['killed'] = True
        else:
            self.last = r
            self.S.save_order(work, self.order)
            e.update({'keep': self.keep, 'bound': r.bound, 'dimV': r.dimV, 'rank': r.rank, 'pivots': r.pivots})
        self._record(r, t, e)
        return r

    def _record(self, r, t, e):
        """Append the history entry of the solve r, started at t, to history.json."""
        e = {'relations': self.nrel, 'rows': len(self.M.rows), 'excess': r.excess, **e}
        e['solve_s'] = round(time.time() - t, 1)
        self.hist.append(e); self.S.save_history(self.hist)
        return e
