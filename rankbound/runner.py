"""The runners: sieve_run (sieve processes, locally or as one shard), run_check (one check with the sieve paused),
and solve_once and solve_given (runs without a sieve)."""
import contextlib, glob, gzip, logging, os, signal, sys, threading, time
from . import Locked, RankboundError, field as fld, linalg, read_json, sieve, write_json
from .rules import check_plan, count_slice, dry, earlier, schedule
from .solver import Solver, too_dense
from .state import GIVEN, State, open_state, slice_name

LOG = logging.getLogger('rankbound').info

MAX_CRASH = 0.1         # stop when las aborts on more than this share
POLL = 20               # seconds between loads of other shards' slices


def slice_failed(S, p, poly, q0, q1, out, tally, log):
    """Handle a sieve process that exited nonzero: skip the slice on a known las abort, unless too many slices abort;
    raise on anything else."""
    if p.stderr.seekable():         # sieve.spawn: an anonymous file
        p.stderr.seek(0)
    err = p.stderr.read()
    err = err.decode(errors='replace') if isinstance(err, bytes) else err
    if not sieve.las_crashed(p.returncode, err):
        raise RuntimeError('The siever failed: %s' % next(
            (l.strip() for l in reversed(err.splitlines()) if l.strip()), 'exit %d' % p.returncode))
    tally['crash'] += 1
    if tally['crash'] > max(10, MAX_CRASH * (tally['crash'] + tally['done'])):
        raise RuntimeError("CADO-NFS's siever crashed on %d of %d parts (a known CADO bug). %s" % (
            tally['crash'], tally['crash'] + tally['done'], sieve.OUT_OF_REACH))
    S.skip_slice(poly, q0, q1, out)

def signal_group(p, sig):
    """Signal a slice's process group, skipping a process that has exited (macOS gives EPERM until it is reaped)."""
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


@contextlib.contextmanager
def sigterm_exits():
    """Turn SIGTERM into SystemExit, so that the cleanup stops the sieve processes, which run in their own sessions."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    old = signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, old)


def kill(p, out):
    """Stop a running or paused slice and delete its partial output."""
    if p.poll() is None:
        signal_group(p, signal.SIGTERM)
        signal_group(p, signal.SIGCONT)
    p.wait()
    p.stderr.close()
    if os.path.exists(out):
        os.remove(out)
    sieve.clean_tmp(out)


def sieve_run(state, jobs=1, shard=(0, 1), solver=None, max_hours=None, log=LOG, until=None):
    """Sieve with jobs processes on the slices j = k mod n of the schedule, until STOP, the goal, a plateau or the time
    limit.  The solver (default: shard 0) checks as relations accumulate, with its sieve processes paused, and writes
    STOP when finished; the other shards then stop.  Returns (outcome or None, Solver)."""
    if jobs < 1:
        raise ValueError('cores must be a positive integer.')
    until = earlier(until, max_hours)
    k, n = shard
    S = State(state)
    for t in range(3):          # others_left holds a free lock briefly
        try:
            lk = S.lock('shard%d_%d.lock' % (k, n))
            break
        except Locked:
            if t == 2:
                raise
            time.sleep(0.5)
    # A local run holds sieve.lock alone; the shards of a run share it.
    try:
        lks = S.lock('sieve.lock', shared=n > 1)
    except Locked:
        os.close(lk)
        raise
    solver = (k == 0) if solver is None else solver
    running = {}            # Popen -> (poly, q0, q1, out)
    outcome = V = None
    t0 = time.time()
    with sigterm_exits():
        try:
            F, prm = open_state(S)
            if S.stopped():
                return (S.stop_outcome() if solver else None), None
            if until is not None and time.time() >= until:
                log('The time limit has already passed; nothing to do.')
                return None, None
            if n == 1:                    # remove what interrupted slices left
                for p in glob.glob(os.path.join(S.sdir, '*.las*')):
                    os.remove(p)
            sv = sieve.siever(prm)
            if solver:
                V = Solver.of(S, F, prm, log, jobs)
                nxt, fresh, polled = V.next_check(), True, 0.0
                due = S.due()             # a check left to this run
            sched = (x for x in schedule(prm) if x[0] % n == k)
            tally = {'done': 0, 'crash': 0}
            waited, missing = 0.0, None
            late = False                # no more checks before the time limit
            checked = False             # this process ran a check
            def fits():
                old = S.unfinished_check()
                fresh = not checked and not tally['done']
                plan, need = check_plan(V.hist, V.nrel, until, t0, time.time(), fresh, old, jobs)
                left = None if until is None else until - time.time()
                if plan == 'too long':
                    th = old.get('threads', '?')
                    raise RankboundError(
                        'The last check did not finish in the time it had (%.1f h on %s core%s). Give more cores or more '
                        'time.' % ((old['until'] - old['start']) / 3600, th, '' if th == 1 else 's'))
                if plan == 'later':
                    S.mark_due(V.nrel)
                    log('Not enough time left for the next check; it runs first when you resume.')
                return plan == 'now'
            def check(dry):
                nonlocal checked, due
                checked, due = True, False
                return run_check(S, V, running, dry=dry, until=until)
            while True:
                if S.stopped() or (until is not None and time.time() > until):
                    break
                # The solver loads the finished slices and checks when due.
                if V and (fresh or (n > 1 and time.time() - polled > POLL)):
                    V.load(); fresh, polled = False, time.time()
                    # A dried-up sieve checks at once; sharded, only after every shard's share is sieved.
                    if not late and (V.nrel >= nxt or due or (dry(tally) and not (
                            n > 1 and prm.get('siever') != 'line' and any(others_left(S, prm, k, n))))):
                        late = not fits()
                        if not late:
                            outcome = check(V.nrel < nxt)
                            if outcome:
                                break
                            nxt = V.next_check()
                while len(running) < jobs and sched is not None:
                    x = next(sched, None)
                    if x is None:               # the schedule is used up
                        sched = None; break
                    j, poly, q0, q1 = x
                    if not S.done(slice_name(poly, q0)):
                        out = os.path.join(S.sdir, slice_name(poly, q0) + '.las')
                        sieve.clean_tmp(out)    # what a killed job left
                        running[sv.start_slice(state, prm, poly, q0, q1, out)] = (poly, q0, q1, out)
                if sched is None and not running:
                    if V and n > 1 and time.time() - waited > POLL:
                        waited = time.time()
                        busy, idle = others_left(S, prm, k, n)
                        if busy or idle:        # the last check waits for every shard
                            if (busy, idle) != missing:
                                missing = (busy, idle)
                                w = sorted(busy + idle)
                                log('Shard %d of %d is done; waiting for shard%s %s.' % (
                                    k, n, 's' if len(w) > 1 else '',
                                    ', '.join(map(str, w[:-1])) + ' and %d' % w[-1] if len(w) > 1 else w[0]))
                                for j in idle:
                                    log('Shard %d is not running: resubmit it.' % j)
                            continue
                    elif V and n > 1:
                        time.sleep(1)
                        continue
                    if V:                       # the last check
                        V.load()
                        if not late and fits():
                            outcome = check(True)
                    break
                time.sleep(1)
                for p in [p for p in running if p.poll() is not None]:
                    poly, q0, q1, out = running.pop(p)
                    try:
                        if p.returncode:
                            slice_failed(S, p, poly, q0, q1, out, tally, log)
                        else:
                            count_slice(tally, S.finish_slice(prm, poly, q0, q1, out)[1])
                    finally:
                        p.stderr.close()
                    fresh = True
        finally:
            for p, v in running.items():
                kill(p, v[3])
            os.close(lks)
            os.close(lk)
    cpu, nrel = S.slice_stats()
    if outcome or S.stopped():
        log('{}: {:,} relations, {:.2f} core-hours.'.format(
            'Sieving finished' if n == 1 else 'Shard %d of %d finished' % (k, n), nrel, cpu / 3600))
    elif n > 1:             # a local run's caller reports Stopped
        log('Sieving paused by the time limit at {:,} relations ({:.2f} core-hours). Run again to resume.'.format(
            nrel, cpu / 3600))
    return outcome, V


def others_left(S, prm, k, n):
    """Return the other shards with unfinished slices in their share, as (running, not running).  A shard that is not
    running is not finished: the last check waits for it."""
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


def run_check(S, V, running, dry=False, until=None):
    """Run one check under run.lock with the sieve processes paused, and write STOP if it gives an outcome.
    check.json records the check while it runs and stays only if the check is killed, which check_plan reads as a
    timeout."""
    for p in running:
        signal_group(p, signal.SIGSTOP)
    V.pids = [p.pid for p in running]
    try:
        fd = S.lock('run.lock')
        try:
            if S.stopped():
                return None
            V.hist = S.history()
            S.begin_check(V.nrel, until, V.threads)
            try:
                oc = V.check(os.path.join(S.path, 'la'), dry=dry)
            except Exception:
                S.end_check()
                raise
            except KeyboardInterrupt:   # Ctrl-C: not a timeout
                S.stop_check()
                raise
            S.end_check()
            S.clear_due()
            if oc:
                S.write_stop(oc)
        finally:
            os.close(fd)
    finally:
        V.pids = ()
        for p in running:
            signal_group(p, signal.SIGCONT)
    return oc


def solve_once(state, log=LOG, threads=1):
    """Load every finished slice and check once; write STOP when finished.  Returns (outcome or None, Solver)."""
    S = State(state)
    F, prm = open_state(S)
    V = Solver.of(S, F, prm, log, threads)
    V.load()
    outcome = run_check(S, V, {})
    if not outcome and S.stopped():     # finished meanwhile by another process
        outcome = S.stop_outcome()
    return outcome, V


def given_params(state):
    """Write params.json for a run on given relations without a sieve, unless the folder has one."""
    path = os.path.join(state, 'params.json')
    if os.path.exists(path):
        return read_json(path)
    with gzip.open(os.path.join(state, 'slices', GIVEN + '.rels.gz'), 'rt') as f:
        top = max((p for l in f for p in sieve.parse_line(l)[2]), default=2)
    prm = sieve.given_params(fld.load(state), top)
    write_json(path, prm)
    return prm


def solve_given(state, log=LOG, threads=1):
    """Run one untrimmed elimination on every finished slice, whatever the excess: the bound the given relations
    prove.  Writes STOP ('goal' or 'relations'); returns (outcome, Solver)."""
    S = State(state)
    F, prm = open_state(S)
    V = Solver.of(S, F, prm, log, threads)
    V.keep = -1
    V.load()
    fd = S.lock('run.lock')
    try:
        r = V._again(os.path.join(S.path, 'la'))
    finally:
        os.close(fd)
    if r.bound is None:
        raise RuntimeError(too_dense(r.dense_mb))
    oc = 'goal' if r.bound <= V.target else 'relations'
    log('{:,} relations: rank <= {} (aim: rank <= {}).'.format(V.nrel, V.rank(r.bound), V.rank(V.target)))
    S.write_stop(oc)
    return oc, V
