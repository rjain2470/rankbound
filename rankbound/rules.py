"""The rules, as functions of their arguments: the special-q schedule, when to check and stop, and the time limit.
A run stops cleanly at its deadline (max_hours, or JOB_MARGIN before a Slurm job's end) and resumes when run again."""
import datetime, logging, math, os, subprocess, time

LOG = logging.getLogger('rankbound').info

SLICE_FRACTION = 1 / 30         # CPU per slice: this share of the sieve time,
SLICE_MIN, SLICE_MAX = 10.0, 300.0     # between these (seconds)
FIRST_CHECK = 0.8               # first check at this share of the relations
SLOPE_DEFAULT = 0.6             # excess per relation, until measured
NEXT_FULL = 1.15                # next full solve at 15% more relations
PLATEAU = 1.14                  # same bound this far apart: a plateau
                                # (below NEXT_FULL, so two solves suffice)
Q_FLOOR = 2 ** 10               # the special-q go down to this


# ---------- the slices and the checks

def slice_cpu(prm):
    """Return the CPU seconds per slice: SLICE_FRACTION of the estimated sieve time, within [SLICE_MIN, SLICE_MAX]."""
    return min(SLICE_MAX, max(SLICE_MIN, prm['model']['core_hours'] * 3600 * SLICE_FRACTION))


def q_ranges(prm, q0, width):
    """Yield the special-q ranges [lo, hi) of one polynomial: up from q0 to 2^lpb0, then down from q0 to Q_FLOOR.  A
    special-q above 2^lpb0 is a large prime of each relation it gives, so most of those would be purged.  The line
    siever's lines b go up from q0 without end."""
    if prm.get('siever') == 'line':
        lo = q0
        while True:
            yield lo, lo + width; lo += width
    top, bot = max(q0, 2 ** prm['lpb0']), min(q0, Q_FLOOR)
    lo = q0
    while lo < top:
        yield lo, min(lo + width, top); lo += width
    hi = q0
    while hi > bot:
        yield max(bot, hi - width), hi; hi -= width


def schedule(prm):
    """Yield (j, poly, q_lo, q_hi): the polynomials interleaved in proportion to their weights, until their ranges are
    used up.  Deterministic, so that shards share it."""
    polys = sorted(prm['polys'].items())
    def width(v):          # special-q (or lines) per slice
        nq = max(1, slice_cpu(prm) / max(v.get('sec_per_q', prm['model']['sec_per_q']), 1e-3))
        return max(1, round(nq)) if prm.get('siever') == 'line' else max(2, round(nq * math.log(prm['q0'])))
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
    return last['relations'] * NEXT_FULL      # after a full solve, or one that did not fit


def decide(history, target):
    """Return 'goal' (bound <= target), 'plateau', or None (go on) after a check."""
    e = history[-1] if history else {}
    if e.get('bound') is None:
        return None
    if target is not None and e['bound'] <= target:
        return 'goal'
    full = [x for x in history if x.get('bound') is not None]
    if len(full) >= 2 and full[-1]['bound'] == full[-2]['bound'] and full[-1]['relations'] >= PLATEAU * full[-2]['relations']:
        return 'plateau'
    return None


# ---------- the time limit

JOB_MARGIN = 300        # seconds kept free before a Slurm job's end


def job_end(env=None, squeue='squeue', log=LOG):
    """Return the end time (epoch seconds) of the Slurm job this process runs in, from SLURM_JOB_END_TIME or
    squeue; None outside Slurm or without a time limit."""
    env = os.environ if env is None else env
    v = env.get('SLURM_JOB_END_TIME', '').strip()
    if v.isdigit() and int(v) > 0:
        return float(v)
    jid = env.get('SLURM_JOB_ID', '').strip()
    if not jid.isdigit():
        return None
    aj, at = env.get('SLURM_ARRAY_JOB_ID', '').strip(), env.get('SLURM_ARRAY_TASK_ID', '').strip()
    if aj.isdigit() and at.isdigit():
        jid = '%s_%s' % (aj, at)
    try:
        r = subprocess.run([squeue, '-h', '-j', jid, '-o', '%e'], capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, SLURM_TIME_FORMAT='standard'))       # local time
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.split()
    try:
        if len(out) != 1:
            raise ValueError('%d lines' % len(out))
        return datetime.datetime.strptime(out[0], '%Y-%m-%dT%H:%M:%S').timestamp()
    except ValueError:
        return None


def deadline(max_hours=None, job=None, env=None, log=LOG):
    """Return the time at which a run stops cleanly, or None: a Slurm job's end less JOB_MARGIN, when it can be
    read, else max_hours from now."""
    env = os.environ if env is None else env
    if job is None:
        job = bool(env.get('SLURM_JOB_ID', '').strip())
    end = job_end(env) if job else None
    if end is not None:
        log('Will stop at %s, %d minutes before the Slurm job ends.' % (
            time.strftime('%H:%M', time.localtime(end - JOB_MARGIN)), JOB_MARGIN // 60))
        return end - JOB_MARGIN
    if job:
        log("Could not read the Slurm job's end time; %s" % (
            'running without a time limit.' if max_hours is None else
            'stopping after --max-hours (%g h) instead.' % max_hours))
    return None if max_hours is None else time.time() + max_hours * 3600


def earlier(until, max_hours):
    """Return the earlier of until and max_hours from now, or None if neither is given."""
    t = [x for x in (until, None if max_hours is None else time.time() + max_hours * 3600) if x is not None]
    return min(t) if t else None


FIRST_FULL = 0.5        # share of a run taken by the first full check


def check_seconds(history, nrel, window=None):
    """Estimate the seconds of a check at nrel relations: the longest full check so far, scaled by the square of the
    relation ratio; before the first one, FIRST_FULL of window.  It only postpones checks, so it never affects a
    bound."""
    checks = []
    for e in history:
        full = e.get('bound') is not None or e.get('dense_MB') is not None
        if checks and checks[-1][0] == e['relations']:
            checks[-1][1] += e.get('solve_s') or 0
            checks[-1][2] |= full
        else:
            checks.append([e['relations'], e.get('solve_s') or 0, full])
    guess = lambda cs: max((s * max(1.0, nrel / max(n, 1)) ** 2 for n, s, _ in cs), default=0.0)
    if any(f for _, _, f in checks):
        return guess([c for c in checks if c[2]])
    return max(guess(checks), FIRST_FULL * (window or 0))


def check_plan(history, nrel, until, start, now, fresh, unfinished=None, threads=1):
    """Decide whether a check due at nrel relations starts now, in a run from start to until; return (plan, the
    estimate in seconds).  'now': it fits, or this process is fresh.  'later': it waits for the next run.  'too
    long': a check with as much time and as many threads did not finish before (unfinished: check.json)."""
    if until is None:
        return 'now', 0.0
    need = check_seconds(history, nrel, until - start)
    if unfinished and unfinished.get('until') and unfinished.get('stopped', math.inf) < unfinished['until'] - JOB_MARGIN:
        unfinished = None
    if unfinished and unfinished.get('until'):         # it had that long and did not finish
        need = max(need, 1.1 * (unfinished['until'] - unfinished['start']))
    if now + need <= until:
        return 'now', need
    if not fresh:
        return 'later', need
    more = threads > unfinished.get('threads', threads) if unfinished else False     # more cores than it had: again
    return ('too long' if unfinished and unfinished.get('until') and not more else 'now'), need


# ---------- the slice tally

MAX_EMPTY = 100         # stop after this many empty slices in a row
DRY_SLICES, DRY_RATIO = 20, 0.1         # dried up: last 20 slices under 10% of the mean


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
    """Return whether the sieve has dried up: the last DRY_SLICES slices gave under DRY_RATIO of the mean relations
    per slice.  A line siever's yield falls fast with b, so the plateau rule's relations may never come."""
    r = tally.get('recent', [])
    return len(r) >= DRY_SLICES and sum(r) < DRY_RATIO * DRY_SLICES * tally['rels'] / tally['done']
