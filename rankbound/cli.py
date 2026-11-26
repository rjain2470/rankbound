"""rankbound: an upper bound on the rank of an elliptic curve over Q, with a witness that the checker verifies.

  rankbound bound CURVE                     e.g. rankbound bound 0,1,1,-2,0
  rankbound bound --curves FILE             the curves of a JSON curve list
  rankbound bound CURVE --estimate          the cost only
  rankbound bound CURVE --shard K/N         sieve one share of the relations on a cluster, then exit
  rankbound status [DIR]                    the runs in a work folder
  rankbound check WITNESS...                check witnesses (PARI/GP)
  rankbound unpack BUNDLE [--ids IDS]       write a bundle's witnesses as files (sha256 checked), to check them

CURVE is a1,a2,a3,a4,a6 or a4,a6.  Run the same command again to resume (a stopped run prints it).
stdout has the results: each curve's bound and witness (or how its run stopped and the command that resumes it),
the estimates, the status table; with --json instead one JSON object per curve (schema rankbound-result-1: bound,
hypothesis, verification, witness, lower bound, outcome, resume).  The progress goes to stderr; -v adds the methods'
inner steps (also in the work folder's run.log).
Exit status: 0 success, 1 failure, 2 usage error, 3 stopped at the time limit (run the same command to resume)."""
# The CLI only parses options and renders lines; the runner, shared with the API, does the work.
import argparse, errno, json, logging, math, os, re, shlex, sys

from .messages import HYP, MESSAGES, STATES, STATUS, core_hours, rank_text, shown, text
from . import core, runner
from .runtime import DETAIL, LOG, interruptible, say

DIR = runner.DIR
EST_ECM_HOURS = 0.05                    # 3 minutes of factoring for --estimate without --ecm-hours
RESULT_HYP = {True: 'assuming the GRH', False: 'unconditionally'}
# the events that are results (stdout without --json); all others are progress (stderr)
RESULTS = frozenset(('end', 'rejected', 'done_before', 'in_table', 'stopped', 'interrupted', 'resume', 'best_so_far',
                     'attempted', 'final_too_long', 'final_budget', 'list_failed', 'list', 'list_unknown', 'list_none'))


def is_result(record):
    """Return whether the log record is a result line."""
    e = getattr(record, 'event', None)
    return e is not None and (e in RESULTS or e.startswith('est_'))
CURVE = re.compile(r'-[0-9]+(,-?[0-9]+)+')      # a curve that argparse would take for an option


INT = re.compile(r'-?[0-9]+')                    # ASCII digits only: no '+', '_', spaces or other scripts' digits
FLOAT = re.compile(r'-?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?')


def positive(kind, zero=False):
    """Return an argparse type for a finite number of kind that is > 0 (>= 0 with zero)."""
    def f(s):
        if len(s) > 40:
            raise argparse.ArgumentTypeError('%r... is too large' % s[:20])
        if not (INT if kind is int else FLOAT).fullmatch(s):
            raise argparse.ArgumentTypeError('%r is not %s' % (s, 'a whole number' if kind is int else
                                                               'a number like 2 or 0.5'))
        x = kind(s)
        if not math.isfinite(x):
            raise argparse.ArgumentTypeError('%r is too large' % s)
        if not (x >= 0 if zero else x > 0):
            raise argparse.ArgumentTypeError('%r must be >= 0' % s if zero else '%r must be positive' % s)
        return x
    f.__name__ = kind.__name__
    return f


def shard(s):
    """Parse --shard K/N, with 1 <= K <= N."""
    m = re.fullmatch(r'([0-9]+)/([0-9]+)', s)
    if not (m and 1 <= int(m[1]) <= int(m[2])):
        raise argparse.ArgumentTypeError('must be K/N with 1 <= K <= N')
    return int(m[1]), int(m[2])


def usage(key, **kw):
    """Stop with a usage error (exit 2)."""
    print(text(key, **kw), file=sys.stderr)
    raise SystemExit(core.EXIT_USAGE)


def first_line(e):
    m = str(e)
    return m.splitlines()[0] if m else type(e).__name__


def curve_text(s):
    """Return the entries of a CURVE, which may be inside [] (as LMFDB, Sage and PARI print it) and spaced."""
    t = s.strip(' \t')
    if t[:1] == '[' and t[-1:] == ']':
        t = t[1:-1]
    return [x.strip(' \t') for x in t.split(',')]


def entries(s):
    """Return the entries of a comma-separated option value, stripped."""
    return [x.strip(' \t') for x in s.split(',')]


def checked_options(a):
    """Return (curve, ids, curves, points, relations) from the options, each None if not given.

    Options that exclude each other, and a value or FILE that cannot be read or parsed, are usage errors.
    """
    given = ([a.curve] if a.curve else []) + (a.curve_arg or [])
    if len(given) > 1:
        usage('one_per_command')
    curve = given[0] if given else None
    if not (curve or a.curves):
        usage('no_curve')
    if curve and a.curves:
        usage('both')
    if a.curves:
        for o, v in (('--bad-primes', a.bad_primes), ('--r', a.r), ('--points', a.points), ('--relations', a.relations),
                     ('--shard', a.shard)):
            if v is not None:
                usage('one_curve', option=o)
    elif a.ids is not None:
        usage('ids_list')
    if a.estimate:
        for o, v in (('--max-hours', a.max_hours is not None), ('--budget', a.budget is not None),
                     ('--relations', a.relations), ('--compute', a.compute), ('--points', a.points),
                     ('--shard', a.shard), ('--json', a.json)):
            if v:
                usage('est_takes', option=o)
    if a.shard:
        for o, v in (('--relations', a.relations), ('--uncond', a.uncond), ('--points', a.points)):
            if v:
                usage('shard_takes', option=o)
    if a.relations and a.uncond:
        usage('rel_uncond')
    if a.ids is not None and not a.ids.strip():
        usage('ids_empty')
    if a.bad_primes is not None and not a.bad_primes.strip(' \t,'):
        usage('bad_primes_empty')
    ids = None if a.ids is None else entries(a.ids)
    if ids is not None and not all(ids):
        raise core.Usage('--ids %r contains an empty id.' % a.ids)
    E = curve_text(curve) if curve else None
    runner.well_formed(E, entries(a.bad_primes) if a.bad_primes else None)    # (a singular curve is a failure)
    got = {}
    for o, f in (('--curves', a.curves), ('--points', a.points), ('--relations', a.relations)):
        if f is None:
            got[o] = None
            continue
        try:
            open(f, 'rb').close()
        except OSError as e:
            usage('no_read', file=f, option=o, reason=e.strerror or type(e).__name__)
        try:                            # malformed contents are a usage error too
            got[o] = runner.read_input(o, f, ids)
        except OSError as e:
            usage('no_read', file=f, option=o, reason=e.strerror or type(e).__name__)
        except core.Usage:
            raise
        except ValueError as e:
            raise core.Usage(str(e))
    if got['--points'] == []:
        usage('no_points', file=a.points)
    curves, points, rels = got['--curves'], got['--points'], got['--relations']
    return E, ids, curves, points, rels


def cmd_bound(a):
    """Run `rankbound bound`: each curve's bound, cost (--estimate) or share of the sieve (--shard)."""
    curve, ids, curves, points, rels = checked_options(a)
    mode = 'estimate' if a.estimate else 'shard' if a.shard else 'run'
    ecm_hours = a.ecm_hours if a.ecm_hours is not None else EST_ECM_HOURS if a.estimate else 1.0
    with runner.factor_note(a.estimate and a.ecm_hours is None):     # --estimate factors briefly, and says so
        return bound_each(a, curve, ids, curves, points, rels, mode, ecm_hours)


def bound_each(a, curve, ids, curves, points, rels, mode, ecm_hours):
    """Run the curves through runner.each, rendering each as it comes; return the exit status."""
    ask = runner.Ask(r=a.r, points=points, relations=rels, uncond=a.uncond,
                     bad_primes=entries(a.bad_primes) if a.bad_primes else None, ecm_hours=ecm_hours,
                     cores=a.cores, max_hours=a.max_hours, budget=a.budget, compute=a.compute, root=a.dir,
                     words={'command': a.resume})
    rc, n, total, failed, unknown = 0, 0, 0.0, [], []
    for it in runner.each(ask, curve=curve, curves=curves, ids=ids, mode=mode, shard=a.shard):
        if it.error is not None:
            if it.alone:
                raise it.error
            rc = 1
            failed.append(it.name)
            say('list_failed', name=it.name, reason=first_line(it.error))
            emit(a, core.outcome_json(it.name, 'failed', error=first_line(it.error), curve=it.curve))
        elif it.stopped is not None:
            stopped(a, it)
            if it.stopped.outcome == 'stopped' and rc == 0:     # at the time limit: resumable
                rc = core.EXIT_STOPPED
        elif mode == 'estimate':
            n += 1
            if it.estimate is None:         # the descent: counted apart, never as 0
                unknown.append(it.name)
            else:
                total += it.estimate
        elif mode == 'shard':
            return it.shard
        elif it.result.origin == 'table':
            in_table(it.result, a, it.curve)
        else:
            rc |= report(it.result, a, it.curve)
    if a.curves and mode == 'estimate':
        sk = ' Skipped %s.' % ' and '.join(', '.join(failed).rsplit(', ', 1)) if failed else ''
        if not unknown:
            say('list', n=n, core_hours=core_hours(total), skipped=sk)
        else:                           # an unknown time is not mentioned
            say('list_unknown' if len(unknown) < n else 'list_none', n=n, known=n - len(unknown),
                core_hours=core_hours(total), skipped=sk)
    return rc


def emit(a, obj):
    """Print obj as a --json line on stdout, if --json."""
    if a.json:
        print(json.dumps(obj), flush=True)


def stopped(a, it):
    """Render a stopped curve: its best checked and attempted bounds, how to go on, and its --json object."""
    e = it.stopped
    best = e.result if e.result is not None and e.result.checked else None
    for key, res in (('best_so_far', best), ('attempted', e.attempted)):
        if res is not None:
            say(key, rank=rank_text(res.rank_bound, res.rank_lower_bound, res.r_proved), hyp=HYP[res.grh],
                path=shown(res.witness))
    cmd = e.resume or a.resume
    final(e.outcome, e.reason, cmd)
    emit(a, core.outcome_json(it.name, e.outcome, resume=cmd, best=best, attempted=e.attempted, reason=e.reason,
                              curve=it.curve))


def final(outcome, reason, cmd):
    """Say how to go on after a stop: what a final stop needs, else the resume command."""
    if outcome in core.FINAL:
        if reason:
            LOG.info(reason)
        say('final_' + outcome)
    elif outcome == 'stopped':
        say('resume', command=cmd)
        say('stopped')


def in_table(known, a, curve=None):
    """Render the table entry of a curve in witnesses.tsv."""
    say('in_table', rank=rank_text(known.rank_bound, known.rank_lower_bound, known.r_proved),
        hyp='assuming the GRH' if known.grh else 'unconditionally')
    emit(a, known.to_json(curve))
    return 0


def report(res, a, curve=None):
    """Render a curve's result and its --json object; return 1 if the checker rejected the witness."""
    if res.checked is False and res.details.get('check') == 'format':     # unreadable: perhaps damaged
        why = (res.details.get('reason') or '').replace('not a valid witness: ', '', 1).rstrip('.')
        say('unreadable', path=shown(res.witness), reason=why or 'its format', folder=shown(os.path.dirname(res.witness)))
    elif res.checked is False:
        say('rejected', check=res.details.get('check') or '?', reason=res.details.get('reason') or '')
    elif not res.details.get('earlier'):
        say('end', rank=rank_text(res.rank_bound, res.rank_lower_bound, res.r_proved), hyp=RESULT_HYP[res.grh],
            path=shown(res.witness), checked=' (checked)' if res.checked else '')
    if res.outcome in core.FINAL:
        final(res.outcome, (res.details.get('final') or {}).get('reason'), res.resume)
    emit(a, dict(res.to_json(curve), resume=res.resume or a.resume))
    return int(res.checked is False)


def cmd_unpack(a):
    """Run `rankbound unpack`: write the bundle's witnesses into DIR, each checked against its sha256."""
    from . import bundle
    try:
        for p in bundle.unpack(a.bundle, a.dir, entries(a.ids) if a.ids else None):
            print(p)
    except (OSError, ValueError, KeyError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


def cmd_status(a):
    """Run `rankbound status`: one line per work folder of DIR."""
    H = MESSAGES['status_head']
    if a.folder is None and not os.path.isdir(DIR):
        raise core.Usage(text('no_status', folder=DIR))
    rows = runner.status(a.folder or DIR)
    print(H.format('name', 'relations', 'core-hours', 'estimate', 'rank bound', 'status'))
    num = lambda x, f: f.format(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else '?'   # a record edited by hand
    for s in rows:
        st = STATUS[4] if s['state'] == 'checked' else STATUS[3] if s['state'] == 'finished' else STATES.get(s['state'], s['state'])
        print(H.format(str(s['name']), num(s['relations'], '{:,}'), num(s['core_hours'], '{:.2f}'),
                       '' if s['estimate'] is None else num(s['estimate'], '{:.2f}'),
                       '' if s['rank_bound'] is None else str(s['rank_bound']), str(st)))
    return 0


def options(s):
    """Add the options of `rankbound bound` to s."""
    s.add_argument('curve', nargs='?', metavar='CURVE', help='a1,a2,a3,a4,a6 or a4,a6')
    s.add_argument('--curve-arg', action='append', help=argparse.SUPPRESS)      # a CURVE starting with a minus sign
    s.add_argument('--curves', metavar='FILE', help='a JSON curve list')
    s.add_argument('--ids', help='the ids of the list to run')
    s.add_argument('--points', metavar='FILE', help='independent points, one "x y" per line')
    s.add_argument('--r', type=positive(int, zero=True), metavar='R', help='a known lower bound on the rank (cited)')
    s.add_argument('--uncond', action='store_true', help="the unconditional bound (Minkowski's) instead of the GRH's")
    s.add_argument('--relations', metavar='FILE', help='prove the bound from these relations')
    s.add_argument('--bad-primes', metavar='P,..', help='the prime factors of the discriminant')
    s.add_argument('--ecm-hours', type=positive(float, zero=True), metavar='H',
                   help='factor the discriminant by GMP-ECM for at most H hours (default 1; with --estimate 3 minutes)')
    s.add_argument('--cores', type=positive(int), metavar='N',
                   help="cores to use (default RANKBOUND_CORES, else this machine's performance cores or the Slurm "
                        "job's allocation)")
    s.add_argument('--max-hours', type=positive(float, zero=True), metavar='H', help='stop after H hours')
    s.add_argument('--budget', type=positive(float), metavar='H',
                   help='the core-hours the sieve may use for a curve in all, across resumes (default 3 times its '
                        'estimate, at least 1); a resume can raise it, not lower it')
    s.add_argument('--dir', default=DIR, help='the work folder (default ./%s)' % DIR)
    s.add_argument('--compute', action='store_true', help='compute even if the curve is in the table')
    s.add_argument('--estimate', action='store_true',
                   help='estimate the cost only (factoring the discriminant if needed, briefly: see --ecm-hours)')
    s.add_argument('--json', action='store_true', help='one JSON line per curve on stdout instead of the result lines')
    s.add_argument('-v', '--verbose', action='store_true', help="also show the methods' inner steps (run.log's)")
    s.add_argument('--shard', type=shard, metavar='K/N', help='sieve share K of N and exit (a cluster job)')


class Parser(argparse.ArgumentParser):
    """argparse with a one-line usage error on stderr (exit 2)."""

    def error(self, message):
        print('%s: error: %s. See %s --help.' % (self.prog, message.rstrip('.'), self.prog), file=sys.stderr)
        raise SystemExit(core.EXIT_USAGE)


def curve_args(argv, takes):
    """Return argv with each curve starting with a minus sign (like -1,0,1,2,3) as --curve-arg=CURVE.

    argparse would take it for an option.  Values of the options in takes (or their prefixes) and anything after
    '--' are left alone.
    """
    out = []
    for k, x in enumerate(argv):
        if x == '--':
            return out + argv[k:]
        prev = out[-1] if out else ''
        value = prev.startswith('-') and '=' not in prev and (prev in takes or (
            prev.startswith('--') and len(prev) > 2 and any(o.startswith(prev) for o in takes)))
        out.append('--curve-arg=' + x if CURVE.fullmatch(x) and not value else x)
    return out


def main(argv=None):
    """Run `rankbound bound` or `rankbound status`; `rankbound check` goes to the checker."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['check']:           # as in __main__
        from .check import main as check
        return check(argv[1:])
    original = list(argv)               # as typed, for the resume command
    p = Parser(prog='rankbound', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--version', action='version', version='rankbound ' + core.VERSION)
    sp = p.add_subparsers(dest='cmd', required=True, metavar='COMMAND', help=argparse.SUPPRESS)
    s = sp.add_parser('bound'); options(s); s.set_defaults(f=cmd_bound)
    if argv[:1] == ['bound']:
        takes = {o for act in s._actions if act.nargs != 0 for o in act.option_strings}
        argv = argv[:1] + curve_args(argv[1:], takes)
    s = sp.add_parser('status'); s.add_argument('folder', nargs='?', metavar='DIR')
    s.set_defaults(f=cmd_status, json=False, verbose=False)
    sp.add_parser('check')              # listed among the commands; handled above
    s = sp.add_parser('unpack'); s.add_argument('bundle', metavar='BUNDLE')
    s.add_argument('--ids', metavar='IDS', help='the ids of the curves to unpack (default: all)')
    s.add_argument('--dir', default='witnesses', metavar='DIR', help='where to write them (default ./witnesses)')
    s.set_defaults(f=cmd_unpack, json=False, verbose=False)
    a, extra = p.parse_known_args(argv)
    if extra:                           # a second curve, or an unknown option
        if a.cmd == 'bound' and not any(x.startswith('-') and not CURVE.fullmatch(x) for x in extra):
            usage('one_per_command')
        sp.choices[a.cmd].error('unrecognized arguments: %s' % ' '.join(extra))
    a.resume = 'rankbound ' + shlex.join(original)
    fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
    progress, results = logging.StreamHandler(sys.stderr), logging.StreamHandler(sys.stdout)
    if not a.json:                      # results on stdout, progress on stderr
        progress.addFilter(lambda rec: not is_result(rec))
        results.addFilter(is_result)
    for h in (progress, results):
        h.setFormatter(fmt)
        h.rankbound = True
    LOG.handlers[:] = [x for x in LOG.handlers if not getattr(x, 'rankbound', False)] + [progress] + \
        ([] if a.json else [results])
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    DETAIL.handlers[:] = [x for x in DETAIL.handlers if not getattr(x, 'rankbound', False)]
    if a.verbose:                       # also the methods' inner steps (DETAIL)
        inner = logging.StreamHandler(sys.stderr)
        inner.setFormatter(fmt)
        inner.rankbound = True
        DETAIL.addHandler(inner)
        DETAIL.setLevel(logging.INFO)
    try:
        with interruptible():
            return a.f(a) or 0
    except KeyboardInterrupt as e:          # Ctrl-C or SIGTERM, after cleanup: like a time limit
        if a.cmd == 'bound':
            say('resume', command=a.resume)
            say('interrupted')
            emit(a, core.outcome_json(None, 'stopped', resume=a.resume))
        raise SystemExit(128 + getattr(e, 'signum', 2))
    except core.Usage as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(core.EXIT_USAGE)
    except (core.RankboundError, ValueError, RuntimeError, OSError) as e:
        msg = str(e)
        if isinstance(e, OSError) and e.errno == errno.ENOSPC:
            msg = text('disk_full', file=e.filename or 'a file')
        elif isinstance(e, OSError) and e.strerror:     # not Python's '[Errno 2] ...'
            msg = '%s%s.' % (e.strerror, ': %s' % e.filename if e.filename else '')
        if a.cmd == 'bound' and a.json:
            emit(a, core.outcome_json(None, 'failed', error=first_line(msg) if msg else type(e).__name__,
                                      resume=a.resume))
        raise SystemExit(msg)


if __name__ == '__main__':
    sys.exit(main())
