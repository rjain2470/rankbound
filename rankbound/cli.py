"""rankbound: an upper bound on the rank of an elliptic curve over Q, under GRH, with a checkable certificate.

  rankbound bound CURVE                     e.g. rankbound bound 0,1,1,-2,0
  rankbound bound --curves FILE             the curves of a JSON curve list
  rankbound estimate CURVE | --curves FILE  the cost only
  rankbound status [DIR]                    progress
  rankbound verify CERT...                  check certificates

For clusters:
  rankbound sieve FOLDER --shard K/N
  rankbound solve FOLDER

CURVE is a1,a2,a3,a4,a6 or a4,a6. Run the same command again to resume.
Exit status: 0 success, 1 failure, 2 usage error."""
# The prover's command line; rankbound/__main__.py sends `rankbound verify` to the verifier first.
import argparse, glob, logging, math, os, re, sys
from . import RankboundError, Stopped, field, gp_message, known, linalg
from .rules import deadline
from .run import STATUS_COLUMNS, VERDICT, Job

log = logging.getLogger('rankbound').info
DIR = 'rankbound_work'
STOPPED = 'Stopped at the time limit. Run the same command again to resume.'
CURVE = re.compile(r'-[0-9]+(,-?[0-9]+)+')      # a curve argparse would take for an option


def positive(kind, zero=False):
    """An argparse type: a finite number of kind, > 0, or >= 0 with zero."""
    def f(s):
        try:
            x = kind(s)
        except ValueError:
            raise argparse.ArgumentTypeError('not a number')
        if not (math.isfinite(x) and (x >= 0 if zero else x > 0)):
            raise argparse.ArgumentTypeError('must be >= 0' if zero else 'must be positive')
        return x
    f.__name__ = kind.__name__
    return f


def shard(s):
    """An argparse type: --shard K/N, integers with 0 <= K < N."""
    m = re.fullmatch(r'([0-9]+)/([0-9]+)', s)
    if not (m and int(m[1]) < int(m[2])):
        raise argparse.ArgumentTypeError('must be K/N with 0 <= K < N')
    return int(m[1]), int(m[2])


def curves(a):
    """Return the Curves of the command's options."""
    curve = a.curve or a.curve_arg
    if not (curve or a.curves):
        raise SystemExit('Give a curve (e.g. 0,1,1,-2,0) or --curves FILE.')
    if curve and a.curves:
        raise SystemExit('Give a curve or --curves FILE, not both.')
    try:
        if a.curves:
            given = [o for o, v in (('--bad-primes', a.bad_primes), ('--name', a.name),
                                    ('--r', a.r if not a.points else None)) if v is not None]
            if given:
                raise SystemExit('%s work%s with a single curve; a curve list gives %s per curve.' % (
                    ' and '.join(given), '' if len(given) > 1 else 's', 'them' if len(given) > 1 else 'it'))
            if a.ids is not None and not a.ids.strip():
                raise SystemExit('--ids is empty. Leave it out to run the whole list.')
            ids = None if a.ids is None else [x.strip() for x in a.ids.split(',')]
            if ids is not None and not all(ids):
                raise SystemExit('--ids %r contains an empty id.' % a.ids)
            cs = field.from_list(a.curves, ids)
        else:
            cs = [field.from_ainvs(curve.split(','), a.bad_primes.split(',') if a.bad_primes else None, a.r, a.name,
                                   factor_seconds=0 if a.ecm_hours else 600, composites=bool(a.ecm_hours))]
        if a.points:
            if len(cs) != 1:
                raise SystemExit('--points works with a single curve only.')
            cs = [field.with_points(cs[0], field.read_points(a.points), a.r)]
        return cs
    except ValueError as e:
        raise SystemExit(gp_message(e))


def cmd_bound(a):
    """bound: with --relations, one curve on those relations; else estimate, then run.  One time limit for all."""
    a.until = deadline(a.max_hours)
    curve = a.curve or a.curve_arg
    if not a.relations:
        if a.keep_sieving:
            raise SystemExit('--keep-sieving needs --relations.')
        if curve and not a.curves and not a.points and not a.compute:     # certified already?
            if a.name is not None:
                field.check_name(a.name)
            field.r_value(a.r)
            row = known.lookup(field.minimal_model(curve.split(','), a.bad_primes.split(',') if a.bad_primes else None))
            if row is not None:                             # found by the minimal model
                for l in known.lines(row, curve):
                    log(l)
                return 0
        return cmd_estimate(a, run=True)
    cs = curves(a)
    if len(cs) != 1:
        raise SystemExit('--relations works with a single curve only.')
    try:
        rels = linalg.relation_input(a.relations)
        J = Job(folder(a, cs[0], rels))
        cs[0] = field.factored(cs[0], a.ecm_hours, J.dir)
        return report(J.run(cs[0], cores=a.cores, mem_gb=a.mem, until=a.until,
                                                      relations=rels, sieve=a.keep_sieving, ecm_hours=a.ecm_hours))
    except Stopped:
        log(STOPPED)


def folder(a, c, rels=None):
    """Return the work folder DIR/NAME of the curve c.  Without --name, a curve given by hand gets the suffixes of its
    r, points and relations, as in rankbound(); --name and curve lists keep their names, so resubmitting resumes."""
    key = c.name
    if not (a.name or a.curves):
        key += '' if c.r is None else '_r%d' % c.r + ('p' if c.proof else '')
        if rels is not None:
            key += '_rel' + rels.digest[:8] + ('_sieve' if a.keep_sieving else '')
    return os.path.join(a.dir, key)


def report(r):
    """Log the RESULT line; return 1 if the certificate failed its verification, else 0."""
    S, lb = r['rank_bound'], r.get('rank_lower_bound')
    equal = S == 0 or (r.get('r_proved') and S == lb)
    log('RESULT %s: rank %s %d under GRH, %s (%.2f core-h)' % (r['name'], '=' if equal else '<=', S,
                                                              VERDICT[r['verified']], r['sieve_core_hours']))
    return int(r['verified'] is False)


def first_line(e):
    m = gp_message(e)
    return m.splitlines()[0] if m else type(e).__name__


def cmd_estimate(a, run=False):
    """estimate: prepare and tune every curve; with run (bound), then run them, cheapest first.  Of several curves,
    one that fails is skipped with its reason.  Returns 1 if a curve was skipped or failed, else 0."""
    cs = curves(a)
    until = a.until if run else deadline()      # estimate: factoring stops before a Slurm job's end too
    plan, failed = [], []
    for c in cs:
        J = Job(folder(a, c))
        try:
            c = field.factored(c, a.ecm_hours, J.dir)      # a list curve without bad_primes
            if os.path.exists(os.path.join(J.dir, 'result.json')):
                plan.append((0, J, c)); continue
            J.prepare(c, ecm_hours=a.ecm_hours, cores=a.cores, mem_gb=a.mem, until=until)
            plan.append((J.tune(a.mem, a.cores)['model']['core_hours'], J, c))
        except Stopped:                     # the time limit, while factoring
            log(STOPPED)
            return int(bool(failed))
        except (RankboundError, RuntimeError, ValueError) as e:
            if len(cs) == 1:
                raise
            failed.append(c.name)
            log('Skipped %s. %s' % (c.name, first_line(e)))
    plan.sort(key=lambda t: t[0])
    if len(cs) > 1:
        log('%d curves; estimated cost %.1f core-hours in all.%s' % (
            len(plan), sum(t[0] for t in plan), ' Skipped: %s.' % ', '.join(failed) if failed else ''))
    rc = int(bool(failed))
    if not run:
        return rc
    for _, J, c in plan:
        try:
            rc |= report(J.run(c, cores=a.cores, mem_gb=a.mem, until=a.until, ecm_hours=a.ecm_hours))
        except Stopped:
            log(STOPPED)
        except (RankboundError, RuntimeError, ValueError) as e:
            if len(plan) == 1:
                raise
            rc = 1
            log('%s failed. %s' % (c.name, first_line(e)))
        finally:
            J.solver = None         # free its matrix before the next curve
    return rc


def folder_job(path):
    """Return the Job of the work folder path; exit if it is not one."""
    if not os.path.exists(os.path.join(path, 'field.json')):
        raise SystemExit('%s is not a rankbound work folder. Start it with rankbound bound or rankbound estimate.' % path)
    return Job(path)


def cmd_sieve(a):
    k, n = a.shard
    solver = (k == 0) if a.solver is None else a.solver == 'yes'
    J = folder_job(a.folder)
    outcome = J.sieve(a.cores, shard=(k, n), solver=solver, until=deadline(a.max_hours))  # the solver: also an earlier STOP
    if outcome and not os.path.exists(os.path.join(a.folder, 'result.json')):
        return report(J.certify(jobs=a.cores, outcome=outcome))
    if not outcome and n == 1 and not os.path.exists(os.path.join(a.folder, 'STOP')):
        log(STOPPED)


def cmd_solve(a):
    J = folder_job(a.folder)
    if os.path.exists(os.path.join(a.folder, 'STOP')) or J.solve(threads=a.cores):
        r = J.certify(jobs=a.cores)
        log('RESULT %s: %s' % (r['name'], r['statement']))


def cmd_status(a):
    if not os.path.isdir(a.dir):
        raise SystemExit('%s is not a folder.' % a.dir)
    print('\t'.join(STATUS_COLUMNS))
    for st in sorted(glob.glob(os.path.join(a.dir, '*', 'field.json'))):
        s = Job(os.path.dirname(st)).status()
        print('\t'.join(str(s[c]) for c in STATUS_COLUMNS))


def curve_options(s, run):
    """Add the options of bound (run) and estimate."""
    s.add_argument('curve', nargs='?', metavar='CURVE', help='a1,a2,a3,a4,a6 or a4,a6')
    s.add_argument('--curve-arg', help=argparse.SUPPRESS)       # a CURVE starting with a minus sign
    s.add_argument('--dir', default=DIR, help='work folder (default ./%s)' % DIR)
    s.add_argument('--name', help='name of the folder and certificate')
    s.add_argument('--bad-primes', metavar='P,..', help='prime factors of the discriminant')
    s.add_argument('--r', type=int, metavar='R', help='a known lower bound on the rank')
    s.add_argument('--curves', metavar='FILE', help='a JSON curve list')
    s.add_argument('--ids', help='ids to run')
    s.add_argument('--cores', type=positive(int), default=1, metavar='N', help='cores to use')
    s.add_argument('--mem', type=positive(float), default=1.0, metavar='GB', help='GB of memory per core')
    s.add_argument('--points', metavar='FILE', help='independent points, one "x y" per line')
    s.add_argument('--ecm-hours', type=positive(float, zero=True), default=1.0, metavar='H',
                   help='factor the discriminant by ECM for at most H hours (default 1; 0: PARI only, 10 min)')
    if run:
        s.add_argument('--max-hours', type=positive(float, zero=True), metavar='H', help='stop after H hours')
        s.add_argument('--relations', metavar='FILE', help='start from these relations')
        s.add_argument('--keep-sieving', action='store_true', help='sieve beyond the given relations')
        s.add_argument('--compute', action='store_true', help='compute even if already certified')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] in (['bound'], ['estimate']):           # a curve such as -1,0,1,2,3 is not an option
        argv = argv[:1] + ['--curve-arg=' + x if CURVE.fullmatch(x) else x for x in argv[1:]]
    p = argparse.ArgumentParser(prog='rankbound', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest='cmd', required=True, metavar='COMMAND', help=argparse.SUPPRESS)
    s = sp.add_parser('bound'); curve_options(s, True); s.set_defaults(f=cmd_bound)
    s = sp.add_parser('estimate'); curve_options(s, False); s.set_defaults(f=cmd_estimate)
    s = sp.add_parser('status'); s.add_argument('dir', nargs='?', default=DIR, metavar='DIR'); s.set_defaults(f=cmd_status)
    s = sp.add_parser('sieve'); s.add_argument('folder', metavar='FOLDER')
    s.add_argument('--shard', type=shard, default=(0, 1), metavar='K/N', help='this worker is shard K of N')
    s.add_argument('--cores', type=positive(int), default=1, metavar='N', help='cores to use')
    s.add_argument('--max-hours', type=positive(float, zero=True), metavar='H', help='stop after H hours')
    s.add_argument('--solver', choices=('yes', 'no'), help='run the checks here')
    s.set_defaults(f=cmd_sieve)
    s = sp.add_parser('solve'); s.add_argument('folder', metavar='FOLDER')
    s.add_argument('--cores', type=positive(int), default=1, metavar='N', help='cores to use')
    s.set_defaults(f=cmd_solve)
    s = sp.add_parser('verify')
    s.set_defaults(f=lambda a: p.error('Run rankbound verify.'))
    a = p.parse_args(argv)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%H:%M:%S', level=logging.INFO, stream=sys.stdout)
    try:
        return a.f(a) or 0
    except (RankboundError, ValueError, RuntimeError, OSError) as e:      # gp's error(): its message, not its trace
        raise SystemExit(gp_message(e))


if __name__ == '__main__':
    sys.exit(main())
