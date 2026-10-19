"""rankbound verify CERT...: check certificates, or folders of them.

Prints PASS, FAIL or NOT CHECKED for each.
Exit status: 0 all pass, 1 one fails, 3 one is over a limit, 4 the verifier could not run, 2 usage error."""
import argparse, glob, json, math, os, sys, time
from . import EXIT, check


def _positive(kind):
    def f(s):
        try:
            x = kind(s)
        except ValueError:
            raise argparse.ArgumentTypeError('not a number')
        if not (math.isfinite(x) and x > 0):
            raise argparse.ArgumentTypeError('must be positive')
        return x
    f.__name__ = kind.__name__
    return f


def _ainvs(s):
    """--curve: a1,a2,a3,a4,a6 (integers; any model)."""
    t = s.split(',')
    try:
        a = [int(x) for x in t]
    except ValueError:
        a = []
    if len(a) != 5 or any(str(x) != y.strip() for x, y in zip(a, t)):
        raise argparse.ArgumentTypeError('must be five integers a1,a2,a3,a4,a6')
    return a


def main(argv=None):
    p = argparse.ArgumentParser(prog='rankbound verify', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cert', nargs='+', metavar='CERT', help='certificates or folders')
    p.add_argument('--cores', type=_positive(int), default=1, metavar='N', help='cores to use')
    p.add_argument('--json', action='store_true', help='one JSON verdict per certificate')
    p.add_argument('--curve', type=_ainvs, metavar='A', help='the curve the certificate must be for')
    p.add_argument('--max-mb', type=_positive(float), metavar='M', help='skip files over M MB')
    p.add_argument('--max-seconds', type=_positive(float), metavar='T', help='time limit per certificate')
    a = p.parse_args(argv)
    certs = [c for x in a.cert for c in (sorted(glob.glob(os.path.join(x, '*.cert.json'))) if os.path.isdir(x) else [x])]
    if not certs:
        p.error('No certificate in %s.' % ', '.join(a.cert))
    if a.curve and len(certs) != 1:
        p.error('--curve works with one certificate only.')
    log = None if a.json else (lambda *x: print(time.strftime('%H:%M:%S'), *x, flush=True))
    verdicts = []
    for c in certs:
        v = check(c, curve=a.curve, cores=a.cores, max_mb=a.max_mb, max_seconds=a.max_seconds, log=log)
        verdicts.append(v['verdict'])
        if a.json:
            print(json.dumps(v), flush=True)
    if log and len(certs) > 1:
        name = lambda c: os.path.basename(c)
        failed = [name(c) for c, v in zip(certs, verdicts) if v == 'rejected']
        unchecked = [name(c) for c, v in zip(certs, verdicts) if v in ('limit', 'error')]
        log('. '.join(['%d of %d certificates pass' % (verdicts.count('accepted'), len(certs))] +
                      (['Failed: ' + ', '.join(failed)] if failed else []) +
                      (['Not checked: ' + ', '.join(unchecked)] if unchecked else [])))
    codes = {EXIT[v] for v in verdicts}
    sys.exit(next((x for x in (1, 4, 3) if x in codes), 0))


if __name__ == '__main__':
    main()
