#!/usr/bin/env python3
"""make test: the test modules (tests/test_*.py) in JOBS parallel shards, one unittest process per module, the
slowest first; a module's lines are printed when it ends, the failures' output again at the end.

    python3 tests/shards.py [--jobs N] [MODULE ...]

Run from the top of the tree (as the Makefile does), with the environment of the Makefile's TEST.  Exit 0 iff every
module passes."""
import argparse, glob, os, re, signal, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
# the slowest modules first (by their time in make test on a MacBook, 3 shards, November 2026): the shards end together
SLOW = ('test_cli', 'test_linalg', 'test_reference', 'test_check_attack', 'test_sieve', 'test_release', 'test_curve',
        'test_descent', 'test_arith', 'test_norm_kernel', 'test_selmer', 'test_check_colblock', 'test_guards',
        'test_stop_resume', 'test_check_form', 'test_check', 'test_check_selmer')


def modules():
    found = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(HERE, 'test_*.py')))
    return [m for m in SLOW if m in found] + [m for m in found if m not in SLOW]


def run(m):
    t = time.time()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([HERE, os.path.dirname(HERE)] +
                                                      [p for p in [os.environ.get('PYTHONPATH')] if p]))
    r = subprocess.run([sys.executable, '-m', 'unittest', '-v', m], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, env=env, cwd=os.path.dirname(HERE))
    return m, r.returncode, r.stdout, time.time() - t


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('modules', nargs='*')
    a = p.parse_args()
    # A job started in the background of a shell (make -j, `cmd &`) has SIGINT ignored, and its children would keep
    # it ignored (Python too): a handler here makes them start with the default, as from a terminal.
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    mods = a.modules or modules()
    t0, n, skipped, bad = time.time(), 0, 0, []
    with ThreadPoolExecutor(max(1, a.jobs)) as ex:
        for f in as_completed([ex.submit(run, m) for m in mods]):
            m, rc, out, secs = f.result()
            k = re.search(r'^Ran (\d+) tests?', out, re.M)
            s = re.search(r'skipped=(\d+)', out.splitlines()[-1] if out.strip() else '')
            n += int(k[1]) if k else 0
            skipped += int(s[1]) if s else 0
            print('%-28s %s in %4.0f s' % (m, 'ok' if rc == 0 else 'FAILED (exit %d)' % rc, secs), flush=True)
            if rc:
                bad.append((m, out))
    for m, out in bad:
        print('\n========== %s\n%s' % (m, out))
    print('Ran %d tests in %d modules in %.0f s on %d shards: %s%s' % (
        n, len(mods), time.time() - t0, a.jobs, 'OK' if not bad else 'FAILED: ' + ', '.join(m for m, _ in bad),
        ' (skipped=%d)' % skipped if skipped else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
