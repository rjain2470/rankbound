#!/usr/bin/env python3
"""The regression set: rankbound bound on every curve of tests/data/quick.tsv, each witness checked by
`rankbound check`, then the bounds compared with the list's expect_S.

    python3 tests/regression.py run [--workers W] [--cores C] [--rb-timeout S] [--verify-timeout S] [--out Q.tsv] [--work DIR]
    python3 tests/regression.py check Q.tsv [--record]

run: one row per curve (tool rankbound) and per witness checked (tool verify) in --out, with the bound S and its
hypothesis, the witness's format, CPU (user + sys of the process and every child it waited for, os.wait4) and wall
seconds, and the checker's verdict.  rankbound runs `python3 -m rankbound bound AINVS --bad-primes .. --r R
--ecm-hours 0 --compute --cores C --dir WORK --json` plus the row's options; its work folder is removed afterwards
unless --keep.  Tasks run cheapest first (by log10 |disc|), W at once.
check: FAIL when a row is missing or not ok, the verdict is not accepted, S < r (unsound), S != expect_S, or the
hypothesis is not the expected one; with --record, write the run's S into expect_S instead.  Exit 1 on any FAIL."""
import argparse, glob, gzip, json, math, os, platform, re, shutil, signal, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
LIST = os.path.join(HERE, 'data', 'quick.tsv')
COLS = ['date', 'machine', 'cpu_model', 'id', 'set', 'tool', 'cores', 'version', 'r', 'two_torsion', 'log10_dK',
        'status', 'S', 'hypothesis', 'method', 'cpu_s', 'wall_s', 'verdict', 'detail', 'source', 'timeout_s',
        'commit', 'rss_mb', 'witness_gz_bytes', 'witness_json_bytes', 'stages', 'settings']
LOCK = threading.Lock()


def read_list(path):
    rows, head = [], None
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        p = line.rstrip('\n').split('\t')
        if head is None:
            head = p
            continue
        rows.append(dict(zip(head, p)))
    return rows


def log10_disc(c):
    a1, a2, a3, a4, a6 = (int(x) for x in c['ainvs'].split(','))
    b2, b4, b6 = a1 * a1 + 4 * a2, 2 * a4 + a1 * a3, a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    d = abs(-b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 * b6 + 9 * b2 * b4 * b6)
    return len(str(d)) - 1 + math.log10(int(str(d)[:15]) / 10 ** (min(15, len(str(d))) - 1))


def cpu_model():
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def version():
    """The digest of rankbound's sources (*.py, *.gp, *.c under rankbound/)."""
    import hashlib
    h = hashlib.sha256()
    root = os.path.join(TREE, 'rankbound')
    for d, ds, fs in sorted(os.walk(root)):
        ds[:] = sorted(x for x in ds if x != '__pycache__')
        for f in sorted(fs):
            if f.endswith(('.py', '.gp', '.c')):
                p = os.path.join(d, f)
                h.update(os.path.relpath(p, root).encode() + b'\0' + open(p, 'rb').read())
    return 'rb-' + h.hexdigest()[:12]


def timed(cmd, timeout, log, info):
    """Run cmd in its own process group; kill the group at timeout.  (rc, cpu, wall, output, timed_out)."""
    t = time.time()
    out = open(log, 'w+')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',           # TREE's rankbound, in place
               PYTHONPATH=os.pathsep.join([TREE] + [x for x in [os.environ.get('PYTHONPATH')] if x]))
    p = subprocess.Popen(cmd, cwd=TREE, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True, env=env)
    killed = []

    def kill():
        killed.append(1)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
    timer = threading.Timer(timeout, kill)
    timer.start()
    _, st, ru = os.wait4(p.pid, 0)
    timer.cancel()
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass
    info['rss_mb'] = round(ru.ru_maxrss / (1 << 20 if sys.platform == 'darwin' else 1 << 10), 1)
    out.seek(0)
    text = out.read()
    out.close()
    return os.waitstatus_to_exitcode(st), round(ru.ru_utime + ru.ru_stime, 3), round(time.time() - t, 3), text, \
        bool(killed)


def row(c, tool, cores, ctx):
    return {'date': time.strftime('%Y-%m-%d %H:%M'), 'machine': ctx['machine'], 'cpu_model': ctx['cpu'],
            'id': c['id'], 'set': c['set'], 'tool': tool, 'cores': str(cores), 'version': ctx['version'], 'r': c['r'],
            'two_torsion': c.get('two_torsion', ''), 'source': 'measured'}


def task(c, a, ctx):
    rb = row(c, 'rankbound', a.cores, ctx)
    work = tempfile.mkdtemp(prefix='rb-' + re.sub(r'[^A-Za-z0-9_.-]', '_', c['id'])[:64] + '-', dir=a.work)
    cmd = [sys.executable, '-m', 'rankbound', 'bound', '--curve-arg=' + c['ainvs'], '--bad-primes', c['bad_primes'],
           '--ecm-hours', '0', '--compute', '--cores', str(a.cores), '--dir', work, '--json']
    if c['r'] != '':
        cmd += ['--r', c['r']]
    cmd += c.get('options', '').split()
    info = {}
    rc, cpu, wall, text, to = timed(cmd, a.rb_timeout, os.path.join(work, 'bound.log'), info)
    rb.update(cpu_s=cpu, wall_s=wall, timeout_s=int(a.rb_timeout), rss_mb=info['rss_mb'],
              settings=' '.join(cmd[3:]).replace(work, '<work>'))
    js = [line for line in text.splitlines() if line.startswith('{')]
    m = json.loads(js[-1]) if js else None
    wits = sorted(glob.glob(os.path.join(work, '**', '*.witness.json'), recursive=True) +
                  glob.glob(os.path.join(work, '**', '*.witness.json.gz'), recursive=True))
    rows = [rb]
    if to:
        rb.update(status='timeout', detail='> %d s' % a.rb_timeout)
    elif not m or not wits or m.get('outcome', 'completed') not in ('completed', 'table'):
        last = [line for line in text.splitlines() if line.strip()][-3:]
        rb.update(status='error', detail=('rc %d: ' % rc + ' | '.join(last))[:300])
    else:
        rb.update(status='ok', S=str(m['rank_bound']), hypothesis='GRH' if m['GRH'] else 'none',
                  verdict={True: 'checked', False: 'rejected', None: 'not checked'}[m['checked']])
        raw = open(wits[0], 'rb').read()
        data = gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw
        wit = json.loads(data)
        rb.update(witness_json_bytes=len(data), witness_gz_bytes=len(raw) if raw[:2] == b'\x1f\x8b' else
                  len(gzip.compress(data)), method=wit.get('format', '').replace('rankbound-', '').replace('-witness', ''))
        dK = (wit.get('field') or {}).get('dK')
        if dK:
            rb['log10_dK'] = '%.1f' % math.log10(abs(int(dK)))
        v = row(c, 'verify', 1, ctx)
        vinfo = {}
        rc, cpu, wall, text, to = timed([sys.executable, '-m', 'rankbound', 'check', wits[0], '--json'],
                                        a.verify_timeout, os.path.join(work, 'check.log'), vinfo)
        v.update(cpu_s=cpu, wall_s=wall, timeout_s=int(a.verify_timeout), rss_mb=vinfo['rss_mb'], method=rb['method'],
                 hypothesis=rb['hypothesis'])
        if to:
            v.update(status='timeout', detail='> %d s' % a.verify_timeout)
        else:
            try:
                j = json.loads([line for line in text.splitlines() if line.startswith('{')][-1])
                v.update(status='ok', verdict=j.get('verdict', '?'), S=rb['S'])
            except (IndexError, ValueError):
                v.update(status='error', detail=('rc %d: ' % rc + text.strip()[-200:]).replace('\n', ' | '))
        rows.append(v)
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return rows


def append(path, rows):
    with LOCK:
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a') as f:
            if new:
                f.write('\t'.join(COLS) + '\n')
            for r in rows:
                f.write('\t'.join(str(r.get(c, '')).replace('\t', ' ').replace('\n', ' ') for c in COLS) + '\n')


def cmd_run(a):
    a.work = os.path.abspath(a.work or os.environ.get('TMPDIR') or tempfile.gettempdir())
    os.makedirs(a.work, exist_ok=True)
    a.out = os.path.abspath(a.out)
    ctx = {'machine': a.machine or ('laptop' if platform.system() == 'Darwin' else 'hpc'), 'cpu': cpu_model(),
           'version': version()}
    cs = sorted(read_list(LIST), key=lambda c: (log10_disc(c), c['id']))
    print('%s: %d curves (%s; rankbound %s), %d workers, out %s' % (ctx['machine'], len(cs), ctx['cpu'],
          ctx['version'], a.workers, a.out), file=sys.stderr, flush=True)
    t0 = time.time()

    def one(c):
        try:
            rows = task(c, a, ctx)
        except Exception as e:                  # the harness's own failure is a failed row, not a crash
            rows = [dict(row(c, 'rankbound', a.cores, ctx), status='error',
                         detail='harness: %s: %s' % (type(e).__name__, str(e)[:200]))]
        append(a.out, rows)
        for r in rows:
            print('%s %-24s %-9s %-7s S=%-3s %s cpu %s wall %s' % (time.strftime('%H:%M:%S'), r['id'], r['tool'],
                  r['status'], r.get('S', ''), r.get('verdict', ''), r.get('cpu_s', ''), r.get('wall_s', '')),
                  file=sys.stderr, flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, cs))
    print('done: %d curves; %.0f s' % (len(cs), time.time() - t0), file=sys.stderr)
    return 0


def cmd_check(a):
    cs = read_list(LIST)
    rows = read_list(a.results)
    ver = next((r['version'] for r in reversed(rows) if r['tool'] == 'rankbound'), None)
    last = {(r['id'], r['tool']): r for r in rows if r['version'] == ver and r['tool'] in ('rankbound', 'verify')}
    bad, out = [], []
    for c in cs:
        rb, v = last.get((c['id'], 'rankbound')), last.get((c['id'], 'verify'))
        why = []
        if rb is None:
            why.append('no rankbound row')
        elif rb['status'] != 'ok':
            why.append('rankbound %s: %s' % (rb['status'], rb['detail'][:120]))
        else:
            S = int(rb['S'])
            if c['r'] != '' and S < int(c['r']):
                why.append('S %d < r %s: UNSOUND' % (S, c['r']))
            if not a.record and c.get('expect_S', '') != '' and S != int(c['expect_S']):
                why.append('S %d, expected %s' % (S, c['expect_S']))
            if rb['hypothesis'] != ('none' if '--uncond' in c.get('options', '') or c['two_torsion'] == 'yes' else 'GRH'):
                why.append('hypothesis %s' % rb['hypothesis'])
            if v is None or v['status'] != 'ok' or v['verdict'] != 'accepted':
                why.append('verify: %s' % ('no row' if v is None else (v['verdict'] or v['status']) + ' ' + v['detail'][:80]))
        cpu = sum(float(x['cpu_s'] or 0) for x in (rb, v) if x)
        out.append((c['id'], c['r'], rb['S'] if rb else '', c.get('expect_S', ''), '%.1f' % cpu,
                    'FAIL: ' + '; '.join(why) if why else 'ok'))
        if why:
            bad.append(c['id'])
    print('version %s, %d curves, results %s' % (ver, len(cs), a.results))
    print('| id | r | S | expect_S | CPU s (bound + verify) | |\n|---|---|---|---|---|---|')
    for o in out:
        print('| %s |' % ' | '.join(o))
    print('%d of %d ok; CPU %.0f s in all%s' % (len(cs) - len(bad), len(cs), sum(float(o[4]) for o in out),
                                              '; FAIL: ' + ', '.join(bad) if bad else ''))
    if a.record:
        S = {o[0]: o[2] for o in out}
        lines, head = open(LIST).read().splitlines(True), None
        with open(LIST, 'w') as f:
            for line in lines:
                if line.startswith('#') or not line.strip():
                    f.write(line)
                    continue
                p = line.rstrip('\n').split('\t')
                if head is None:
                    head = p
                else:
                    p[head.index('expect_S')] = S.get(p[0], '')
                f.write('\t'.join(p) + '\n')
        print('expect_S recorded in %s' % LIST)
    return int(bool(bad))


def main(argv=None):
    p = argparse.ArgumentParser(prog='tests/regression.py', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest='cmd', required=True)
    s = sp.add_parser('run')
    s.add_argument('--workers', type=int, default=1, help='curves at once')
    s.add_argument('--cores', type=int, default=1, help='cores of one rankbound run')
    s.add_argument('--rb-timeout', type=float, default=1800, help='seconds for one rankbound bound')
    s.add_argument('--verify-timeout', type=float, default=1800, help='seconds for one rankbound check')
    s.add_argument('--out', default='quick.tsv')
    s.add_argument('--work', help='work folder (default $TMPDIR)')
    s.add_argument('--machine', help="the machine's name in the results (default laptop on macOS, else hpc)")
    s.add_argument('--keep', action='store_true', help='keep the work folders')
    s = sp.add_parser('check')
    s.add_argument('results')
    s.add_argument('--record', action='store_true', help="write the run's S into the list's expect_S")
    a = p.parse_args(argv)
    return {'run': cmd_run, 'check': cmd_check}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
