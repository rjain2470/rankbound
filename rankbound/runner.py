"""The one path from a curve to a published, checked bound, shared by the API and the CLI.

normalize -> look up (witnesses.tsv) -> route -> execute -> verify -> bind -> publish

The runner writes a route's witness once and commits it before the check, so a run stopped before the verdict resumes by
checking that very witness. A verdict counts only for the bytes the checker read and the route's claim.
"""
import contextlib, dataclasses, json, os, shutil, subprocess, sys, tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from . import core, machine, runtime
from .core import GRH, UNCONDITIONAL, LowerBound, ProofData, Request, Resources, Result, Stopped, Verdict

DESCENT, SPLIT, SIEVE = 'descent', 'SPLIT', 'sieve'
DIR = 'rankbound_work'                  # the default root of the work folders (the CLI's --dir)
METHOD_OF = {DESCENT: 'descent', SPLIT: 'selmer', SIEVE: 'selmer'}
PROOF = 'proof'                         # the artifact of the proof data and its witness's digest
PROOF_SCHEMA = 'rankbound-proof-1'
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- routing

def route(curve, uncond=False):
    """Return the route of the classified curve: DESCENT, SPLIT (uncond) or SIEVE."""
    if curve.two_torsion():
        return DESCENT
    return SPLIT if uncond else SIEVE


def folder(meth, path):
    """Return the route meth's work folder at path."""
    if meth == DESCENT:
        from .descent import Descent
        return Descent(path)
    from .selmer import Job, Selmer
    return Selmer(path) if meth == SPLIT else Job(path)


def hypothesis_of(meth):
    """Return the hypothesis of a route's bound."""
    return GRH if meth == SIEVE else UNCONDITIONAL


# ---------- normalize

def well_formed(E=None, bad_primes=None):
    """Raise core.Usage unless E and the bad primes are written as integers; singular curves fail later."""
    from . import curve as cv
    try:
        if E is not None:
            cv.ainvs_given(E)
        for p in bad_primes or ():
            cv.canon(p, 'bad prime')
    except ValueError as e:
        raise core.Usage(str(e)) from None


def model(E, bad_primes=None):
    """Return (the curve as given, its minimal model, the map [u, r, s, t]) as tuples of integer strings."""
    from . import curve as cv
    a = cv._ainvs5(E)
    mm, urst = cv.minimal(a, [int(p) for p in cv._bad_primes(bad_primes or [])])
    return tuple(str(x) for x in a), tuple(str(x) for x in mm), tuple(str(x) for x in urst)


def lower_of(curve):
    """Return the Curve's rank lower bound, proved by points or cited, or None."""
    if curve.r is None:
        return None
    if curve.proof:
        return LowerBound(curve.r, True, 'points', _points_digest(curve.proof))
    return LowerBound(curve.r, False, curve.source or 'cited')


def _points_digest(proof):
    """Return the digest of the points proving r and of their model."""
    import hashlib
    return hashlib.sha256(json.dumps([proof.get('points'), proof.get('ainvs')]).encode()).hexdigest()


def request_of(curve, meth, workdir, relations=None, options=None, model_map=None, budget=None):
    """Return the normalized Request of the Curve on the route meth, in workdir.

    A run does not compute model_map, since that could factor the discriminant again.
    """
    mm = tuple(str(x) for x in curve.minimal_model) if curve.minimal_model is not None else None
    return Request(source=tuple(str(x) for x in curve.ainvs), minimal_model=mm,
                   model_map=None if model_map is None else tuple(str(x) for x in model_map),
                   hypothesis=hypothesis_of(meth), relations=relations, lower=lower_of(curve), name=curve.name,
                   output=os.path.abspath(workdir),
                   bad_primes=None if curve.bad_primes is None else tuple(str(p) for p in curve.bad_primes),
                   options=dict(options or {}), budget=budget)


# ---------- preflight: before any expensive work

NATIVE = ('sieve', 'f2elim', 'f2kernel', os.path.join('check', 'check'))


GP_MIN = (2, 17)                        # the PARI/GP version the checker needs
_GP = {}


def gp_found():
    """Return (path, version) of the gp on the PATH, or (None, None); the prover imports nothing of the checker."""
    path = shutil.which('gp')
    if path is None:
        return None, None
    if path not in _GP:
        try:
            v = subprocess.run([path, '--version-short'], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            v = ''
        _GP[path] = v or None
    return path, _GP[path]


def gp_usable():
    """Return whether the gp on the PATH is at least GP_MIN."""
    path, v = gp_found()
    try:
        return path is not None and tuple(int(x) for x in v.split('.')[:2]) >= GP_MIN
    except (AttributeError, ValueError):
        return False


def preflight(root, checker=None):
    """Raise core.Unprepared before any work if a native program is missing or root is not writable.

    Without a usable PARI/GP the run goes ahead, and its witness is checked on a resume with one.
    """
    for prog in NATIVE:
        path = os.path.join(HERE, prog)
        if not (os.path.isfile(path) and os.access(path, os.X_OK)):
            raise core.Unprepared(runtime.text('no_native', program=path))
    d = os.path.abspath(root)
    while not os.path.isdir(d) and os.path.dirname(d) != d:
        if os.path.lexists(d):              # a file or a broken link where the folder would be
            raise core.Unprepared(runtime.text('no_write', folder=root, reason='not a folder' if os.path.exists(d)
                                               else 'a link to nothing'))
        d = os.path.dirname(d)              # the run makes the folder in its nearest existing parent
    try:                                    # each name the run makes there must fit NAME_MAX
        top = os.pathconf(d, 'PC_NAME_MAX')
    except (OSError, ValueError):
        top = 255
    if any(len(n.encode()) > top for n in os.path.relpath(os.path.abspath(root), d).split(os.sep)):
        raise core.Unprepared(runtime.text('no_write', folder=root, reason='a name in it is longer than %d bytes' % top))
    try:
        fd, probe = tempfile.mkstemp(prefix='.preflight.', dir=d)
        os.close(fd)
        os.unlink(probe)
    except OSError as e:
        raise core.Unprepared(runtime.text('no_write', folder=root, reason=e.strerror or str(e)))
    if checker is None and not gp_usable():
        path, v = gp_found()
        if path is None:
            runtime.say('no_gp')
        else:
            runtime.say('old_gp' if v else 'odd_gp', version=v, path=path)


# ---------- look up

def lookup(mm, uncond=False, given_r=None):
    """Return the witnesses.tsv Result for mm, or None; raise ValueError if given_r exceeds its bound."""
    from . import curve as cv
    row = cv.table_row(list(mm))
    if row is None or (uncond and row.get(cv.COL_GRH) != 'no'):
        return None
    cv.table_r(row, given_r)
    S, r, eq = int(row[cv.COL_UB]), row[cv.COL_LB], row[cv.COL_EQ]
    r_proved = eq in ('yes', 'no')
    if not r and given_r is not None:
        r = str(given_r)
    tor = row.get(cv.COL_TOR)
    grh = row.get(cv.COL_GRH, 'yes') != 'no'
    return Result(name=row['name'], rank_bound=S, rank_lower_bound=int(r) if r else None, r_proved=r_proved,
                  proved_equal=S == 0 or (r_proved and r != '' and S == int(r)), grh=grh,
                  method=cv.table_method(row), statement=cv.table_statement(row), witness=row['file'],
                  checked=None, minimal_model=list(mm),
                  torsion=[int(x) for x in tor.strip('[]').split(',') if x.strip()] if tor else None, known=True,
                  hypothesis=GRH if grh else UNCONDITIONAL, lower_bound=('proved' if r_proved else 'cited') if r else None,
                  outcome='table', verification='unavailable', origin='table', details={'row': dict(row)})


# ---------- verify

def checker_command():
    """Return the checker's command line."""
    return [sys.executable, '-m', 'rankbound', 'check']


def verify(path, curve=None, budget=None, cores=1, logfile=None, checker=None):
    """Return the checker's Verdict on the witness at path.

    The bytes are copied to a private snapshot while hashed, and the checker reads the snapshot, so the verdict is on
    exactly those bytes.
    """
    b = runtime.current() if budget is None else budget
    if checker is None and not gp_usable():
        return Verdict('unavailable', runtime.sha256_file(path)[0])
    b.check()
    tmp = tempfile.mkdtemp(prefix='.verify.', dir=os.path.dirname(os.path.abspath(path)))
    try:
        snap = os.path.join(tmp, os.path.basename(path))
        with open(snap, 'wb') as f:
            sha, _ = runtime.sha256_file(path, f)
        cmd = list(checker or checker_command()) + ['--json', '--cores', str(cores)]
        if curve is not None:
            cmd += ['--curve', ','.join(str(x) for x in curve)]
        left = b.left()
        if left is not None:
            cmd += ['--max-seconds', str(max(1, int(left)))]
        cmd.append(snap)
        root = os.path.dirname(HERE)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([root] + [p for p in [os.environ.get('PYTHONPATH')] if p]))
        rc, out, err = runtime.run(cmd, budget=b, env=env)
        if logfile:
            with open(logfile, 'w') as f:
                f.write((err or '') + (out or ''))
        lines = [l for l in (out or '').splitlines() if l.startswith('{')]
        if not lines:
            return Verdict('error', sha, reason='The checker failed (exit %d). See %s.' % (rc, logfile or 'its log'))
        try:
            d = json.loads(lines[-1])
            d['witness'] = str(path)        # the checker read a snapshot of path
            v = Verdict.of(d, sha)
        except (ValueError, TypeError):
            return Verdict('error', sha, reason='The checker printed no verdict (exit %d).' % rc)
        if v.state == 'accepted' and v.sha256 != sha:
            return Verdict('error', sha, reason='The checker read other bytes than the witness it was given.',
                           raw=v.raw)
        return v
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------- bind and publish

def against(proof, v):
    """Return v for the route's claim: accepting another bound or hypothesis is an error."""
    if v.state == 'accepted':
        ub, grh = v.raw.get('rank_upper_bound'), v.raw.get('GRH')
        if (ub is not None and ub != proof.rank_bound) or (grh is not None and (grh == 'yes') != (proof.hypothesis == GRH)):
            return Verdict('error', v.sha256, reason='The checker accepted another claim (rank <= %s, GRH %s) than '
                           "the route's (rank <= %d, %s)." % (ub, grh, proof.rank_bound, proof.hypothesis), raw=v.raw)
        lb, rp = v.raw.get('rank_lower_bound'), bool(v.raw.get('r_proved'))
        if lb != proof.rank_lower_bound or rp != bool(proof.r_proved):
            return Verdict('error', v.sha256, reason='The checker accepted another lower bound (r = %s, %s) than the '
                           "route's (r = %s, %s)." % (lb, 'proved' if rp else 'not proved', proof.rank_lower_bound,
                                                      'proved' if proof.r_proved else 'not proved'), raw=v.raw)
    return v


def committed(v, sha):
    """Return v for the witness committed with sha256 sha: a verdict on other bytes is an error."""
    if sha is not None and v.sha256 is not None and v.sha256 != sha:
        return Verdict('error', v.sha256, reason='The checker read other bytes (sha256 %s) than the witness committed '
                       '(%s).' % (v.sha256, sha), raw=v.raw)
    return v


def bind(req, proof, verdict, workdir, origin='computed', final=None, prov=None):
    """Return (run.json record, Result) of the proof with its verdict."""
    v = against(proof, verdict)
    S, lb = proof.rank_bound, proof.rank_lower_bound
    rec = dict(proof.record)
    rec.update(name=req.name, GRH=proof.hypothesis == GRH, rank_bound=S, rank_lower_bound=lb, r_proved=proof.r_proved,
               witness=proof.witness, method=proof.method, checked=core.checked_of(v.state), verification=v.state,
               witness_sha256=v.sha256, checker=v.checker, run_outcome=final['outcome'] if final else 'completed',
               request=req.key(), statement=v.statement or proof.statement, final=final)
    if prov is not None:
        rec['provenance'] = prov
    rec.setdefault('outcome', proof.outcome)
    rec.setdefault('cost', proof.cost)
    if v.state not in ('accepted', 'unavailable'):
        rec.update(check=v.check or v.state, reason=v.reason)
    return rec, result_of(rec, workdir, req, origin)


def result_of(rec, workdir, req=None, origin='computed'):
    """Return the Result of a run.json record."""
    from .messages import METHODS
    S, lb = rec['rank_bound'], rec['rank_lower_bound']
    rp = bool(rec.get('r_proved'))
    ver = rec.get('verification') or {True: 'accepted', False: 'rejected'}.get(rec.get('checked'), 'unavailable')
    tor = rec.get('torsion')
    tor = [int(x) for x in (tor if isinstance(tor, list) else str(tor or '').strip('[]').split(',')) if str(x).strip()]
    return Result(name=rec['name'], rank_bound=S, rank_lower_bound=lb, r_proved=rp,
                  proved_equal=ver == 'accepted' and (S == 0 or (rp and S == lb)),
                  grh=bool(rec['GRH']), method=METHODS.get(rec.get('method'), rec.get('method') or ''),
                  statement=rec['statement'], witness=rec['witness'], checked=core.checked_of(ver),
                  core_hours=rec.get('core_hours') or 0.0, workdir=workdir, minimal_model=rec.get('minimal_model'),
                  conductor=rec.get('conductor'), torsion=tor, G=rec.get('G'), details=rec,
                  hypothesis=GRH if rec['GRH'] else UNCONDITIONAL,
                  lower_bound=None if lb is None else 'proved' if rp else 'cited',
                  outcome=rec.get('run_outcome') or 'completed', verification=ver, origin=origin,
                  witness_sha256=rec.get('witness_sha256'), cost=rec.get('cost') or {},
                  resume=req.resume_command() if req is not None else None)


def proof_record(proof):
    """Return the proof data as JSON, without the witness."""
    return {'witness': os.path.basename(proof.witness), 'rank_bound': proof.rank_bound, 'hypothesis': proof.hypothesis,
            'statement': proof.statement, 'method': proof.method, 'rank_lower_bound': proof.rank_lower_bound,
            'r_proved': proof.r_proved, 'record': proof.record, 'cost': proof.cost, 'outcome': proof.outcome,
            'final': proof.final, 'reason': proof.reason}


def save_proof(workdir, req, proof):
    """Commit the proof data as the artifact 'proof', bound to the request and the witness's bytes."""
    if os.path.dirname(os.path.abspath(proof.witness)) != os.path.abspath(workdir):
        raise ValueError('The witness %s is not in the work folder %s.' % (proof.witness, workdir))
    name = os.path.basename(proof.witness)
    with runtime.transaction(workdir, PROOF, PROOF_SCHEMA, req.key()) as tx:
        tx.adopt(name)
        tx.write(PROOF + '.json', json.dumps(proof_record(proof), indent=1))
    return runtime.inspect(workdir, PROOF, digests=False)[1]['files'][name]['sha256']


def pending_proof(workdir, req):
    """Return (ProofData, sha256) of the request's committed, unchecked witness, or (None, None)."""
    m = runtime.load(workdir, PROOF, PROOF_SCHEMA, req.key())
    if m is None:
        return None, None
    try:
        d = runtime.read_json(os.path.join(workdir, PROOF + '.json'))
        return ProofData(witness=os.path.join(workdir, d['witness']), rank_bound=d['rank_bound'],
                         hypothesis=d['hypothesis'], statement=d['statement'], method=d['method'],
                         rank_lower_bound=d['rank_lower_bound'], r_proved=d['r_proved'], record=d['record'],
                         cost=d['cost'], outcome=d['outcome'], final=d.get('final'), reason=d.get('reason')), \
            m['files'][d['witness']]['sha256']
    except (OSError, ValueError, KeyError, TypeError):
        runtime.invalidate(workdir, PROOF)
        return None, None


# ---------- execute: one request, start to end

def initial(req, meth):
    """Return the fields of a new run.json."""
    return dict(name=req.name, method=METHOD_OF[meth], GRH=req.hypothesis == GRH,
                lower_bound=None if req.lower is None else req.lower.r, stage='not started', estimate=None,
                core_hours=0.0, relations=0, rank_bound=None, result=None)


def bind_folder(J, curve, given=None):
    """Bind J's folder to the Curve for a step that publishes nothing, and set J.key.

    Another r or other points share the work and leave run.json as it is.
    """
    gd = given.digest if given is not None else (J.info().get('key') or {}).get('relations')
    req = request_of(curve, J.route, J.dir, gd)
    J.key = req.identity()
    J.store.begin(J.key, **dict(initial(req, J.route), method=J.method))


def begin(store, req, meth):
    """Bind the folder to the request's identity and return the key.

    A request that differs only in r or its points takes over the folder and drops the old result.
    """
    key = req.identity()
    try:
        store.begin(key, rebind=True, **initial(req, meth))
    except runtime.Mismatch as e:
        old = e.old
        if isinstance(old, dict) and old.get('hypothesis') != req.hypothesis and dict(old, hypothesis=req.hypothesis) == key:
            raise runtime.Mismatch(runtime.text('other_grh' if old['hypothesis'] == GRH else 'other_unc'), old, key)
        raise
    return key


def rebased(rec, workdir):
    """Return rec with its witness in workdir, where every published witness lies, so a copied folder uses its own."""
    if rec and rec.get('witness'):
        w = os.path.join(workdir, os.path.basename(rec['witness']))
        if os.path.abspath(w) != os.path.abspath(rec['witness']):
            rec = dict(rec, witness=w)
    return rec


def changed(rec):
    """Return whether a published witness's bytes differ from its recorded sha256."""
    w, sha = rec.get('witness'), rec.get('witness_sha256')
    return bool(w and sha and os.path.exists(w) and runtime.sha256_file(w)[0] != sha)


def done_before(rec, req, say=True):
    """Return the Result of an earlier published result, reported again if say."""
    res = result_of(dict(rec, earlier=True), req.output, req, 'resumed')
    if say:
        runtime.say('done_before', rank=rank_words(res), hyp='assuming the GRH' if res.grh else 'unconditionally',
                    path=shown(res.witness), checked=' (checked)' if res.checked else '')
    return res


def proof_of_record(rec, meth):
    """Return the ProofData of a published record."""
    fin = rec.get('final') or {}
    return ProofData(witness=rec['witness'], rank_bound=rec['rank_bound'], hypothesis=GRH if rec['GRH'] else UNCONDITIONAL,
                     statement=rec['statement'], method=rec.get('method') or METHOD_OF[meth],
                     rank_lower_bound=rec['rank_lower_bound'], r_proved=bool(rec.get('r_proved')),
                     record={k: v for k, v in rec.items() if k not in ('checked', 'verification', 'check', 'reason',
                                                                     'witness_sha256', 'checker', 'earlier', 'final',
                                                                     'run_outcome')},
                     cost=rec.get('cost') or {}, outcome=rec.get('outcome') or 'completed', final=fin.get('outcome'),
                     reason=fin.get('reason'))


def attempted_of(proof, req):
    """Return the pending Result of a witness written but not checked."""
    rec = dict(proof.record, name=req.name, rank_bound=proof.rank_bound, rank_lower_bound=proof.rank_lower_bound,
               r_proved=proof.r_proved, GRH=proof.hypothesis == GRH, statement=proof.statement, witness=proof.witness,
               method=proof.method, verification='pending', run_outcome='stopped')
    return result_of(rec, req.output, req, 'resumed')


def final_of(outcome, reason, store, resources, window):
    """Return run.json's record of a final stop, with the budget, hours and cores it had."""
    return {'outcome': outcome, 'reason': reason,
            'budget_core_hours': store.info().get('budget_core_hours', resources.budget_core_hours),
            'window_h': None if window is None else round(window / 3600, 4), 'cores': resources.cores}


def kept(store, key, rec, req):
    """Return the checked final result rec as the best bound so far, or None.

    Its witness is copied to NAME.best.witness..., so the continuing run cannot change the bytes it stands for.
    """
    if not rec.get('checked'):
        return None
    d, b = os.path.split(rec['witness'])
    if '.best.' not in b:
        dst = os.path.join(d, b.replace('.witness.', '.best.witness.', 1) if '.witness.' in b else 'best.' + b)
        shutil.copyfile(rec['witness'], dst + '.part')
        os.replace(dst + '.part', dst)
        if runtime.sha256_file(dst)[0] != rec.get('witness_sha256'):
            return None
        rec = dict(rec, witness=dst)
        store.publish(key, rec)
    return result_of(rec, req.output, req, 'resumed')


def grown(fin, resources, window):
    """Return whether this run has more budget, time or cores than the final stop fin had."""
    if fin.get('outcome') == 'budget':
        b, old = resources.budget_core_hours, fin.get('budget_core_hours')
        return b is not None and (old is None or b > old)
    old = fin.get('window_h')
    return resources.cores > (fin.get('cores') or 0) or (old is not None and (window is None or window / 3600 > old * 1.05))


def conclude(store, key, req, proof, budget=None, cores=1, checker=None, origin='computed', sha=None, final=None,
             resources=None):
    """Check proof's committed witness, bind the verdict and publish under key.

    If the checker cannot finish, the witness stays committed and the next run checks it.
    """
    b = runtime.current() if budget is None else budget
    store.update(key, stage='checking')
    from . import curve as cv               # the checker's --curve needs all five a-invariants
    five = cv.ainvs_given(list(req.source))
    v = committed(against(proof, verify(proof.witness, five, b, cores, store.path('check.log'), checker)), sha)
    if v.state in ('limit', 'error'):
        runtime.raise_if_stopped(b)
    if v.state != 'unavailable':
        runtime.write_json(store.path('check.json'), dict(v.raw, sha256=v.sha256), None)
    if v.state in ('limit', 'error'):
        raise RuntimeError(v.reason)
    prov = provenance(req, v, store, resources if resources is not None else Resources(cores))
    if 'spent' in (proof.record or {}):         # include the check in the spending
        proof = dataclasses.replace(proof, record=dict(proof.record, spent=runtime.spent_now(store)))
    rec, res = bind(req, proof, v, req.output, origin, final, prov)
    store.publish(key, rec, stage='finished', rank_bound=rec['rank_bound'],
                  core_hours=round(rec.get('core_hours') or 0.0, 3), final=final)
    return res


PROVENANCE = 'rankbound-provenance-1'
ENV_PREFIXES = ('RANKBOUND_', 'CHECK_', 'OMP_NUM_THREADS', 'SLURM_JOB_ID', 'SLURM_CPUS_ON_NODE', 'SLURM_CPUS_PER_TASK')


def provenance(req, v, store, resources):
    """Return the run's provenance: versions, parameters, seeds and artifacts.

    It identifies the run but is not bit-for-bit reproducibility: relations, times and the sieve's order vary. A
    different seed can change what a cut-short search finds, never whether a checked bound holds.
    """
    import platform
    seeds = {'f2kernel': '1, 2, ... (one per try, in order)', 'sieve_trial': 1, 'majorant': 1,
             'descent_squares': 'random.Random(p) at each prime p',
             'unseeded': 'ECM curves and roots mod p (factoring: their results are checked exactly)'}
    arts = {}
    for f in sorted(os.listdir(req.output)):
        if f.endswith('.manifest.json'):
            try:
                m = runtime.read_json(os.path.join(req.output, f))
                arts[m['name']] = {'schema': m.get('schema'), 'fingerprint': m.get('fingerprint'),
                                   'files': {n: e.get('sha256') for n, e in sorted((m.get('files') or {}).items())}}
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                arts[f[:-len('.manifest.json')]] = None     # unreadable: unknown
    env = {k: os.environ[k] for k in sorted(os.environ) if k.startswith(ENV_PREFIXES)}
    return {'format': PROVENANCE,
            'versions': {'rankbound': core.VERSION, 'prover_sha256': _sources_sha256(),
                         'python': platform.python_version(), 'platform': '%s-%s' % (platform.system(), platform.machine()),
                         'checker': v.checker or v.raw.get('checker'), 'gp': v.raw.get('gp'),
                         'native': {p: _native_id(os.path.join(HERE, p)) for p in NATIVE}},
            'parameters': {'options': {k: x for k, x in sorted(req.options.items()) if x is not None},
                           'cores': resources.cores, 'cores_from': resources.cores_from,
                           'ecm_hours': resources.ecm_hours,
                           'max_hours': resources.max_hours, 'budget_core_hours': store.info().get('budget_core_hours'),
                           'hypothesis': req.hypothesis, 'environment': env},
            'seeds': seeds, 'artifacts': arts}


_IDS = {}


def _native_id(path):
    """Return {'sha256', 'compiler'} of a native program, or None if missing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    k = (path, st.st_size, st.st_mtime_ns)
    if k not in _IDS:
        with open(path, 'rb') as f:
            data = f.read()
        import hashlib, re
        m = re.search(rb'(GCC: \([^\x00]{1,120}|clang version [^\x00]{1,120})', data)
        _IDS[k] = {'sha256': hashlib.sha256(data).hexdigest(), 'compiler': m.group(1).decode('ascii', 'replace') if m else None}
    return _IDS[k]


def _sources_sha256():
    """Return the sha256 of the prover's sources."""
    if 'src' not in _IDS:
        import hashlib
        h = hashlib.sha256()
        for f in sorted(os.listdir(HERE)):
            if f.endswith('.py'):
                with open(os.path.join(HERE, f), 'rb') as fh:
                    h.update(f.encode() + b'\0' + hashlib.sha256(fh.read()).digest())
        _IDS['src'] = h.hexdigest()
    return _IDS['src']


def execute(curve, req, resources, meth, route_obj=None, budget=None, checker=None, cores=None, inputs=None):
    """Run the request on its route to a published result, resuming the work folder, and return the Result.

    A published result is returned again, checked first if it never was; a committed witness is checked, not proved
    again; a final stop is repeated unless this run has more. The run's CPU time, the checker's included, is counted.
    """
    from . import curve as cv
    with cv.analysis():
        return _execute(curve, req, resources, meth, route_obj, budget, checker, cores, inputs)


def _execute(curve, req, resources, meth, route_obj, budget, checker, cores, inputs):
    from . import witness
    J = route_obj if route_obj is not None else folder(meth, req.output)
    store = runtime.RunStore(req.output)
    cores = resources.cores if cores is None else cores
    b = runtime.current() if budget is None else budget
    window = b.left()                       # compared with a final 'too_long' stop's
    J.precheck(curve)
    if curve is not None and curve.bad_primes is not None and not curve.composites:
        from . import curve as cv, field
        field.check_bad_primes(cv.ainvs_given(list(curve.ainvs)), [int(p) for p in curve.bad_primes])
    from .messages import core_hours, duration, plural, shown
    runtime.say('setup', dir=shown(req.output), cores=plural(cores, 'core'),
                limit='' if window is None or window == float('inf') else ', at most ' + duration(window),
                budget='' if req.budget is None else ', a budget of ' + core_hours(req.budget))
    best = old = proof = None
    try:
        with J.logging(), store.own('bound'), runtime.scope(b), runtime.metered(store):
            key = begin(store, req, meth)
            J.key = key
            old = rebased(store.stored(key), req.output)
            if old and old.get('witness') and not os.path.exists(old['witness']):     # deleted: prove again
                runtime.say('witness_missing', path=shown(old['witness']))
                runtime.invalidate(req.output, PROOF)
                old = None
            fin = store.info().get('final')
            if fin and grown(fin, resources, window):
                store.update(key, final=None)
                runtime.invalidate(req.output, PROOF)
                fin = None
            if fin and not old:             # a final stop with no bound to show
                raise Stopped(req.output, outcome=fin['outcome'], reason=fin.get('reason'))
            if old and not fin and old.get('run_outcome') in core.FINAL:    # past a final stop: keep its bound as best
                best = kept(store, key, old, req)
                old = None
            if old and changed(old):        # the published witness changed: check it again
                runtime.say('witness_changed', path=shown(old['witness']))
                proof, sha = proof_of_record(old, meth), runtime.sha256_file(old['witness'])[0]
                return conclude(store, key, req, proof, b, cores, checker, 'resumed', sha, None, resources)
            if old and (old.get('checked') is not None or (checker is None and not gp_usable())):
                return done_before(old, req)
            proof, sha = pending_proof(req.output, req)
            origin = 'resumed' if proof is not None or old else 'computed'
            if proof is None and old:
                proof, sha = proof_of_record(old, meth), old.get('witness_sha256')
            if proof is None:
                try:
                    proof = J.prove(curve, req, resources, b, **(inputs or {}))
                except Stopped as e:
                    if e.outcome in core.FINAL:
                        store.update(key, final=final_of(e.outcome, e.reason, store, resources, window))
                    raise
                if proof is None:           # another process published the result meanwhile
                    return done_before(store.stored(key), req, say=False)
                if proof.data is not None:
                    witness.write(proof.data, proof.witness)        # the witness's only write
                sha = save_proof(req.output, req, proof)
            final = None
            if proof.final:
                final = fin if fin and fin['outcome'] == proof.final else final_of(proof.final, proof.reason, store,
                                                                                    resources, window)
            return conclude(store, key, req, proof, b, cores, checker, origin, sha, final, resources)
    except Stopped as e:
        e.workdir = req.output
        if e.result is None:
            e.result = best
        if e.attempted is None and proof is not None:
            e.attempted = attempted_of(proof, req)
        e.resume = req.resume_command()
        raise


def run_folder(J, curve, resources, *, relations=None, checker=None, options=None, budget=None):
    """Build the Curve's Request for the work folder J and execute it."""
    meth = J.route
    given = gd = None
    if relations is not None:
        from .sieve import relation_input
        given = relation_input(relations)
        gd = given.digest
    elif meth == SIEVE:
        gd = (J.info().get('key') or {}).get('relations')
    req = request_of(curve, meth, J.dir, gd, options, budget=resources.budget_core_hours)
    return execute(curve, req, resources, meth, J, budget, checker,
                   inputs={'given': given} if given is not None else None)


def rank_words(res):
    """Return 'Rank 0', 'Rank at most 3', ... for a Result."""
    from .messages import rank_text
    return rank_text(res.rank_bound, res.rank_lower_bound, res.r_proved)


def shown(path):
    from .messages import shown as sh
    return sh(path)


# ---------- status: what the work folders hold

def status(root=DIR):
    """Return `rankbound status`'s rows, one per work folder of root (or root itself)."""
    if not os.path.isdir(root):
        raise core.RankboundError(runtime.text('not_folder', folder=root))
    dirs = [os.path.join(root, n) for n in sorted(os.listdir(root)) if os.path.isfile(os.path.join(root, n, 'run.json'))]
    if not dirs and os.path.isfile(os.path.join(root, 'run.json')):
        dirs = [root]
    rows = []
    for d in dirs:
        store = runtime.RunStore(d)
        s = store.info()
        try:                                # do not show a result for another run
            res = store.stored(None, s) or {}
        except core.RankboundError:
            res = {}
        state = s.get('stage', 'not started')
        if (s.get('final') or {}).get('outcome') in core.FINAL:
            state = s['final']['outcome']
        elif res:
            state = 'checked' if res.get('checked') else 'finished'
        elif runtime.inspect(d, PROOF, digests=False)[0] == 'valid':
            state = 'pending'
        rows.append({'name': s.get('name', ''), 'relations': s.get('relations') or 0,
                     'core_hours': s.get('core_hours') or 0.0, 'estimate': s.get('estimate'),
                     'rank_bound': res.get('rank_bound', s.get('rank_bound')), 'state': state})
    return rows


# ---------- the entry: rankbound() and `rankbound bound`

@dataclass
class Ask:
    """The caller's request, the same for every curve of a list: the API's keywords or the CLI's options."""
    r: Optional[int] = None
    points: Any = None
    relations: Any = None
    uncond: bool = False
    bad_primes: Optional[list] = None
    ecm_hours: float = 1.0
    cores: Optional[int] = None
    max_hours: Optional[float] = None
    budget: Optional[float] = None
    compute: bool = False
    root: str = DIR
    workdir: Optional[str] = None
    checker: Optional[list] = None
    folder_of: Any = None
    words: Dict = field(default_factory=dict)
    cores_from: Optional[str] = None


@dataclass
class Item:
    """One curve yielded by each(); alone means its failure is the request's."""
    name: Optional[str]
    curve: Optional[list]
    alone: bool = True
    result: Any = None
    estimate: Optional[float] = None
    shard: Optional[int] = None
    stopped: Any = None
    error: Any = None


@contextlib.contextmanager
def factor_note(on):
    """While on, factoring messages say --estimate factors only briefly."""
    from . import curve as cv
    token = cv.FACTOR_NOTE.set(bool(on))
    try:
        yield
    finally:
        cv.FACTOR_NOTE.reset(token)


def read_input(option, path, ids=None):
    """Return the parsed contents of the file given with option, or raise ValueError naming it."""
    from . import curve as cv
    if option == '--curves':
        return cv.from_list(path, ids)
    if option == '--points':
        pts = cv.read_points(path)
        return cv.points_of(pts) if pts else pts
    from .sieve import relation_input
    return relation_input(path)


def check_options(cores=None, max_hours=None, ecm_hours=1.0, relations=None, uncond=False, budget=None, compute=False):
    """Raise ValueError for options out of range, of the wrong type, or exclusive."""
    if cores is not None and (type(cores) is not int or cores < 1):
        raise ValueError('cores must be a positive integer.')
    for name, v in (('max_hours', max_hours), ('ecm_hours', ecm_hours), ('budget', budget)):
        if v is not None and not (isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v < float('inf')):
            raise ValueError('%s must be a number >= 0.' % name)
    if budget is not None and budget <= 0:
        raise ValueError('budget must be a number > 0.')
    for name, v in (('uncond', uncond), ('compute', compute)):
        if type(v) is not bool:
            raise ValueError('%s must be True or False, not %.40r.' % (name, v))
    if relations is not None and uncond:
        raise ValueError(runtime.text('rel_uncond'))


def start(max_hours=None):
    """Return the run's budget from the request's start; its end raises Stopped."""
    return runtime.Budget.for_run(seconds=runtime.run_seconds(max_hours))


def bound(E, *, r=None, points=None, relations=None, uncond=False, bad_primes=None, ecm_hours=1.0, cores=None,
          max_hours=None, budget=None, workdir=None, compute=False, checker=None, folder_of=None, clock=None):
    """Return the Result for the curve E on `rankbound bound`'s path; see rankbound.rankbound."""
    from . import curve as cv
    from .messages import FOLDER_WORD
    check_options(cores, max_hours, ecm_hours, relations, uncond, budget, compute)
    cv.seq(E, 'ainvs')
    if points is not None and not isinstance(points, str):
        points = cv.points_of(points)
    b = start(max_hours) if clock is None else clock
    words = dict(r=r, points=points, relations=relations if relations is None or isinstance(relations, str) else '...',
                 uncond=uncond, bad_primes=bad_primes, compute=compute, cores=cores,
                 max_hours=max_hours, budget=budget, ecm_hours=None if ecm_hours == 1.0 else ecm_hours)
    ask = Ask(r=r, points=points, relations=relations, uncond=uncond, bad_primes=bad_primes, ecm_hours=ecm_hours,
              cores=cores, max_hours=max_hours, budget=budget, compute=compute, workdir=workdir, checker=checker,
              root=os.path.dirname(os.path.abspath(workdir)) if workdir else DIR, folder_of=folder_of, words=words)
    token = FOLDER_WORD.set('workdir')
    try:
        with runtime.interruptible():       # SIGTERM raises as SIGINT does
            it = next(each(ask, curve=E, clock=b))
    finally:
        FOLDER_WORD.reset(token)
    if it.error is not None or it.stopped is not None:
        raise it.error or it.stopped
    return it.result


def each(ask, *, curve=None, curves=None, ids=None, mode='run', shard=None, clock=None):
    """Yield an Item for the curve, or for each curve of a list, preparing each only after the last is yielded.

    A failed curve is yielded and the list goes on; a stop ends the list.
    """
    from . import curve as cv
    runtime.clean_environ()
    b = runtime.current() if mode == 'estimate' else start(ask.max_hours) if clock is None else clock
    if ask.cores_from is None:
        try:
            c = machine.cores(ask.cores)
        except ValueError as e:
            raise core.Usage(str(e))
        ask = dataclasses.replace(ask, cores=c.n, cores_from=c.source)
    else:
        c = None
    have = machine.allowed_cores() if mode != 'estimate' else None
    if have is not None and ask.cores > have:   # clamp to the allocation
        runtime.say('cores_clamped', asked=ask.cores, have=have)
        ask, c = dataclasses.replace(ask, cores=have), None
    with cv.analysis():                     # compute each model's minimal model and torsion once
        if curve is not None and mode == 'run' and not (ask.compute or ask.relations is not None or
                                                         ask.points is not None):     # the table first
            cv.r_value(ask.r)
            known = lookup(model(curve, ask.bad_primes)[1], ask.uncond, ask.r)
            if known is not None:
                yield Item(known.name, list(curve), result=known)
                return
        if c is not None and c.source not in ('option', 'environment'):
            from .messages import CORES_WHY, plural
            runtime.say('cores_auto', cores=plural(c.n, 'core'), why=CORES_WHY[c.source].format(total=c.total))
        preflight(ask.root, ask.checker)    # before factoring or any other work
        Es = (curves if isinstance(curves, list) else cv.from_list(curves, ids)) if curves else [curve]
        for E in Es:
            c, alone = (E if isinstance(E, cv.Curve) else None), len(Es) == 1
            try:
                with runtime.scope(b):
                    c = curve_of(E, ask)
                    if cv.hashed(c.name, c.minimal_model):      # a hashed name: say which curve it is first
                        runtime.say('curve_named', name=c.name, ainvs=cv.short_ainvs(c.ainvs))
                    out = one(c, ask, b, mode, shard)
            except core.Usage:
                raise
            except Stopped as e:
                yield Item(c and c.name, list(c.ainvs if c else E), alone, stopped=e)
                return
            except (core.RankboundError, RuntimeError, ValueError, OSError) as e:
                yield Item(c and c.name, list(c.ainvs if c else E), alone, error=e)
                continue
            yield Item(c.name, list(c.ainvs), alone, **{'result' if mode == 'run' else mode: out})


def curve_of(E, ask):
    """Return the classified Curve of E, its discriminant factored as ask says."""
    from . import curve as cv
    if isinstance(E, cv.Curve):
        c = cv.factored(E, ask.ecm_hours, ask.workdir or os.path.join(ask.root, E.name))
        if c.minimal_model is None:         # the request's identity needs it
            c = dataclasses.replace(c, minimal_model=list(model(c.ainvs, c.bad_primes)[1]))
    else:
        c = cv.from_ainvs(E, bad_primes=ask.bad_primes, r=ask.r, factor_seconds=0 if ask.ecm_hours else 600,
                          composites=bool(ask.ecm_hours))
    c = cv.classified(c)
    if ask.points is not None:
        c = cv.with_points(c, cv.read_points(ask.points) if isinstance(ask.points, str) else ask.points, ask.r)
    return c


def one(c, ask, b, mode='run', shard=None):
    """Return the Curve c's Result, estimate or shard status in budget b."""
    meth = route(c, ask.uncond)
    if ask.relations is not None and meth == DESCENT:
        raise core.Usage(runtime.text('rel_torsion'))
    if mode == 'shard' and meth == DESCENT:
        raise core.Usage(runtime.text('shard_torsion'))
    if ask.budget is not None and meth != SIEVE:      # only the sieve has a budget
        raise core.Usage(runtime.text('budget_no_sieve'))
    J = (ask.folder_of or folder)(meth, ask.workdir or os.path.join(ask.root, c.name))
    if mode in ('estimate', 'shard') and meth == SIEVE:
        from .sieve import relation_input
        given = relation_input(ask.relations) if ask.relations is not None else None
        bind_folder(J, c, given)
    if mode == 'estimate':
        return J.estimate(c, cores=ask.cores, ecm_hours=ask.ecm_hours, given=ask.relations)
    if mode == 'shard':
        return J.sieve_shard(c, cores=ask.cores, shard=shard, budget=b, budget_core_hours=ask.budget)
    return run_folder(J, c, Resources(ask.cores, ask.max_hours, ask.ecm_hours, ask.budget, ask.cores_from),
                      relations=ask.relations, checker=ask.checker, options=ask.words, budget=b)
