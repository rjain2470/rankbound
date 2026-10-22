"""The staged Job over a state folder: prepare, tune, sieve and check until the goal or a plateau, then certify and
verify.  Every stage is checkpointed in the folder, so calling run() again resumes."""
import dataclasses, logging, math, os, time
from . import FactorisationNeeded, RankboundError, Stopped, factor, field as fld, gp_message, linalg, read_json, sieve
from . import write_json
from . import verify as verifier
from .field import Curve
from .rules import earlier
from .runner import given_params, sieve_run, solve_given, solve_once
from .solver import DriedUp, Solver, too_dense
from .state import State

LOG = logging.getLogger('rankbound').info
VERDICT = {True: 'certificate verified', False: 'certificate NOT verified', None: 'certificate not verified'}
STATUS = {True: 'finished, verified', False: 'finished, NOT verified', None: 'finished, NOT verified'}
WORKED_ON = 'was started for another curve (or another r, or other relations). Use another --dir or --name.'


# ---------- the job

def aim_text(F):
    """Return the run's aim as a log line."""
    r, g, u, n, w = F['rank_lower_bound'], fld.goal(F), F['u'], F['n'], F['w']
    best = fld.parity_bound(g, u, n, w)
    if r is None:
        return 'Aim: rank <= %d (the least possible). Stops there or when the bound stops improving.' % best
    if best == r:
        return 'Aim: rank <= %d = r, which proves rank = %d.' % (best, r)
    if w != (-1) ** r and best == r + 1:
        return 'Aim: rank <= %d = r + 1 (the root number rules out r).' % best
    return 'Aim: rank <= %d; r = %d cannot be reached here (u + n = %d).' % (best, r, u + n)


class Job:
    """The staged job over a state folder; rankbound() is Job(workdir).run(curve, ...)."""

    def __init__(self, workdir):
        self.dir = os.path.abspath(workdir)     # absolute: the sievers run in other folders
        self.solver = None                     # the Solver after a sieve or solve in this process
        self.outcome = None
        self.prm = None                        # params.json, once tune() ran in this process

    def _path(self, name):
        return os.path.join(self.dir, name)

    def _cost(self, **kw):
        p = self._path('cost.json')
        c = read_json(p, {})
        c.update(kw)
        write_json(p, c)
        return c

    def curve(self):
        return Curve.from_dict(read_json(self._path('curve.json')))

    def check_curve(self, curve):
        """Raise RankboundError if this folder holds another curve (another minimal model) or another r."""
        p = self._path('field.json')
        if curve.r is not None and os.path.exists(p) and read_json(p).get('rank_lower_bound') != curve.r:
            raise RankboundError('%s %s' % (self.dir, WORKED_ON))
        old = None
        for f in ('field.json', 'curve.json'):
            if not old and os.path.exists(self._path(f)):
                old = read_json(self._path(f)).get('minimal_model')
        if not old:
            return
        new = curve.minimal_model
        if new is None:
            p = self._path('curve.json')
            if os.path.exists(p) and [str(x) for x in read_json(p)['ainvs']] == [str(x) for x in curve.ainvs]:
                return
            new = fld.from_ainvs(curve.ainvs, bad_primes=curve.bad_primes).minimal_model
        if [str(x) for x in new] != [str(x) for x in old]:
            raise RankboundError('%s %s' % (self.dir, WORKED_ON))

    def prepare(self, curve, ecm_hours=1.0, cores=1, mem_gb=1.0, until=None):
        """Write the field data and curve.json, after checking the proof of r, if any.  Composites left in the
        discriminant are factored first (factor.split: for at most ecm_hours on cores processes, stopping at until)."""
        fld.checked(curve)                  # also a Curve made by hand
        self.check_curve(curve)
        new = not os.path.exists(self._path('field.json'))
        if curve.composites and not new:    # factored earlier: the field has every prime
            curve = dataclasses.replace(curve, bad_primes=fld.load(self.dir)['bad_primes'], composites=[])
        if curve.composites:
            if not ecm_hours:
                raise FactorisationNeeded('The discriminant is not fully factored. Give all its prime factors in '
                                          '"bad_primes".', [{'n': c, 'digits': len(c)} for c in curve.composites])
            D = abs(fld.disc(fld._ainvs5(curve.ainvs)))
            for c in curve.composites:
                if int(c) < 2:
                    raise ValueError('The composite %s is not an integer above 1.' % c)
                if D % int(c):
                    raise ValueError('The composite %s does not divide the discriminant.' % c)
            curve = self.factor(curve, ecm_hours, cores, mem_gb, until)
        t = time.time()
        F = fld.prepare(self.dir, curve.ainvs, curve.bad_primes, r=curve.r, name=curve.name,
                        minimal_model=curve.minimal_model)
        if new:
            self._cost(field=round(time.time() - t, 1))
            LOG('Conductor: %d digits, %d bad primes; root number %+d.' % (len(fld.conductor(F)), len(F['bad_primes']),
                                                                         F['w']))
        self.check_proof(curve)             # before curve.json is written
        write_json(self._path('curve.json'), curve.to_dict())
        LOG(aim_text(F))
        return F

    def factor(self, curve, hours, cores=1, mem_gb=1.0, until=None):
        """Return curve with the primes of its composites added to its bad primes (factor.split, effort kept in
        factor.json)."""
        os.makedirs(self.dir, exist_ok=True)
        t = time.time()
        try:
            P = factor.split(self._path('factor.json'), curve.composites, hours=hours, cores=cores, mem_gb=mem_gb,
                             until=until, log=LOG)
        finally:
            self._cost(factor=round(read_json(self._path('factor.json'), {}).get('wall_s', time.time() - t), 1))
        bad = sorted({int(p) for p in curve.bad_primes} | set(P))
        return dataclasses.replace(curve, bad_primes=[str(p) for p in bad], composites=[])

    def check_given(self, given):
        """Raise RankboundError if this folder holds other given relations, or is finished without them."""
        if not os.path.isdir(self.dir):
            return
        old = State(self.dir).given()
        if old is not None and old != given.digest:
            raise RankboundError('%s %s' % (self.dir, WORKED_ON))
        if old is None and os.path.exists(self._path('result.json')):
            raise RankboundError('%s %s' % (self.dir, WORKED_ON))

    def check_sieve(self):
        """Raise RankboundError if this folder ran on given relations without a sieve."""
        if not os.path.isdir(self.dir):
            return
        p, S = self._path('params.json'), State(self.dir)
        if (os.path.exists(p) and read_json(p).get('siever') == 'given') or (S.stopped() and S.stop_outcome() == 'relations'):
            raise RankboundError('%s %s' % (self.dir, WORKED_ON))

    def check_proof(self, curve):
        """Raise RankboundError, before any sieving, if the points do not prove the rank lower bound on the
        certificate's model."""
        F = fld.load(self.dir)
        if not curve.proof or F.get('rank_lower_bound') is None:      # no r: no rank_lower_bound
            return
        dK = int(F['dK'])
        if F['rank_lower_bound'] and dK > 0 and math.isqrt(dK) ** 2 == dK:        # r = 0 needs no prime
            # K is cyclic: the 2-division cubic has 0 or 3 roots mod every good p, so no #E(F_p) = 2 mod 4.
            raise RankboundError('Points cannot prove the rank of this curve (its cubic field is cyclic). Give r with '
                                 '--r instead; it will be cited, not proved.')
        try:
            linalg.rank_proof(F['minimal_model'], F['bad_primes'], F['rank_lower_bound'],
                              dict(curve.proof, ainvs=curve.proof.get('ainvs') or curve.ainvs))
        except RuntimeError as e:           # also Rejected, from field.gp's rkmap
            raise RankboundError(gp_message(e))

    def give(self, given):
        """Write the given relations as the finished slice slices/given.rels.gz, factoring F(a, b) where needed."""
        S = State(self.dir)
        if S.given() == given.digest:
            return False
        fd = S.lock('run.lock')
        try:
            if S.given() == given.digest:
                return False
            lines, nf = linalg.given_lines(fld.load(self.dir), given.rels)
            S.write_given(lines, given.digest)
        finally:
            os.close(fd)
        return True

    def tune(self, mem_gb=1.0, cores=1, lpb_min=None):
        """Run the trial sieve, write params.json and log the cost estimate."""
        F = fld.load(self.dir)
        prm = sieve.tune(self.dir, F, mem_gb=mem_gb, log=LOG, lpb_min=lpb_min)
        for l in sieve.estimate_lines(F, prm, cores):
            LOG(l)
        self.prm = prm
        return prm

    def retune(self, err, mem_gb=1.0, cores=1):
        """Re-tune at a larger lpb after the sieve dried up; the finished slices stay valid."""
        path = self._path('params.json')
        prm = read_json(path)
        lpb = prm['lpb0'] + 2
        if prm.get('siever') == 'given' or lpb > sieve.LPB_MAX:
            raise err
        LOG('Too few relations at these settings; re-tuning and continuing (the relations so far are kept).')
        os.replace(path, self._path('params_lpb%d.json' % prm['lpb0']))
        self.prm = None
        new = self.tune(mem_gb, cores, lpb_min=lpb)
        if new['lpb0'] <= prm['lpb0']:           # never loop
            raise err
        return new

    def sieve(self, cores=1, shard=(0, 1), max_hours=None, solver=None, until=None):
        """Sieve and check with cores processes, as shard (k, n), until the earlier of until and max_hours from now.
        Returns 'goal', 'plateau' or None."""
        self.outcome, self.solver = sieve_run(self.dir, jobs=cores, shard=shard, solver=solver, max_hours=max_hours,
                                              log=LOG, until=until)
        return self.outcome

    def solve(self, threads=1):
        """Run one check on every finished slice; returns 'goal', 'plateau' or None."""
        self.outcome, self.solver = solve_once(self.dir, log=LOG, threads=threads)
        return self.outcome

    def certify(self, verify=True, jobs=1, outcome=None):
        """Write the certificate and, with verify, check it (verify.log, verify.json); write and return result.json.
        Raises RuntimeError if the verification cannot run or is over a limit."""
        S = State(self.dir)
        fd = S.lock('run.lock')       # one certifier at a time, and no check
        try:
            return self._certify(S, verify, jobs, outcome)
        finally:
            os.close(fd)

    def _certify(self, S, verify, jobs, outcome):
        outcome = outcome or self.outcome or read_json(self._path('STOP'))['outcome']
        F = fld.load(self.dir)
        cur = read_json(self._path('curve.json'))
        prm = read_json(self._path('params.json'))
        V = self.solver
        # Resumed after STOP: solve again on the slices of the last full solve, in their order, so that its
        # elimination is reused.
        if V is None or V.last is None:
            V = Solver.of(S, F, prm, LOG, jobs)
            V.load(S.load_order(self._path('la')))
            V.last = linalg.solve(V.M, self._path('la'), V.target, keep=V.keep, threads=jobs)
            if V.last.bound is None:
                raise RuntimeError(too_dense(V.last.dense_mb))
        cpu, nrel = S.slice_stats()
        path = self._path(F['name'] + '.cert.json')
        t = time.time()
        proof = cur.get('proof')
        # The points are on the model the curve was given on, which can differ from the minimal model.
        if proof and not proof.get('ainvs'):
            proof = dict(proof, ainvs=cur['ainvs'])
        c = linalg.certificate(self.dir, V.M, V.last, path, proof, cur.get('source', 'cited'))
        tc = time.time() - t
        LOG('Certificate: %s' % path)
        good, st, tv = None, c['statement'], None
        if verify:
            t = time.time()
            with open(self._path('verify.log'), 'w') as vl:
                v = verifier.check(path, cores=jobs, log=lambda *a: print(time.strftime('%H:%M:%S'), *a, file=vl,
                                                                         flush=True))
            write_json(self._path('verify.json'), v, None)
            if v['verdict'] in ('limit', 'error'):
                raise RuntimeError(v['reason'])
            good, st = v['verdict'] == 'accepted', v['statement'] or st
            tv = round(time.time() - t, 1)
        cost = self._cost(trial=prm['trial']['cpu_s'], sieve=round(cpu, 1),
                          solve=round(sum(h.get('solve_s', 0) for h in S.history()), 1),
                          certify=round(tc, 1), verify=tv)
        r = F['rank_lower_bound']
        result = {'name': F['name'], 'outcome': outcome, 'verified': good, 'verify_s': tv,
                  'G': c['claims']['G'], 'rank_bound': c['claims']['S'], 'certificate': path,
                  'rank_lower_bound': r, 'g_target': F['g_target'], 'statement': st,
                  'r_proved': r is not None and (r == 0 or 'points' in c.get('rank_lower_bound', {})),
                  'relations': nrel, 'sieve_core_hours': round(cpu / 3600, 3), 'estimate_core_hours': prm['model']['core_hours'],
                  'cost': cost}
        write_json(self._path('result.json'), result)
        return result

    def run(self, curve=None, cores=1, mem_gb=1.0, max_hours=None, verify=True, relations=None, sieve=True, until=None,
            ecm_hours=1.0):
        """Run every stage, resuming, and return result.json's dict; curve is needed on the first call.  With
        relations and sieve=False, certify the bound those relations prove, without sieving.  The factor step and the
        sieve stop (Stopped) at the earlier of until and max_hours from now; the factor step gives up
        (FactorisationNeeded) after ecm_hours."""
        until = earlier(until, max_hours)
        if curve is not None:
            self.check_curve(curve)
        given = None if relations is None else linalg.relation_input(relations)
        if given is not None:
            self.check_given(given)
        elif not sieve and not (os.path.isdir(self.dir) and State(self.dir).given()):
            raise ValueError('sieve=False needs relations')
        if sieve:
            self.check_sieve()
        rp = self._path('result.json')
        if os.path.exists(rp):
            r = read_json(rp)
            LOG('%s: done earlier: %s' % (r['name'], r['statement']))
            return r
        if self.prm is None:
            self.prepare(curve or self.curve(), ecm_hours=ecm_hours, cores=cores, mem_gb=mem_gb, until=until)
        new = given is not None and self.give(given)
        if not sieve:
            given_params(self.dir)
            if new or not State(self.dir).stopped():
                self.outcome, self.solver = solve_given(self.dir, log=LOG, threads=cores)
            return self.certify(verify=verify, jobs=cores)
        if self.prm is None:
            self.tune(mem_gb, cores)
        while True:
            try:
                outcome = self.sieve(cores, until=until)
                break
            except DriedUp as e:                # each re-tune raises lpb by >= 2, up to LPB_MAX
                self.retune(e, mem_gb, cores)
        if not outcome:
            if not State(self.dir).stopped():
                raise Stopped(self.dir)
            # Another process finished the run (STOP) while this one sieved.
            if os.path.exists(rp):
                LOG('Another process has already finished this curve.')
                return read_json(rp)
            LOG('Another process finished sieving; writing the certificate.')
        return self.certify(verify=verify, jobs=cores)

    def status(self):
        """Return the row of `rankbound status` for this folder."""
        F = fld.load(self.dir)
        prm = read_json(self._path('params.json'), None)
        S = State(self.dir)
        cpu, n = S.slice_stats()
        res = read_json(self._path('result.json'), {})
        G = next((e['bound'] for e in reversed(S.history()) if e.get('bound') is not None), None)
        bound = res['rank_bound'] if res else '' if G is None else fld.parity_bound(G, F['u'], F['n'], F['w'])
        return {'name': F['name'], 'relations': n, 'core-hours': round(cpu / 3600, 2),
                'estimated': prm['model']['core_hours'] if prm else '', 'rank bound': bound,
                'status': STATUS[res['verified']] if res else 'sieving' if n or S.history() else 'not started'}


STATUS_COLUMNS = ('name', 'relations', 'core-hours', 'estimated', 'rank bound', 'status')
