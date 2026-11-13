"""Curves with a rational point of order 2: the unconditional bound by 2-isogeny descent (verify/two_descent.gp has
the mathematics), its certificate (format verify.two_descent.FORMAT) and its work folder

    DIR/curve.json                  the Curve
    DIR/NAME.cert.json              the certificate
    DIR/verify.log, verify.json     the verifier's lines and verdict
    DIR/result.json                 the result

The steps follow the mathematics: descent() computes, for each rational root e of the 2-division cubic, the local
images and the Selmer groups of the 2-isogeny with kernel (e, 0) and its dual, the isogeny pairings Theta_1 and
Theta_1' on them (with their witnesses), the bound B_e = s + s' - 2 - rank Theta_1 - rank Theta_1', and the lower
bound from points; certificate() writes one entry per isogeny."""
import dataclasses, datetime, json, logging, os, re, time
from . import FactorisationNeeded, RankboundError, gp_message, read_json, write_json
from . import field as fld
from . import verify as verifier
from .verify import Rejected, parity_bound, run_gp
from .verify.two_descent import FORMAT, statement

LOG = logging.getLogger('rankbound').info
FIELD_GP = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'field.gp')      # rkmap
PRIME_SEARCH = 10 ** 5      # primes searched for the characters of the points
METHOD = '2-isogeny descent'
WORKED_ON = 'was started for another curve (or another r). Use another --dir or --name.'


def gp_vec(v):
    return '[%s]' % ','.join(str(x) for x in v)


NUMBER = re.compile(r'-?[0-9]+(?:/[0-9]+)?')


def gp_list(text):
    """Return a GP vector as printed (integers, fractions, nested [...]) as nested lists of number strings."""
    return json.loads(NUMBER.sub(lambda m: '"%s"' % m.group(), text))


def ct_block(rank, text):
    """Return the certificate's Cassels-Tate entry from tdct's rank and [quartics, local points, pairs]."""
    Q, loc, pairs = gp_list(text)
    return {'quartics': Q, 'local': loc,
            'pairs': [{'i': int(p[0]), 'j': int(p[1]), 'sum': p[2], 'm': p[3], 'sum_local': p[4], 'primes': p[5],
                       'points': p[6]} for p in pairs], 'rank': int(rank)}


def descent(ainvs, bad_primes, proof=None, r=None, gp='gp'):
    """Return the 2-descent of the curve ainvs (any model, with all the primes of its discriminant) as a dict:
    minimal_model, torsion, w, conductor, isogenies [{e, a, b, selmer, selmer_isogenous, s, s_isogenous, pairing,
    cassels_tate, B}], cassels_tate (of E, or None), B, S, and, with the proof's points (on proof['ainvs'], else on
    ainvs), the rank lower bound {r, points, primes} on the minimal model.  The Cassels-Tate pairings are computed
    only while S exceeds the known lower bound (the number of points, else r, else 0).  Raises RankboundError if the points do not prove rank >= r, RuntimeError if an internal check
    fails (a bug: the local images of the two isogenies are not dual, or b, b' are not in the Selmer groups)."""
    a = [fld.canon(x, 'a-invariant') for x in fld._ainvs5(ainvs)]
    P = [fld.canon(p, 'bad prime') for p in bad_primes]
    pts = (proof or {}).get('points') or []
    src = fld._ainvs5(proof.get('ainvs') or ainvs) if pts else a
    body = '{ my(P = %s, mm, Q); addprimes(P); mm = ellminimalmodel(ellinit(%s))[1..5];\n' % (gp_vec(P), gp_vec(a))
    body += '  Q = rkmap(%s, mm, P, [%s]);\n' % (gp_vec(src), ','.join('[%s,%s]' % tuple(
        fld.canon(x, 'proof point', fld.RAT) for x in p) for p in pts)) if pts else '  Q = [];\n'
    body += '  tdprove(mm, P, Q, %s, %d, %d); }' % (gp_vec((proof or {}).get('primes') or []), PRIME_SEARCH,
                                                    len(pts) if pts else r or 0)
    try:
        out = run_gp(body, libs=('verify.gp', FIELD_GP, 'two_descent.gp'), seconds=7200, gp=gp)
    except Rejected as e:           # the bad primes (as field.gp's sfield), or rkmap and tdprimes: the points
        msg = gp_message(e).splitlines()[0]
        if msg.startswith('The bad prime'):
            raise ValueError(msg)
        raise RankboundError(msg)
    D = {'isogenies': [], 'points': [], 'cassels_tate': None}
    V = places(P)
    for l in out.splitlines():
        t = l.split('\t')
        if t[0] == 'MM':
            D['minimal_model'] = t[1].split(',')
        elif t[0] == 'TOR':
            D['torsion'] = [int(x) for x in t[1].split(',') if x]
        elif t[0] == 'W':
            D['w'] = int(t[1])
        elif t[0] == 'CN':
            D['conductor'] = t[1].strip()
        elif t[0] == 'ISOG':
            e, a_, b_, sel, sel2, s, s2, B, dual, tors = t[1:]
            if dual != '1' or tors != '1':
                raise RuntimeError('two_descent.gp: at the root %s the internal checks fail (local duality %s, '
                                   'b and b\' in the Selmer groups %s). This is a bug: please report it.' % (e, dual, tors))
            D['isogenies'].append({'e': e, 'a': a_, 'b': b_, 'selmer': [x for x in sel.split(',') if x],
                                   'selmer_isogenous': [x for x in sel2.split(',') if x], 's': int(s),
                                   's_isogenous': int(s2),
                                   'pairing': {'places': V, 'conic': [], 'local': [], 'rank': None,
                                               'conic_isogenous': [], 'local_isogenous': [], 'rank_isogenous': None},
                                   'cassels_tate': None, 'B': None})
        elif t[0] == 'PC':          # a conic point and the local points of a Selmer basis element
            pr = D['isogenies'][int(t[1]) - 1]['pairing']
            g = '' if t[2] == '1' else '_isogenous'
            pr['conic' + g].append(t[3].split(','))
            pr['local' + g].append([z.split(':') for z in t[4].strip().split(',')])
        elif t[0] == 'TH':
            i = D['isogenies'][int(t[1]) - 1]
            i['pairing']['rank'], i['pairing']['rank_isogenous'], i['B'] = int(t[2]), int(t[3]), int(t[4])
        elif t[0] == 'CTI':         # the Cassels-Tate pairing on Sel_2 of the isogenous curve
            D['isogenies'][int(t[1]) - 1]['cassels_tate'] = ct_block(t[2], t[3])
        elif t[0] == 'CTE':
            D['cassels_tate'] = ct_block(t[1], t[2])
        elif t[0] == 'CTSKIP':
            LOG('The Cassels-Tate pairing on %s was not computed: %s' % ('E' if t[1] == 'E' else "E' (isogeny %s)" % t[1],
                                                                       t[2].strip()))
        elif t[0] == 'BS':
            D['B'], D['S'] = int(t[1]), int(t[2])
        elif t[0] == 'PT':
            D['points'].append([fld.canon(t[1].strip(), 'mapped point', fld.RAT), fld.canon(t[2].strip(), 'mapped point', fld.RAT)])
        elif t[0] == 'PR':
            D['primes'] = [int(x) for x in t[1].split(',') if x]
    if not D['isogenies'] or D['B'] != min(i['B'] for i in D['isogenies']) or D['S'] != parity_bound(D['B'], 0, 0, D['w']):
        raise RuntimeError('two_descent.gp: the bound does not add up. This is a bug: please report it.')
    if proof is not None:
        D['rank_lower_bound'] = {'r': len(pts), 'points': D.pop('points'), 'primes': D.pop('primes', [])}
    return D


def places(bad_primes):
    """Return the places at which the pairings are evaluated, as the certificate lists them: R, then the primes of
    2 * the bad primes and 3, 5, 7 in increasing order."""
    return ['R'] + [str(p) for p in sorted({2, 3, 5, 7} | {int(p) for p in bad_primes})]


def certificate(path, name, bad_primes, D, rl):
    """Write the certificate path for the descent D (from descent()) with the rank lower bound rl ({r, points,
    primes}, {r, source} or None), and return it as a dict.  Raises RankboundError, writing nothing, if S < r."""
    r = rl['r'] if rl else None
    S, st = statement(D['B'], D['w'], r, None if rl is None or 'points' in rl else rl['source'])
    if r is not None and S < r:
        raise RankboundError('The 2-descent proves rank <= %d, but r = %d was given. No certificate written. %s' % (
            S, r, 'This contradicts the points: please report it.' if 'points' in rl else 'Check r, or run without it.'))
    mm = D['minimal_model']
    c = {'name': name, 'curve': {'ainvs': mm, 'minimal_model': mm, 'bad_primes': [str(p) for p in bad_primes]},
         'statement': st, 'torsion': D['torsion'], 'GRH': 'no'}
    if rl:
        c['rank_lower_bound'] = rl
    c.update(descent=D['isogenies'], cassels_tate=D['cassels_tate'], local={'w': D['w']}, claims={'B': D['B'], 'S': S},
             created=datetime.date.today().isoformat(), format=FORMAT)
    write_json(path, c)
    return c


class Descent:
    """The work folder of a curve with 2-torsion; rankbound() is Descent(workdir).run(curve, ...)."""

    def __init__(self, workdir):
        self.dir = os.path.abspath(workdir)
        self.solver = None              # as Job's, for the command line

    def _path(self, name):
        return os.path.join(self.dir, name)

    def check_curve(self, curve):
        """Raise RankboundError if this folder holds another curve or another r."""
        p = self._path('curve.json')
        if not os.path.exists(p):
            return
        old = fld.Curve.from_dict(read_json(p))
        if [str(x) for x in old.ainvs] != [str(x) for x in curve.ainvs] or old.r != curve.r:
            raise RankboundError('%s %s' % (self.dir, WORKED_ON))

    def run(self, curve, cores=1, verify=True, ecm_hours=1.0, mem_gb=1.0, until=None, **_):
        """Bound the rank of curve, write the certificate and, with verify, check it; write and return result.json's
        dict (read back if the folder is finished).  Composites left in the discriminant are factored first, as by
        Job.  Other options of Job.run are ignored: there is nothing to sieve."""
        fld.checked(curve)
        self.check_curve(curve)
        rp = self._path('result.json')
        if os.path.exists(rp):
            r = read_json(rp)
            LOG('%s: done earlier: %s' % (r['name'], r['statement']))
            return r
        os.makedirs(self.dir, exist_ok=True)
        if curve.composites:
            if not ecm_hours:
                raise FactorisationNeeded('The discriminant is not fully factored. Give all its prime factors in '
                                              '"bad_primes".', [{'n': c, 'digits': len(c)} for c in curve.composites])
            from .run import Job
            curve = Job(self.dir).factor(curve, ecm_hours, cores, mem_gb, until)
        t = time.time()
        proof = dict(curve.proof, ainvs=curve.proof.get('ainvs') or curve.ainvs) if curve.proof else None
        D = descent(curve.ainvs, curve.bad_primes, proof, curve.r)
        k = len(curve.bad_primes)
        LOG('Conductor: %d digits, %d bad prime%s; root number %+d.' % (len(D['conductor']), k, 's' * (k != 1), D['w']))
        LOG('E(Q)_tors = %s: E(Q)[2] != 0, so the rank is bounded by 2-isogeny descent, unconditionally.' % (
            ' x '.join('Z/%d' % n for n in D['torsion'])))
        if curve.minimal_model is not None and [str(x) for x in curve.minimal_model] != D['minimal_model']:
            raise ValueError('The minimal_model given for %s is wrong; it should be [%s]. Correct it or leave it out.' % (
                curve.name, ','.join(D['minimal_model'])))
        rl = D.get('rank_lower_bound') or ({'r': curve.r, 'source': curve.source} if curve.r is not None else None)
        write_json(self._path('curve.json'), dataclasses.replace(curve, torsion=D['torsion']).to_dict())
        path = self._path(curve.name + '.cert.json')
        c = certificate(path, curve.name, curve.bad_primes, D, rl)
        tc = time.time() - t
        LOG('Certificate: %s' % path)
        good, st, tv = None, c['statement'], None
        if verify:
            t = time.time()
            with open(self._path('verify.log'), 'w') as vl:
                v = verifier.check(path, cores=cores, log=lambda *a: print(time.strftime('%H:%M:%S'), *a, file=vl,
                                                                          flush=True))
            write_json(self._path('verify.json'), v, None)
            if v['verdict'] in ('limit', 'error'):
                raise RuntimeError(v['reason'])
            good, st = v['verdict'] == 'accepted', v['statement'] or st
            tv = round(time.time() - t, 1)
        r = rl['r'] if rl else None
        res = {'name': curve.name, 'outcome': 'descent', 'method': METHOD, 'GRH': 'no', 'verified': good,
               'verify_s': tv, 'B': D['B'], 'rank_bound': c['claims']['S'], 'certificate': path,
               'rank_lower_bound': r, 'r_proved': r is not None and (r == 0 or 'points' in rl), 'statement': st,
               'torsion': D['torsion'], 'w': D['w'], 'minimal_model': D['minimal_model'], 'conductor': D['conductor'],
               'isogenies': [{'e': i['e'], 's': i['s'], 's_isogenous': i['s_isogenous'],
                              'rank_theta': i['pairing']['rank'], 'rank_theta_isogenous': i['pairing']['rank_isogenous'],
                              'rank_ct_isogenous': (i['cassels_tate'] or {}).get('rank'), 'B': i['B']}
                             for i in D['isogenies']],
               'rank_ct': (D['cassels_tate'] or {}).get('rank'),
               'sieve_core_hours': 0.0, 'cost': {'descent': round(tc, 1), 'verify': tv}}
        write_json(rp, res)
        return res

    def status(self):
        """Return the row of `rankbound status` for this folder."""
        cur = read_json(self._path('curve.json'))
        res = read_json(self._path('result.json'), {})
        from .run import STATUS
        return {'name': cur['name'], 'relations': '', 'core-hours': 0.0, 'estimated': 0.0,
                'rank bound': res.get('rank_bound', ''),
                'status': (STATUS[res['verified']] + ' (2-descent)') if res else 'not started'}


def result(res, workdir):
    """Return the Result of the result.json dict res of a 2-descent."""
    from . import Result
    S, r = res['rank_bound'], res['rank_lower_bound']
    return Result(name=res['name'], minimal_model=res['minimal_model'], rank_bound=S, G=None, u=None, n=None,
                  w=res['w'], selmer_bound=None, rank_lower_bound=r, r_proved=res['r_proved'],
                  proved_equal=S == 0 or (res['r_proved'] and S == r), outcome=res['outcome'],
                  statement=res['statement'], certificate=res['certificate'], verified=res['verified'],
                  cost=res.get('cost', {}), workdir=workdir, conductor=res['conductor'], torsion=res['torsion'],
                  grh=False, method=METHOD)
