"""rankbound: an upper bound on rank E(Q), with a certificate: under GRH if E(Q)[2] = 0, unconditionally if not.

    from rankbound import rankbound
    res = rankbound([0, -1, 1, -929, -10595])      # 571a1: res.rank_bound == 2, res.verified
    rankbound([0, -1, 1, -929, -10595], relations=res.certificate)     # the same bound from those relations alone
    rankbound([0, 0, 1, -1, 0], points=[(0, 0)])   # 37a1: rank >= 1 proved (r_proved), rank = 1 (proved_equal)
    rankbound([0, 0, 0, -25, 0], points=[(-4, 6)])  # E(Q)[2] != 0: rank = 1, unconditionally (res.grh False)

The methods and their hypotheses are described in the README.  The subpackage rankbound.verify checks a certificate
and imports nothing from the rest of rankbound; the prover's modules load only when used."""
import json, os
from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional

__version__ = '2.0.0'


# The errors come first: the submodules import them.
class RankboundError(Exception):
    """Base class of rankbound's errors."""


class NotApplicable(RankboundError):
    """The curve has a rational point of order 2, and the class group method was asked for (Job)."""


class FactorisationNeeded(RankboundError):
    """The discriminant could not be factored; composites lists what is left, with the effort spent on each."""

    def __init__(self, msg, composites=()):
        super().__init__(msg)
        self.composites = list(composites)


class Locked(RankboundError):
    """Another process is using this folder."""


class Stopped(RankboundError):
    """Stopped at the time limit."""

    def __init__(self, workdir):
        super().__init__("Stopped at the time limit. Call rankbound() again with workdir='%s' to resume." % workdir)
        self.workdir = workdir


def write_json(path, obj, indent=1):
    """Write obj to path atomically, so that no other process reads it half-written."""
    tmp = '%s.tmp%d' % (path, os.getpid())
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def read_json(path, *default):
    """Return the JSON file at path, or the default, if one is given and the file does not exist."""
    if default and not os.path.exists(path):
        return default[0]
    with open(path) as f:
        return json.load(f)


def gp_message(e):
    """Return the text of the error e; for a GP error(), its message without gp's trace."""
    return str(e).split('user error: ')[-1].strip()


def __getattr__(name):
    """Import rankbound.Job and rankbound.Curve on first use."""
    if name == 'Job':
        from .run import Job
        return Job
    if name == 'Curve':
        from .field import Curve
        return Curve
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


@dataclass(frozen=True)
class Result:
    name: str
    minimal_model: List[str]
    rank_bound: int                 # S: rank E(Q) <= S (under GRH if grh)
    G: Optional[int]                # dim Cl(K)[2] <= G; None (to w) if 'known'
    u: Optional[int]
    n: Optional[int]
    w: Optional[int]
    selmer_bound: Optional[int]     # dim Sel_2(E) <= G + u + n; None for a 2-descent (B + t bounds r + t, not Sel_2)
    rank_lower_bound: Optional[int] # r, if given
    r_proved: bool                  # rank E(Q) >= r is proved by the certificate
    proved_equal: bool              # rank E(Q) = S is proved (under GRH if grh)
    outcome: str                    # 'goal', 'plateau', 'relations', 'known' or 'descent'
    statement: str
    certificate: str                # the certificate file
    verified: Optional[bool]        # None if not verified here
    cost: Dict[str, Optional[float]] = _field(default_factory=dict)   # seconds per stage
    workdir: str = ''
    conductor: Optional[str] = None # N, if computed here
    torsion: Optional[List[int]] = None     # E(Q)_tors as from elltors, if known
    grh: bool = True                # whether rank_bound assumes GRH
    method: str = 'class group'     # or '2-isogeny descent' (E(Q)[2] != 0)


def rankbound(E, *, relations=None, r=None, points=None, bad_primes=None, sieve=None, workdir=None, cores=1,
              mem_gb=1.0, max_hours=None, verify=True, name=None, compute=False, ecm_hours=1.0):
    """Return a certified bound rank E(Q) <= S, as a Result: under GRH by the class group method if E(Q)[2] = 0,
    unconditionally by 2-isogeny descent if not (then relations are refused and the sieve options unused).

    E is [a1, a2, a3, a4, a6] or [a4, a6], on any model.  The keyword arguments are the options of `rankbound bound`
    (README, Usage); calling again with the same workdir resumes.  Without bad_primes, what trial division leaves
    of the discriminant is factored by ECM for at most ecm_hours (0: PARI's factor() for 10 minutes per cofactor,
    as in 1.0); FactorisationNeeded then names what is left, and calling again continues."""
    if type(cores) is not int or cores < 1:
        raise ValueError('cores must be a positive integer.')
    if not (isinstance(mem_gb, (int, float)) and 0 < mem_gb < float('inf')):
        raise ValueError('mem_gb must be a positive number.')
    if max_hours is not None and not (isinstance(max_hours, (int, float)) and not isinstance(max_hours, bool)
                                      and 0 <= max_hours < float('inf')):
        raise ValueError('max_hours must be None or a number >= 0.')     # a NaN limit would never expire
    if not (isinstance(ecm_hours, (int, float)) and not isinstance(ecm_hours, bool) and 0 <= ecm_hours < float('inf')):
        raise ValueError('ecm_hours must be a number >= 0.')
    if workdir is not None and not isinstance(workdir, (str, os.PathLike)):
        raise ValueError('workdir must be a folder name.')
    from . import field, linalg, run
    if sieve is None:
        sieve = relations is None
    if not sieve and relations is None:
        raise ValueError('sieve=False needs relations=....')
    if isinstance(E, field.Curve):
        E = field.checked(E)
    # A curve in certificates.tsv is looked up by its minimal model, recomputed even for a Curve.
    if not compute and relations is None and points is None:
        if not isinstance(E, field.Curve):
            field.r_value(r)
            if name is not None:
                field.check_name(name)
        a = (E.ainvs, E.bad_primes) if isinstance(E, field.Curve) else (E, bad_primes)
        res = known_result(field.minimal_model(*a), ','.join(map(str, a[0])))
        if res is not None:
            return res
    c = field.classified(E if isinstance(E, field.Curve) else field.from_ainvs(
        E, bad_primes=bad_primes, r=r, name=name, factor_seconds=0 if ecm_hours else 600, composites=bool(ecm_hours)))
    if points is not None:
        c = field.with_points(c, points, r)
    key = c.name + ('' if c.r is None else '_r%d' % c.r + ('p' if c.proof else ''))   # a state keeps its r and proof
    if c.two_torsion():
        if relations is not None:
            raise ValueError('This curve has a rational point of order 2: its bound comes from a 2-isogeny descent, '
                             'not from relations.')
        from . import two_descent
        from .rules import earlier
        D = two_descent.Descent(workdir or os.path.join('rankbound_work', key))
        return two_descent.result(D.run(c, cores=cores, mem_gb=mem_gb, verify=verify, ecm_hours=ecm_hours,
                                        until=earlier(None, max_hours)), D.dir)
    given = None if relations is None else linalg.relation_input(relations)
    J = run.Job(workdir or os.path.join('rankbound_work', key + ('_rel' + given.digest[:8] + ('_sieve' if sieve else '')
                                                             if given else '')))
    res = J.run(c, cores=cores, mem_gb=mem_gb, max_hours=max_hours, verify=verify, relations=given, sieve=sieve,
                ecm_hours=ecm_hours)
    F = field.load(J.dir)
    r, S = res['rank_lower_bound'], res['rank_bound']
    r_proved = res.get('r_proved')
    if r_proved is None:            # result.json without r_proved: from curve.json
        r_proved = r is not None and (r == 0 or bool(read_json(os.path.join(J.dir, 'curve.json')).get('proof')))
    return Result(name=res['name'], minimal_model=F['minimal_model'], rank_bound=S, G=res['G'],
                  u=F['u'], n=F['n'], w=F['w'], selmer_bound=res['G'] + F['u'] + F['n'],
                  rank_lower_bound=r, r_proved=r_proved, proved_equal=S == 0 or (r_proved and S == r),
                  outcome=res['outcome'], statement=res['statement'], certificate=res['certificate'],
                  verified=res['verified'], cost=res.get('cost', {}), workdir=J.dir, conductor=field.conductor(F),
                  torsion=F.get('torsion'))


def known_result(mm, curve=''):
    """Return the Result (outcome 'known') of the certificates.tsv row for the minimal model mm, or None."""
    import logging
    from . import known
    row = known.lookup(mm)
    if row is None:
        return None
    for l in known.lines(row, curve):
        logging.getLogger('rankbound').info(l)
    S, r, eq = int(row[known.UB]), row[known.LB], row[known.EQ]
    r_proved = eq in ('yes', 'no')
    tor = row.get(known.TOR)
    return Result(name=row['name'], minimal_model=list(mm), rank_bound=S, G=None, u=None, n=None, w=None,
                  selmer_bound=None, rank_lower_bound=int(r) if r else None, r_proved=r_proved,
                  proved_equal=S == 0 or (r_proved and r != '' and S == int(r)), outcome='known',
                  statement=known.statement(row), certificate=row['file'], verified=None, cost={}, workdir='',
                  torsion=[int(x) for x in tor.strip('[]').split(',') if x.strip()] if tor else None,
                  grh=row.get(known.GRH, 'yes') != 'no',
                  method='2-isogeny descent' if row.get(known.GRH) == 'no' else 'class group')


__all__ = ['rankbound', 'Job', 'Result', 'Curve', 'RankboundError', 'NotApplicable', 'FactorisationNeeded', 'Locked',
           'Stopped', '__version__']
