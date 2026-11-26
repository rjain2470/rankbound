"""rankbound: an upper bound on rank E(Q), with a witness: unconditionally, or under GRH (stated) for some curves
with E(Q)[2] = 0.

    from rankbound import rankbound
    res = rankbound([0, -1, 1, -929, -10595])      # 571a1: res.rank_bound == 0, res.checked
    rankbound([0, 0, 1, -1, 0], points=[(0, 0)])   # 37a1: rank >= 1 proved (r_proved), rank = 1 (proved_equal)
    rankbound([0, 0, 0, -25, 0], points=[(-4, 6)])  # E(Q)[2] != 0: rank = 1, unconditionally (res.grh False)

The checker (rankbound/check/) imports nothing from the prover, and the prover's modules load only when used."""
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

if TYPE_CHECKING:
    from .core import Result

ERRORS = ('RankboundError', 'NotApplicable', 'FactorisationNeeded', 'TimeLimit', 'Rejected', 'Locked', 'Stopped')


def __getattr__(name: str) -> Any:
    """Import Curve, Result, the errors and __version__ on first use, so importing the package loads no prover code."""
    if name == 'Curve':
        from .curve import Curve
        return Curve
    if name == '__version__':
        from .core import VERSION
        return VERSION
    if name == 'Result' or name in ERRORS:
        from . import core
        return getattr(core, name)
    raise AttributeError('module %r has no attribute %r' % (__name__, name))


def rankbound(E: Sequence[Union[int, str]], *, r: Optional[int] = None,
              points: Optional[Sequence[Sequence[Union[int, str]]]] = None, relations: Optional[str] = None,
              uncond: bool = False, bad_primes: Optional[Sequence[Union[int, str]]] = None, ecm_hours: float = 1.0,
              cores: Optional[int] = None, max_hours: Optional[float] = None, budget: Optional[float] = None,
              workdir: Optional[str] = None, compute: bool = False) -> 'Result':
    """Return a certified bound rank E(Q) <= S as a Result; this is `rankbound bound` as a function.

    E is [a1, a2, a3, a4, a6] or [a4, a6], on any model.  If E(Q)[2] != 0, a 2-isogeny descent bounds the rank
    unconditionally.  Otherwise the 2-Selmer group does, assuming the GRH for the sieve's (or the given) relations,
    or unconditionally with uncond (Minkowski's bound; refused when out of reach).

    workdir is this curve's own folder (default rankbound_work/NAME); calling again with it resumes.  points are
    pairs (x, y) of ints, Fractions or strings like '-3/4'.  The run raises Stopped at max_hours or at the Slurm
    job's end, whichever is first.  budget is the core-hours allowed across resumes (default 3 times the estimate,
    at least 1); when it is spent the result's outcome is 'budget'.  cores defaults to RANKBOUND_CORES, else what
    this machine or job provides (see rankbound/machine.py).  The other keyword arguments are the options of the
    same name of `rankbound bound`.
    """
    from . import runner
    return runner.bound(E, r=r, points=points, relations=relations, uncond=uncond, bad_primes=bad_primes,
                        ecm_hours=ecm_hours, cores=cores, max_hours=max_hours, budget=budget, workdir=workdir,
                        compute=compute)


__all__ = ['rankbound', 'Result', 'Curve', 'RankboundError', 'NotApplicable', 'FactorisationNeeded', 'Locked',
           'Stopped', '__version__']
