"""The data passed between rankbound's parts: errors, the request, proof data, the verdict and the result.

A route returns ProofData, an unchecked witness, which the runner alone writes, checks and publishes.
"""
import hashlib, json, re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


VERSION = '3.0.0'


# ---------- the errors

class RankboundError(Exception):
    """Base class of rankbound's errors."""


class NotApplicable(RankboundError):
    """The class group method was asked for a curve with a rational point of order 2."""


class FactorisationNeeded(RankboundError):
    """The discriminant could not be factored; composites lists the unfactored parts and the effort spent on each."""

    def __init__(self, msg, composites=()):
        super().__init__(msg)
        self.composites = list(composites)


class TimeLimit(RankboundError):
    """A computation did not finish within its time limit."""

    def __init__(self, msg, seconds=0):
        super().__init__(msg)
        self.seconds = seconds


class Rejected(RankboundError, RuntimeError):
    """The input is not what it should be (bad primes, points)."""


class Locked(RankboundError):
    """Another process is using this folder."""


class Stopped(RankboundError):
    """The run stopped before a result.

    outcome is 'stopped' (the time limit; run again to resume) or a final stop: 'too_long' (a needed step cannot
    fit in one run; reason says what it needs) or 'budget' (the core-hour budget is spent).  result is the best
    checked Result so far, attempted a written but unchecked one, and resume the command that resumes the run.
    """

    def __init__(self, workdir, result=None, outcome='stopped', reason=None):
        super().__init__(reason or 'Stopped at the time limit. Run the same command again to resume.')
        self.workdir = workdir
        self.result = result
        self.outcome = outcome
        self.reason = reason
        self.attempted = None
        self.resume = None


class Damaged(RankboundError):
    """A stored file failed validation; it is rebuilt, never reused."""

    def __init__(self, msg, path=''):
        super().__init__(msg)
        self.path = path


# ---------- shared with the witnesses (the checker keeps its own copy)

MAX_DIGITS = 4000           # numbers the verifier reads
NUM = re.compile(r'0|-?[1-9][0-9]*')
RAT = re.compile(r'0|-?[1-9][0-9]*(/[1-9][0-9]*)?')
NAME = re.compile(r'[A-Za-z0-9_.-]{1,64}')
FORMAT2 = 'rankbound-2descent-witness-1'
FORMAT4 = 'rankbound-selmer-witness-4'
FORMAT5 = 'rankbound-selmer-witness-5'
FORMATS3 = ('rankbound-selmer-certificate-4', 'rankbound-selmer-certificate-3', 'rankbound-selmer-certificate-2')
# rankbound 2's formats, whose relations can still be given


def parity_bound(G: int, u: int, n: int, w: int) -> int:
    """Return the largest S <= G + u + n with (-1)^S = w (G + u + n if w is not +-1)."""
    return parity_floor(G + u + n, w)


def parity_floor(S: int, w: int) -> int:
    """Return the largest integer <= S with (-1)^S = w (S if w is not +-1).

    This is sound because (-1)^{dim Sel_2} = w (Monsky).
    """
    return S - 1 if w in (1, -1) and (-1) ** S != w else S


# ---------- the vocabulary of results

GRH, UNCONDITIONAL = 'GRH', 'none'                  # Request.hypothesis, in the witnesses' words
# Result.outcome.  'stopped' is the time limit (resumable); FINAL are stops that a resume repeats unless given
# more time, cores or budget, and they report the best checked bound, if any.
OUTCOMES = ('completed', 'stopped', 'refused', 'failed', 'table', 'too_long', 'budget')
FINAL = ('too_long', 'budget')
# Result.verification is 'accepted', 'rejected', 'unavailable' (no PARI/GP; also a table row), 'pending',
# 'limit' (the checker stopped at its limits) or 'error'.  Result.origin is 'computed', 'resumed' or 'table'.

# The stable exit status of `rankbound bound`: 0 every curve has its result (a final stop included), 1 a
# failure or a rejected witness, 2 a usage error, 3 stopped at the time limit.  A signal ends it as the signal does.
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_STOPPED = 0, 1, 2, 3
RESULT_SCHEMA = 'rankbound-result-1'     # one --json object per curve on stdout; progress goes to stderr


class Unprepared(RankboundError):
    """A check before any work failed; the message says the remedy."""


class Usage(RankboundError, ValueError):
    """Options that exclude each other for this curve: a usage error in the CLI, a ValueError in the API."""


def outcome_json(name, outcome, *, error=None, resume=None, best=None, attempted=None, reason=None, curve=None):
    """Return the --json object of a curve without a result: stopped, a final stop, or 'failed' (with error).

    best is the best checked Result so far, attempted a written but unchecked one.
    """
    return {'schema': RESULT_SCHEMA, 'name': name, 'curve': curve, 'outcome': outcome, 'rank_bound': None,
            'hypothesis': None, 'GRH': None, 'verification': None, 'checked': None, 'witness': None,
            'error': error, 'resume': resume, 'reason': reason, 'best': None if best is None else best.to_json(),
            'attempted': None if attempted is None else attempted.to_json()}


def digest(obj: Any) -> str:
    """Return the hex sha256 of obj's canonical JSON text (sorted keys, no spaces)."""
    text = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(text.encode()).hexdigest()


def checked_of(verification):
    """Return Result.checked: True if accepted, False if rejected, else None."""
    return True if verification == 'accepted' else False if verification == 'rejected' else None


# ---------- the request

@dataclass(frozen=True)
class LowerBound:
    """rank E(Q) >= r, proved by points (proof is their digest) or cited (source)."""
    r: int
    proved: bool = False
    source: str = 'cited'
    proof: Optional[str] = None


@dataclass(frozen=True)
class Resources:
    """What a run may use; not part of the request's identity.  The time limit is the run's runtime.Budget."""
    cores: int = 1
    max_hours: Optional[float] = None
    ecm_hours: float = 1.0
    budget_core_hours: Optional[float] = None   # across resumes; None for the default
    cores_from: Optional[str] = None            # machine.Cores.source


@dataclass(frozen=True)
class Request:
    """A normalized request: what is to be proved, and the work folder (output).

    minimal_model and model_map ([u, r, s, t]) are None until computed; hypothesis is the effective one. options hold
    the caller's words for the resume command. budget, options and bad_primes are not part of the identity, so they may
    change on resume.
    """
    source: Tuple[str, ...]
    minimal_model: Optional[Tuple[str, ...]] = None
    model_map: Optional[Tuple[str, ...]] = None
    hypothesis: str = GRH
    relations: Optional[str] = None
    lower: Optional[LowerBound] = None
    name: str = ''
    output: str = ''
    bad_primes: Optional[Tuple[str, ...]] = None      # asserted by the caller, then checked
    options: Dict = field(default_factory=dict, compare=False, hash=False)
    budget: Optional[float] = field(default=None, compare=False, hash=False)

    def identity(self) -> Dict[str, Any]:
        """Return what the work folder's result must have been computed for, as JSON values."""
        lo = self.lower
        return {'schema': 1, 'curve': list(self.source),
                'minimal_model': None if self.minimal_model is None else list(self.minimal_model),
                'model_map': None if self.model_map is None else list(self.model_map),
                'hypothesis': self.hypothesis, 'relations': self.relations,
                'lower': None if lo is None else {'r': lo.r, 'proved': lo.proved, 'source': lo.source, 'proof': lo.proof},
                'name': self.name}

    def key(self) -> str:
        """Return the identity's digest."""
        return digest(self.identity())

    def resume_command(self) -> str:
        """Return the command that resumes this request: the CLI line as typed, or the API call."""
        o = dict(self.options)
        if o.get('command'):
            return o['command']
        o['workdir'] = self.output
        args = ', '.join('%s=%r' % (k, v) for k, v in o.items() if v is not None and v is not False)
        return 'rankbound([%s]%s)' % (', '.join(self.source), ', ' + args if args else '')


# ---------- proof data, verdict

@dataclass
class ProofData:
    """What a route returns: a witness, not yet written or checked.

    data is the witness for witness.write (None if the route wrote it already). final is a FINAL outcome when the route
    stopped for good with this as its best bound, and reason that stop's words.
    """
    witness: str
    rank_bound: int
    hypothesis: str
    statement: str
    method: str
    rank_lower_bound: Optional[int] = None
    r_proved: bool = False
    record: Dict = field(default_factory=dict)
    cost: Dict = field(default_factory=dict)
    outcome: str = 'completed'
    data: Optional[dict] = None
    final: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class Verdict:
    """The checker's verdict on a witness's bytes (sha256), for the expected curve.

    state is a Result.verification value, check the first failed check, raw the checker's JSON.
    """
    state: str
    sha256: Optional[str] = None
    statement: Optional[str] = None
    reason: str = ''
    check: Optional[str] = None
    checker: Optional[str] = None
    raw: Dict = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def of(cls, v, sha256=None):
        """Return the Verdict of the checker's JSON v, or 'unavailable' for None (no PARI/GP)."""
        if v is None:
            return cls('unavailable', sha256)
        state = v.get('verdict')
        if state not in ('accepted', 'rejected', 'limit', 'error'):
            state = 'error'
        failed = [c['check'] for c in v.get('checks') or [] if isinstance(c, dict) and not c.get('ok')]
        return cls(state, v.get('sha256') or sha256, v.get('statement'), v.get('reason') or '',
                   (failed[0] if failed else state) if state != 'accepted' else None, v.get('checker'), dict(v))


# ---------- the result (rankbound.Result)

@dataclass(frozen=True)
class Result:
    """A bound rank E(Q) <= rank_bound (assuming the GRH if grh), and how it was reached.

    hypothesis is 'GRH' or 'none'; lower_bound is 'proved' (by the witness's points), 'cited' or None.  outcome,
    verification and origin are independent: a completed run can be unchecked or rejected.  witness_sha256 is
    the hash of the bytes the checker read; resume is the command that continues or re-checks the run.
    """
    name: str
    rank_bound: int                 # S: rank E(Q) <= S (assuming the GRH if grh)
    rank_lower_bound: Optional[int] # r, if given or proved
    r_proved: bool                  # rank E(Q) >= r is proved by the witness
    proved_equal: bool              # rank E(Q) = S is proved (assuming the GRH if grh)
    grh: bool                       # whether rank_bound assumes the GRH
    method: str                     # '2-isogeny descent' or '2-Selmer group'
    statement: str
    witness: str                    # the witness file
    checked: Optional[bool]         # the checker's verdict; None without PARI/GP or for a curve of the table
    core_hours: float = 0.0
    workdir: str = ''
    minimal_model: Optional[List[str]] = None
    conductor: Optional[str] = None
    torsion: Optional[List[int]] = None
    G: Optional[int] = None         # dim Cl(K)[2] <= G (2-Selmer group)
    known: bool = False             # from the table of witnesses (witnesses.tsv): nothing computed
    details: Dict = field(default_factory=dict)    # the rest of the run's record
    hypothesis: str = GRH
    lower_bound: Optional[str] = None
    outcome: str = 'completed'
    verification: str = 'pending'
    origin: str = 'computed'
    witness_sha256: Optional[str] = None
    cost: Dict = field(default_factory=dict)
    resume: Optional[str] = None

    def to_json(self, curve: Optional[Sequence[Union[int, str]]] = None) -> Dict[str, Any]:
        """Return the --json object of this result (RESULT_SCHEMA).

        A table row has origin 'table' and verification 'unavailable': a catalogue entry, not a new proof.
        """
        return {'schema': RESULT_SCHEMA, 'name': self.name, 'curve': curve,
                'minimal_model': self.minimal_model, 'outcome': self.outcome, 'rank_bound': self.rank_bound,
                'hypothesis': self.hypothesis, 'GRH': self.grh, 'verification': self.verification,
                'checked': self.checked, 'witness': self.witness, 'witness_sha256': self.witness_sha256,
                'statement': self.statement, 'method': self.method, 'rank_lower_bound': self.rank_lower_bound,
                'lower_bound': self.lower_bound, 'r_proved': self.r_proved, 'proved_equal': self.proved_equal,
                'origin': self.origin, 'known': self.known, 'core_hours': self.core_hours, 'workdir': self.workdir,
                'resume': self.resume, 'error': self.details.get('reason') if self.verification == 'rejected' else None,
                'reason': (self.details.get('final') or {}).get('reason') if self.outcome in FINAL else None,
                'provenance': self.details.get('provenance')}
