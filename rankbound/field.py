"""The curve and its cubic field: the checks of the input, the minimal model, and the field data from field.gp in
STATE/field.json and STATE/small.txt."""
import dataclasses, fractions, gzip, hashlib, json, os
from dataclasses import dataclass, field
from typing import List, Optional
from .verify import MAX_DIGITS, NAME, NUM, RAT, Rejected, parity_bound, run_gp
from . import FactorisationNeeded, NotApplicable, gp_message, read_json, write_json

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT = 'rankbound-selmer-field-1'


# ---------- values that reach gp

WHAT = {'ainvs': 'a-invariant', 'bad_primes': 'bad prime', 'minimal_model': 'minimal_model value',
        'proof point': 'point coordinate', 'mapped point': 'point coordinate'}    # how messages name each value

def canon(x, what, pat=NUM):
    """Return x, a JSON integer or a canonical decimal string, as a string; raise ValueError otherwise."""
    # gp evaluates what it reads, so every value pasted into gp passes here first.
    if type(x) is int:
        s = str(x) if x.bit_length() <= 14000 else None      # str() refuses over 4,300 digits
    else:
        s = x if isinstance(x, str) and pat.fullmatch(x) else ''
    what = WHAT.get(what, what)
    if s is None or len(s) > MAX_DIGITS:
        raise ValueError("%s %s has more than %d digits, the verifier's limit." % (
            'An' if what[0] in 'aeiou' else 'A', what, MAX_DIGITS))
    if s:
        return s
    if pat is NUM:
        raise ValueError('The %s %.80r is not an integer.' % (what, x))
    raise ValueError('The %s %.80r is not written like -3/4.' % (what, x))


def check_name(name):
    """Return name if the verifier accepts it (at most 64 letters, digits and _.-), else raise ValueError."""
    if isinstance(name, str) and NAME.fullmatch(name) and name not in ('.', '..'):
        return name
    raise ValueError('The name %.80r is not allowed: use at most 64 letters, digits, _ . and -.' % (name,))


def r_value(r):
    """Return r if it is None or an integer >= 0, else raise ValueError."""
    if r is None or (type(r) is int and r >= 0):
        return r
    raise ValueError('r must be an integer >= 0.')


PROOF = '"proof" must be {"points": [[x, y], ...], "primes": [p, ...]}.'


def check_proof(proof):
    """Return the proof of a rank lower bound (points, and optionally primes and ainvs) or None, checked as the
    verifier checks it; raise ValueError if it is malformed."""
    if proof is None:
        return None
    if not isinstance(proof, dict):
        raise ValueError(PROOF)
    pts, prs = proof.get('points'), proof.get('primes', [])
    if not (isinstance(pts, list) and (prs is None or isinstance(prs, list))):
        raise ValueError(PROOF)
    prs = prs or []
    if proof.get('ainvs') is not None:
        _ainvs5(proof['ainvs'])
    for pt in pts:
        if not (isinstance(pt, list) and len(pt) == 2):
            raise ValueError(PROOF)
        for x in pt:
            canon(x, 'proof point', RAT)
    for p in prs:
        if type(p) is not int:
            raise ValueError(PROOF)
    return proof


def point_value(x):
    """Return a point coordinate (an int, a Fraction or a string like '-3/4') as a canonical string."""
    if isinstance(x, fractions.Fraction):
        x = str(x)
    return canon(x, 'point coordinate', RAT)


def read_points(path):
    """Return the points of a --points file, one 'x y' or 'x,y' per line; blank and '#' lines are skipped."""
    pts = []
    with open(path) as f:
        try:
            for k, l in enumerate(f, 1):
                t = l.replace(',', ' ').split()
                if not t or t[0].startswith('#'):
                    continue
                if len(t) != 2:
                    raise ValueError('%s, line %d: not a point "x y".' % (path, k))
                pts.append(t)
        except UnicodeDecodeError:
            raise ValueError('%s is not UTF-8 text.' % path)
    return pts


def with_points(curve, points, r=None):
    """Return curve with r = len(points), to be proved by the points, which are on the given model and must be
    independent modulo 2E(Q)."""
    if not all(isinstance(p, (list, tuple)) for p in seq(points, 'points')):
        raise ValueError('Each point must be a pair (x, y).')
    pts = [[point_value(x) for x in p] for p in points]
    if any(len(p) != 2 for p in pts):
        raise ValueError('Each point must be a pair (x, y).')
    if r is not None and r_value(r) != len(pts):
        raise ValueError('r = %s but %d points were given.' % (r, len(pts)))
    return dataclasses.replace(curve, r=len(pts), proof={'points': pts, 'ainvs': list(curve.ainvs)}, source='cited')


# ---------- the curve

@dataclass(frozen=True)
class Curve:
    name: str
    ainvs: List[str]                          # the input model
    bad_primes: List[str]
    r: Optional[int] = None                   # a known rank lower bound
    proof: Optional[dict] = None              # points (and primes) that prove r
    source: str = 'cited'                     # where r is from, without a proof
    minimal_model: Optional[List[str]] = None
    composites: List[str] = field(default_factory=list)

    def to_dict(self):
        d = {'name': self.name, 'ainvs': self.ainvs, 'bad_primes': self.bad_primes, 'composites': self.composites,
             'r': self.r, 'proof': self.proof, 'source': self.source}
        if self.minimal_model is not None:
            d['minimal_model'] = self.minimal_model
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(name=d['name'], ainvs=d['ainvs'], bad_primes=d['bad_primes'], r=d.get('r'), proof=d.get('proof'),
                   source=d.get('source', 'cited'), minimal_model=d.get('minimal_model'), composites=d.get('composites', []))

    def key(self):
        """Return the default name of the curve."""
        return default_name(self.minimal_model or from_ainvs(self.ainvs, bad_primes=self.bad_primes).minimal_model)


def default_name(mm):
    """Return the minimal model's a-invariants joined by '_', or a 12-digit hash of them if that is over 64
    characters."""
    s = '_'.join(str(x) for x in mm)
    return s if len(s) <= 64 else hashlib.sha256(','.join(str(x) for x in mm).encode()).hexdigest()[:12]


def checked(c):
    """Return c, a Curve made by hand, after checking every field as from_ainvs would; raise ValueError otherwise."""
    if not isinstance(c, Curve):
        raise ValueError('Expected a Curve.')
    check_name(c.name)
    _ainvs5(c.ainvs)
    _bad_primes(c.bad_primes)
    for x in seq(c.composites, 'composites'):
        canon(x, 'composite')
    if c.minimal_model is not None:
        [canon(x, 'minimal_model') for x in seq(c.minimal_model, 'minimal_model')]
    r_value(c.r)
    check_proof(c.proof)
    if not isinstance(c.source, str):
        raise ValueError('source must be a string.')
    return c


# The minimal model, whether E(Q)[2] = 0, and the bad primes: trial division to 10^6, cheap splits of what is left
# (perfect powers, the gcd with c4, which the primes of additive reduction divide), then factor() for at most S seconds
# per cofactor (none with S = 0: the composites go to the ECM step).
CURVE_GP = '''{ my(E = ellminimalmodel(ellinit([%s])), T = x^3 + E.b2*x^2 + 8*E.b4*x + 16*E.b6, P = List(), C = List(), G,
     J = (v -> strjoin(apply(z -> Str(z), Vec(v)), ",")), S = %d);
  if(%d, my(F = factor(abs(E.disc), 10^6), todo = List());
    for(i = 1, #F~, my(p = F[i, 1]); if(p < 10^6 || ispseudoprime(p), listput(P, p), listput(todo, p)));
    while(#todo, my(q = todo[#todo], k, g); listpop(todo);
      if(q == 1, next);
      if(ispseudoprime(q), listput(P, q); next);
      k = ispower(q, , &g); if(k, listput(todo, g); next);
      g = gcd(q, E.c4); if(g > 1 && g < q, listput(todo, g); listput(todo, q / g); next);
      G = if(S, iferr(alarm(S, factor(q)), e, 0), 0);
      if(type(G) != "t_MAT", listput(C, q); next);
      for(j = 1, #G~, if(ispseudoprime(G[j, 1]), listput(P, G[j, 1]), listput(C, G[j, 1])))));
  print("E2\\t", polisirreducible(T), "\\t", J(E[1..5]), "\\t", J(Set(Vec(P))), "\\t", J(Set(Vec(C)))); }'''


def seq(v, what):
    """Return v if it is a list or tuple, else raise ValueError (a string would be read character by character)."""
    if isinstance(v, (list, tuple)):
        return v
    raise ValueError('%s must be a list.' % what)


def disc(a):
    """Return the discriminant of the a-invariants a."""
    a1, a2, a3, a4, a6 = a
    b2, b4, b6 = a1 * a1 + 4 * a2, 2 * a4 + a1 * a3, a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    return -b2 * b2 * b8 - 8 * b4 ** 3 - 27 * b6 * b6 + 9 * b2 * b4 * b6


def _ainvs5(ainvs):
    """Return the 5 integers of ainvs (5 a-invariants or [a4, a6]); raise ValueError if malformed or singular."""
    a = [int(canon(x, 'a-invariant')) for x in seq(ainvs, 'ainvs')]
    if len(a) == 2:
        a = [0, 0, 0] + a
    if len(a) != 5:
        raise ValueError('Give the curve as a1,a2,a3,a4,a6 or as a4,a6.')
    if disc(a) == 0:
        raise ValueError('The a-invariants [%s] give a singular curve (discriminant 0).' % ','.join(map(str, a)))
    return a


def _bad_primes(bad_primes):
    """Return bad_primes as canonical strings, each >= 2; raise ValueError otherwise."""
    bad_primes = [canon(p, 'bad prime') for p in seq(bad_primes, 'bad_primes')]
    for p in bad_primes:
        if int(p) < 2:
            raise ValueError('The bad prime %s is not prime.' % p)
    return bad_primes


def minimal_model(ainvs, bad_primes=None, gp='gp'):
    """Return the reduced minimal model of ainvs as 5 strings; the discriminant is factored only as far as
    ellminimalmodel needs."""
    a = _ainvs5(ainvs)
    pre = '' if bad_primes is None else 'addprimes([%s]); ' % ','.join(_bad_primes(bad_primes))
    out = run_gp(pre + 'print("MM\\t", strjoin(apply(z -> Str(z), Vec(ellminimalmodel(ellinit([%s]))[1..5])), ","));'
                 % ','.join(map(str, a)), seconds=7200, gp=gp)
    return next(l.split('\t')[1].strip() for l in out.splitlines() if l.startswith('MM')).split(',')


NOT_APPLICABLE = ('This curve has a rational point of order 2, so the method does not apply (use a 2-isogeny descent, '
                  'e.g. mwrank).')


def from_ainvs(ainvs, bad_primes=None, r=None, name=None, gp='gp', factor_seconds=600, composites=False):
    """Return the Curve of ainvs, on any model.  Without bad_primes the discriminant is factored, by PARI for at most
    factor_seconds per cofactor; a composite left raises FactorisationNeeded or, with composites, is returned in
    Curve.composites, for Job.prepare's ECM step.  Raises NotApplicable if E(Q)[2] != 0."""
    a = _ainvs5(ainvs)
    r = r_value(r)
    if name is not None:
        check_name(name)
    body = CURVE_GP % (','.join(map(str, a)), int(factor_seconds), bad_primes is None)
    if bad_primes is not None:
        bad_primes = _bad_primes(bad_primes)
        body = 'addprimes([%s]);\n' % ','.join(bad_primes) + body
    out = run_gp(body, seconds=7200 + int(factor_seconds) * 100, gp=gp)
    t = next(l.split('\t') for l in out.splitlines() if l.startswith('E2'))
    if t[1].strip() != '1':
        raise NotApplicable(NOT_APPLICABLE)
    left = []
    if bad_primes is None:
        left = [x for x in t[4].strip().split(',') if x]
        if left and not composites:
            raise FactorisationNeeded('PARI could not factor the discriminant. Give its prime factors with '
                                      '--bad-primes P1,P2,...', [{'n': c, 'digits': len(c)} for c in left])
        bad_primes = [x for x in t[3].strip().split(',') if x]
    mm = t[2].strip().split(',')
    return Curve(name=default_name(mm) if name is None else name, ainvs=[str(x) for x in a], bad_primes=[str(p) for p in bad_primes], r=r,
                 minimal_model=mm, composites=left)


def from_list(path, ids=None):
    """Return the curves of a JSON curve list (optionally gzipped) whose id is in ids (None: all).  Each needs id (or
    icarm_id), ainvs and bad_primes (and composites, if bad_primes are not all the primes); a ValueError names the
    curve."""
    try:
        with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
            d = json.load(f)
    except UnicodeDecodeError:
        raise ValueError('%s is not UTF-8 text.' % path)
    except json.JSONDecodeError as e:
        raise ValueError('%s is not valid JSON at line %d, column %d.' % (path, e.lineno, e.colno))
    except (gzip.BadGzipFile, EOFError):
        raise ValueError('%s is not a valid gzip file.' % path)
    src = os.path.basename(path)
    if not (isinstance(d, dict) and isinstance(d.get('curves'), list)
            and all(isinstance(c, dict) for c in d['curves'])):
        raise ValueError('%s is not a curve list; see the README for the format.' % src)
    out, found, seen = [], set(), set()
    for k, c in enumerate(d['curves'], 1):
        icarm = 'icarm_id' in c
        i = c['icarm_id'] if icarm else c.get('id')
        if i is None:
            raise ValueError('Entry %d of %s has no "id".' % (k, src))
        if str(i) in seen:
            raise ValueError('%s has two curves with the id %.80s.' % (src, i))
        seen.add(str(i))
        if ids and str(i) not in {str(x) for x in ids}:
            continue
        found.add(str(i))
        try:
            ai = [canon(x, 'ainvs') for x in seq(c['ainvs'], 'ainvs')]
            _ainvs5(ai)                                         # a singular curve is refused here
            mm = c.get('minimal_model')
            mm = None if mm is None else [canon(x, 'minimal_model') for x in seq(mm, 'minimal_model')]
            pf = check_proof(c.get('proof'))
            # The list's points are on its ainvs; record that, so the certificate can map them.
            if pf is not None and pf.get('ainvs') is None:
                pf = dict(pf, ainvs=ai)
            r = r_value(c.get('r'))
            if r is None and pf is not None and pf.get('points'):   # r defaults to the number of points
                r = len(pf['points'])
            out.append(Curve(name=check_name(c.get('name', ('icarm%s' if icarm else 'curve%s') % i)), ainvs=ai,
                             bad_primes=[canon(p, 'bad_primes') for p in seq(c['bad_primes'], 'bad_primes')],
                             composites=list(seq(c.get('composites', []), 'composites')), r=r, proof=pf,
                             source='ICARM #%s' % i if icarm else 'curve %s of %s' % (i, src), minimal_model=mm))
        except (ValueError, TypeError) as e:
            raise ValueError('%s, curve %s: %s' % (src, i, e))
        except KeyError as e:
            raise ValueError('Curve %s of %s has no "%s".' % (i, src, e.args[0]))
    if ids and found != {str(x) for x in ids}:
        miss = sorted({str(x) for x in ids} - found)
        raise ValueError('%s has no curve with the id%s %s.' % (src, 's' if len(miss) > 1 else '', ', '.join(miss)))
    names = [c.name for c in out]
    if len(set(names)) < len(names):
        raise ValueError('%s has two curves named %s.' % (src, sorted(n for n in names if names.count(n) > 1)[0]))
    if not out:
        raise ValueError('%s has no curves.' % src)
    return out


# ---------- the field

def compute(ainvs, bad_primes, gp='gp', seconds=7200):
    """Return (the field data, the small labels (p, r)) from field.gp; raise ValueError, with field.gp's message, if
    the bad primes are wrong."""
    _ainvs5(ainvs)                  # every input passes here
    body = 'sfield([%s], [%s]);' % (','.join(canon(a, 'a-invariant') for a in ainvs),
                                    ','.join(canon(p, 'bad prime') for p in bad_primes))
    try:
        out = run_gp(body, libs=('verify.gp', os.path.join(HERE, 'field.gp')), seconds=seconds, gp=gp)
    except Rejected as e:           # field.gp's error() on bad input
        msg = gp_message(e).splitlines()[0]
        if msg.startswith('The bad prime'):
            raise ValueError(msg)
        if msg.startswith('E2 '):
            raise NotApplicable(NOT_APPLICABLE)
        raise
    sf, small, ram = {}, [], []
    for l in out.splitlines():
        t = l.split('\t')
        if t[0] == 'SF':
            sf[t[1]] = t[2] if len(t) > 2 else ''
        elif t[0] == 'S':
            small.append((int(t[1]), int(t[2])))
        elif t[0] == 'RAM':
            ram.append({'p': t[1], 'all_degree_1': t[2] == '1',
                        'labels': [[int(x) for x in z.split(':')] for z in t[3].split(',') if z]})
    if int(sf['ns']) != len(small):
        raise RuntimeError('field.gp: ns %s but %d labels' % (sf['ns'], len(small)))
    lst = lambda s: [x for x in s.split(',') if x]
    F = {
        'format': FORMAT,
        'minimal_model': lst(sf['ainvs']),
        'bad_primes': [str(p) for p in bad_primes],
        'poly': [int(c) for c in lst(sf['poly'])],
        'dK': sf['dK'],
        'log2_dK': float(sf['log2dK']),
        'r1': int(sf['r1']),
        'GB': int(sf['GB']),
        'bach': int(sf['bach']),
        'ns': len(small),
        'log2_size': float(sf['log2size']),
        'log2_skew': float(sf['log2skew']),
        'real_roots': [int(x) for x in lst(sf['real_roots'])],
        'J': [[int(y) for y in z.split(':')] for z in lst(sf['J']) if ':k' not in z],
        'J_large': [z for z in lst(sf['J']) if ':k' in z],
        'ramified': ram,
        'u': int(sf['u']),
        'n': int(sf['n']),
        'bk_primes': [dict(zip(('p', 'reduction', 'v_disc', 'contribution'), z.split(':'))) for z in lst(sf['bk_primes'])],
        'w': int(sf['w']),
        'sign_disc': int(sf['signD']),
    }
    F['conductor'] = conductor(F)
    return F, small


def conductor(F):
    """Return the conductor prod p^f_p, from the field data's Brumer-Kramer primes (f_p: their 'reduction')."""
    N = 1
    for b in F['bk_primes']:
        N *= int(b['p']) ** int(b['reduction'])
    return str(N)


def target(F, r):
    """Return g* = r - u - n, the least dim Cl(K)[2] that rank >= r allows, or None without r."""
    return None if r is None else r - F['u'] - F['n']


def goal(F):
    """Return the bound on G at which the run stops: the largest G that gives the least possible S.  S >= r, S = r
    needs w = (-1)^r, and G >= 0 gives S >= parity_bound(0, u, n, w)."""
    r, u, n, w = F.get('rank_lower_bound'), F['u'], F['n'], F['w']
    best = parity_bound(0, u, n, w)
    if r is not None:
        best = max(best, r if w == (-1) ** r else r + 1)
    return best + 1 - u - n


def prepare(state, ainvs, bad_primes, r=None, name=None, gp='gp', minimal_model=None):
    """Write STATE/field.json and small.txt, once, and return the field data.  Raises ValueError if minimal_model is
    wrong, or if r exceeds ns + u + n, the bound without relations."""
    path = os.path.join(state, 'field.json')
    if os.path.exists(path):
        return load(state)
    name = check_name(name or os.path.basename(os.path.abspath(state)))
    F, small = compute(ainvs, bad_primes, gp=gp)
    if minimal_model is not None and [str(x) for x in minimal_model] != F['minimal_model']:
        raise ValueError('The minimal_model given for %s is wrong; it should be [%s]. Correct it or leave it out.' % (
            name, ','.join(F['minimal_model'])))
    if r is not None and r > F['ns'] + F['u'] + F['n']:
        raise ValueError('r = %d is impossible for this curve: under GRH its rank is at most %d. Check r.' % (
            r, F['ns'] + F['u'] + F['n']))
    os.makedirs(state, exist_ok=True)
    F['name'] = name
    F['input_ainvs'] = [str(a) for a in ainvs]
    F['rank_lower_bound'] = r
    F['g_target'] = target(F, r)
    with open(os.path.join(state, 'small.txt.tmp'), 'w') as f:
        f.writelines('%d %d\n' % s for s in small)
    os.replace(os.path.join(state, 'small.txt.tmp'), os.path.join(state, 'small.txt'))
    write_json(path, F)
    return F


def load(state):
    return read_json(os.path.join(state, 'field.json'))


def write_split(poly, pmax, out, extra=(), gp='gp'):
    """Write to out, in makefb's format, the primes p < pmax and the primes in extra at which F splits completely:
    the free relations of a run on given relations."""
    extra = sorted({int(p) for p in extra if int(p) >= pmax})
    tmp = '%s.tmp%d.gz' % (out, os.getpid())
    with open(tmp + '.gpv', 'w') as f:            # readvec: the list may be long
        f.writelines('%d\n' % p for p in extra)
    try:
        body = 'sfsplit([%s], %d, readvec("%s"));' % (','.join(str(int(c)) for c in poly), int(pmax), tmp + '.gpv')
        out_txt = run_gp(body, libs=(os.path.join(HERE, 'field.gp'),), seconds=7200, gp=gp)
    finally:
        os.remove(tmp + '.gpv')
    with gzip.open(tmp, 'wt') as f:
        f.write('# primes below %d (and %d larger ones) where F splits completely (rankbound, field.gp sfsplit)\n' % (
            pmax, len(extra)))
        f.write(out_txt)
    os.replace(tmp, out)
    return out


def load_small(state):
    with open(os.path.join(state, 'small.txt')) as f:
        return [tuple(map(int, l.split())) for l in f]


def shifted(poly, k):
    """Return the coefficients c0..c3 of F(x + k, 1)."""
    c0, c1, c2, c3 = poly
    return [c0 + c1 * k + c2 * k * k + c3 * k ** 3, c1 + 2 * c2 * k + 3 * c3 * k * k, c2 + 3 * c3 * k, c3]


def norm(poly, a, b):
    c0, c1, c2, c3 = poly
    return ((c3 * a + c2 * b) * a + c1 * b * b) * a + c0 * b ** 3
