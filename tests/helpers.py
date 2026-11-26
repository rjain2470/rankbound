"""Shared by the tests: the honest witnesses of every format (corpus) and paths in them (paths, get, put, drop),
the format-3 witness of ICARM #0 (wit0), a Solver without __init__ (bare_solver), the failed
checks of a witness (fails), a route's run through the runner (run_route), the GP function td1 of the 2-isogeny
descent (DESCENT) and PARI's rows of relations (REFROWS)."""
import atexit, copy, json, os, shutil, tempfile
import rankbound

TESTS = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.abspath(rankbound.__file__))      # the package under test (rankbound in a checkout)
SRC = os.path.dirname(PKG)                                      # the folder that holds it: a subprocess's PYTHONPATH
# ICARM #0 (11a3, rank 0) and relations (a, b) from its witness
ICARM0 = [0, -1, 1, 0, 0]
ICARM0_RELS = ('-83,1 -7,1 -4,1 -3,1 -2,1 2,1 3,1 4,1 5,1 7,1 39,1 45,1 117,1 -25,2 3,2 5,2 29,2 -5,3 1,3 2,3 5,3 -5,4 '
               '-1,4 3,4 -2,5 -1,5 1,5 2,5 7,5 58,5 507,5 -1,6 -43,7 -5,7 1,7 3,7 -4,9 -7,11 -5,11 -8,15 -9,16 91,31 '
               '-17,43 130,97').split()
_WIT0 = []


def wit0():
    """Return the path of a copy of ICARM #0's format-3 witness (rank <= 0; tests/data/icarm0.f3.cert.json, written
    by rankbound 2's class group method from ICARM0_RELS, which no longer writes format 3), removed at exit."""
    if not _WIT0:
        d = tempfile.mkdtemp(prefix='rankbound_wit0')
        atexit.register(shutil.rmtree, d, True)
        p = os.path.join(d, 'icarm0.cert.json')
        shutil.copyfile(os.path.join(TESTS, 'data', 'icarm0.f3.cert.json'), p)
        _WIT0.append(p)
    return _WIT0[0]


def bare_solver(S, M=None, **kw):
    """Return a Solver without __init__ on state S and matrix M: no relations or checks, target 0 (a goal that
    stops), u = n = 0 and w = +1; KW overrides any attribute."""
    from rankbound import linalg
    from rankbound import sieve as jobs
    V = object.__new__(jobs.Solver)
    V.S, V.M = S, M if M is not None else linalg.Matrix({'J': []}, [])
    V.nrel, V.nfree, V.target, V.goal_stop, V.prm = 0, 0, 0, True, {'siever': 'given'}
    V.hist, V.order, V.last, V.loaded, V.last_rows, V.first = [], [], None, set(), 0, frozenset()
    V.F = {'u': 0, 'n': 0, 'w': 1}
    V.__dict__.update(kw)
    return V


def fails(c, d, name='t'):
    """Return the names of the failed checks of the witness dict c (written to the folder d)."""
    path = os.path.join(d, name + '.cert.json')
    with open(path, 'w') as f:
        json.dump(c, f)
    from rankbound import check as verifier
    v = verifier.check(path)
    assert v['verdict'] in ('accepted', 'rejected'), v['reason']
    return [x['check'] for x in v['checks'] if not x['ok']]


# td1(ai) = [B, S, t, 1 if every isogeny passes the duality and torsion checks], by the 2-isogeny descent alone
DESCENT = r'''
td1(ai) =
{
  my(E = ellinit(ai), P = factor(abs(E.disc))[, 1]~, M = tdmodel(ai, P), S = Set(concat([2], P)), D);
  D = vector(#M[3], i, tdisogeny(M[2], M[3][i], S));
  concat(tdbound(apply(d -> d[8], D), ellrootno(M[1])), [#M[3] \ 2 + 1, vecmin(apply(d -> d[9] && d[10], D))]);
}
'''


# refrows(K, rels): the odd-valuation primes of each relation [a, b, H] (a - b alpha), [p] (the ideal (p)) or
# [c0, c1, c2, d, H] ((c0 + c1 alpha + c2 alpha^2)/d) by PARI's
# idealval, named by the verifier's ckcol ("p:r", "p::f"); K = ckinit(f, P) (selmer.gp).  The reference of the prover's
# row rule (test_reference) and of the verifier's relcheck (test_verify_relcheck).
REFROWS = r"""
refrows(K, rels) =
{
  my(nf = K[1], al = K[2], Jl = K[3]);
  for(i = 1, #rels,
    my(r = rels[i], cols = List(), ps, el);
    if(#r == 1, ps = [r[1]]; el = r[1],
       #r == 5, ps = Set(concat([r[5], Jl, if(r[4] > 1, factor(r[4])[, 1]~, [])])); el = (r[1] + r[2] * al + r[3] * al^2) / r[4],
       ps = Set(concat(r[3], Jl)); el = r[1] - r[2] * al);
    foreach(ps, p, my(dec = idealprimedec(nf, p));
      for(k = 1, #dec, if(idealval(nf, el, dec[k]) % 2, listput(cols, ckcol(K, p, dec[k])))));
    print("R ", i - 1, " ", strjoin(Vec(cols), " ")));
}
"""


def corpus():
    """Return {name: (the true rank, the witness)}: the honest witnesses of tests/data/corpus.json (the prover's:
    formats 4 and 2-isogeny, every bound the true rank) and of ICARM #0 (format 2) and the format-3 files."""
    with open(os.path.join(TESTS, 'data', 'corpus.json')) as f:
        out = {k: (v['rank'], v['witness']) for k, v in json.load(f).items()}
    for name, rank in (('37a1.f3', 1), ('icarm0.f3', 0)):
        with open(os.path.join(TESTS, 'data', name + '.cert.json')) as f:
            out[name] = (rank, json.load(f))
    return out


def paths(x, path=()):
    """Yield the path (keys and indices) of every value inside x."""
    for k, y in (x.items() if isinstance(x, dict) else enumerate(x) if isinstance(x, list) else ()):
        yield path + (k,)
        yield from paths(y, path + (k,))


def get(c, path):
    for k in path:
        c = c[k]
    return c


def put(c, path, value):
    """Return a copy of c with value at path."""
    c = copy.deepcopy(c)
    get(c, path[:-1])[path[-1]] = value
    return c


def drop(c, path):
    c = copy.deepcopy(c)
    get(c, path[:-1]).pop(path[-1])
    return c


def run_route(J, curve, cores=1, budget=None, relations=None):
    """The record of the route J's run of the classified Curve curve through the runner (runner.run_folder)."""
    from rankbound import core, runner
    return runner.run_folder(J, curve, core.Resources(cores=cores), relations=relations, budget=budget).details
