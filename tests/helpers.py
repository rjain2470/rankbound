"""Shared by the tests: the certificate of ICARM #0 (cert0) and a Solver without __init__ (bare_solver)."""
import atexit, os, shutil, tempfile
from rankbound import solver

TESTS = os.path.dirname(os.path.abspath(__file__))
# ICARM #0 (11a3, rank 0) and relations (a, b) from its certificate
ICARM0 = [0, -1, 1, 0, 0]
ICARM0_RELS = ('-83,1 -7,1 -4,1 -3,1 -2,1 2,1 3,1 4,1 5,1 7,1 39,1 45,1 117,1 -25,2 3,2 5,2 29,2 -5,3 1,3 2,3 5,3 -5,4 '
               '-1,4 3,4 -2,5 -1,5 1,5 2,5 7,5 58,5 507,5 -1,6 -43,7 -5,7 1,7 3,7 -4,9 -7,11 -5,11 -8,15 -9,16 91,31 '
               '-17,43 130,97').split()
_CERT0 = []


def cert0():
    """Return the path of ICARM #0's certificate (rank <= 0), built once from ICARM0_RELS and removed at exit."""
    if not _CERT0:
        from rankbound import rankbound
        d = tempfile.mkdtemp(prefix='rankbound_cert0')
        atexit.register(shutil.rmtree, d, True)
        res = rankbound(ICARM0, relations=ICARM0_RELS, points=[], name='icarm0', workdir=os.path.join(d, 'icarm0'),
                        verify=False)
        _CERT0.append(res.certificate)
    return _CERT0[0]


def bare_solver(S, M=None, **kw):
    """Return a Solver without __init__ on state S and matrix M: no relations or checks, target 0, u = n = 0 and
    w = +1; KW overrides any attribute."""
    from rankbound import linalg
    V = object.__new__(solver.Solver)
    V.S, V.M = S, M if M is not None else linalg.Matrix({'J': []}, [])
    V.nrel, V.nfree, V.target, V.keep, V.log = 0, 0, 0, 1000, lambda *a: None
    V.hist, V.order, V.last, V.loaded = [], [], None, set()
    V.F = {'u': 0, 'n': 0, 'w': 1}
    V.__dict__.update(kw)
    return V
