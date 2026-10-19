"""Curves already certified: their rows of certificates.tsv, found by the reduced minimal model.  Nothing is
computed or checked here; `rankbound verify` checks the certificate itself."""
import os

INDEX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'certificates.tsv')
LB, UB, EQ, MM = 'rank lower bound', 'rank upper bound under GRH', 'rank = r?', 'minimal model'    # its columns
ORDER = {'yes': 0, 'no': 1, 'r cited': 2, 'no (r cited)': 3}      # among the rows of one curve, if it has several


def lookup(minimal_model, index=None):
    """Return the row of certificates.tsv for the minimal model, or None; of several rows, the least S, then r
    proved before r cited."""
    path = index or INDEX
    if not os.path.isfile(path):
        return None
    key = ','.join(str(x) for x in minimal_model)
    with open(path) as f:
        head = f.readline().rstrip('\n').split('\t')
        if MM not in head:
            return None
        rows = [dict(zip(head, l.rstrip('\n').split('\t'))) for l in f if l.strip()]
    hit = sorted((r for r in rows if r.get(MM) == key and r.get('file') and r.get(EQ) in ORDER),
                 key=lambda r: (int(r[UB]), ORDER[r[EQ]], r['name']))
    return dict(hit[0]) if hit else None


def statement(row):
    """Return what the row's certificate proves, in words."""
    S, r, eq = row[UB], row[LB], row[EQ]
    if S == '0' or eq == 'yes':
        return 'rank E(Q) = %s under GRH' % S
    if eq == 'no':
        return '%s <= rank E(Q) <= %s under GRH' % (r, S)
    return 'rank E(Q) <= %s under GRH%s' % (S, ' (r = %s is cited, not proved in the certificate)' % r if r else '')


def lines(row, curve):
    """Return the lines `rankbound bound CURVE` prints for the row."""
    return ['%s is already certified: %s. Nothing was computed.' % (row['name'], statement(row)),
            '  Certificate %s (sha256 %s). Check it with: rankbound verify %s' % (row['file'], row['sha256'], row['file']),
            '  To compute the upper bound anyway: rankbound bound %s --compute' % curve]
