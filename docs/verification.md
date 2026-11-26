# Verification

Each bound that `rankbound` proves is written into a *witness*. Each witness can be verified using `rankbound check`, which is independent of `rankbound`.

## Format

A witness is a file of the form `<curve>.witness.json` or `<curve>.witness.json.gz`. Its `format` key specifies the method used to obtain the bound.

For a curve with 2-torsion, the witness gives, for each 2-isogeny, bases of both Selmer groups, the points from which the Cassels pairings are computed,
an optional Cassels–Tate section, and the bound from that isogeny. The checker then recomputes both Selmer groups from their local images, recomputes the ranks
of the pairings, and takes the least bound, lowered by one when its parity disagrees with the root number.

For curves without 2-torsion, each witness is computed assuming the GRH and contains the following:

- the cubic field $K$, given by an index form $F$, and the bound of \[BDF08\];
- the relations, each a string such as `"a,b:hexprimes"`, which denotes $a - b\alpha$ for a root $\alpha$ of $F(x, 1)$,
  followed by the primes of its norm;
- the steps of the Gaussian elimination over $`\mathbb{F}_2`$ that combine the relations;
- elements of $`K^*/K^{*2}`$ built from the relations, and local points;
- optionally, a Cassels–Tate section;
- the bounds it claims.

If a curve without 2-torsion has a sufficiently small cubic field, its rank bound can be computed unconditionally. In
this case, its witness contains the same data, except Minkowski's bound replaces the bound of \[BDF08\].

## Checking

Each witness gets one of the following verdicts.

| verdict | meaning |
|---|---|
| `accepted` | Every check passed. |
| `rejected` | A check failed. |
| `limit` | The witness is over `--max-mb` or `--max-seconds`, and is not checked. |
| `error` | The checker could not run. |
