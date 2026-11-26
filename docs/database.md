# Database

The database covers the 8,172 curves submitted to ICARM's
[Elliptic Curve Rank Leaderboard](https://elliptic-rank.icarm.cloud/) as of 2026-11-26 00:00 UTC, each with the rank $r$
listed there at that time. Every witness was computed and checked with `rankbound`.

For the 4,419 curves with rational 2-torsion, the witness proves an unconditional upper bound by descent via 2-isogeny. Since 
every one of these equals their lower bound, the rank of each of these curves is proved exactly. For the 3,753 curves without
rational 2-torsion, the witness proves a bound assuming the GRH. This bound is sharp for 3,745 of them, and exceeds the lower bound for the following four:

| curve | $r$ | $S$ |
|---|---|---|
| #442 | 19 | 20 |
| #3839 | 28 | 30 |
| #809 | 29 | 31 |
| #7333 | 30 | 32 |

The remaining four curves have no witness. The discriminant of #289 has a cofactor of 432 digits that we could not
factor, so its bad primes are not known. The curves #374 and #443 give too few relations, even at the largest setting
of the sieve. Finally, the cubic field of #568 is defined by a polynomial with a coefficient of 1,441 bits, which is
more than the sieve can take.

## The record

The witnesses are stored on [Zenodo](https://doi.org/10.5281/zenodo.23479585). Since most of them are small, they are
grouped by rank into the following files.

| file | contents |
|---|---|
| `icarmN.json.gz` | the witness of curve #N, for each of the 31 curves of rank at least 31 |
| `icarm_rank28.json.gz`, `icarm_rank29.json.gz`, `icarm_rank30.json.gz` | the witnesses of rank 28, 29 and 30 |
| `icarm_rank00-27.json.gz` | the witnesses of rank at most 27 |
| `icarm.tsv` | the table below |
| `README.md` | a description of the record |
| `SHA256SUMS` | the checksums of the other files |

The file `icarm.tsv` has one row per curve with the following columns.

| column | value |
|---|---|
| `name`, `icarm_id` | `icarmN` and $N$, the curve's id on the leaderboard |
| `file` | the file holding the witness |
| `rank lower bound`, `rank upper bound` | Lower and upper bounds $r$ and $S$ |
| `rank = r?` | `yes` when $S = r$, so that the rank is proved |
| `GRH` | `yes` when the bound assumes the GRH, `no` when it is unconditional |
| `torsion` | $`E(\mathbb{Q})_{\mathrm{tors}}`$ as given by PARI's `elltors`, e.g. `[]`, `[5]`, `[2, 2]` |
| `minimal model` | the reduced minimal model, `a1,a2,a3,a4,a6` |
| `sha256` | the sha256 of the uncompressed witness |
| `verified` | when and how the witness was checked: the date, PARI/GP's version and the checker's hash |

The repository's `witnesses.tsv` has the same format except there, `file` names the witness as written by `rankbound unpack`. The format of each witness is described in [verification.md](verification.md#format).

To check a witness, first extract it from its file. For example, `rankbound unpack icarm_rank28.json.gz --ids 3862`
writes the witness of rank-28 curve #3862 to `./witnesses/icarm3862.witness.json.gz` and checks it against its
sha256. Then, `rankbound check` takes the witnesses and prints a verdict for each. With `--json`, it prints one JSON
verdict per witness with exit status 0 if every witness passes.
