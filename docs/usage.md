# Usage

`rankbound` has the following commands.

| command | description |
|---|---|
| `rankbound bound <a-invariants>` | Bounds the rank of a curve and writes a witness. |
| `rankbound bound --estimate <a-invariants>` | Estimates the cost of a run. |
| `rankbound status [<folder>]` | Shows the progress of the runs in a work folder. |
| `rankbound check <witness>...` | Checks witnesses. |
| `rankbound unpack <file>` | Writes the witnesses of a database file to their own files. |

Their arguments are the following.

| argument | value |
|---|---|
| `<a-invariants>` | A list of the form `a1,a2,a3,a4,a6`, `a4,a6`, `[a1, a2, a3, a4, a6]`. |
| `<folder>` | A work folder set to `./rankbound_work` by default. |
| `<witness>` | A `.witness.json` file (or a `.witness.json.gz` file, or a `.cert.json` file of rankbound 2), or a folder of them. |

For each command, `rankbound <command> --help` lists every option. Each run first prints its work folder and the
number of cores it uses, and this line is left out of the examples below.

## Bounding a rank

If a run of `rankbound bound` is interrupted, running the same command again resumes it. The command exits with status
0 when every curve has its result, 1 when a curve failed or its witness was rejected, 2 for a usage error, and 3 when a
run stopped at its time limit. Curves listed in the repository's `witnesses.tsv` are looked up rather than computed,
unless `--compute` is given. An installed copy of `rankbound` does not include `witnesses.tsv`, and so computes every
curve.

Given $r$ independent rational points, the witness also proves that $\mathrm{rank}~E(\mathbb{Q}) \ge r$. For
example, the following code proves that the curve $E: y^2 = x^3 - 25x$ has rank 1 using the generator $(-4, 6)$ of $E(\mathbb{Q})$:

```
$ printf '%s\n' '-4 6' > points.txt
$ rankbound bound 0,0,0,-25,0 --points points.txt
10:45:11 Conductor 800, two bad primes, root number -1, torsion Z/2 x Z/2.
10:45:11 Rank at least 1.
10:45:11 Rank at most 1. Verification method: 2-isogeny descent (unconditional).
10:45:11 Rank 1, unconditionally. Witness: rankbound_work/0_0_0_-25_0/0_0_0_-25_0.witness.json (checked)
```

Moreover, consider $E: y^2 + y = x^3 - x^2 - 929x - 10595$. Since the cubic field of $E$ is small, `--uncond` proves that it has rank 0 unconditionally:

```
$ rankbound bound 0,-1,1,-929,-10595 --uncond --dir unc
10:45:14 Conductor 571, one bad prime, root number +1, no rational 2-torsion.
10:45:14 Searching for relations up to Minkowski's bound for an unconditional result. This should take about 1 second on 1 core.
10:45:14 Rank at most 2. Verification method: 2-Selmer group (unconditional).
10:45:15 Rank at most 0. Verification method: Cassels-Tate pairing (unconditional).
10:45:15 Rank 0, unconditionally. Witness: unc/0_-1_1_-929_-10595/0_-1_1_-929_-10595.witness.json.gz (checked)
```

Elliptic curves of high rank take longer. For example, for the rank-31 curve ICARM #846, `rankbound bound` with `--r 31` on 16 cores proves
that $\mathrm{rank}~E(\mathbb{Q}) \le 31$, assuming the GRH, in 7 minutes and 1.2 core-hours of sieving. For the rank-32 curve ICARM #6782, it proves that
$\mathrm{rank}~E(\mathbb{Q}) \le 32$ assuming the GRH, and the checker takes 36 minutes at 17 GB on 32 cores.

`rankbound bound` supports the following options:

```
--r R               a known lower bound r on the rank, cited: stop once the bound is r
--points FILE       r independent points, one "x y" per line: also prove rank >= r
--uncond            the unconditional bound (Minkowski's), refused with its cost if out of reach
--relations FILE    prove the bound from these relations (e.g. a witness), without sieving
--bad-primes P,...  the primes of the discriminant, if it is hard to factor
--ecm-hours H       factor the discriminant by ECM for at most H hours (default 1; 3 minutes with --estimate)
--curves FILE       several curves from a JSON list (select with --ids)
--cores N           number of cores (default: the performance cores, or a Slurm job's allocation)
--max-hours H       stop after H hours (run again to resume)
--budget H          core-hours the sieve may use for a curve in all (default 3 times its estimate, at least 1)
--dir DIR           the work folder
--compute           compute even if the curve is in the table
--json              one JSON result per curve on stdout (schema rankbound-result-1) instead of the result lines
-v                  also show the methods' inner steps (the work folder's run.log)
--shard K/N         on a cluster: sieve share K of N of the curve's relations
```

To bound several curves at once, give `--curves` a JSON file of the form
`{"curves": [{"id": 1, "ainvs": [...], "bad_primes": [...]}, ...]}`, where `bad_primes` is optional. If the bad primes
are not given, `rankbound` factors the discriminant by trial division and ECM, and skips the curve if this takes longer
than `--ecm-hours`. On a cluster, `rankbound bound` stops five minutes before the time limit of a Slurm job, and
resubmitting the job resumes the run. The relations of a large field can be split over several machines with
`--shard K/N`, using one work folder.

One can also run `rankbound` in Python. For example, the following code bounds the rank of the curve of the README and
checks its witness:

```python
from rankbound import rankbound

res = rankbound([0, -1, 1, -929, -10595])
print(res.rank_bound, res.checked)        # 0 True
```

By default, the work folder is `rankbound_work/NAME`, and `workdir=` names another, one per curve.

## Checking witnesses

The witness checker in `rankbound/check/` is independent of `rankbound`. To check a submitted witness against its
curve, run

```
rankbound check --json --curve a1,a2,a3,a4,a6 --max-mb 50 --max-seconds 600 NAME.witness.json
```
Each witness gets a verdict given by its exit status. A witness passes (0) when every check passes, and fails (1) if a check fails, the file is malformed, or when the file is for a different curve. A file that cannot be opened is a usage error (2), and a witness is not checked (3) when it is over `--max-mb` or `--max-seconds`, or when its rank check does not fit in the memory of the machine. Finally, the status is (4) when the checker could not run. With `--json`, `rankbound check` prints a JSON object per witness listing every check.
