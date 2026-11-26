"""The text of every line the user sees, and the wording of numbers, times and ranks.

Data and string formatting only, shared by the CLI and runtime.say.
"""
import contextvars, os, time

MESSAGES = {
    # ---- bound: the curve (the first line of a run)
    'curve': 'Conductor {conductor}, {bad} bad prime{s}, root number {w:+d}, {torsion}.',
    # ---- the rank lower bound: points proved, or --r cited
    'lower': 'Rank at least {r}.',
    'lower_cited': 'Rank at least {r}, as given with --r. This is cited, not proved.',
    # ---- the time limit: the Slurm job's end, or --max-hours
    'slurm_end': 'This job ends at {end}, so rankbound will stop at {stop}.',
    'slurm_unknown': 'Could not read the end time of the Slurm job, so rankbound runs without a time limit.',
    'slurm_unknown_max': 'Could not read the end time of the Slurm job, so rankbound stops after {hours} as set by '
                         '--max-hours.',
    # ---- factoring the discriminant (its primes not given)
    'factor_ecm': 'Factoring a {digits}-digit number with GMP-ECM on {cores} for at most {time}{note}.',
    'factor_own': 'Factoring a {digits}-digit number for at most {time}, since GMP-ECM is not installed{note}.',
    'factor_found': 'Found {digits}-digit = {parts} after {time}.',
    'factor_next': 'The {digits}-digit number did not split in {time}, so GMP-ECM runs next.',
    'factor_skip': 'Curve {name} is skipped. Its discriminant could not be factored in {time}. Give its primes with '
                   '--bad-primes or more time with --ecm-hours.',
    'factor_skip_ecm': 'Curve {name} is skipped. Its discriminant could not be factored in {time}. Give its primes with '
                       '--bad-primes, more time with --ecm-hours, or install GMP-ECM.',
    'factor_est': ' (the time --estimate gives it; --ecm-hours H for more)',
    'curve_named': 'Curve {name} is {ainvs}.',
    # ---- how the bound will be computed
    'how_sieve': 'Sieving for relations. This should take {about}.',
    'how_uncond': "Searching for relations up to Minkowski's bound for an unconditional result. This should take "
                  'about {time} on {cores}.',
    # ---- progress of the sieve
    'found': 'Found {n:,} relations so far in {core_hours}.',
    'solving': 'Solving the linear system of {n:,} relations. This should take about {time}.',
    'solving_first': 'Solving the linear system of {n:,} relations.',      # the first check: no estimate yet
    'last_check': 'The bound stopped improving, so one last check uses every relation.',
    'retune': 'Too few relations at these settings, so the sieve is retuned. The relations found so far are kept.',
    'sieved': 'Sieving is finished with {n:,} relations in {core_hours}.',
    'witness_start': 'Writing the witness from the relations of the last check.',
    'late_witness': 'Not enough time is left to write the witness. It is written first when you resume.',
    # ---- bounds (S the bound; method as in METHODS; hyp: unconditional or assuming the GRH)
    'bound': 'Rank at most {S}. Verification method: {method} ({hyp}).',
    'ct_time': 'The Cassels-Tate pairing did not finish in {seconds} seconds, so the bound stays at rank at most {S}.',
    'ct_fail': 'The Cassels-Tate pairing could not be computed, so the bound stays at rank at most {S}. {reason}',
    # ---- the end (checked: " (checked)" when the checker accepted the witness, empty without PARI/GP)
    'end': '{rank}, {hyp}. Witness: {path}{checked}',
    'rejected': 'The checker rejected this witness at the check {check}. Please report this. {reason}',
    'unreadable': 'The checker cannot read the witness {path} ({reason}). If it was changed or damaged after it was '
                  'written, restore it, or delete {folder} and run the same command again; else please report this.',
    'done_before': 'Already done earlier. {rank}, {hyp}. Witness: {path}{checked}',
    'in_table': 'This curve is already in the table of witnesses. {rank}, {hyp}. Add --compute to compute it '
                'again.',
    # ---- stopping and resuming
    'stopped': 'Stopped at the time limit. Run the same command again to resume.',
    'resume': 'To resume: {command}',
    'best_so_far': 'The best checked bound so far: {rank}, {hyp}. Witness: {path}',
    'paused': 'Sieving paused at the time limit with {n:,} relations. Run the same command again to resume.',
    'late': 'Not enough time is left for the next check. It runs first when you resume.',
    'too_long': 'The last check did not finish in the {time} it had on {cores}. Give it more cores or more time.',
    'passed': 'The time limit has already passed, so there is nothing to do.',
    # ---- shards (--shard K/N)
    'shard_done': 'Shard {k} of {n} is done.',
    'shard_others': 'Shard {k} of {n} is done. {others} still running.',
    'shard_idle': 'Shard {k} is not running. Submit it again.',
    'other_sieved': 'Another process finished sieving, so this one writes the witness.',
    'other_done': 'Another process already finished this curve.',
    # ---- a list of curves (--curves)
    'list': '{n} curves, estimated {core_hours} in all.{skipped}',
    'list_failed': 'Curve {name} failed. {reason}',
    # ---- bound --estimate
    'est_sieve': '{name} has no rational 2-torsion. Sieving for relations should take {about}{wall}.',
    'est_uncond': "An unconditional bound for {name} needs Minkowski's bound {B}. This should take about {time} on "
                  '{cores}.',
    'est_uncond_far': "An unconditional bound for {name} needs Minkowski's bound {B}. This should take about {time} on "
                      '{cores}, beyond the {limit}-hour limit, so --uncond would refuse it.',
    # ---- errors: usage (exit 2)
    'no_curve': 'Give a curve, like 0,1,1,-2,0, or a list of curves with --curves.',
    'both': 'Give a curve or --curves, not both.',
    'one_per_command': 'Give one curve per command, or several with --curves FILE.',
    'ids_list': '--ids works with --curves only.',
    'bad_primes_empty': '--bad-primes is empty. Give the primes, like --bad-primes 2,3,37, or leave it out.',
    'no_points': 'The file {file} given with --points has no points. Give one "x y" per line, or leave it out.',
    'one_curve': '{option} works with one curve only.',
    'ids_empty': '--ids is empty. Leave it out to run the whole list.',
    'no_read': 'The file {file} given with {option} cannot be read ({reason}).',
    'est_takes': '--estimate only estimates the cost, so it takes no {option}.',
    'shard_takes': '--shard only sieves, so it takes no {option}.',
    'rel_uncond': 'Given relations give bounds assuming the GRH only. Drop --uncond or --relations.',
    'rel_other_curve': 'The witness {file} is for another curve (minimal model {curve}), so its relations are not for '
                       'this one.',
    'rel_other_form': 'The relations of the witness {file} are written for the index form {poly}, and this run uses '
                      '{ours}, so they cannot be read here. Give relations "a,b:hexprimes" for {ours}, or run without '
                      '--relations.',
    'rel_hard': 'Relation {k}{ab} has no primes, and |F(a, b)| ({digits} digits) does not factor quickly. Give '
                'its primes, as "a,b:hexprimes", or leave it out.',
    'rel_torsion': 'This curve has rational 2-torsion, so its bound comes from a 2-isogeny descent and not from '
                   'relations. Run it without --relations.',
    'shard_torsion': 'This curve has rational 2-torsion, so its bound comes from a 2-isogeny descent, which has '
                     'nothing to sieve. Run it without --shard.',
    'not_folder': '{folder} is not a rankbound work folder.',
    'no_status': 'There is no work folder ./{folder} here. Give one: rankbound status DIR.',
    # ---- errors: the work folder
    'other_run': 'This work folder was started for another curve, other points or other relations. Use another --dir.',
    'other_result': 'The result recorded in {folder}/run.json is not for the run recorded there. Use another --dir.',
    'other_grh': 'This work folder holds a bound assuming the GRH, and this run asks for an unconditional one. Use '
                 'another --dir.',
    'other_unc': 'This work folder holds an unconditional bound, and this run asks for one assuming the GRH. Use '
                 'another --dir.',
    'damaged': 'The file {file} is damaged. Delete it and {file2}, and that part is sieved again.',
    'damaged_run': 'The file {file} is damaged. Delete it and run the same command again; the other files of the folder '
                   'are kept, and used again where they are valid.',
    'busy': 'Another rankbound process is working in {folder}. Wait for it, or stop it.',
    # ---- errors: resources
    'disk': 'The last step needs more temporary disk than {folder} has free. Free some disk there, or set F2ELIM_TMPDIR '
            'to a folder with more.',
    'no_bound': 'The relations given ({rels}) give no bound: too few are left once those with a prime in no other relation are '
                'removed. Give more relations, or run without --relations to sieve.',
    'memory': 'The last step needs {gb:.1f} GB of memory, more than this machine has. Run it on a machine with more '
              'memory.',
    'uncond_far': "An unconditional bound needs Minkowski's bound {B}, about {time} on {cores}, beyond the {limit}-hour "
                  'limit. Run without --uncond or give more cores.',
    'ecm_bad': 'RANKBOUND_ECM={path} is not a program.',
    'no_start': 'Could not start {program}. {reason}',
    # ---- errors: before any work (preflight)
    'no_native': 'The native program {program} is missing or not executable. Build it with `make build` in the '
                 'source tree, or reinstall rankbound with pip.',
    'no_write': 'The work folder {folder} cannot be written ({reason}). Choose another with --dir.',
    'no_gp': 'PARI/GP (gp) is not on the PATH, so the witness will be written but not checked. Install PARI/GP and '
             'run the same command again to check it.',
    'old_gp': 'PARI/GP {version} ({path}) is older than 2.17, so the witness will be written but not checked. Install '
              'PARI/GP 2.17 or later and run the same command again to check it.',
    'odd_gp': 'The gp at {path} does not say its version, so the witness will be written but not checked. Install '
              'PARI/GP 2.17 or later and run the same command again to check it.',
    'disk_full': 'No space left on the device while writing {file}. What was saved is kept: free some space and run '
                 'the same command again to resume.',
    'failed': '{program} failed. {reason}',
    'too_few': 'The sieve finds too few relations for this curve, even at its largest setting. This curve is out of '
               'reach for now.',
    'poly_big': 'The cubic field\'s polynomial has a coefficient of {bits} bits, more than the sieve can take. This '
                'curve is out of reach for now.',
    # ---- errors: the given rank
    'r_high': 'The bound is rank at most {S}, but r = {r} was given. No witness was written. Check r.',
    'points_high': 'The bound is rank at most {S}, but the {n} points given are independent. No witness was written. '
                   'Please report this.',
    'cyclic': 'rankbound cannot prove the rank of this curve from points because its cubic field is cyclic. Give '
              'the rank with --r instead. It will be cited, not proved.',
    'chars_high': 'The characters show a larger class group than the relations allow. Please report this.',
    # ---- status (the table's last column: STATUS)
    'status_head': '{:<24} {:>12} {:>12} {:>10} {:>12}   {}',
    # ---- the stopping rules (a pending check, a check no run can do, the request budget)
    'pending_check': 'The check left from the last run runs first, on the same {n:,} relations.',
    'shard_pending': 'A check is waiting, so this shard does not sieve. Run the job that checks first.',
    'check_too_long': 'The next check needs about {need} on {cores}, more than the {window} a run has here. Run it '
                      'with more time (--max-hours) or more cores.',
    'budget_spent': 'The budget of {budget} for this curve is spent ({spent} so far), so sieving stops here. Give a '
                    'larger one with --budget to go on.',
    'budget_none': 'The budget of {budget} for this curve was spent before any check gave a bound. Give a larger one '
                   'with --budget to go on.',
}
METHODS = {'descent': '2-isogeny descent', 'selmer': '2-Selmer group', 'parity': '2-Selmer group and the root number',
           'ct': 'Cassels-Tate pairing'}
HYP = {True: 'assuming the GRH', False: 'unconditional'}
STATUS = ('not started', 'sieving', 'solving', 'finished', 'finished and checked')

# The name of the work-folder option in messages: the CLI's --dir, or the API's workdir= (set by runner.bound).
FOLDER_WORD = contextvars.ContextVar('rankbound_folder_word', default='--dir')


def text(key, **kw):
    """Return the message key filled with kw, naming the work-folder option as FOLDER_WORD says."""
    m = MESSAGES[key].format(**kw)
    w = FOLDER_WORD.get()
    return m if w == '--dir' else m.replace('with --dir', 'with ' + w + '=').replace('--dir', w)


WORDS = ('no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')


def count(n):
    """Return n in words below 10, else in digits."""
    return WORDS[n] if 0 <= n < 10 else '{:,}'.format(n)


def plural(n, what):
    """'1 core', '4 cores'."""
    return '%d %s%s' % (n, what, '' if n == 1 else 's')


def core_hours(h):
    """'1.6 core-hours', '0.00 core-hours', '120 core-hours'."""
    return ('%.2f' if h < 0.1 else '%.1f' if h < 100 else '%.0f') % h + ' core-hours'


def about(h):
    """An estimate of h core-hours in words: 'about 1.6 core-hours', or 'under a minute' below 0.01 core-hours."""
    return 'under a minute' if h < 0.01 else 'about ' + core_hours(h)


def duration(s):
    """A time in words: '40 seconds', '12 minutes', '2.0 hours', '3 days', '40 years'."""
    if s < 90:
        return plural(max(1, round(s)), 'second')
    if s < 5400:
        return plural(round(s / 60), 'minute')
    if s < 48 * 3600:
        h = s / 3600
        return ('%.1f hours' % h) if h < 10 else plural(round(h), 'hour')
    if s < 3.15e7:
        return plural(round(s / 86400), 'day')
    y = float('%.2g' % (s / 3.15e7))           # two digits: '40 years', '1,500 years', '1.2e12 years'
    return ('{:,} years'.format(round(y)) if y < 1e6 else big(round(y)) + ' years') if y >= 10 else '%g years' % y


def big(n):
    """A large number briefly: 2,048 or 1.2e9."""
    if n < 10 ** 6:
        return '{:,}'.format(n)
    e = len(str(int(n))) - 1
    return '%.1fe%d' % (n / 10 ** e, e)


def clock(t):
    """The time of day of the epoch time t: '5:36 PM'."""
    return time.strftime('%I:%M %p', time.localtime(t)).lstrip('0')


def conductor_text(N):
    """The conductor, exact when short, else ≈ 1eN."""
    N = str(N)
    return N if len(N) <= 15 else '≈ 1e%d' % (len(N) - 1)


def torsion_text(tors):
    """'no rational 2-torsion', or 'torsion Z/2 x Z/4' for a torsion subgroup with 2-torsion."""
    if not any(n % 2 == 0 for n in tors or []):
        return 'no rational 2-torsion'
    return 'torsion ' + ' x '.join('Z/%d' % n for n in tors)


def shown(path):
    """A path as the user sees it: relative to the current folder when inside it."""
    try:
        r = os.path.relpath(path)
    except ValueError:
        return path
    return path if r.startswith('..') else r


def rank_text(S, lower, proved):
    """'Rank 0', 'Rank 31' (a proved lower bound equals S), or 'Rank at most 31' (also when a cited r equals S)."""
    return 'Rank %d' % S if S == 0 or (proved and lower == S) else 'Rank at most %d' % S


MESSAGES.update({
    # ---- a Cassels-Tate pairing of the 2-isogeny descent left out, and why (run.log; descent.ct_allowance)
    'ct_skip': "The Cassels-Tate pairing on {curve} was not computed: {why}",
    'ct_skip_time': 'its time limit was reached ({detail}); the bound without it stands.',
    'ct_skip_cost': 'its cost, projected from its first pairs, is {detail}, above what is left of its allowance; the '
                    'bound without it stands.',
    'ct_skip_fail': 'its search failed ({detail}); the bound without it stands.',
    # ---- a stop's attempted bound, the final stops (core.FINAL)
    'attempted': 'Written but not yet checked: {rank}, {hyp}. Witness: {path}. It is checked first when you resume.',
    'final_too_long': 'This curve needs a longer run than this one. Run the same command with a larger --max-hours or '
                      'more --cores to go on.',
    'final_budget': 'The core-hour budget of this curve is spent. Run the same command with a larger --budget to go on.',
    # ---- a work folder of an older version; a published witness changed since it was checked
    'old_folder': 'This work folder was started by an older version of rankbound, and this version cannot resume it. '
                  'Use a new --dir (the old folder is left as it is).',
    'witness_missing': 'The witness {path} is missing, so the bound is proved again.',
    'witness_changed': 'The witness {path} has changed since it was checked, so it is checked again.',
    # ---- estimates (an unknown time is not mentioned, and never counted as 0)
    'est_descent_unknown': '{name} has rational 2-torsion, so a 2-isogeny descent bounds its rank.',
    'list_unknown': '{n} curves: {known} estimated at {core_hours} in all.{skipped}',
    'list_none': '{n} curves.{skipped}',
})
# the status of a witness not yet checked, and of the final stops
STATES = {'pending': 'witness not yet checked', 'too_long': 'needs a longer run', 'budget': 'budget spent'}

# a stop by a signal; a final stop's sieve; a work folder shared by requests that differ only in r or its points
MESSAGES.update({
    'interrupted': 'Interrupted. Run the same command again to resume.',
    'sieve_stopped': 'Sieving stopped with {n:,} relations in {core_hours}.',
    'other_work': 'This work folder was started for another curve or other relations. Use another --dir.',
    'other_model': 'This work folder was started for this curve on another model, {model}. Give the curve as {model} '
                   'to resume it, or use another --dir.',
    'due_check': 'The work folder already has the {n:,} relations of the next check, so they are checked first, '
                 'before any sieving.',
    'budget_no_sieve': '--budget applies only to the sieve, which this curve does not use.',
    'budget_kept': 'This curve already has a budget of {budget}, and a resume only raises it, so --budget {given} is '
                   'not used.',
    'cores_clamped': 'More cores were asked for ({asked}) than this machine or job allows; it runs on {have}.',
    'cores_auto': 'Using {cores} ({why}). --cores N or RANKBOUND_CORES=N to change.',
    'setup': 'Work folder {dir}, {cores}{limit}{budget}.',
})
# where the cores come from when neither --cores nor RANKBOUND_CORES is set (machine.Cores.source)
CORES_WHY = {'performance': 'the performance cores of {total}', 'physical': 'the physical cores of {total} hardware threads',
             'slurm': 'the Slurm allocation', 'quota': 'the cgroup CPU quota',
             'affinity': 'the CPUs this process may use, of {total}', 'machine': 'the cores of this machine'}
