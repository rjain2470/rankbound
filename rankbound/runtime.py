"""What every route shares at run time: the budget, the work folder, artifacts, subprocesses and progress.

A request has one monotonic deadline; a stage's child budget never outlives its parent. When budgets end, the outermost
decides: the run's end raises Stopped (resume later), a stage's own limit TimeLimit (the run goes on). An artifact's
manifest is written last and is its commit, so data without a valid manifest is never reused. fcntl.flock locks and
renames hold across the nodes of a shared filesystem as on one machine.
"""
import contextlib, contextvars, ctypes, ctypes.util, fcntl, hashlib, json, logging, math, os, resource, signal, subprocess, \
    sys, tempfile, threading, time

from . import machine
from .core import Damaged, Locked, RankboundError, Stopped, TimeLimit, digest
from .messages import HYP, METHODS, clock, core_hours, conductor_text, count, duration, rank_text, shown, text, torsion_text

LOG = logging.getLogger('rankbound')                 # what the user sees
DETAIL = logging.getLogger('rankbound.detail')       # the work folder's run.log only
DETAIL.propagate = False


# ---------- progress: events rendered from messages.py's templates

def say(key, **kw):
    """Log the progress event key, keeping the event and values on the record."""
    LOG.info(text(key, **kw), extra={'event': key, 'values': kw})


def say_curve(N, nbad, w, tors):
    say('curve', conductor=conductor_text(N), bad=count(nbad), s='' if nbad == 1 else 's', w=w, torsion=torsion_text(tors))


def say_lower(r, proved):
    if r:
        say('lower' if proved else 'lower_cited', r=r)


def say_bound(S, method, grh):
    say('bound', S=S, method=METHODS[method], hyp=HYP[grh])


def detail(msg, *args):
    """Log a line to run.log only."""
    DETAIL.info(msg, *args)


# ---------- the budget

class Budget:
    """An absolute deadline on the monotonic clock, or none, and the exception raised when it is reached."""

    def __init__(self, seconds=None, parent=None, stop=None, label='', masks=False):
        self.own = None if seconds is None else time.monotonic() + max(0.0, float(seconds))
        self.parent = parent
        self.stop = stop
        self.label = label
        self.seconds = seconds
        self.masks = masks          # any end reached through this budget raises its own stop()
        ends = [e for e in (self.own, parent.end if parent is not None else None) if e is not None]
        self.end = min(ends) if ends else None

    @classmethod
    def for_run(cls, *, seconds=None, workdir=''):
        """Return the budget of a run that stops in seconds (None: no limit)."""
        return cls(seconds, stop=lambda: Stopped(workdir), label='run')

    def child(self, seconds=None, stop=None, label='', masks=False):
        """Return a stage's budget of seconds, never past this one's end."""
        return Budget(seconds, self, stop, label, masks)

    def left(self):
        """Return the seconds left (>= 0), or None without a limit."""
        return None if self.end is None else max(0.0, self.end - time.monotonic())

    def wall(self):
        """Return the end as a wall-clock time, for a program, or None."""
        return None if self.end is None else time.time() + (self.end - time.monotonic())

    def expired(self):
        return self.end is not None and time.monotonic() >= self.end

    def check(self):
        """Raise the budget's exception if its end has passed."""
        if self.end is not None and time.monotonic() >= self.end:
            raise self.error()

    def limit(self, seconds):
        """Return seconds cut to what is left (at least 1), as a program's time limit; raise if the end passed."""
        self.check()
        left = self.left()
        return seconds if left is None else max(1.0, left if seconds is None else min(seconds, left))

    def error(self):
        """Return the exception of the outermost budget whose own end has passed."""
        if self.masks and self.stop is not None:
            return self.stop()
        chain, b = [], self
        while b is not None:
            chain.append(b)
            b = b.parent
        now = time.monotonic()
        for b in reversed(chain):
            if b.own is not None and now >= b.own:
                return b.stop() if b.stop else TimeLimit('The time limit was reached.', b.seconds or 0)
        return TimeLimit('The time limit was reached.', self.seconds or 0)


JOB_MARGIN = 300        # seconds kept free before a Slurm job's end


def run_seconds(max_hours=None, job=None, env=None, quiet=False):
    """Return the seconds until the run stops: the earlier of max_hours and the Slurm job's end less JOB_MARGIN."""
    env = os.environ if env is None else env
    if job is None:
        job = bool(env.get('SLURM_JOB_ID', '').strip())
    end = machine.job_end(env) if job else None
    ends = [] if max_hours is None else [time.time() + max_hours * 3600]
    if end is not None:
        if not quiet:
            say('slurm_end', end=clock(end), stop=clock(min([end - JOB_MARGIN] + ends)))
        ends.append(end - JOB_MARGIN)
    elif job and not quiet:
        if max_hours is None:
            say('slurm_unknown')
        else:
            say('slurm_unknown_max', hours=duration(max_hours * 3600))
    return min(ends) - time.time() if ends else None


UNLIMITED = Budget()
_CURRENT = contextvars.ContextVar('rankbound_budget', default=UNLIMITED)


def current():
    """Return the budget of the code running now (UNLIMITED outside scope())."""
    return _CURRENT.get()


@contextlib.contextmanager
def scope(budget):
    """Make budget the current one while this runs, restoring the enclosing one after."""
    tok = _CURRENT.set(budget)
    try:
        yield budget
    finally:
        _CURRENT.reset(tok)


def check():
    """Check the current budget; the prover's loops call this through arith.tick."""
    b = _CURRENT.get()
    if b.end is not None and time.monotonic() >= b.end:
        raise b.error()


class Interrupted(KeyboardInterrupt):
    """SIGTERM raised like KeyboardInterrupt, so that every cleanup runs."""
    signum = signal.SIGTERM


_TERM = {'depth': 0, 'fired': False}


@contextlib.contextmanager
def interruptible():
    """Make SIGTERM raise Interrupted in the main thread; a second one during cleanup is ignored."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_term(*_):
        if not _TERM['fired']:
            _TERM['fired'] = True
            raise Interrupted()
    old = signal.signal(signal.SIGTERM, on_term)
    _TERM['depth'] += 1
    try:
        yield
    finally:
        _TERM['depth'] -= 1
        if not _TERM['depth']:
            _TERM['fired'] = False
        signal.signal(signal.SIGTERM, old)


@contextlib.contextmanager
def alarm(budget=None, on_fire=None):
    """Raise the budget's exception by SIGALRM in code that does not poll; main thread only."""
    b = current() if budget is None else budget
    if b.end is None or threading.current_thread() is not threading.main_thread():
        yield
        return
    old = signal.getsignal(signal.SIGALRM)
    old_left = signal.getitimer(signal.ITIMER_REAL)[0]
    old_end = time.monotonic() + old_left if old_left > 0 else None

    def fire(*a):
        if old_end is not None and old_end <= b.end and callable(old):
            return old(*a)          # the enclosing alarm is due first
        if on_fire:
            on_fire()
        raise b.error()
    signal.signal(signal.SIGALRM, fire)
    end = b.end if old_end is None else min(b.end, old_end)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, end - time.monotonic()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        if old_end is not None:
            signal.setitimer(signal.ITIMER_REAL, max(0.01, old_end - time.monotonic()))


def run_budget(budget=None):
    """Return the first-ending run budget among budget and its ancestors, or None."""
    b, best = current() if budget is None else budget, None
    while b is not None:
        if b.label == 'run' and b.own is not None and (best is None or b.own < best.own):
            best = b
        b = b.parent
    return best


def stopped(budget=None, slack=1.0):
    """Return whether the run's end has passed, within slack seconds.

    A stage that failed then was stopped by the run's end, not by its own fault.
    """
    r = run_budget(budget)
    return r is not None and time.monotonic() >= r.own - slack


def raise_if_stopped(budget=None, slack=1.0):
    """Raise the run's Stopped if its end has passed."""
    if stopped(budget, slack):
        raise run_budget(budget).stop()


@contextlib.contextmanager
def limited(seconds=None, error=None):
    """Cut the current budget to seconds while this runs; any end reached inside raises error()."""
    with scope(current().child(seconds, stop=error, masks=error is not None)) as b:
        yield b


# ---------- files

def write_json(path, obj, indent=1):
    """Write obj to path atomically and durably."""
    write_bytes(path, (json.dumps(obj, indent=indent) + ('\n' if indent is not None else '')).encode())


def write_bytes(path, data):
    """Write the bytes data to path atomically and durably."""
    with atomic_file(path) as f:
        f.write(data)


@contextlib.contextmanager
def atomic_file(path):
    """Yield a temporary file that is fsynced and renamed onto path at the end, or removed on error."""
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix='.' + os.path.basename(path) + '.', suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'wb') as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    sync_dir(d)


def read_json(path, *default):
    """Return the JSON file at path, or the default if given and the file does not exist."""
    if default and not os.path.exists(path):
        return default[0]
    with open(path) as f:
        return json.load(f)


def sync_dir(d):
    with contextlib.suppress(OSError):
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def sha256_file(path, out=None):
    """Return the hex sha256 and size of path, copying the bytes read to out if given."""
    h, n = hashlib.sha256(), 0
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 16), b''):
            h.update(b)
            n += len(b)
            if out is not None:
                out.write(b)
    return h.hexdigest(), n


# ---------- locks

def lock_fd(path, shared=False, wait=0.0, busy=None):
    """Lock the file path and return the fd; raise Locked(busy) after wait seconds.

    With flock, two fds of one process exclude each other too.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    end = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= end:
                os.close(fd)
                raise Locked(busy or 'Another rankbound process holds %s.' % path)
            time.sleep(0.05 if wait <= 5 else 0.25)


@contextlib.contextmanager
def lock(path, shared=False, wait=0.0, busy=None):
    """Hold the lock of lock_fd while this runs."""
    fd = lock_fd(path, shared, wait, busy)
    try:
        yield fd
    finally:
        os.close(fd)


class Mismatch(RankboundError):
    """The work folder holds another request."""

    def __init__(self, msg, old=None, new=None):
        super().__init__(msg)
        self.old, self.new = old, new


RESULT_ONLY = ('lower',)     # the parts of a request's identity that bind only its result


def same_work(old, new):
    """Return whether old and new differ only in RESULT_ONLY, so they share the folder's sieve work."""
    if not (isinstance(old, dict) and isinstance(new, dict) and 'schema' in old and 'schema' in new):
        return False
    return {k: v for k, v in old.items() if k not in RESULT_ONLY} == {k: v for k, v in new.items() if k not in RESULT_ONLY}


def mismatch(old, new):
    """Return the Mismatch for a folder holding old, asked for new."""
    if isinstance(new, dict) and 'schema' in new and not (isinstance(old, dict) and 'schema' in old):
        return Mismatch(text('old_folder'), old, new)
    if isinstance(old, dict) and isinstance(new, dict) and 'GRH' in old and dict(old, GRH=new.get('GRH')) == new:
        return Mismatch(text('other_grh' if old['GRH'] else 'other_unc'), old, new)
    if isinstance(old, dict) and isinstance(new, dict) and 'hypothesis' in old and \
            dict(old, hypothesis=new.get('hypothesis')) == new:
        return Mismatch(text('other_grh' if old['hypothesis'] == 'GRH' else 'other_unc'), old, new)
    if isinstance(old, dict) and isinstance(new, dict) and isinstance(old.get('curve'), list) and \
            old.get('minimal_model') is not None and old.get('minimal_model') == new.get('minimal_model') and \
            old.get('curve') != new.get('curve') and \
            {k: v for k, v in old.items() if k not in ('curve', 'model_map', 'lower')} == \
            {k: v for k, v in new.items() if k not in ('curve', 'model_map', 'lower')}:
        return Mismatch(text('other_model', model=','.join(map(str, old['curve']))), old, new)
    return Mismatch(text('other_work'), old, new)


class RunStore:
    """The work folder's run.json: the request's identity, the run's progress and its result.

    Every change is a read-modify-write under key.lock, held for milliseconds; runs and stages are owned through own(),
    never through key.lock.
    """
    META_WAIT = 120.0

    def __init__(self, workdir):
        self.dir = os.path.abspath(workdir)

    def path(self, name):
        return os.path.join(self.dir, name)

    @contextlib.contextmanager
    def meta(self):
        """Hold key.lock while run.json is read, checked and written."""
        with lock(self.path('key.lock'), wait=self.META_WAIT, busy=text('busy', folder=self.dir)):
            yield

    def info(self):
        """Return run.json ({} before the run starts); raise Damaged if it is damaged."""
        try:
            d = read_json(self.path('run.json'), {})
        except ValueError:
            raise Damaged(text('damaged_run', file=self.path('run.json')), self.path('run.json'))
        return d if isinstance(d, dict) else {}

    def _write(self, d):
        write_json(self.path('run.json'), d)
        return d

    def begin(self, key, rebind=False, **init):
        """Bind the folder to key, or check that it is bound to it, and return run.json.

        A key differing only in RESULT_ONLY shares the work; with rebind it takes the folder and the old result is
        dropped, since a result is never reused for another r. Any other key raises Mismatch.
        """
        os.makedirs(self.dir, exist_ok=True)
        with self.meta():
            d = self.info()
            old = d.get('key')
            if old is None:
                return self._write(dict(init, key=key))
            if old != key:
                if not same_work(old, key):
                    raise mismatch(old, key)
                if rebind:
                    d.update(key=key, result=None, final=None, rank_bound=None, lower_bound=init.get('lower_bound'))
                    d.pop('result_key', None)
                    d = self._write(d)
            return d

    def update(self, key=None, **kw):
        """Update run.json with kw if still bound to key (None: any), and return it."""
        with self.meta():
            d = self.info()
            if key is not None and d.get('key') != key:
                raise mismatch(d.get('key'), key)
            d.update(kw)
            return self._write(d)

    def stored(self, key=None, d=None):
        """Return the result in run.json (or d) if it is key's, else None; raise if it is another run's."""
        d = self.info() if d is None else d
        res = d.get('result')
        if not res:
            return None
        ok = d.get('key') is not None and (key is None or d['key'] == key)
        if 'result_key' in d:
            ok = ok and d['result_key'] == d['key']
        else:
            ok = ok and res.get('name') == d.get('name')
        if not ok:
            raise RankboundError(text('other_result', folder=self.dir))
        return res

    def publish(self, key, result, **kw):
        """Record result as key's if run.json is still bound to key."""
        with self.meta():
            d = self.info()
            if d.get('key') is None or (key is not None and d['key'] != key):
                raise RankboundError(text('other_run'))
            d.update(kw, result=result, result_key=d['key'])
            return self._write(d)

    @contextlib.contextmanager
    def own(self, stage, shared=False, wait=0.0):
        """Own the stage while this runs, exclusively or shared."""
        with lock(self.path(stage + '.lock'), shared=shared, wait=wait, busy=text('busy', folder=self.dir)):
            yield

    # ---- the request budget, across resumes

    def spent(self, d=None):
        """Return the core-hours, wall seconds and runs spent by this request so far."""
        d = self.info() if d is None else d
        sp = d.get('spent') or {}
        return {'core_hours': float(sp.get('core_hours') or 0.0), 'wall_s': float(sp.get('wall_s') or 0.0),
                'runs': int(sp.get('runs') or 0)}

    def add_spent(self, core_hours=0.0, wall_s=0.0, runs=0):
        """Add to the spending in one locked step and return the new total."""
        with self.meta():
            d = self.info()
            sp = self.spent(d)
            sp = {'core_hours': round(sp['core_hours'] + core_hours, 6), 'wall_s': round(sp['wall_s'] + wall_s, 3),
                  'runs': sp['runs'] + runs}
            d['spent'] = sp
            self._write(d)
            return sp

    def set_budget(self, estimate, given=None):
        """Set and return the request's core-hour budget: given, else BUDGET_FACTOR times the estimate.

        A resume only raises it to a larger given one. None means no cap.
        """
        with self.meta():
            d = self.info()
            old = d.get('budget_core_hours')
            new = old
            if given is not None and (old is None or given > old):
                new = float(given)
            elif given is not None and given < old:
                say('budget_kept', budget=core_hours(old), given=core_hours(given))
            elif old is None and estimate:
                new = round(max(BUDGET_FACTOR * float(estimate), BUDGET_FLOOR), 6)
            if new != old:
                d['budget_core_hours'] = new
                self._write(d)
            return new


# ---------- the work folder of every route: run.json

class Folder:
    """A work folder DIR/NAME, the base class of each route; runner.execute alone drives its lifecycle."""
    method = ''
    route = ''

    def __init__(self, workdir):
        self.dir = os.path.abspath(workdir)     # the sievers run in other folders
        self.store = RunStore(self.dir)

    def path(self, name):
        return os.path.join(self.dir, name)

    key = None          # this process's request key, once begin has checked or written it

    def info(self):
        return self.store.info()

    def meta(self):
        """Hold key.lock while run.json is read, checked and written."""
        return self.store.meta()

    def note(self, **kw):
        """Update run.json with kw."""
        return self.store.update(None, **kw)

    def stored(self, d=None):
        """Return the result in run.json (or d) for this process's request."""
        return self.store.stored(self.key, d)

    def earlier(self):
        """Return the result of a finished run, reported again, or None."""
        res = self.stored()
        if res:
            say('done_before', rank=rank_text(res['rank_bound'], res['rank_lower_bound'], res['r_proved']),
                hyp='assuming the GRH' if res.get('GRH') else 'unconditionally', path=shown(res['witness']),
                checked=' (checked)' if res['checked'] else '')
            return dict(res, earlier=True)
        return None

    @contextlib.contextmanager
    def logging(self):
        """Send the inner steps to run.log while this runs."""
        os.makedirs(self.dir, exist_ok=True)
        h = logging.FileHandler(self.path('run.log'))
        h.setFormatter(logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        DETAIL.addHandler(h)
        DETAIL.setLevel(logging.INFO)
        try:
            yield
        finally:
            DETAIL.removeHandler(h)
            h.close()

    def precheck(self, curve):
        """Refuse the curve before any work (ValueError, RankboundError)."""


BUDGET_FACTOR = 3.0     # the default request budget, times the estimate
BUDGET_FLOOR = 1.0      # the least default budget, in core-hours


def final_stop(workdir, outcome, reason):
    """Return core.Stopped for a final stop without a bound to certify."""
    return Stopped(workdir, outcome=outcome, reason=reason)


class Meter:
    """The CPU time of one run of the request, this process's and its children's, not cores x wall time.

    CPU time is the unit of the sieve's estimate. A thread adds it to run.json every EVERY seconds.
    """
    EVERY = 600.0

    def __init__(self, store, cpu=None):
        self.store, self.cpu = store, cpu or cpu_seconds
        self.saved = time.monotonic()
        self.cpu_saved = self.cpu()
        self.base = store.spent()['core_hours']         # spent by earlier and concurrent runs, at the last read
        self.budget = store.info().get('budget_core_hours')
        self.closed = False
        self.lock = threading.Lock()
        self.done = threading.Event()
        threading.Thread(target=self._every, daemon=True).start()

    def _every(self):
        while not self.done.wait(self.EVERY):
            with contextlib.suppress(OSError, RankboundError):
                self.tick()

    def core_hours(self):
        """Return the request's core-hours, including this run's unwritten time."""
        return self.base + (self.cpu() - self.cpu_saved) / 3600

    def flush(self, runs=0):
        with self.lock:
            now, cpu = time.monotonic(), self.cpu()
            dt, self.saved = now - self.saved, now
            dc, self.cpu_saved = cpu - self.cpu_saved, cpu
            sp = self.store.add_spent(dc / 3600, dt, runs)
            self.base = sp['core_hours']
            self.budget = self.store.info().get('budget_core_hours')
            return sp

    def tick(self):
        if time.monotonic() - self.saved >= self.EVERY:
            self.flush()

    def over(self):
        """Return whether the request budget is spent."""
        self.tick()
        return self.budget is not None and self.core_hours() >= self.budget

    def close(self):
        """Write this run's spending, once, and count the run."""
        if not self.closed:
            self.closed = True
            self.done.set()
            self.flush(runs=1)


_METERS = {}            # work folder -> this process's Meter in it


def spent_now(store):
    """Return the recorded spending plus this process's unwritten time."""
    m = _METERS.get(store.dir)
    if m is not None and not m.closed:
        return m.flush()
    return store.spent()


@contextlib.contextmanager
def metered(store):
    """Yield this process's Meter for store's folder; nested calls share it, so nothing is counted twice."""
    m = _METERS.get(store.dir)
    if m is not None and not m.closed:
        yield m
        return
    m = _METERS[store.dir] = Meter(store)
    try:
        yield m
    finally:
        _METERS.pop(store.dir, None)
        m.close()


# ---------- artifacts: data first, the manifest last

MANIFEST = 'rankbound-manifest-1'


def manifest_path(folder, name):
    return os.path.join(folder, name + '.manifest.json')


class Transaction:
    """The files of one artifact being written, committed with the manifest by commit()."""

    def __init__(self, folder, name, schema, fingerprint, counts=None):
        self.folder, self.name, self.schema, self.fingerprint = os.path.abspath(folder), name, schema, fingerprint
        self.counts = dict(counts or {})
        self.files = {}             # file name -> its temporary path
        self.adopted = []           # files already in place, written atomically by their owner
        os.makedirs(self.folder, exist_ok=True)

    def path(self, fname):
        """Return the temporary path for the artifact's file fname."""
        if fname != os.path.basename(fname) or fname.endswith('.manifest.json'):
            raise ValueError('An artifact file is a plain name: %r.' % fname)
        if fname not in self.files:
            fd, tmp = tempfile.mkstemp(prefix='.%s.' % fname, suffix='.part', dir=self.folder)
            os.close(fd)
            self.files[fname] = tmp
        return self.files[fname]

    def adopt(self, fname):
        """Record the file fname, already in place, in the manifest."""
        if fname != os.path.basename(fname) or not os.path.isfile(os.path.join(self.folder, fname)):
            raise ValueError('No file %r in %s to record.' % (fname, self.folder))
        self.adopted.append(fname)

    def write(self, fname, data):
        """Write data (bytes or str) as the file fname."""
        with open(self.path(fname), 'wb') as f:
            f.write(data.encode() if isinstance(data, str) else data)

    def commit(self):
        """Make the files durable, remove the old manifest, rename the files into place and write the manifest."""
        entries = {}
        for fname, tmp in sorted(self.files.items()):
            with open(tmp, 'rb+') as f:
                os.fsync(f.fileno())
            h, n = sha256_file(tmp)
            entries[fname] = {'bytes': n, 'sha256': h}
        for fname in self.adopted:
            p = os.path.join(self.folder, fname)
            with open(p, 'rb') as f:
                os.fsync(f.fileno())
            h, n = sha256_file(p)
            entries[fname] = {'bytes': n, 'sha256': h}
        mp = manifest_path(self.folder, self.name)
        if os.path.exists(mp):
            os.remove(mp)       # until the new manifest, the artifact is absent, never mixed
            sync_dir(self.folder)
        for fname, tmp in sorted(self.files.items()):
            os.replace(tmp, os.path.join(self.folder, fname))
        sync_dir(self.folder)
        m = {'format': MANIFEST, 'name': self.name, 'schema': self.schema, 'fingerprint': self.fingerprint,
             'counts': self.counts, 'files': entries, 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        write_json(mp, m)
        self.files = {}
        return m

    def abort(self):
        for tmp in self.files.values():
            with contextlib.suppress(OSError):
                os.remove(tmp)
        self.files = {}


@contextlib.contextmanager
def transaction(folder, name, schema, fingerprint, counts=None):
    """Yield a Transaction, committed when the block ends and removed if it raises."""
    tx = Transaction(folder, name, schema, fingerprint, counts)
    try:
        yield tx
        tx.commit()
    except BaseException:
        tx.abort()
        raise


def inspect(folder, name, schema=None, fingerprint=None, digests=True):
    """Return (state, manifest), state 'valid', 'missing', 'stale' or 'damaged'."""
    mp = manifest_path(folder, name)
    try:
        with open(mp) as f:
            m = json.load(f)
    except FileNotFoundError:
        return 'missing', None
    except (OSError, ValueError):
        return 'damaged', None
    try:
        if m.get('format') != MANIFEST or m.get('name') != name or not isinstance(m.get('files'), dict):
            return 'damaged', m
        for fname, e in m['files'].items():
            p = os.path.join(folder, fname)
            if os.path.getsize(p) != e['bytes'] or digests and sha256_file(p)[0] != e['sha256']:
                return 'damaged', m
    except (OSError, KeyError, TypeError, AttributeError):
        return 'damaged', m
    if schema is not None and m.get('schema') != schema or fingerprint is not None and m.get('fingerprint') != fingerprint:
        return 'stale', m
    return 'valid', m


def load(folder, name, schema=None, fingerprint=None, digests=True):
    """Return the artifact's manifest if valid, else None; a damaged manifest is set aside."""
    state, m = inspect(folder, name, schema, fingerprint, digests)
    if state == 'damaged':
        mp = manifest_path(folder, name)
        with contextlib.suppress(OSError):
            os.replace(mp, mp + '.damaged')
        detail('artifact %s in %s is damaged: rebuilt', name, folder)
    return m if state == 'valid' else None


def invalidate(folder, name):
    """Remove the artifact's manifest, so its files no longer count."""
    with contextlib.suppress(FileNotFoundError):
        os.remove(manifest_path(folder, name))
        sync_dir(folder)


def fingerprint(**inputs):
    """Return the digest of every mathematical input and format convention of an artifact."""
    return digest(inputs)


# ---------- subprocesses: started in their own process group, always reaped

_CHILDREN = set()
_CHILD_LOCK = threading.Lock()


def cpu_seconds():
    """Return the CPU seconds of this process and its children, running ones included where visible."""
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    with _CHILD_LOCK:
        live = [p.pid for p in _CHILDREN if p.returncode is None]
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime + sum(_live_cpu(pid) for pid in live)


_TICK = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') and 'SC_CLK_TCK' in os.sysconf_names else 100
_LIBPROC = []           # macOS: [proc_pid_rusage, seconds per unit of its times]


def _live_cpu(pid):
    """Return the CPU seconds of the running child pid and its reaped children, or 0 where unknown."""
    try:
        if sys.platform.startswith('linux'):
            with open('/proc/%d/stat' % pid, 'rb') as f:
                st = f.read()
            utime, stime, cutime, cstime = (int(x) for x in st[st.rindex(b')') + 2:].split()[11:15])
            return (utime + stime + cutime + cstime) / _TICK
        if sys.platform == 'darwin':
            if not _LIBPROC:
                lib = ctypes.CDLL(ctypes.util.find_library('proc') or '/usr/lib/libproc.dylib')

                class Timebase(ctypes.Structure):
                    _fields_ = [('numer', ctypes.c_uint32), ('denom', ctypes.c_uint32)]
                tb = Timebase()
                ctypes.CDLL(None).mach_timebase_info(ctypes.byref(tb))
                # The times are in kernel ticks: the timebase's, or 125/3 ns on Apple silicon (also under Rosetta,
                # whose timebase says 1/1).  Pick the one this process's own times match.
                info = (ctypes.c_uint64 * 20)()
                me = resource.getrusage(resource.RUSAGE_SELF)
                if lib.proc_pid_rusage(os.getpid(), 0, info) != 0 or not info[2] + info[3]:
                    raise OSError('proc_pid_rusage')
                ratio = (me.ru_utime + me.ru_stime) / (info[2] + info[3])
                _LIBPROC.extend([lib.proc_pid_rusage, min((tb.numer / tb.denom / 1e9, 125 / 3 / 1e9),
                                                          key=lambda x: abs(math.log(x / ratio)))])
            info = (ctypes.c_uint64 * 20)()         # rusage_info_v0: a 16-byte uuid, then user and system time
            if _LIBPROC[0](pid, 0, info) != 0:
                return 0.0
            return (info[2] + info[3]) * _LIBPROC[1]
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    return 0.0


# The native programs' internal variables are set per call, so an inherited one could change a run.  clean_environ()
# drops them at a request's start, except NATIVE_OVERRIDES (for tests and measurements).
NATIVE_ENV = ('F2ELIM_', 'F2KERNEL_', 'RANKBOUND_PARENT')
NATIVE_OVERRIDES = frozenset(('F2KERNEL_DENSE_N', 'F2KERNEL_PAR_WORK', 'F2ELIM_TMPDIR', 'F2ELIM_TMP_MB',
                              'F2ELIM_BLOCK_WORDS', 'F2ELIM_BLOCK_MB', 'F2ELIM_DENSE_MB'))


def clean_environ(env=None):
    """Remove the native programs' internal variables from env and return their names."""
    env = os.environ if env is None else env
    gone = sorted(k for k in env if k.startswith(NATIVE_ENV) and k not in NATIVE_OVERRIDES)
    for k in gone:
        del env[k]
    return gone


def start(cmd, **kw):
    """Start cmd in a new process group, so that terminate() reaches its children."""
    p = subprocess.Popen(cmd, start_new_session=True, **kw)
    with _CHILD_LOCK:
        _CHILDREN.add(p)
    return p


def terminate(p, grace=5.0):
    """Stop p and its group with SIGTERM, then SIGKILL after grace seconds, and reap it."""
    try:
        if p.poll() is None:
            for sig, t in ((signal.SIGTERM, grace), (signal.SIGKILL, None)):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(p.pid, sig)
                try:
                    p.wait(t)
                    break
                except subprocess.TimeoutExpired:
                    continue
        else:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, signal.SIGKILL)       # the group's leftovers
    finally:
        with _CHILD_LOCK:
            _CHILDREN.discard(p)


def reap(p):
    """Forget the finished process p."""
    with _CHILD_LOCK:
        _CHILDREN.discard(p)


def run(cmd, budget=None, input=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=None, cwd=None, grace=5.0,
        text=True, deadline_env=None, slack=0.0):
    """Run cmd within the budget and return (returncode, stdout, stderr).

    At the budget's end or an interruption, the program's group is stopped and reaped. A program told its deadline
    through deadline_env is stopped only slack seconds after it.
    """
    b = current() if budget is None else budget
    b.check()
    if deadline_env is not None:
        env = dict(os.environ if env is None else env)
        env.pop(deadline_env, None)
        if b.end is not None:
            env[deadline_env] = '%.3f' % b.wall()
    p = start(cmd, stdin=subprocess.PIPE if input is not None else None, stdout=stdout, stderr=stderr, env=env,
              cwd=cwd, text=text)
    try:
        left = b.left()
        with interruptible():           # SIGTERM does not reach the program's own group, so stop it here
            out, err = p.communicate(input, timeout=None if left is None else left + slack)
    except BaseException as e:
        terminate(p, grace)
        for f in (p.stdin, p.stdout, p.stderr):
            if f is not None:
                with contextlib.suppress(OSError):
                    f.close()
        if isinstance(e, subprocess.TimeoutExpired):
            raise b.error()
        raise
    reap(p)
    return p.returncode, out, err
