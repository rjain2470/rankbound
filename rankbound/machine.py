"""The resources of this machine or job: cores, memory and a Slurm job's end time.

The prover asks only here; the checker does not import this module.  A run's cores are --cores N, else
RANKBOUND_CORES, else detect(): the smallest of the sources that apply, namely
- a Slurm job's allocation (SLURM_CPUS_PER_TASK, else SLURM_CPUS_ON_NODE);
- outside Slurm, the performance cores of a Mac (hw.perflevel0.physicalcpu, else hw.physicalcpu), or on Linux
  the CPUs of the process's affinity, only the performance cores of a hybrid CPU, one per physical core;
- on Linux, the affinity and a cgroup CPU quota, also inside Slurm;
- else os.cpu_count().
"""
import math, os, subprocess, sys
from typing import Callable, Iterable, List, Mapping, NamedTuple, Optional, Set, Tuple

ENV = 'RANKBOUND_CORES'
Sysctl = Callable[[str], Optional[int]]


class Cores(NamedTuple):
    """The cores of a run.

    source is 'option', 'environment', or a detected source: 'slurm', 'quota', 'performance', 'physical',
    'affinity' or 'machine'.  total is the machine's logical CPU count, for messages.
    """
    n: int
    source: str
    total: Optional[int] = None


def cores(given: Optional[int] = None, env: Optional[Mapping[str, str]] = None) -> Cores:
    """Return given, else RANKBOUND_CORES (ValueError unless a positive integer), else detect()."""
    if given is not None:
        return Cores(given, 'option')
    env = os.environ if env is None else env
    v = env.get(ENV, '').strip()
    if v:
        if not v.isdigit() or int(v) < 1:
            raise ValueError('%s=%s is not a positive integer.' % (ENV, v))
        return Cores(int(v), 'environment')
    return detect(env)


def detect(env: Optional[Mapping[str, str]] = None, system: Optional[str] = None, root: str = '/',
           sysctl: Optional[Sysctl] = None, affinity: Optional[Set[int]] = None,
           cpu_count: Optional[int] = None) -> Cores:
    """Return the smallest core count of the sources that apply (see the module docstring).

    A tie goes to the first of slurm, quota, the host's own count, affinity.  The arguments stand in for the
    real system (root is where /proc and /sys are) and default to it.
    """
    env = os.environ if env is None else env
    system = sys.platform if system is None else system
    sysctl = _sysctl if sysctl is None else sysctl
    aff = _affinity() if affinity is None else affinity
    mac, linux = system == 'darwin', system.startswith('linux')
    total = (sysctl('hw.logicalcpu') if mac else None) or (os.cpu_count() if cpu_count is None else cpu_count)
    found: List[Tuple[int, str]] = []
    slurm = slurm_cpus(env)
    if slurm:
        found.append((slurm, 'slurm'))
    if linux:
        q = cgroup_quota(root)
        if q:
            found.append((q, 'quota'))
    if not slurm:
        if mac:
            found += mac_cores(sysctl, total)
        elif linux:
            found += linux_cores(root, aff, total)
    if aff and (slurm or not linux):
        found.append((len(aff), 'affinity' if total and len(aff) < total else 'machine'))
    if not found:
        found.append((total or 1, 'machine'))
    n, source = min(found, key=lambda f: f[0])
    return Cores(n, source, total or n)


def slurm_cpus(env: Mapping[str, str]) -> Optional[int]:
    """Return the Slurm task's CPUs, or None outside Slurm."""
    per, node = _positive(env.get('SLURM_CPUS_PER_TASK')), _positive(env.get('SLURM_CPUS_ON_NODE'))
    return min(per, node) if per and node else per or node


def mac_cores(sysctl: Sysctl, total: Optional[int]) -> List[Tuple[int, str]]:
    """Return the performance cores of Apple silicon, else the physical cores of an Intel Mac."""
    for name, source in (('hw.perflevel0.physicalcpu', 'performance'), ('hw.physicalcpu', 'physical')):
        n = sysctl(name)
        if n:
            return [(n, source if total and n < total else 'machine')]
    return []


def linux_cores(root: str, aff: Optional[Set[int]], total: Optional[int]) -> List[Tuple[int, str]]:
    """Return the affinity's CPUs on Linux, only performance cores of a hybrid CPU, one per physical core."""
    cpus = set(aff) if aff else set(range(total or 0))
    if not cpus:
        return []
    perf = cpu_list(_read(root, 'sys/devices/cpu_core/cpus'))
    mine = cpus & perf if perf and cpus & perf else cpus
    phys = physical_cores(root, mine)
    n = phys or len(mine)
    source = 'performance' if mine != cpus else 'physical' if n < len(mine) else \
        'affinity' if total and len(cpus) < total else 'machine'
    return [(n, source)]


def physical_cores(root: str, cpus: Iterable[int]) -> Optional[int]:
    """Return the number of physical cores among cpus, or None if the topology is unreadable."""
    cores_ = set()
    for c in cpus:
        sib = cpu_list(_read(root, 'sys/devices/system/cpu/cpu%d/topology/thread_siblings_list' % c))
        if not sib:
            return None
        cores_.add(min(sib))
    return len(cores_) or None


def cgroup_quota(root: str = '/') -> Optional[int]:
    """Return the smallest cgroup CPU quota (v2 or v1) of this process and its ancestors, rounded up, or None."""
    best = None
    for line in (_read(root, 'proc/self/cgroup') or '').splitlines():
        parts = line.split(':', 2)
        if len(parts) != 3:
            continue
        _, ctl, path = parts
        if ctl == '':
            mounts = ['sys/fs/cgroup']
        elif 'cpu' in ctl.split(','):
            mounts = ['sys/fs/cgroup/' + ctl, 'sys/fs/cgroup/cpu']
        else:
            continue
        for m in mounts:
            for d in _ancestors(path):
                q = _quota(root, m + d, v2=ctl == '')
                if q is not None:
                    best = q if best is None else min(best, q)
    return best


def _quota(root, d, v2):
    try:
        if v2:
            q, p = (_read(root, d + '/cpu.max') or '').split()[:2]
        else:
            q, p = _read(root, d + '/cpu.cfs_quota_us'), _read(root, d + '/cpu.cfs_period_us')
        q, p = int(q), int(p)
    except (ValueError, TypeError):
        return None
    return max(1, math.ceil(q / p)) if q > 0 and p > 0 else None


def _ancestors(path):
    """'/a/b' -> ['/a/b', '/a', '']: the cgroup, its parents, the mount's root."""
    parts = [x for x in path.strip().split('/') if x]
    return ['/' + '/'.join(parts[:i]) if i else '' for i in range(len(parts), -1, -1)]


def cpu_list(s: Optional[str]) -> Set[int]:
    """Linux's CPU list '0-3,8,10-11' as a set; empty if unreadable."""
    out: Set[int] = set()
    try:
        for part in (s or '').strip().split(','):
            if part:
                a, _, b = part.partition('-')
                out.update(range(int(a), int(b or a) + 1))
    except ValueError:
        return set()
    return out


def allowed_cores(env: Optional[Mapping[str, str]] = None, cpu_count: Optional[int] = None) -> Optional[int]:
    """Return the most cores this process may use, or None if unknown; larger requests are clamped to it.

    This is the fewer of the CPU affinity and SLURM_CPUS_ON_NODE, else (macOS) the logical CPU count.
    """
    env = os.environ if env is None else env
    aff = _affinity()
    caps = [c for c in (len(aff) if aff else None, _positive(env.get('SLURM_CPUS_ON_NODE'))) if c]
    if not caps:
        n = os.cpu_count() if cpu_count is None else cpu_count
        return n if n and n > 0 else None
    return min(caps)


def memory(env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """Return the job's memory in bytes: physical memory capped by Slurm's limit, or None if unreadable."""
    try:
        mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        return None
    e = os.environ if env is None else env
    try:
        node, cpu, cpus = (int(e.get(k) or 0) for k in ('SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU', 'SLURM_CPUS_ON_NODE'))
        if node > 0:
            mem = min(mem, node << 20)
        elif cpu > 0 and cpus > 0:
            mem = min(mem, (cpu * cpus) << 20)
    except ValueError:
        pass
    return mem


def job_end(env: Optional[Mapping[str, str]] = None, squeue: str = 'squeue') -> Optional[float]:
    """Return the Slurm job's end as epoch seconds, or None outside Slurm or without a time limit."""
    import datetime
    env = os.environ if env is None else env
    v = env.get('SLURM_JOB_END_TIME', '').strip()
    if v.isdigit() and int(v) > 0:
        return float(v)
    jid = env.get('SLURM_JOB_ID', '').strip()
    if not jid.isdigit():
        return None
    aj, at = env.get('SLURM_ARRAY_JOB_ID', '').strip(), env.get('SLURM_ARRAY_TASK_ID', '').strip()
    if aj.isdigit() and at.isdigit():
        jid = '%s_%s' % (aj, at)
    try:
        r = subprocess.run([squeue, '-h', '-j', jid, '-o', '%e'], capture_output=True, text=True, timeout=60,
                           env=dict(os.environ, SLURM_TIME_FORMAT='standard'))       # local time
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.split()
    try:
        if len(out) != 1:
            raise ValueError('%d lines' % len(out))
        return datetime.datetime.strptime(out[0], '%Y-%m-%dT%H:%M:%S').timestamp()
    except ValueError:
        return None


# ---------- the system's answers (stubbed in the tests)

def _positive(v: Optional[str]) -> Optional[int]:
    try:
        n = int((v or '').strip())
    except ValueError:
        return None
    return n if n > 0 else None


def _read(root: str, rel: str) -> Optional[str]:
    try:
        with open(os.path.join(root, rel)) as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def _affinity() -> Optional[Set[int]]:
    """Return the CPU ids this process may run on, or None where the system does not say."""
    if hasattr(os, 'sched_getaffinity'):
        try:
            return set(os.sched_getaffinity(0))
        except OSError:
            pass
    return None


def _sysctl(name: str) -> Optional[int]:
    """Return an integer macOS sysctl, or None if absent or unreadable."""
    if sys.platform != 'darwin':
        return None
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        buf, size = ctypes.create_string_buffer(8), ctypes.c_size_t(8)
        if libc.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
        v = int.from_bytes(buf.raw[:size.value], sys.byteorder)
    except (OSError, AttributeError, ValueError, TypeError):
        return None
    return v if v > 0 else None
