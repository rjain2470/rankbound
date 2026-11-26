"""machine.py: the cores of a run (--cores, RANKBOUND_CORES, then what the machine or job gives: Slurm, a cgroup CPU
quota, the performance cores of a Mac or a hybrid Intel CPU, the physical cores, the affinity), the most cores
allowed, the memory and the words of each source.  The systems are faked: environments, a /proc and /sys under a
temporary root, and a stubbed sysctl."""
import os, sys, tempfile, unittest
from unittest import mock

from rankbound import machine, messages


def sysctl(**values):
    """A stub of machine._sysctl: hw_logicalcpu=8 stands for hw.logicalcpu."""
    return lambda name: values.get(name.replace('.', '_'))


class FakeRoot:
    """A temporary root with the files of a Linux machine: CPUs and their SMT siblings, cgroups."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def write(self, rel, text):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(text + '\n')
        return self

    def smt(self, pairs):
        """Each CPU's thread_siblings_list from groups of CPUs that share a core."""
        for group in pairs:
            for c in group:
                self.write('sys/devices/system/cpu/cpu%d/topology/thread_siblings_list' % c, ','.join(map(str, group)))
        return self

    def close(self):
        self.tmp.cleanup()


class TestPrecedence(unittest.TestCase):
    def test_given_env_detected(self):
        """--cores N (given), then RANKBOUND_CORES, then the detection; an empty RANKBOUND_CORES is not set."""
        with mock.patch.object(machine, 'detect', lambda env=None: machine.Cores(6, 'performance', 8)):
            self.assertEqual(machine.cores(3, {'RANKBOUND_CORES': '5'}), (3, 'option', None))
            self.assertEqual(machine.cores(None, {'RANKBOUND_CORES': '5'}), (5, 'environment', None))
            self.assertEqual(machine.cores(None, {'RANKBOUND_CORES': ' '}), (6, 'performance', 8))
            self.assertEqual(machine.cores(None, {}), (6, 'performance', 8))
            for bad in ('0', '-1', 'four', '2.5'):
                with self.assertRaisesRegex(ValueError, '^RANKBOUND_CORES=%s is not a positive integer.$' % bad):
                    machine.cores(None, {'RANKBOUND_CORES': bad})

    def test_real(self):
        """On this machine: at least one core, no more than os.cpu_count(), a known source; the memory."""
        c = machine.detect({})
        self.assertTrue(1 <= c.n <= (os.cpu_count() or c.n), c)
        self.assertIn(c.source, messages.CORES_WHY)
        self.assertGreater(machine.memory({}) or 0, 1 << 20)
        if sys.platform == 'darwin':
            self.assertEqual(machine._sysctl('hw.logicalcpu'), os.cpu_count())
            self.assertIsNone(machine._sysctl('hw.no_such_name'))
        else:
            self.assertIsNone(machine._sysctl('hw.logicalcpu'))


class TestMac(unittest.TestCase):
    def detect(self, env=None, **values):
        return machine.detect(env or {}, system='darwin', sysctl=sysctl(**values), affinity=set(), cpu_count=8)

    def test_apple_silicon(self):
        """The performance cores (perflevel0), of the logical CPUs."""
        self.assertEqual(self.detect(hw_logicalcpu=8, hw_perflevel0_physicalcpu=6, hw_physicalcpu=8),
                         (6, 'performance', 8))
        self.assertEqual(self.detect(hw_logicalcpu=24, hw_perflevel0_physicalcpu=24, hw_physicalcpu=24),
                         (24, 'machine', 24))           # one performance level: all of them

    def test_intel(self):
        """No perflevels: the physical cores (hyperthreads not counted)."""
        self.assertEqual(self.detect(hw_logicalcpu=16, hw_physicalcpu=8), (8, 'physical', 16))
        self.assertEqual(self.detect(hw_logicalcpu=4, hw_physicalcpu=4), (4, 'machine', 4))

    def test_unreadable(self):
        """sysctl failing: os.cpu_count()."""
        self.assertEqual(self.detect(), (8, 'machine', 8))

    def test_slurm(self):
        """In a Slurm job the allocation, as Slurm counts it."""
        self.assertEqual(self.detect({'SLURM_CPUS_ON_NODE': '4'}, hw_logicalcpu=8, hw_perflevel0_physicalcpu=6),
                         (4, 'slurm', 8))


class TestLinux(unittest.TestCase):
    def setUp(self):
        self.fs = FakeRoot()
        self.addCleanup(self.fs.close)

    def detect(self, env=None, affinity=None, cpus=8):
        return machine.detect(env or {}, system='linux', root=self.fs.root, sysctl=sysctl(),
                              affinity=set(range(cpus)) if affinity is None else affinity, cpu_count=cpus)

    def test_affinity(self):
        """No topology to read: the CPUs of the affinity (all of them: the machine's)."""
        self.assertEqual(self.detect(), (8, 'machine', 8))
        self.assertEqual(self.detect(affinity={0, 1, 2}), (3, 'affinity', 8))
        self.assertEqual(machine.detect({}, system='linux', root=self.fs.root, sysctl=sysctl(), affinity=set(),
                                        cpu_count=5), (5, 'machine', 5))

    def test_smt(self):
        """One per physical core: SMT siblings counted once; within the affinity."""
        self.fs.smt([(0, 4), (1, 5), (2, 6), (3, 7)])
        self.assertEqual(self.detect(), (4, 'physical', 8))
        self.assertEqual(self.detect(affinity={0, 4, 1}), (2, 'physical', 8))
        self.assertEqual(self.detect(affinity={0, 1}), (2, 'affinity', 8))

    def test_hybrid(self):
        """A hybrid Intel CPU (/sys/devices/cpu_core/cpus): its performance cores, one per physical core."""
        self.fs.smt([(2 * i, 2 * i + 1) for i in range(6)] + [(c,) for c in range(12, 20)])
        self.fs.write('sys/devices/cpu_core/cpus', '0-11')
        self.assertEqual(self.detect(cpus=20), (6, 'performance', 20))
        self.assertEqual(self.detect(affinity=set(range(12, 20)), cpus=20), (8, 'affinity', 20))   # E-cores only

    def test_cgroup_v2(self):
        """cgroup v2's cpu.max of the process's cgroup or an ancestor, rounded up; 'max' is no quota."""
        self.fs.write('proc/self/cgroup', '0::/system.slice/run.service')
        self.fs.write('sys/fs/cgroup/system.slice/cpu.max', '250000 100000')
        self.assertEqual(self.detect(), (3, 'quota', 8))
        self.fs.write('sys/fs/cgroup/system.slice/run.service/cpu.max', '100000 100000')
        self.assertEqual(self.detect(), (1, 'quota', 8))
        for f in ('system.slice/run.service/cpu.max', 'system.slice/cpu.max'):
            self.fs.write('sys/fs/cgroup/' + f, 'max 100000')
        self.assertEqual(self.detect(), (8, 'machine', 8))
        self.fs.write('proc/self/cgroup', '0::/')            # a container's own namespace: the mount's root
        self.fs.write('sys/fs/cgroup/cpu.max', '400000 100000')
        self.assertEqual(self.detect(), (4, 'quota', 8))
        self.assertEqual(machine.cgroup_quota(self.fs.root), 4)

    def test_cgroup_v1(self):
        """cgroup v1: cpu.cfs_quota_us over cpu.cfs_period_us (-1: none)."""
        self.fs.write('proc/self/cgroup', '5:memory:/docker/ab\n4:cpu,cpuacct:/docker/ab')
        self.fs.write('sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us', '200000')
        self.fs.write('sys/fs/cgroup/cpu,cpuacct/cpu.cfs_period_us', '100000')
        self.assertEqual(self.detect(), (2, 'quota', 8))
        self.fs.write('sys/fs/cgroup/cpu,cpuacct/cpu.cfs_quota_us', '-1')
        self.assertEqual(self.detect(), (8, 'machine', 8))

    def test_slurm(self):
        """In a Slurm job: its allocation (SLURM_CPUS_PER_TASK when set), never halved for SMT; at most the
        affinity and a quota."""
        self.fs.smt([(0, 4), (1, 5), (2, 6), (3, 7)])
        self.assertEqual(self.detect({'SLURM_JOB_ID': '7', 'SLURM_CPUS_ON_NODE': '8'}), (8, 'slurm', 8))
        self.assertEqual(self.detect({'SLURM_CPUS_ON_NODE': '16', 'SLURM_CPUS_PER_TASK': '4'}, cpus=16),
                         (4, 'slurm', 16))
        self.assertEqual(self.detect({'SLURM_CPUS_ON_NODE': '4', 'SLURM_CPUS_PER_TASK': '8'}), (4, 'slurm', 8))
        self.assertEqual(self.detect({'SLURM_CPUS_ON_NODE': '16'}, affinity=set(range(6)), cpus=16), (6, 'affinity', 16))
        self.assertEqual(self.detect({'SLURM_CPUS_ON_NODE': 'x'}), (4, 'physical', 8))       # unreadable: not Slurm
        self.fs.write('proc/self/cgroup', '0::/job')
        self.fs.write('sys/fs/cgroup/job/cpu.max', '200000 100000')
        self.assertEqual(self.detect({'SLURM_CPUS_ON_NODE': '8'}), (2, 'quota', 8))


class TestLimits(unittest.TestCase):
    def test_allowed(self):
        """The most cores allowed (more asked are clamped): the affinity and SLURM_CPUS_ON_NODE, the fewer; where
        neither is known, the logical CPUs."""
        with mock.patch.object(machine, '_affinity', lambda: set(range(8))):
            self.assertEqual(machine.allowed_cores({}), 8)
            self.assertEqual(machine.allowed_cores({'SLURM_CPUS_ON_NODE': '3'}), 3)
            self.assertEqual(machine.allowed_cores({'SLURM_CPUS_ON_NODE': '30'}), 8)
        with mock.patch.object(machine, '_affinity', lambda: None):     # macOS: the logical CPUs
            self.assertEqual(machine.allowed_cores({}, cpu_count=8), 8)
            self.assertEqual(machine.allowed_cores({}), os.cpu_count())
            self.assertIsNone(machine.allowed_cores({}, cpu_count=0))
            self.assertEqual(machine.allowed_cores({'SLURM_CPUS_ON_NODE': '3'}, cpu_count=8), 3)

    def test_memory(self):
        """The machine's memory, at most Slurm's (per node, or per CPU times the CPUs; 0: the node's)."""
        with mock.patch.object(os, 'sysconf', lambda k: {'SC_PAGE_SIZE': 4096, 'SC_PHYS_PAGES': 1 << 22}[k]):
            self.assertEqual(machine.memory({}), 16 << 30)
            self.assertEqual(machine.memory({'SLURM_MEM_PER_NODE': '1024'}), 1 << 30)
            self.assertEqual(machine.memory({'SLURM_MEM_PER_NODE': '0'}), 16 << 30)
            self.assertEqual(machine.memory({'SLURM_MEM_PER_CPU': '512', 'SLURM_CPUS_ON_NODE': '4'}), 2 << 30)
            self.assertEqual(machine.memory({'SLURM_MEM_PER_NODE': 'x'}), 16 << 30)

        def fail(k):
            raise ValueError(k)
        with mock.patch.object(os, 'sysconf', fail):
            self.assertIsNone(machine.memory({}))

    def test_cpu_list(self):
        self.assertEqual(machine.cpu_list('0-3,8,10-11\n'), {0, 1, 2, 3, 8, 10, 11})
        self.assertEqual((machine.cpu_list(''), machine.cpu_list(None), machine.cpu_list('a-b')), (set(), set(), set()))


class TestWords(unittest.TestCase):
    def test_line(self):
        """The first line of a run for each detected source."""
        def line(n, source, total):
            return messages.text('cores_auto', cores=messages.plural(n, 'core'),
                                 why=messages.CORES_WHY[source].format(total=total))
        self.assertEqual(line(6, 'performance', 8),
                         'Using 6 cores (the performance cores of 8). --cores N or RANKBOUND_CORES=N to change.')
        self.assertEqual(line(64, 'slurm', 128),
                         'Using 64 cores (the Slurm allocation). --cores N or RANKBOUND_CORES=N to change.')
        self.assertEqual(line(1, 'quota', 8),
                         'Using 1 core (the cgroup CPU quota). --cores N or RANKBOUND_CORES=N to change.')
        self.assertEqual(set(messages.CORES_WHY),
                         {'slurm', 'quota', 'performance', 'physical', 'affinity', 'machine'})


if __name__ == '__main__':
    unittest.main()
