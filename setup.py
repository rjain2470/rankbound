"""Compiles the prover's f2elim, f2kernel and siever and the checker's check during `pip install`, as `make build` does
(the list of programs and the flags are in rankbound/native.mk, which the Makefile includes too), so that an install
from a checkout or from GitHub (`pip install git+https://github.com/rjain2470/rankbound`) is ready to run.  The compiled
programs are native code: the wheel is tagged for the platform that built it (py3-none-PLATFORM), never py3-none-any,
and the source distribution carries no binaries.  The metadata is in pyproject.toml."""
import os
import platform
import re
import subprocess
import sys

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py
try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:                      # setuptools < 70.1: the wheel package
    from wheel.bdist_wheel import bdist_wheel

HERE = os.path.dirname(os.path.abspath(__file__))


def native():
    """native.mk as a dict: NAME = value lines."""
    d = {}
    for line in open(os.path.join(HERE, 'rankbound', 'native.mk')):
        m = re.match(r'([A-Z_]+)\s*=\s*(.*?)\s*$', line)
        if m:
            d[m.group(1)] = m.group(2).split()
    return d


NATIVE = native()
PROGRAMS = NATIVE['PROGRAMS']


class BuildPy(build_py):
    def run(self):
        flags = NATIVE['OPT'] + NATIVE['WARN'] + NATIVE['THREADS']      # no -Werror: another compiler may warn
        for prog in PROGRAMS:
            out, src = os.path.join(HERE, prog), os.path.join(HERE, prog + '.c')
            if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
                cc = os.environ.get('CC', 'cc')
                print('compiling %s with %s' % (prog, cc))
                try:
                    subprocess.check_call([cc] + flags + ['-o', out, src] + NATIVE['LIBS'])
                except OSError:
                    sys.exit('rankbound needs a C compiler to build its native programs (macOS: xcode-select '
                             '--install; Debian/Ubuntu: apt install gcc). Set CC to use another; %s was not found.' % cc)
                except subprocess.CalledProcessError as e:
                    sys.exit('rankbound: %s could not compile %s.c (exit %d); its messages are above.'
                             % (cc, prog, e.returncode))
        super().run()
        for prog in PROGRAMS:            # the programs go into the wheel directly: they are not in the sdist's manifest
            self.copy_file(os.path.join(HERE, prog), os.path.join(self.build_lib, prog))


class Wheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):                   # Python-version independent, platform specific
        _, _, plat = super().get_tag()
        return 'py3', 'none', machine_tag(plat)


def machine_tag(plat, arch=None):
    """The wheel's platform tag for the programs as compiled here: a macOS Python built for both architectures
    (macosx_10_9_universal2) gives programs of this machine's architecture only (arch, default platform.machine();
    arm64 from macOS 11)."""
    m = re.fullmatch(r'macosx_(\d+)_(\d+)_universal2', plat)
    if not m:
        return plat
    arch = arch or platform.machine()
    v = (int(m.group(1)), int(m.group(2)))
    if arch == 'arm64' and v < (11, 0):
        v = (11, 0)
    return 'macosx_%d_%d_%s' % (v[0], v[1], arch)


class NativeDistribution(Distribution):
    def has_ext_modules(self):           # makes the wheel platform specific
        return True


setup(cmdclass={'build_py': BuildPy, 'bdist_wheel': Wheel}, distclass=NativeDistribution)
