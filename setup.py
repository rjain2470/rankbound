"""Compiles the prover's f2elim and the verifier's f2rank during `pip install`, as `make build` does, so that an
install from a checkout or from GitHub (`pip install git+https://github.com/rjain2470/rankbound`) is ready to run.
The metadata is in pyproject.toml."""
import os
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAMS = ['rankbound/f2elim', 'rankbound/verify/f2rank']


class BuildPy(build_py):
    def run(self):
        for prog in PROGRAMS:
            out, src = os.path.join(HERE, prog), os.path.join(HERE, prog + '.c')
            if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
                cc = os.environ.get('CC', 'cc')
                print('compiling %s with %s' % (prog, cc))
                subprocess.check_call([cc, '-O3', '-pthread', '-o', out, src])
        super().run()


setup(cmdclass={'build_py': BuildPy})
