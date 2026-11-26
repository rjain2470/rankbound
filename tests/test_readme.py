"""The examples of the README and docs/usage.md are what this version prints: every code block that starts with a
`$ rankbound` (or `$ printf`) command is run in order, in one temporary folder per file, and its lines are compared
with the file's, apart from what changes from run to run (times, hashes, dates, seconds, core-hours and PARI/GP's version).  The
files leave out each run's first line (its work folder and cores), and so does this test."""
import os, re, shlex, shutil, subprocess, sys, tempfile, unittest
from helpers import SRC, TESTS

ROOT = os.path.dirname(TESTS)
FILES = [os.path.join(ROOT, 'README.md'), os.path.join(ROOT, 'docs', 'usage.md')]
VARY = [(re.compile(r'^\d\d:\d\d:\d\d '), ''), (re.compile(r'\b[0-9a-f]{64}\b'), 'SHA256'),
        (re.compile(r'\b[0-9a-f]{16}\b'), 'HASH'), (re.compile(r'PARI/GP \d+\.\d+\.\d+'), 'PARI/GP VERSION'),
        (re.compile(r'\d{1,2} [A-Z][a-z]+ \d{4} at \d{1,2}:\d\d [AP]M'), 'DATE'),
        (re.compile(r'\b\d+\.\d\d core-hours\b'), 'N core-hours'), (re.compile(r'\bin \d+ seconds?\b'), 'in N seconds')]


def norm(line):
    for rx, to in VARY:
        line = rx.sub(to, line)
    return line


def examples(text):
    """[(command, [expected lines])], in the file's order, from the code blocks that start with '$ '."""
    out = []
    for block in re.findall(r'^```\n(.*?)^```', text, re.S | re.M):
        lines = block.rstrip('\n').split('\n')
        if not lines[0].startswith('$ '):
            continue
        for line in lines:
            if line.startswith('$ '):
                out.append((line[2:], []))
            else:
                out[-1][1].append(line)
    return out


@unittest.skipUnless(shutil.which('gp'), 'needs PARI/GP (the examples check their witnesses)')
class TestReadme(unittest.TestCase):
    def test_examples(self):
        for path in FILES:
            with self.subTest(os.path.relpath(path, ROOT)):
                self.run_file(path)

    def run_file(self, path):
        with open(path, encoding='utf-8') as f:
            runs = examples(f.read())
        self.assertGreaterEqual(len(runs), 2)
        d = tempfile.mkdtemp(prefix='rb_readme')
        self.addCleanup(shutil.rmtree, d, True)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('SLURM_', 'RANKBOUND_'))}
        env.update(PYTHONPATH=SRC, RANKBOUND_CORES='1')
        for cmd, want in runs:
            if not want and cmd.startswith('rankbound '):
                continue                    # a command shown without its output
            if cmd.startswith('rankbound '):
                argv = [sys.executable, '-m', 'rankbound'] + shlex.split(cmd)[1:]
                r = subprocess.run(argv, cwd=d, env=env, capture_output=True, text=True, timeout=600)
                got = [l for l in (r.stderr + r.stdout).split('\n') if l and ' Work folder ' not in l[:24]]
                self.assertEqual(r.returncode, 0, cmd + '\n' + r.stderr)
                self.assertEqual([norm(l) for l in got], [norm(l) for l in want], cmd)
            else:
                r = subprocess.run(cmd, shell=True, cwd=d, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, cmd)


if __name__ == '__main__':
    unittest.main()
