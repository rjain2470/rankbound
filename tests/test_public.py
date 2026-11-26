"""The public tree says nothing internal: no plan or audit labels, development rounds or branches, private paths,
and no TODO/FIXME notes.  It reads every file git tracks (outside tests/data), so it runs in a checkout; an installed
copy has no tests.  A development checkout adds its own words from bench/public_words.txt (not in the public tree)."""
import os, re, subprocess, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = [r'/Users/', r'/home/[a-z]', r'\b[Rr]ound \d+\b', r'\b[Rr]\d-[a-z]{3,}', r'\bpub-[a-z]{3,}', r'\bnot done\b',
         r'\bTODO\b', r'\bFIXME\b', r'[a-z,] \((?:[ADFLOPWZ]\d{1,2})\)', r'\b[Dd]ecision [A-Z]?\d']
SKIP = ('tests/data/', 'bench/')
SKIP_FILES = {'tests/test_public.py'}
LOCAL = os.path.join(ROOT, 'bench', 'public_words.txt')


def local_words():
    """The development repository's own words and skipped files (bench/public_words.txt), if it is there."""
    words, skip = [], set()
    if os.path.exists(LOCAL):
        with open(LOCAL, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('skip: '):
                    skip.add(line[6:].strip())
                elif line and not line.startswith('#'):
                    words.append(line)
    return words, skip


def tracked():
    try:
        r = subprocess.run(['git', '-C', ROOT, 'ls-files', '-z'], capture_output=True)
    except OSError:                                 # no git
        return None
    return [f for f in r.stdout.decode().split('\0') if f] if r.returncode == 0 else None


class TestPublic(unittest.TestCase):
    def test_no_internal_words(self):
        files = tracked()
        if files is None:
            self.skipTest('not a git checkout')
        more, skip = local_words()
        rx = re.compile('|'.join('(?:%s)' % w for w in WORDS + more))
        found = []
        for f in files:
            if f in SKIP_FILES | skip or f.startswith(SKIP):
                continue
            try:
                with open(os.path.join(ROOT, f), encoding='utf-8') as h:
                    text = h.read()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(text.split('\n'), 1):
                found += ['%s:%d: %s' % (f, i, m.group(0)) for m in rx.finditer(line)]
        self.assertEqual(found, [], '\n'.join(found))


if __name__ == '__main__':
    unittest.main()
