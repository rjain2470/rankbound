"""The command `rankbound`: `rankbound verify` goes to the verifier's command line, everything else to the
prover's."""
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['verify']:
        from .verify.__main__ import main as verify
        return verify(argv[1:])
    from .cli import main as prover
    return prover(argv)


if __name__ == '__main__':
    sys.exit(main())
