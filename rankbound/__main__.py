"""The command `rankbound`: `rankbound check` runs the checker, anything else the prover's CLI."""
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['check']:
        from .check import main as check
        return check(argv[1:])
    from .cli import main as prover
    return prover(argv)


if __name__ == '__main__':
    sys.exit(main())
