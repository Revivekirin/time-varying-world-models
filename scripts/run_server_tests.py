"""File-backed, spawn-safe unit-test gate; skipped tests fail server validation."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        print('Server gate requires all tests passing without skips', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
