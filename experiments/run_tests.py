"""Deterministic test harness; suppress Windows working-set eviction in tests only.

The OS call forcibly pages out imported libraries hundreds of times in tiny
fixtures. It is not a numerical operation. Garbage collection and every
scientific test/assertion remain active; production execution is unchanged.
"""
import os
import sys
import unittest

from .common import ROOT, load_config, seed_execution


def main():
    os.chdir(ROOT)
    seed_execution(load_config(ROOT / "configs/rack_kv_v1.yaml"))
    if os.name == "nt":
        import ctypes
        ctypes.windll.psapi.EmptyWorkingSet = lambda _handle: 1
        print("Test-only OS shim: EmptyWorkingSet suppressed; scientific functions and gc.collect unchanged.", flush=True)
    pattern = "test_reproduce_v1.py" if "--invariants" in sys.argv else "test_*.py"
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
