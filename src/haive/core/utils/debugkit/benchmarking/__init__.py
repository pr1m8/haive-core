"""
Benchmarking utilities submodule.

This submodule provides benchmarking and performance measurement
capabilities including timing, load testing, and comparison utilities.
"""

try:
    from .core import BenchmarkSuite
    from .load import LoadTester
    from .timing import TimingBenchmark
except ImportError:
    from ..fallbacks import FallbackBenchmark as BenchmarkSuite

    TimingBenchmark = None
    LoadTester = None

# Create default benchmark instance
benchmark = BenchmarkSuite()

__all__ = [
    "benchmark",
    "BenchmarkSuite",
    "TimingBenchmark",
    "LoadTester",
]
