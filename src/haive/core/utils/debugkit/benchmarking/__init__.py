"""
Benchmarking utilities submodule.

This submodule provides benchmarking and performance measurement
capabilities including timing, load testing, and comparison utilities.
"""

try:
    from haive.core.utils.debugkit.benchmarking.core import BenchmarkSuite
    from haive.core.utils.debugkit.benchmarking.load import LoadTester
    from haive.core.utils.debugkit.benchmarking.timing import TimingBenchmark
except ImportError:
    from haive.core.utils.debugkit.fallbacks import FallbackBenchmark as BenchmarkSuite

    TimingBenchmark = None
    LoadTester = None

# Create default benchmark instance
benchmark = BenchmarkSuite()

__all__ = [
    "BenchmarkSuite",
    "LoadTester",
    "TimingBenchmark",
    "benchmark",
]
